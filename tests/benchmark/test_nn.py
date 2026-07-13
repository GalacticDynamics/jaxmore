"""Benchmarks for `jaxmore.nn`.

The optimizer-passing strategies (see README) are plumbing, not hot paths: the
differences between them are dominated by the forward/backward pass. These
benchmarks exist to keep that claim honest rather than to pick a winner.

Each benchmark measures a *warmed* (already-traced) training run, so we time
execution rather than compilation.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from jaxtyping import Array

from jaxmore.nn import AbstractScanNNTrainer, masked_mean

N_SAMPLES = 128
BATCH_SIZE = 16
NUM_EPOCHS = 5


def _mlp(key: Array) -> eqx.nn.MLP:
    return eqx.nn.MLP(
        in_size=2, out_size=1, width_size=32, depth=2, activation=jax.nn.relu, key=key
    )


def _masked_mse(model: Any, X_batch: Array, y_batch: Array, batch_mask: Array) -> Array:
    preds = jax.vmap(model)(X_batch).squeeze(-1)
    return masked_mean((preds - y_batch) ** 2, batch_mask)


def _data() -> tuple[Array, Array]:
    key = jr.key(0)
    X = jr.normal(key, (N_SAMPLES, 2))
    y = 2 * X[:, 0] + X[:, 1]
    return X, y


class _Trainer(AbstractScanNNTrainer):
    """Optimizer lives outside the carry."""

    def init(self, *, key: Array, X: Array, y: Array) -> tuple[tuple, tuple]:
        model_key, data_key = jr.split(key)
        model = _mlp(model_key)
        opt_state = optax.adam(1e-2).init(eqx.filter(model, eqx.is_array))
        return (model, opt_state, jr.fold_in(data_key, 0)), (
            jnp.ones(len(X), dtype=bool),
            (X, y),
        )

    def pack_carry_state(self, carry: tuple) -> tuple[tuple, dict]:
        model, opt_state, key = carry
        model_dyn, model_static = eqx.partition(model, eqx.is_array)
        return (model_dyn, opt_state, key), {"model_static": model_static}

    def unpack_carry_state(self, carry: tuple, static: dict) -> tuple:
        model_dyn, opt_state, key = carry
        return (eqx.combine(model_dyn, static["model_static"]), opt_state, key)


class _TrainerWithOptimizer(AbstractScanNNTrainer):
    """Optimizer travels in the carry (static side)."""

    def init(
        self, *, key: Array, X: Array, y: Array, optimizer: optax.GradientTransformation
    ) -> tuple[tuple, tuple]:
        model_key, data_key = jr.split(key)
        model = _mlp(model_key)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        return (model, opt_state, optimizer, jr.fold_in(data_key, 0)), (
            jnp.ones(len(X), dtype=bool),
            (X, y),
        )

    def pack_carry_state(self, carry: tuple) -> tuple[tuple, dict]:
        model, opt_state, optimizer, key = carry
        model_dyn, model_static = eqx.partition(model, eqx.is_array)
        return (
            (model_dyn, opt_state, key),
            {"model_static": model_static, "optimizer": optimizer},
        )

    def unpack_carry_state(self, carry: tuple, static: dict) -> tuple:
        model_dyn, opt_state, key = carry
        model = eqx.combine(model_dyn, static["model_static"])
        return (model, opt_state, static["optimizer"], key)


def _step_closure(carry: tuple, batch_inputs: tuple) -> tuple[Array, tuple]:
    model, opt_state, key = carry
    batch_mask, (X_batch, y_batch) = batch_inputs
    loss, grads = eqx.filter_value_and_grad(_masked_mse)(
        model, X_batch, y_batch, batch_mask
    )
    updates, opt_state = optax.adam(1e-2).update(grads, opt_state)
    return loss, (eqx.apply_updates(model, updates), opt_state, key)


def _step_carry(carry: tuple, batch_inputs: tuple) -> tuple[Array, tuple]:
    model, opt_state, optimizer, key = carry
    batch_mask, (X_batch, y_batch) = batch_inputs
    loss, grads = eqx.filter_value_and_grad(_masked_mse)(
        model, X_batch, y_batch, batch_mask
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    return loss, (eqx.apply_updates(model, updates), opt_state, optimizer, key)


def _step_kw(
    carry: tuple, batch_inputs: tuple, *, optimizer: optax.GradientTransformation
) -> tuple[Array, tuple]:
    model, opt_state, key = carry
    batch_mask, (X_batch, y_batch) = batch_inputs
    loss, grads = eqx.filter_value_and_grad(_masked_mse)(
        model, X_batch, y_batch, batch_mask
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    return loss, (eqx.apply_updates(model, updates), opt_state, key)


@pytest.mark.benchmark
def test_bench_optimizer_in_closure(benchmark) -> None:
    """Option 1: optimizer captured in a closure."""
    X, y = _data()
    trainer = _Trainer(make_step=_step_closure, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=X, y=y)

    run = eqx.filter_jit(
        lambda c, d: trainer.run(
            c, d, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, key=jr.key(1)
        )
    )
    jax.block_until_ready(run(carry, epoch_data))  # warm up (compile)

    benchmark(lambda: jax.block_until_ready(run(carry, epoch_data)))


@pytest.mark.benchmark
def test_bench_optimizer_in_carry(benchmark) -> None:
    """Option 2: optimizer carried in the (static) carry state."""
    X, y = _data()
    trainer = _TrainerWithOptimizer(make_step=_step_carry, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(
        key=jr.key(0), X=X, y=y, optimizer=optax.adam(1e-2)
    )

    run = eqx.filter_jit(
        lambda c, d: trainer.run(
            c, d, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, key=jr.key(1)
        )
    )
    jax.block_until_ready(run(carry, epoch_data))

    benchmark(lambda: jax.block_until_ready(run(carry, epoch_data)))


@pytest.mark.benchmark
def test_bench_optimizer_via_step_kw(benchmark) -> None:
    """Option 3: optimizer forwarded through `run(step_kw=...)`."""
    X, y = _data()
    trainer = _Trainer(make_step=_step_kw, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=X, y=y)

    run = eqx.filter_jit(
        lambda c, d: trainer.run(
            c,
            d,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            key=jr.key(1),
            step_kw={"optimizer": optax.adam(1e-2)},
        )
    )
    jax.block_until_ready(run(carry, epoch_data))

    benchmark(lambda: jax.block_until_ready(run(carry, epoch_data)))


@pytest.mark.benchmark
def test_bench_empty_batch_skipping(benchmark) -> None:
    """Mostly-ignorable data: the empty-batch skip should keep this cheap."""
    X, y = _data()
    mask = jnp.arange(len(X)) < 32  # 32 usable of 128 -> several empty batches

    trainer = _Trainer(make_step=_step_closure, loss_agg_fn=masked_mean)
    carry, _ = trainer.init(key=jr.key(0), X=X, y=y)
    epoch_data = (mask, (X, y))

    run = eqx.filter_jit(
        lambda c, d: trainer.run(
            c, d, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, key=jr.key(1)
        )
    )
    jax.block_until_ready(run(carry, epoch_data))

    benchmark(lambda: jax.block_until_ready(run(carry, epoch_data)))
