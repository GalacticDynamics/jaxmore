"""End-to-end usage tests for `jaxmore.nn`.

These mirror the patterns documented in the README: a full Equinox + Optax
training run, and the three ways of getting an optimizer into `make_step`.
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

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def data() -> tuple[Array, Array]:
    """Build a simple linear regression problem: y = 2*x0 + x1 + noise."""
    key = jr.key(0)
    X = jr.normal(key, (100, 2))
    y = 2 * X[:, 0] + X[:, 1] + 0.1 * jr.normal(jr.fold_in(key, 1), (100,))
    return X, y


def _mlp(key: Array, in_size: int) -> eqx.nn.MLP:
    return eqx.nn.MLP(
        in_size=in_size,
        out_size=1,
        width_size=32,
        depth=2,
        activation=jax.nn.relu,
        key=key,
    )


def _masked_mse(model: Any, X_batch: Array, y_batch: Array, batch_mask: Array) -> Array:
    """MSE over the *real* rows of the batch only."""
    preds = jax.vmap(model)(X_batch).squeeze(-1)
    return masked_mean((preds - y_batch) ** 2, batch_mask)


# ============================================================================
# The canonical trainer (optimizer in a closure) -- README's flagship example
# ============================================================================


class NNTrainer(AbstractScanNNTrainer):
    """Trainer for a small feed-forward regressor."""

    def init(
        self, *, key: Array, X: Array, y: Array, mask: Array | None = None
    ) -> tuple[tuple, tuple]:
        model_key, data_key = jr.split(key)
        model = _mlp(model_key, X.shape[1])
        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        initial_carry = (model, opt_state, jr.fold_in(data_key, 0))

        if mask is None:
            mask = jnp.ones(len(X), dtype=bool)
        return initial_carry, (mask, (X, y))

    def pack_carry_state(self, carry: tuple) -> tuple[tuple, dict]:
        model, opt_state, key = carry
        model_dyn, model_static = eqx.partition(model, eqx.is_array)
        return (model_dyn, opt_state, key), {"model_static": model_static}

    def unpack_carry_state(self, carry: tuple, static: dict) -> tuple:
        model_dyn, opt_state, key = carry
        model = eqx.combine(model_dyn, static["model_static"])
        return (model, opt_state, key)


def test_end_to_end_training_reduces_loss(data: tuple[Array, Array]) -> None:
    """A full training run converges and leaves a usable model behind."""
    X, y = data
    optimizer = optax.adam(1e-2)

    def make_step(carry: tuple, batch_inputs: tuple) -> tuple[Array, tuple]:
        model, opt_state, key = carry
        batch_mask, (X_batch, y_batch) = batch_inputs

        loss, grads = eqx.filter_value_and_grad(_masked_mse)(
            model, X_batch, y_batch, batch_mask
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        return loss, (eqx.apply_updates(model, updates), opt_state, key)

    trainer = NNTrainer(make_step=make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=X, y=y)
    final_carry, losses = trainer.run(
        carry, epoch_data, num_epochs=20, batch_size=16, key=jr.key(1)
    )

    assert losses.shape == (20,)
    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < losses[0]

    # The final carry is a *usable* model, not a partitioned husk.
    model, _, _ = final_carry
    preds = jax.vmap(model)(X).squeeze(-1)
    assert preds.shape == y.shape
    assert jnp.all(jnp.isfinite(preds))


def test_training_survives_mostly_ignorable_data(data: tuple[Array, Array]) -> None:
    """Regression (B3 + B4): ignorable samples cluster into all-False batches.

    `shuffle_and_batch` sorts usable samples first, so with 40 usable of 100 and
    `batch_size=16` the trailing batches contain *no* usable rows at all. A
    mask-aware loss then calls `masked_mean` with an empty mask.

    Before the fix this produced NaN gradients (eager) and burned a full
    forward/backward pass on padding. Nothing here may be NaN.
    """
    X, y = data
    mask = jnp.arange(len(X)) < 40  # 40 usable, 60 ignorable
    optimizer = optax.adam(1e-2)

    def make_step(carry: tuple, batch_inputs: tuple) -> tuple[Array, tuple]:
        model, opt_state, key = carry
        batch_mask, (X_batch, y_batch) = batch_inputs

        loss, grads = eqx.filter_value_and_grad(_masked_mse)(
            model, X_batch, y_batch, batch_mask
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        return loss, (eqx.apply_updates(model, updates), opt_state, key)

    trainer = NNTrainer(make_step=make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=X, y=y, mask=mask)
    final_carry, losses = trainer.run(
        carry, epoch_data, num_epochs=10, batch_size=16, key=jr.key(1)
    )

    assert jnp.all(jnp.isfinite(losses)), f"NaN epoch loss: {losses}"
    assert losses[-1] < losses[0]

    model, _, _ = final_carry
    weights = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    assert not any(bool(jnp.any(jnp.isnan(w))) for w in weights), "NaN in weights"


def test_unmasked_loss_trains_on_padding(data: tuple[Array, Array]) -> None:
    """Demonstrate *why* the mask matters, by pinning the wrong behaviour.

    An unmasked loss sees the zero-padded rows as real `(0, 0) -> 0` samples.
    With N=100 and batch_size=16 there are 112 slots, i.e. 12 fabricated rows.
    This test documents the trap the README warns about.
    """
    X, y = data
    optimizer = optax.adam(1e-2)

    def make_step_unmasked(carry: tuple, batch_inputs: tuple) -> tuple[Array, tuple]:
        model, opt_state, key = carry
        _batch_mask, (X_batch, y_batch) = batch_inputs

        def loss_fn(m: Any) -> Array:
            preds = jax.vmap(m)(X_batch).squeeze(-1)
            return jnp.mean((preds - y_batch) ** 2)  # <- ignores the mask

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        return loss, (eqx.apply_updates(model, updates), opt_state, key)

    def make_step_masked(carry: tuple, batch_inputs: tuple) -> tuple[Array, tuple]:
        model, opt_state, key = carry
        batch_mask, (X_batch, y_batch) = batch_inputs

        loss, grads = eqx.filter_value_and_grad(_masked_mse)(
            model, X_batch, y_batch, batch_mask
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        return loss, (eqx.apply_updates(model, updates), opt_state, key)

    def train(make_step: Any) -> Any:
        trainer = NNTrainer(make_step=make_step, loss_agg_fn=masked_mean)
        carry, epoch_data = trainer.init(key=jr.key(0), X=X, y=y)
        final_carry, _ = trainer.run(
            carry, epoch_data, num_epochs=20, batch_size=16, key=jr.key(1)
        )
        return final_carry[0]

    # Evaluate both models on the *true* data, with a clean masked metric.
    full_mask = jnp.ones(len(X), dtype=bool)
    err_unmasked = _masked_mse(train(make_step_unmasked), X, y, full_mask)
    err_masked = _masked_mse(train(make_step_masked), X, y, full_mask)

    # Training on 12 fabricated rows measurably hurts the fit.
    assert err_masked < err_unmasked


# ============================================================================
# Optimizer passing strategies (README: three patterns)
# ============================================================================


class TrainerWithOptimizer(AbstractScanNNTrainer):
    """Carries the optimizer as part of the training state (README Option 2)."""

    def init(
        self, *, key: Array, X: Array, y: Array, optimizer: optax.GradientTransformation
    ) -> tuple[tuple, tuple]:
        model_key, data_key = jr.split(key)
        model = _mlp(model_key, X.shape[1])
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        initial_carry = (model, opt_state, optimizer, jr.fold_in(data_key, 0))
        return initial_carry, (jnp.ones(len(X), dtype=bool), (X, y))

    def pack_carry_state(self, carry: tuple) -> tuple[tuple, dict]:
        model, opt_state, optimizer, key = carry
        model_dyn, model_static = eqx.partition(model, eqx.is_array)
        # An optax GradientTransformation is functions, not arrays -> static.
        return (
            (model_dyn, opt_state, key),
            {"model_static": model_static, "optimizer": optimizer},
        )

    def unpack_carry_state(self, carry: tuple, static: dict) -> tuple:
        model_dyn, opt_state, key = carry
        model = eqx.combine(model_dyn, static["model_static"])
        return (model, opt_state, static["optimizer"], key)


def test_optimizer_in_closure(data: tuple[Array, Array]) -> None:
    """Option 1: `make_step` closes over the optimizer."""
    X, y = data
    optimizer = optax.adam(1e-2)

    def make_step(carry: tuple, batch_inputs: tuple) -> tuple[Array, tuple]:
        model, opt_state, key = carry
        batch_mask, (X_batch, y_batch) = batch_inputs
        loss, grads = eqx.filter_value_and_grad(_masked_mse)(
            model, X_batch, y_batch, batch_mask
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        return loss, (eqx.apply_updates(model, updates), opt_state, key)

    trainer = NNTrainer(make_step=make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=X, y=y)
    _, losses = trainer.run(
        carry, epoch_data, num_epochs=10, batch_size=16, key=jr.key(1)
    )

    assert losses[-1] < losses[0]


@pytest.mark.parametrize("opt_name", ["adam", "sgd", "adamw"])
def test_optimizer_in_carry(data: tuple[Array, Array], opt_name: str) -> None:
    """Option 2: one trainer, any optimizer -- only `init()` changes."""
    X, y = data
    optimizer = {
        "adam": optax.adam(1e-2),
        "sgd": optax.sgd(1e-2),
        "adamw": optax.adamw(1e-2),
    }[opt_name]

    def make_step(carry: tuple, batch_inputs: tuple) -> tuple[Array, tuple]:
        model, opt_state, optimizer, key = carry
        batch_mask, (X_batch, y_batch) = batch_inputs
        loss, grads = eqx.filter_value_and_grad(_masked_mse)(
            model, X_batch, y_batch, batch_mask
        )
        # `adamw` needs the current params; others ignore them.
        params = eqx.filter(model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return loss, (eqx.apply_updates(model, updates), opt_state, optimizer, key)

    trainer = TrainerWithOptimizer(make_step=make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=X, y=y, optimizer=optimizer)
    _, losses = trainer.run(
        carry, epoch_data, num_epochs=10, batch_size=16, key=jr.key(1)
    )

    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < losses[0]


def test_optimizer_via_step_kw(data: tuple[Array, Array]) -> None:
    """Option 3: the optimizer is forwarded through `run(step_kw=...)`."""
    X, y = data

    def make_step(
        carry: tuple, batch_inputs: tuple, *, optimizer: optax.GradientTransformation
    ) -> tuple[Array, tuple]:
        model, opt_state, key = carry
        batch_mask, (X_batch, y_batch) = batch_inputs
        loss, grads = eqx.filter_value_and_grad(_masked_mse)(
            model, X_batch, y_batch, batch_mask
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        return loss, (eqx.apply_updates(model, updates), opt_state, key)

    trainer = NNTrainer(make_step=make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=X, y=y)
    _, losses = trainer.run(
        carry,
        epoch_data,
        num_epochs=10,
        batch_size=16,
        key=jr.key(1),
        step_kw={"optimizer": optax.adam(1e-2)},
    )

    assert losses[-1] < losses[0]


def test_all_three_strategies_agree(data: tuple[Array, Array]) -> None:
    """The three patterns are plumbing: given the same optimizer they agree."""
    X, y = data

    def closure_step(carry: tuple, batch_inputs: tuple) -> tuple[Array, tuple]:
        model, opt_state, key = carry
        batch_mask, (X_batch, y_batch) = batch_inputs
        loss, grads = eqx.filter_value_and_grad(_masked_mse)(
            model, X_batch, y_batch, batch_mask
        )
        updates, opt_state = optax.adam(1e-2).update(grads, opt_state)
        return loss, (eqx.apply_updates(model, updates), opt_state, key)

    def kw_step(
        carry: tuple, batch_inputs: tuple, *, optimizer: optax.GradientTransformation
    ) -> tuple[Array, tuple]:
        model, opt_state, key = carry
        batch_mask, (X_batch, y_batch) = batch_inputs
        loss, grads = eqx.filter_value_and_grad(_masked_mse)(
            model, X_batch, y_batch, batch_mask
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        return loss, (eqx.apply_updates(model, updates), opt_state, key)

    kw = {"num_epochs": 5, "batch_size": 16, "key": jr.key(1)}

    t1 = NNTrainer(make_step=closure_step, loss_agg_fn=masked_mean)
    c1, d1 = t1.init(key=jr.key(0), X=X, y=y)
    _, losses_closure = t1.run(c1, d1, **kw)

    t3 = NNTrainer(make_step=kw_step, loss_agg_fn=masked_mean)
    c3, d3 = t3.init(key=jr.key(0), X=X, y=y)
    _, losses_kw = t3.run(c3, d3, step_kw={"optimizer": optax.adam(1e-2)}, **kw)

    assert jnp.allclose(losses_closure, losses_kw, rtol=1e-5)


# ============================================================================
# Per-epoch hooks (the patterns `phasecurvefit` needs)
# ============================================================================


def test_scheduled_loss_weight(data: tuple[Array, Array]) -> None:
    """Ramp a loss weight across epochs via `prepare_step_kw`.

    This is `phasecurvefit`'s `lambda_p` schedule: a velocity-alignment weight
    interpolated from `lam_min` to `lam_max` over training.
    """
    X, y = data
    lam_min, lam_max, num_epochs = 1.0, 5.0, 8
    optimizer = optax.adam(1e-2)

    class RampTrainer(NNTrainer):
        def prepare_step_kw(
            self, /, *, epoch_idx: Array, num_epochs: int, epoch_key: Array
        ) -> dict[str, Any]:
            del epoch_key
            frac = epoch_idx / (num_epochs - 1) if num_epochs > 1 else 0.0
            return {"lam": lam_min + (lam_max - lam_min) * frac}

    seen: list[float] = []

    def make_step(
        carry: tuple, batch_inputs: tuple, *, lam: Array
    ) -> tuple[Array, tuple]:
        model, opt_state, key = carry
        batch_mask, (X_batch, y_batch) = batch_inputs
        jax.debug.callback(lambda v: seen.append(float(v)), lam)

        def loss_fn(m: Any) -> Array:
            return lam * _masked_mse(m, X_batch, y_batch, batch_mask)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        return loss, (eqx.apply_updates(model, updates), opt_state, key)

    trainer = RampTrainer(make_step=make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=X, y=y)
    _, losses = trainer.run(
        carry, epoch_data, num_epochs=num_epochs, batch_size=16, key=jr.key(1)
    )
    jax.block_until_ready(losses)

    # 100 samples / batch 16 -> 7 batches per epoch; take the first of each.
    per_epoch = seen[::7]
    expected = [
        lam_min + (lam_max - lam_min) * i / (num_epochs - 1) for i in range(num_epochs)
    ]
    assert per_epoch == pytest.approx(expected, rel=1e-5)


def test_resampled_negatives_each_epoch(data: tuple[Array, Array]) -> None:
    """Draw fresh random negatives every epoch via `prepare_data_args`.

    This is `phasecurvefit`'s `order_net` pattern: `random_ws` is resampled
    uniformly within the data bounds once per epoch, then batched alongside the
    real (ordered) samples.
    """
    X, y = data
    optimizer = optax.adam(1e-2)
    lo, hi = X.min(axis=0), X.max(axis=0)

    class NegativesTrainer(NNTrainer):
        def prepare_data_args(
            self,
            carry: tuple,
            data_args: tuple,
            /,
            *,
            epoch_idx: Array,
            num_epochs: int,
            epoch_key: Array,
        ) -> tuple[Array, ...]:
            del carry, epoch_idx, num_epochs
            X_real, y_real = data_args
            negatives = jr.uniform(epoch_key, shape=X_real.shape, minval=lo, maxval=hi)
            return (X_real, y_real, negatives)

    def make_step(carry: tuple, batch_inputs: tuple) -> tuple[Array, tuple]:
        model, opt_state, key = carry
        batch_mask, (X_batch, y_batch, X_neg) = batch_inputs

        def loss_fn(m: Any) -> Array:
            fit = _masked_mse(m, X_batch, y_batch, batch_mask)
            # Push predictions on the negatives toward zero.
            neg = masked_mean(jax.vmap(m)(X_neg).squeeze(-1) ** 2, batch_mask)
            return fit + 0.01 * neg

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        return loss, (eqx.apply_updates(model, updates), opt_state, key)

    trainer = NegativesTrainer(make_step=make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=X, y=y)
    _, losses = trainer.run(
        carry, epoch_data, num_epochs=15, batch_size=16, key=jr.key(1)
    )

    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < losses[0]


# ============================================================================
# Freezing part of the model (non-trivial filter_spec)
# ============================================================================


class _Encoder(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key: Array) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=2,
            out_size=4,
            width_size=8,
            depth=1,
            activation=jax.nn.relu,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        return self.mlp(x)


class _AutoEncoder(eqx.Module):
    encoder: _Encoder
    decoder: eqx.nn.MLP

    def __init__(self, key: Array) -> None:
        ekey, dkey = jr.split(key)
        self.encoder = _Encoder(ekey)
        self.decoder = eqx.nn.MLP(
            in_size=4,
            out_size=1,
            width_size=8,
            depth=1,
            activation=jax.nn.relu,
            key=dkey,
        )

    def __call__(self, x: Array) -> Array:
        return self.decoder(self.encoder(x))


def _trainable_spec(model: _AutoEncoder) -> Any:
    """Every array is trainable *except* those under the encoder subtree.

    `eqx.is_array` alone would make the whole model trainable. Building the spec
    by hand and then zeroing out one subtree with `eqx.tree_at` is how you freeze
    part of a model.
    """
    spec = jax.tree_util.tree_map(eqx.is_array, model)
    encoder_frozen = jax.tree_util.tree_map(lambda _: False, model.encoder)
    return eqx.tree_at(lambda m: m.encoder, spec, encoder_frozen)


class FrozenEncoderTrainer(AbstractScanNNTrainer):
    """Trains the decoder only; the encoder is held fixed.

    Frozen arrays land in the *static* half of the partition, so the optimizer
    never sees them and `scan` carries them as constants.
    """

    def init(self, *, key: Array, X: Array, y: Array) -> tuple[tuple, tuple]:
        model_key, data_key = jr.split(key)
        model = _AutoEncoder(model_key)

        # The optimizer is initialized on the *trainable* leaves only, so its
        # state has exactly the shape of the gradients `make_step` will hand it.
        trainable, _ = eqx.partition(model, _trainable_spec(model))
        opt_state = optax.adam(1e-2).init(trainable)

        initial_carry = (model, opt_state, jr.fold_in(data_key, 0))
        return initial_carry, (jnp.ones(len(X), dtype=bool), (X, y))

    def pack_carry_state(self, carry: tuple) -> tuple[tuple, dict]:
        model, opt_state, key = carry
        model_dyn, model_static = eqx.partition(model, _trainable_spec(model))
        return (model_dyn, opt_state, key), {"model_static": model_static}

    def unpack_carry_state(self, carry: tuple, static: dict) -> tuple:
        model_dyn, opt_state, key = carry
        model = eqx.combine(model_dyn, static["model_static"])
        return (model, opt_state, key)


def test_frozen_encoder_is_not_updated(data: tuple[Array, Array]) -> None:
    """A frozen subtree must come out of training bit-for-bit unchanged."""
    X, y = data
    optimizer = optax.adam(1e-2)

    def make_step(carry: tuple, batch_inputs: tuple) -> tuple[Array, tuple]:
        model, opt_state, key = carry
        batch_mask, (X_batch, y_batch) = batch_inputs

        # Differentiate w.r.t. the trainable half only. Taking the gradient of
        # the *whole* model would produce encoder gradients that `opt_state`
        # has no slot for.
        trainable, frozen = eqx.partition(model, _trainable_spec(model))

        def loss_fn(t: Any) -> Array:
            return _masked_mse(eqx.combine(t, frozen), X_batch, y_batch, batch_mask)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
        updates, opt_state = optimizer.update(grads, opt_state)
        trainable = eqx.apply_updates(trainable, updates)

        return loss, (eqx.combine(trainable, frozen), opt_state, key)

    trainer = FrozenEncoderTrainer(make_step=make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=X, y=y)
    model_before = carry[0]

    final_carry, losses = trainer.run(
        carry, epoch_data, num_epochs=20, batch_size=16, key=jr.key(1)
    )
    model_after = final_carry[0]

    # The encoder is untouched...
    enc_before = jax.tree_util.tree_leaves(
        eqx.filter(model_before.encoder, eqx.is_array)
    )
    enc_after = jax.tree_util.tree_leaves(eqx.filter(model_after.encoder, eqx.is_array))
    for b, a in zip(enc_before, enc_after, strict=True):
        assert jnp.array_equal(b, a), "frozen encoder weights changed"

    # ...while the decoder did train.
    dec_before = jax.tree_util.tree_leaves(
        eqx.filter(model_before.decoder, eqx.is_array)
    )
    dec_after = jax.tree_util.tree_leaves(eqx.filter(model_after.decoder, eqx.is_array))
    assert any(
        not jnp.array_equal(b, a) for b, a in zip(dec_before, dec_after, strict=True)
    ), "decoder did not train"

    assert losses[-1] < losses[0]
