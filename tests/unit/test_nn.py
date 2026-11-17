"""Unit tests for `jaxmore.nn`."""

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array, TypeCheckError

from jaxmore.nn import AbstractScanNNTrainer, masked_mean, shuffle_and_batch

# ============================================================================
# masked_mean
# ============================================================================


def test_masked_mean() -> None:
    """Mean is taken over the True entries only."""
    arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = jnp.array([True, True, False, True, False])
    expected = (1.0 + 2.0 + 4.0) / 3.0
    assert jnp.allclose(masked_mean(arr, mask), expected)


def test_masked_mean_all_true_matches_plain_mean() -> None:
    """With an all-True mask this is just `jnp.mean`."""
    arr = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask = jnp.ones(4, dtype=bool)
    assert jnp.allclose(masked_mean(arr, mask), jnp.mean(arr))


def test_masked_mean_empty_mask_is_nan() -> None:
    """An empty mask has no well-defined mean, so it is NaN."""
    arr = jnp.array([1.0, 2.0, 3.0])
    assert jnp.isnan(masked_mean(arr, jnp.zeros(3, dtype=bool)))


def test_masked_mean_empty_mask_has_finite_gradient() -> None:
    """Regression: an empty mask must not produce a NaN *gradient*.

    The naive ``jnp.where(count > 0, total / count, jnp.nan)`` evaluates
    ``total / count`` on both branches, so ``count == 0`` computes ``0 / 0``;
    `jnp.where`'s VJP then multiplies that branch's ``inf`` derivative by zero
    and yields NaN. `masked_mean` clamps the denominator to avoid this.

    This is reachable in real training: `shuffle_and_batch` groups ignorable
    samples together, so a batch can contain no usable samples at all.
    """
    arr = jnp.array([1.0, 2.0, 3.0])
    grad = jax.grad(masked_mean)(arr, jnp.zeros(3, dtype=bool))

    assert not jnp.any(jnp.isnan(grad)), f"NaN gradient on empty mask: {grad}"
    assert jnp.allclose(grad, 0.0)


def test_masked_mean_gradient_ignores_masked_entries() -> None:
    """Gradient flows only to the unmasked entries."""
    arr = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask = jnp.array([True, False, True, False])
    grad = jax.grad(masked_mean)(arr, mask)

    assert jnp.allclose(grad, jnp.array([0.5, 0.0, 0.5, 0.0]))


# ============================================================================
# shuffle_and_batch
# ============================================================================


def test_shuffle_and_batch_shapes() -> None:
    """Arrays are padded up to a whole number of batches."""
    n = 10
    mask = jnp.array([True] * 7 + [False] * 3)
    x = jnp.arange(n, dtype=jnp.float32)
    y = jnp.arange(n, 2 * n, dtype=jnp.float32)

    batch_mask, (batch_x, batch_y) = shuffle_and_batch(
        mask, x, y, key=jr.key(42), batch_size=4
    )

    assert batch_mask.shape == (3, 4)  # ceil(10 / 4) == 3
    assert batch_x.shape == (3, 4)
    assert batch_y.shape == (3, 4)


def test_shuffle_and_batch_preserves_usable_count() -> None:
    """Every usable sample survives shuffling; padding is never marked usable."""
    mask = jnp.array([True] * 7 + [False] * 3)
    x = jnp.arange(10, dtype=jnp.float32)

    batch_mask, (batch_x,) = shuffle_and_batch(mask, x, key=jr.key(0), batch_size=4)

    assert int(jnp.sum(batch_mask)) == 7
    # The usable rows are exactly the original usable values, up to permutation.
    assert set(map(float, batch_x[batch_mask])) == set(map(float, x[mask]))


def test_shuffle_and_batch_pads_with_pad_value() -> None:
    """Padded rows carry `pad_value` and are masked out."""
    mask = jnp.ones(5, dtype=bool)
    x = jnp.arange(5, dtype=jnp.float32)

    batch_mask, (batch_x,) = shuffle_and_batch(
        mask, x, key=jr.key(0), batch_size=4, pad_value=-999.0
    )

    assert batch_mask.shape == (2, 4)
    assert int(jnp.sum(batch_mask)) == 5  # 3 padded slots
    assert jnp.all(batch_x[~batch_mask] == -999.0)


def test_shuffle_and_batch_ignorable_samples_cluster_into_empty_batches() -> None:
    """Ignorable samples sort to the end, so whole batches can be all-False.

    This documents the behaviour that makes `masked_mean`'s empty-mask handling
    and the trainer's empty-batch skip necessary.
    """
    n, usable = 100, 40
    mask = jnp.arange(n) < usable
    x = jnp.zeros((n, 2))

    batch_mask, _ = shuffle_and_batch(mask, x, key=jr.key(1), batch_size=16)

    usable_per_batch = jnp.sum(batch_mask, axis=1)
    assert int(jnp.sum(usable_per_batch == 0)) > 0, "expected some all-False batches"
    assert int(jnp.sum(usable_per_batch)) == usable


def test_shuffle_and_batch_is_deterministic_given_key() -> None:
    """Same key, same batching."""
    mask = jnp.ones(10, dtype=bool)
    x = jnp.arange(10.0)

    m1, (x1,) = shuffle_and_batch(mask, x, key=jr.key(3), batch_size=4)
    m2, (x2,) = shuffle_and_batch(mask, x, key=jr.key(3), batch_size=4)

    assert jnp.array_equal(m1, m2)
    assert jnp.array_equal(x1, x2)


def test_shuffle_and_batch_rejects_mismatched_leading_dim() -> None:
    """A wrong-length arg fails fast, not with an opaque shape error in scan.

    Under this repo's pytest config jaxtyping's runtime checker catches the bad
    shape from the annotation first. Without a typechecker hook installed -- i.e.
    in ordinary use -- the explicit guard in `shuffle_and_batch` raises instead.
    Either way the caller gets an immediate, legible error.
    """
    mask = jnp.ones(10, dtype=bool)

    with pytest.raises((ValueError, TypeCheckError)):
        shuffle_and_batch(
            mask, jnp.arange(10.0), jnp.arange(9.0), key=jr.key(0), batch_size=4
        )


@pytest.mark.parametrize("batch_size", [0, -1])
def test_shuffle_and_batch_rejects_bad_batch_size(batch_size: int) -> None:
    """`batch_size` must be a positive int."""
    mask = jnp.ones(4, dtype=bool)
    with pytest.raises(ValueError, match="positive int"):
        shuffle_and_batch(mask, jnp.arange(4.0), key=jr.key(0), batch_size=batch_size)


# ============================================================================
# AbstractScanNNTrainer -- structure
# ============================================================================


def test_abstract_methods() -> None:
    """The three subclass hooks are abstract."""
    abstract = AbstractScanNNTrainer.__abstractmethods__
    assert "pack_carry_state" in abstract
    assert "unpack_carry_state" in abstract
    assert "init" in abstract


def test_make_step_is_a_field_not_an_abstract_method() -> None:
    """`make_step` / `loss_agg_fn` are constructor args, not subclass hooks."""
    abstract = AbstractScanNNTrainer.__abstractmethods__
    assert "make_step" not in abstract
    assert "loss_agg_fn" not in abstract


def test_run_signature() -> None:
    """`run` takes the loop config; `loss_agg_fn` lives on the constructor."""
    params = list(inspect.signature(AbstractScanNNTrainer.run).parameters)

    assert {"num_epochs", "batch_size", "key", "step_kw", "show_pbar"} <= set(params)
    assert "loss_agg_fn" not in params


def test_show_pbar_defaults_to_false() -> None:
    """`jax_tqdm` is an optional extra, so the default path must not need it."""
    params = inspect.signature(AbstractScanNNTrainer.run).parameters
    assert params["show_pbar"].default is False


def test_jit_defaults_to_true() -> None:
    """`run` reuses its compiled program unless told not to."""
    params = inspect.signature(AbstractScanNNTrainer.run).parameters
    assert params["jit"].default is True


# ============================================================================
# AbstractScanNNTrainer -- behaviour
# ============================================================================


class _TrivialTrainer(AbstractScanNNTrainer):
    """Stateless trainer whose loss is just a function of the batch mask."""

    def init(self, *, key: Array, X: Array) -> tuple[tuple, tuple]:
        del key
        return (), (jnp.ones(len(X), dtype=bool), (X,))

    def pack_carry_state(self, carry: Any) -> tuple[Any, Any]:
        return carry, ()

    def unpack_carry_state(self, carry: Any, static: Any) -> Any:
        del static
        return carry


def test_step_kw_is_forwarded_to_make_step() -> None:
    """`run(step_kw=...)` reaches `make_step` as keyword arguments."""

    def _make_step(
        carry: Any, batch_inputs: tuple, *, scale: float = 1.0
    ) -> tuple[Array, Any]:
        batch_mask, _ = batch_inputs
        return scale * jnp.sum(batch_mask.astype(jnp.float32)), carry

    trainer = _TrivialTrainer(make_step=_make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=jnp.arange(4.0))

    _, losses = trainer.run(
        carry,
        epoch_data,
        num_epochs=1,
        batch_size=2,
        key=jr.key(0),
        step_kw={"scale": 3.0},
    )

    assert losses.shape == (1,)
    # Two full batches of 2 usable rows each -> 3.0 * 2 == 6.0 per batch.
    assert jnp.allclose(losses[0], 6.0)


def test_prepare_data_args_default_is_noop() -> None:
    """The default hook returns `data_args` unchanged, by identity."""

    def _make_step(carry: Any, batch_inputs: tuple) -> tuple[Array, Any]:
        batch_mask, _ = batch_inputs
        return jnp.sum(batch_mask.astype(jnp.float32)), carry

    trainer = _TrivialTrainer(make_step=_make_step, loss_agg_fn=masked_mean)
    data_args = (jnp.arange(4.0),)

    out = trainer.prepare_data_args(
        (), data_args, epoch_idx=jnp.asarray(0), num_epochs=1, epoch_key=jr.key(0)
    )

    assert out is data_args


def test_prepare_step_kw_default_is_empty() -> None:
    """The default hook contributes no keyword arguments."""

    def _make_step(carry: Any, batch_inputs: tuple) -> tuple[Array, Any]:
        batch_mask, _ = batch_inputs
        return jnp.sum(batch_mask.astype(jnp.float32)), carry

    trainer = _TrivialTrainer(make_step=_make_step, loss_agg_fn=masked_mean)

    out = trainer.prepare_step_kw(
        epoch_idx=jnp.asarray(0), num_epochs=1, epoch_key=jr.key(0)
    )

    assert out == {}


def test_prepare_data_args_override_injects_epoch_data() -> None:
    """An override can add per-epoch derived inputs for `make_step`.

    This is the `order_net` pattern: fresh random negatives every epoch.
    """

    class _AugTrainer(_TrivialTrainer):
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
            x = data_args[0]
            return (x, jr.uniform(epoch_key, shape=x.shape))

    def _make_step(carry: Any, batch_inputs: tuple) -> tuple[Array, Any]:
        batch_mask, (_x, random_like) = batch_inputs
        has_aug = jnp.any(random_like != 0)
        loss = jnp.where(has_aug, 1.0, 0.0) * jnp.any(batch_mask)
        return loss.astype(jnp.float32), carry

    trainer = _AugTrainer(make_step=_make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=jnp.arange(4.0))

    _, losses = trainer.run(
        carry, epoch_data, num_epochs=1, batch_size=2, key=jr.key(5)
    )

    assert losses.shape == (1,)
    assert jnp.allclose(losses[0], 1.0)


def test_prepare_data_args_resamples_each_epoch() -> None:
    """The hook gets a fresh key per epoch, so resampled data actually differs."""

    class _AugTrainer(_TrivialTrainer):
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
            x = data_args[0]
            return (jr.uniform(epoch_key, shape=x.shape),)

    def _make_step(carry: Any, batch_inputs: tuple) -> tuple[Array, Any]:
        batch_mask, (sampled,) = batch_inputs
        return masked_mean(sampled, batch_mask), carry

    trainer = _AugTrainer(make_step=_make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=jnp.arange(8.0))

    _, losses = trainer.run(
        carry, epoch_data, num_epochs=4, batch_size=4, key=jr.key(1)
    )

    # Different draws each epoch -> the mean of the sample differs each epoch.
    assert len(jnp.unique(losses)) == 4


def test_prepare_step_kw_schedules_a_scalar_across_epochs() -> None:
    """Regression: the epoch index must reach `make_step` as a schedulable value.

    This is `phasecurvefit`'s `lambda_p` ramp: a loss weight interpolated
    linearly from `lam_min` to `lam_max` over training. Expressing it requires
    both `epoch_idx` (traced) and `num_epochs` (static).
    """
    lam_min, lam_max, num_epochs = 1.0, 5.0, 5

    class _RampTrainer(_TrivialTrainer):
        def prepare_step_kw(
            self, /, *, epoch_idx: Array, num_epochs: int, epoch_key: Array
        ) -> dict[str, Any]:
            del epoch_key
            frac = epoch_idx / (num_epochs - 1) if num_epochs > 1 else 0.0
            return {"lam": lam_min + (lam_max - lam_min) * frac}

    def _make_step(carry: Any, batch_inputs: tuple, *, lam: Array) -> tuple[Array, Any]:
        # Emit `lam` as the loss so we can read the schedule off `epoch_losses`.
        del batch_inputs
        return jnp.asarray(lam, dtype=float), carry

    trainer = _RampTrainer(make_step=_make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=jnp.arange(8.0))

    _, losses = trainer.run(
        carry, epoch_data, num_epochs=num_epochs, batch_size=4, key=jr.key(1)
    )

    expected = jnp.linspace(lam_min, lam_max, num_epochs)
    assert jnp.allclose(losses, expected), f"{losses} != {expected}"


def test_prepare_step_kw_overrides_static_step_kw() -> None:
    """Per-epoch hook values take precedence over `run(step_kw=...)`."""

    class _RampTrainer(_TrivialTrainer):
        def prepare_step_kw(
            self, /, *, epoch_idx: Array, num_epochs: int, epoch_key: Array
        ) -> dict[str, Any]:
            del num_epochs, epoch_key
            return {"lam": jnp.asarray(epoch_idx, dtype=float)}

    def _make_step(
        carry: Any, batch_inputs: tuple, *, lam: Array, other: float
    ) -> tuple[Array, Any]:
        del batch_inputs
        return jnp.asarray(lam, dtype=float) + other, carry

    trainer = _RampTrainer(make_step=_make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=jnp.arange(8.0))

    _, losses = trainer.run(
        carry,
        epoch_data,
        num_epochs=3,
        batch_size=4,
        key=jr.key(1),
        step_kw={"lam": 99.0, "other": 10.0},  # `lam` must lose to the hook
    )

    assert jnp.allclose(losses, jnp.array([10.0, 11.0, 12.0]))


def test_empty_batches_are_skipped() -> None:
    """Regression: `make_step` must not run on batches with no usable samples.

    With 4 usable of 16 samples and `batch_size=4`, the usable rows sort into
    batch 0 and batches 1-3 are entirely ignorable. Those must be skipped.
    """
    n, usable, batch_size = 16, 4, 4
    calls: list[int] = []

    class _CountingTrainer(AbstractScanNNTrainer):
        def init(self, *, key: Array, X: Array) -> tuple[tuple, tuple]:
            del key
            return (), (jnp.arange(len(X)) < usable, (X,))

        def pack_carry_state(self, carry: Any) -> tuple[Any, Any]:
            return carry, ()

        def unpack_carry_state(self, carry: Any, static: Any) -> Any:
            del static
            return carry

    def _make_step(carry: Any, batch_inputs: tuple) -> tuple[Array, Any]:
        # Traced once per *branch*, not per batch -- so instead of counting
        # traces we assert on the observable result below.
        calls.append(1)
        batch_mask, (x,) = batch_inputs
        return masked_mean(x, batch_mask), carry

    trainer = _CountingTrainer(make_step=_make_step, loss_agg_fn=masked_mean)
    carry, epoch_data = trainer.init(key=jr.key(0), X=jnp.ones(n))

    _, losses = trainer.run(
        carry, epoch_data, num_epochs=2, batch_size=batch_size, key=jr.key(1)
    )

    # The all-ignorable batches would each yield a NaN loss if they were run and
    # aggregated. They are skipped, so the epoch losses stay finite.
    assert jnp.all(jnp.isfinite(losses)), f"NaN leaked from an empty batch: {losses}"
    assert jnp.allclose(losses, 1.0)


# ============================================================================
# AbstractScanNNTrainer -- compile cache
# ============================================================================


@contextmanager
def count_compiles() -> Iterator[list[int]]:
    """Count XLA backend compiles inside the block."""
    count = [0]

    def listener(event: str, _: float, **__: object) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            count[0] += 1

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        yield count
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)


def _sum_step(carry: Any, batch_inputs: tuple, **kw: Any) -> tuple[Array, Any]:
    """Module-level, so it compares equal across `run` calls."""
    batch_mask, (x,) = batch_inputs
    return jnp.sum(jnp.where(batch_mask, x, 0.0)) + kw.get("offset", 0.0), carry


@dataclass(frozen=True, slots=True)
class _OffsetTrainer(_TrivialTrainer):
    """Trainer with an array field, passed to `make_step` each epoch."""

    offset: Array

    def prepare_step_kw(self, /, **kw: Any) -> dict[str, Any]:
        del kw
        return {"offset": self.offset}


def _run(trainer: AbstractScanNNTrainer, **kw: Any) -> Array:
    carry, epoch_data = trainer.init(key=jr.key(0), X=jnp.arange(4.0))
    _, losses = trainer.run(
        carry, epoch_data, num_epochs=2, batch_size=2, key=jr.key(1), **kw
    )
    return losses


def test_run_reuses_compiled_program() -> None:
    """A second identical `run` compiles nothing and gives the same result."""
    trainer = _TrivialTrainer(make_step=_sum_step, loss_agg_fn=masked_mean)
    first = _run(trainer)
    with count_compiles() as n:
        second = _run(trainer)
    assert n[0] == 0
    assert jnp.array_equal(first, second)


def test_run_matches_eager() -> None:
    """``jit=True`` and ``jit=False`` give the same losses."""
    trainer = _TrivialTrainer(make_step=_sum_step, loss_agg_fn=masked_mean)
    assert jnp.allclose(_run(trainer), _run(trainer, jit=False))


def test_run_traces_array_fields() -> None:
    """An array field is traced: a new value reuses the program and is used."""
    _run(_OffsetTrainer(_sum_step, masked_mean, jnp.asarray(0.0)))
    trainer = _OffsetTrainer(_sum_step, masked_mean, jnp.asarray(10.0))
    eager = _run(trainer, jit=False)
    with count_compiles() as n:
        losses = _run(trainer)
    assert n[0] == 0
    assert jnp.allclose(losses, eager)
    assert jnp.all(losses >= 10.0)


def test_run_unhashable_static_part_runs_eagerly() -> None:
    """A static part that can't be hashed falls back to eager, with no error."""
    trainer = _TrivialTrainer(make_step=_sum_step, loss_agg_fn=masked_mean)
    losses = _run(trainer, step_kw={"tags": {"unhashable"}})
    assert jnp.allclose(losses, _run(trainer, jit=False))


def test_run_show_pbar_under_jit() -> None:
    """The jax_tqdm progress bar works under jit and on a cache hit."""
    pytest.importorskip("jax_tqdm")
    trainer = _TrivialTrainer(make_step=_sum_step, loss_agg_fn=masked_mean)
    first = _run(trainer, show_pbar=True)
    with count_compiles() as n:
        second = _run(trainer, show_pbar=True)
    assert n[0] == 0
    assert jnp.allclose(first, second)
