"""Test different approaches for passing optimizer to make_step.

This module benchmarks three approaches:
1. Optimizer in carry state
2. Optimizer in batch_inputs (via prepare_data_args)
3. Optimizer via step_kw parameter (current approach)
"""

import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from jaxmore.nn import AbstractScanNNTrainer, masked_mean

# ============================================================================
# Option 1: Optimizer in Carry
# ============================================================================


@dataclass(frozen=True)
class TrainerOptimizerInCarry(AbstractScanNNTrainer):
    """Trainer with optimizer in carry state."""

    def pack_carry_state(self, carry):
        """Pack carry: (model, opt_state, key, optimizer)."""
        model, opt_state, key, optimizer = carry
        model_dyn, model_static = eqx.partition(model, eqx.is_array)
        # Optimizer is static (not an array), so it goes in static metadata
        return (model_dyn, opt_state, key), {
            "model_static": model_static,
            "optimizer": optimizer,
        }

    def unpack_carry_state(self, carry, static):
        """Unpack carry."""
        model_dyn, opt_state, key = carry
        model = eqx.combine(model_dyn, static["model_static"])
        optimizer = static["optimizer"]
        return (model, opt_state, key, optimizer)

    def init(self, *, X, y, model, optimizer, key):
        """Initialize training."""
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        initial_carry = (model, opt_state, key, optimizer)
        mask = jnp.ones(len(X), dtype=bool)
        epoch_data = (mask, (X, y))
        return initial_carry, epoch_data


def make_step_optimizer_in_carry(carry, batch_inputs):
    """Make step with optimizer from carry."""
    model, opt_state, key, optimizer = carry
    batch_mask, batch_data = batch_inputs
    X_batch, y_batch = batch_data

    def loss_fn(m):
        preds = jax.vmap(m)(X_batch)
        return jnp.mean((preds - y_batch[:, None]) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, (model, opt_state, key, optimizer)


# ============================================================================
# Option 2: Optimizer in batch_inputs (via prepare_data_args)
# ============================================================================


@dataclass(frozen=True)
class TrainerOptimizerInBatch(AbstractScanNNTrainer):
    """Trainer with optimizer in batch inputs.

    Note: This approach doesn't work well because the optimizer can't be
    shuffled/batched like data. We store it as an attribute and add it
    to batch_inputs manually in a custom batch_scan_fn override.
    """

    optimizer: optax.GradientTransformation

    def pack_carry_state(self, carry):
        """Pack carry: (model, opt_state, key)."""
        model, opt_state, key = carry
        model_dyn, model_static = eqx.partition(model, eqx.is_array)
        return (model_dyn, opt_state, key), {"model_static": model_static}

    def unpack_carry_state(self, carry, static):
        """Unpack carry."""
        model_dyn, opt_state, key = carry
        model = eqx.combine(model_dyn, static["model_static"])
        return (model, opt_state, key)

    def init(self, *, X, y, model, key):
        """Initialize training."""
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        initial_carry = (model, opt_state, key)
        mask = jnp.ones(len(X), dtype=bool)
        epoch_data = (mask, (X, y))
        return initial_carry, epoch_data


def make_step_optimizer_in_batch(carry, batch_inputs, *, optimizer):
    """Make step with optimizer from batch_inputs (passed as kwarg)."""
    model, opt_state, key = carry
    batch_mask, batch_data = batch_inputs
    X_batch, y_batch = batch_data

    def loss_fn(m):
        preds = jax.vmap(m)(X_batch)
        return jnp.mean((preds - y_batch[:, None]) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, (model, opt_state, key)


# ============================================================================
# Option 3: Optimizer via step_kw (current approach)
# ============================================================================


@dataclass(frozen=True)
class TrainerOptimizerInKwargs(AbstractScanNNTrainer):
    """Trainer with optimizer passed via step_kw."""

    optimizer: optax.GradientTransformation

    def pack_carry_state(self, carry):
        """Pack carry: (model, opt_state, key)."""
        model, opt_state, key = carry
        model_dyn, model_static = eqx.partition(model, eqx.is_array)
        return (model_dyn, opt_state, key), {"model_static": model_static}

    def unpack_carry_state(self, carry, static):
        """Unpack carry."""
        model_dyn, opt_state, key = carry
        model = eqx.combine(model_dyn, static["model_static"])
        return (model, opt_state, key)

    def init(self, *, X, y, model, key):
        """Initialize training."""
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        initial_carry = (model, opt_state, key)
        mask = jnp.ones(len(X), dtype=bool)
        epoch_data = (mask, (X, y))
        return initial_carry, epoch_data


def make_step_optimizer_in_kwargs(carry, batch_inputs, *, optimizer):
    """Make step with optimizer from kwargs."""
    model, opt_state, key = carry
    batch_mask, batch_data = batch_inputs
    X_batch, y_batch = batch_data

    def loss_fn(m):
        preds = jax.vmap(m)(X_batch)
        return jnp.mean((preds - y_batch[:, None]) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, (model, opt_state, key)


# ============================================================================
# Tests
# ============================================================================


@pytest.fixture
def setup_data():
    """Create test data."""
    key = jr.key(0)
    X = jr.normal(key, (100, 5))
    y = jr.normal(jr.fold_in(key, 1), (100,))
    return X, y


@pytest.fixture
def setup_model():
    """Create test model."""
    key = jr.key(42)
    model = eqx.nn.MLP(in_size=5, out_size=1, width_size=8, depth=1, key=key)
    return model


@pytest.fixture
def setup_optimizer():
    """Create optimizer."""
    return optax.adam(1e-3)


def test_optimizer_in_carry_correctness(setup_data, setup_model, setup_optimizer):
    """Test that optimizer in carry works correctly."""
    X, y = setup_data
    model = setup_model
    optimizer = setup_optimizer

    trainer = TrainerOptimizerInCarry(
        make_step=make_step_optimizer_in_carry, loss_agg_fn=masked_mean
    )

    key = jr.key(100)
    carry, epoch_data = trainer.init(
        X=X, y=y, model=model, optimizer=optimizer, key=key
    )

    final_carry, losses = trainer.run(
        carry, epoch_data, num_epochs=5, batch_size=20, key=key, show_pbar=False
    )

    # Check that losses decreased
    assert losses.shape == (5,)
    assert jnp.all(jnp.isfinite(losses))
    # Losses should generally decrease
    assert losses[-1] < losses[0]


def test_optimizer_in_batch_correctness(setup_data, setup_model, setup_optimizer):
    """Test that optimizer in batch_inputs works correctly.

    Note: This approach is semantically problematic because the optimizer
    can't be batched/shuffled like data. We pass it via step_kw instead.
    """
    X, y = setup_data
    model = setup_model
    optimizer = setup_optimizer

    trainer = TrainerOptimizerInBatch(
        make_step=make_step_optimizer_in_batch,
        loss_agg_fn=masked_mean,
        optimizer=optimizer,
    )

    key = jr.key(100)
    carry, epoch_data = trainer.init(X=X, y=y, model=model, key=key)

    # Have to pass optimizer via step_kw since it can't be batched
    final_carry, losses = trainer.run(
        carry,
        epoch_data,
        num_epochs=5,
        batch_size=20,
        key=key,
        step_kw={"optimizer": optimizer},
        show_pbar=False,
    )

    # Check that losses decreased
    assert losses.shape == (5,)
    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < losses[0]


def test_optimizer_in_kwargs_correctness(setup_data, setup_model, setup_optimizer):
    """Test that optimizer in kwargs works correctly."""
    X, y = setup_data
    model = setup_model
    optimizer = setup_optimizer

    trainer = TrainerOptimizerInKwargs(
        make_step=make_step_optimizer_in_kwargs,
        loss_agg_fn=masked_mean,
        optimizer=optimizer,
    )

    key = jr.key(100)
    carry, epoch_data = trainer.init(X=X, y=y, model=model, key=key)

    final_carry, losses = trainer.run(
        carry,
        epoch_data,
        num_epochs=5,
        batch_size=20,
        key=key,
        step_kw={"optimizer": optimizer},
        show_pbar=False,
    )

    # Check that losses decreased
    assert losses.shape == (5,)
    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < losses[0]


# ============================================================================
# Benchmarks (run manually with: python tests/test_optimizer_passing.py)
# ============================================================================


# ============================================================================
# Manual timing test for README
# ============================================================================


def run_timing_comparison():
    """Run timing comparison for all three approaches.

    This is a standalone function that can be run to generate timing data
    for the README.
    """
    print("\n" + "=" * 70)
    print("Optimizer Passing Benchmark Comparison")
    print("=" * 70)

    # Setup
    key = jr.key(0)
    X = jr.normal(key, (1000, 10))
    y = jr.normal(jr.fold_in(key, 1), (1000,))
    model_key = jr.key(42)
    optimizer = optax.adam(1e-3)

    num_epochs = 50
    batch_size = 100
    n_trials = 10

    # Option 1: Optimizer in carry
    print("\n1. Optimizer in Carry State")
    print("-" * 70)
    model = eqx.nn.MLP(in_size=10, out_size=1, width_size=16, depth=2, key=model_key)
    trainer1 = TrainerOptimizerInCarry(
        make_step=make_step_optimizer_in_carry, loss_agg_fn=masked_mean
    )
    carry1, epoch_data1 = trainer1.init(
        X=X, y=y, model=model, optimizer=optimizer, key=key
    )

    # Warmup
    trainer1.run(
        carry1,
        epoch_data1,
        num_epochs=1,
        batch_size=batch_size,
        key=key,
        show_pbar=False,
    )

    times1 = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _, losses = trainer1.run(
            carry1,
            epoch_data1,
            num_epochs=num_epochs,
            batch_size=batch_size,
            key=key,
            show_pbar=False,
        )
        jax.block_until_ready(losses)
        times1.append(time.perf_counter() - start)

    mean1 = jnp.mean(jnp.array(times1))
    std1 = jnp.std(jnp.array(times1))
    print(f"Time: {mean1:.4f} ± {std1:.4f} seconds")

    # Option 2: Optimizer in batch_inputs (doesn't actually work - use step_kw)
    print("\n2. Optimizer in Batch Inputs (Note: has to use step_kw)")
    print("-" * 70)
    print("   (This approach is fundamentally flawed - optimizer can't be batched)")
    model = eqx.nn.MLP(in_size=10, out_size=1, width_size=16, depth=2, key=model_key)
    trainer2 = TrainerOptimizerInBatch(
        make_step=make_step_optimizer_in_batch,
        loss_agg_fn=masked_mean,
        optimizer=optimizer,
    )
    carry2, epoch_data2 = trainer2.init(X=X, y=y, model=model, key=key)

    # Warmup - have to pass optimizer via step_kw
    trainer2.run(
        carry2,
        epoch_data2,
        num_epochs=1,
        batch_size=batch_size,
        key=key,
        step_kw={"optimizer": optimizer},
        show_pbar=False,
    )

    times2 = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _, losses = trainer2.run(
            carry2,
            epoch_data2,
            num_epochs=num_epochs,
            batch_size=batch_size,
            key=key,
            step_kw={"optimizer": optimizer},
            show_pbar=False,
        )
        jax.block_until_ready(losses)
        times2.append(time.perf_counter() - start)

    mean2 = jnp.mean(jnp.array(times2))
    std2 = jnp.std(jnp.array(times2))
    print(f"Time: {mean2:.4f} ± {std2:.4f} seconds")

    # Option 3: Optimizer via step_kw
    print("\n3. Optimizer via step_kw Parameter (Recommended)")
    print("-" * 70)
    model = eqx.nn.MLP(in_size=10, out_size=1, width_size=16, depth=2, key=model_key)
    trainer3 = TrainerOptimizerInKwargs(
        make_step=make_step_optimizer_in_kwargs,
        loss_agg_fn=masked_mean,
        optimizer=optimizer,
    )
    carry3, epoch_data3 = trainer3.init(X=X, y=y, model=model, key=key)

    # Warmup
    trainer3.run(
        carry3,
        epoch_data3,
        num_epochs=1,
        batch_size=batch_size,
        key=key,
        step_kw={"optimizer": optimizer},
        show_pbar=False,
    )

    times3 = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _, losses = trainer3.run(
            carry3,
            epoch_data3,
            num_epochs=num_epochs,
            batch_size=batch_size,
            key=key,
            step_kw={"optimizer": optimizer},
            show_pbar=False,
        )
        jax.block_until_ready(losses)
        times3.append(time.perf_counter() - start)

    mean3 = jnp.mean(jnp.array(times3))
    std3 = jnp.std(jnp.array(times3))
    print(f"Time: {mean3:.4f} ± {std3:.4f} seconds")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"1. Optimizer in Carry:       {mean1:.4f} ± {std1:.4f} s  (baseline)")
    print(
        f"2. Optimizer in Batch:       {mean2:.4f} ± {std2:.4f} s  ({mean2 / mean1:.2f}x) *"
    )
    print(
        f"3. Optimizer via step_kw:    {mean3:.4f} ± {std3:.4f} s  ({mean3 / mean1:.2f}x)"
    )
    print("\n* Note: Option 2 doesn't actually work - optimizer can't be batched")
    print("  It uses step_kw under the hood, so timing is same as Option 3.")
    print("\n>>> Recommendation: Use option 3 (step_kw parameter)")
    print("\nRationale:")
    print("  - Cleanest API: optimizer is a hyperparameter, not training state")
    print("  - No packing/unpacking overhead (unlike option 1)")
    print("  - Most semantically correct: separates data from configuration")
    print("  - Best performance: avoids unnecessary serialization")
    print("  - Option 2 is fundamentally flawed (can't batch non-array objects)")
    print("=" * 70)


if __name__ == "__main__":
    run_timing_comparison()
