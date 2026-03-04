"""Tests for AbstractScanNNTrainer and related utilities."""

import inspect
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array

from jaxmore.nn import AbstractScanNNTrainer, masked_mean, shuffle_and_batch


def test_masked_mean() -> None:
    """Test masked_mean function."""
    # Input array of values to average
    arr: Array = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Boolean mask indicating which elements to include in the mean
    mask: Array = jnp.array([True, True, False, True, False])
    # Computed mean of masked elements
    result: Array = masked_mean(arr, mask)
    # Expected value: (1.0 + 2.0 + 4.0) / 3.0 = 2.333...
    expected: float = (1.0 + 2.0 + 4.0) / 3.0
    assert jnp.allclose(result, expected), "masked_mean failed"


def test_shuffle_and_batch() -> None:
    """Test shuffle_and_batch function."""
    # Random key for reproducible shuffling
    key: Array = jr.key(42)
    # Total number of samples
    n: int = 10
    # Boolean mask: True for 7 samples, False for 3 samples
    mask: Array = jnp.array([True] * 7 + [False] * 3)
    # Input feature array: [0.0, 1.0, ..., 9.0]
    x: Array = jnp.arange(n, dtype=jnp.float32)
    # Input target array: [10.0, 11.0, ..., 19.0]
    y: Array = jnp.arange(n, 2 * n, dtype=jnp.float32)
    # Batched mask and data with shape (3 batches, 4 batch_size)
    batch_mask: Array
    batch_x: Array
    batch_y: Array
    batch_mask, (batch_x, batch_y) = shuffle_and_batch(
        mask, x, y, key=key, batch_size=4
    )
    assert batch_mask.shape == (3, 4), "Incorrect batch mask shape"
    assert batch_x.shape == (3, 4), "Incorrect batch x shape"
    assert batch_y.shape == (3, 4), "Incorrect batch y shape"


def test_nnt_base_abstract() -> None:
    """Test AbstractScanNNTrainer is abstract."""
    # Verify that required abstract methods are not implemented
    assert "pack_carry_state" in AbstractScanNNTrainer.__abstractmethods__
    assert "unpack_carry_state" in AbstractScanNNTrainer.__abstractmethods__


def test_nnt_run_signature() -> None:
    """Test AbstractScanNNTrainer.run method signature."""
    # Extract the signature of the run method
    sig: inspect.Signature = inspect.signature(AbstractScanNNTrainer.run)
    # Get all parameter names from the signature
    params: list[str] = list(sig.parameters.keys())
    # Verify expected training loop parameters are present
    assert "num_epochs" in params
    assert "batch_size" in params
    # Random key for reproducibility
    assert "key" in params
    # Keyword arguments to forward to make_step
    assert "step_kw" in params
    # Progress bar flag
    assert "show_pbar" in params
    # loss_agg_fn should not be in run parameters (it's in __init__)
    assert "loss_agg_fn" not in params


def test_nnt_init_is_abstract() -> None:
    """Test AbstractScanNNTrainer.init is abstract."""
    # Verify that init method must be implemented by subclasses
    assert "init" in AbstractScanNNTrainer.__abstractmethods__


def test_step_kw_forwarding() -> None:
    """Test that run forwards step_kw to make_step."""

    class _KwTrainer(AbstractScanNNTrainer):
        def init(self, *, key: Array, X: Array) -> tuple[tuple, tuple]:
            """Initialize with dummy data."""
            del key
            # Boolean mask: True for first 2 samples, False for last 2
            mask: Array = jnp.array([True, True, False, False])
            # Tuple of (mask, data_tuple) for batching
            epoch_data: tuple = (mask, (X,))
            return (), epoch_data

        def pack_carry_state(self, carry: Any) -> tuple[Any, Any]:
            # No dynamic state to separate; return empty static dict
            return carry, ()

        def unpack_carry_state(self, carry: Any, static: Any) -> Any:
            del static
            return carry

    def _make_step(
        carry: Any, batch_inputs: tuple, *, scale: float = 1.0
    ) -> tuple[Array, Any]:
        """Compute loss for one batch with optional scaling."""
        batch_mask: Array
        _batch_data: tuple
        batch_mask, _batch_data = batch_inputs
        # Loss is scaled sum of batch mask (proxy for training objective)
        loss: Array = scale * jnp.sum(batch_mask.astype(jnp.float32))
        return loss, carry

    # Create trainer instance with custom step function
    trainer: _KwTrainer = _KwTrainer(make_step=_make_step, loss_agg_fn=masked_mean)
    # Simple input array: [0.0, 1.0, 2.0, 3.0]
    x: Array = jnp.arange(4.0)

    # Initialize training state and epoch data
    carry: tuple
    epoch_data: tuple
    carry, epoch_data = trainer.init(key=jr.key(0), X=x)
    # Run training for 1 epoch with scale=3.0 in step_kw
    _final_carry: tuple
    epoch_losses: Array
    _final_carry, epoch_losses = trainer.run(
        carry,
        epoch_data,
        num_epochs=1,
        batch_size=2,
        key=jr.key(0),
        step_kw={"scale": 3.0},
        show_pbar=False,
    )

    assert epoch_losses.shape == (1,)
    # Expected loss: scale * sum(batch_mask) = 3.0 * 2 = 6.0
    assert jnp.allclose(epoch_losses[0], 6.0)


def test_prepare_data_args_default_noop() -> None:
    """Default prepare_data_args should be a no-op."""

    class _Trainer(AbstractScanNNTrainer):
        def init(self, *, key: Array, X: Array) -> tuple[None, tuple]:
            del key
            # Mask with all True values (include all samples)
            mask: Array = jnp.ones(len(X), dtype=bool)
            # Epoch data tuple: (mask, (data,))
            return None, (mask, (X,))

        def pack_carry_state(self, carry: None) -> tuple[None, None]:
            return carry, None

        def unpack_carry_state(self, carry: None, static: None) -> None:
            del static
            return carry

    def _make_step(carry: None, batch_inputs: tuple) -> tuple[Array, None]:
        # Unpack batch data
        batch_mask: Array
        _batch_data: tuple
        batch_mask, _batch_data = batch_inputs
        # Return loss as sum of batch mask
        return jnp.sum(batch_mask.astype(jnp.float32)), carry

    # Create trainer with no data augmentation
    trainer: _Trainer = _Trainer(make_step=_make_step, loss_agg_fn=masked_mean)
    # Input data: tuple containing single array
    data_args: tuple[Array, ...] = (jnp.arange(4.0),)
    # Call prepare_data_args (should return unchanged)
    out: tuple[Array, ...] = trainer.prepare_data_args(
        None, data_args, epoch_key=jr.key(0)
    )
    # Verify default implementation is identity function
    assert out is data_args


def test_prepare_data_args_override_injects_epoch_data() -> None:
    """Override can inject derived data into make_step inputs."""

    class _Trainer(AbstractScanNNTrainer):
        def init(self, *, key: Array, X: Array) -> tuple[tuple, tuple]:
            del key
            # Mask with all True values
            mask: Array = jnp.ones(len(X), dtype=bool)
            return (), (mask, (X,))

        def pack_carry_state(self, carry: tuple) -> tuple[tuple, tuple]:
            return carry, ()

        def unpack_carry_state(self, carry: tuple, static: tuple) -> tuple:
            del static
            return carry

        def prepare_data_args(
            self, carry: tuple, data_args: tuple, /, *, epoch_key: Array
        ) -> tuple[Array, ...]:
            """Inject per-epoch random augmentation data."""
            del carry
            # Original input array
            x: Array = data_args[0]
            # Generate random augmentation parameters with shape matching x
            random_like: Array = jr.uniform(epoch_key, shape=x.shape)
            # Return both original and augmented data
            return (x, random_like)

    def _make_step(carry: tuple, batch_inputs: tuple) -> tuple[Array, tuple]:
        """Compute loss using both original and augmented data."""
        batch_mask: Array
        x: Array
        random_like: Array
        batch_mask, (x, random_like) = batch_inputs
        # Check if augmentation data is non-zero (has aug applied)
        has_aug: Array = jnp.any(random_like != 0)
        # Loss is 1.0 if augmentation was applied and mask is non-empty, else 0.0
        loss: Array = jnp.where(has_aug, 1.0, 0.0) * jnp.any(batch_mask)
        return loss.astype(jnp.float32), carry

    # Create trainer with custom data prep
    trainer: _Trainer = _Trainer(make_step=_make_step, loss_agg_fn=masked_mean)
    # Initialize with simple array [0.0, 1.0, 2.0, 3.0]
    carry: tuple
    epoch_data: tuple
    carry, epoch_data = trainer.init(key=jr.key(0), X=jnp.arange(4.0))

    # Run training; prepare_data_args will inject random augmentation
    _final_carry: tuple
    epoch_losses: Array
    _final_carry, epoch_losses = trainer.run(
        carry,
        epoch_data,
        num_epochs=1,
        batch_size=2,
        key=jr.key(5),
        show_pbar=False,
    )
    assert epoch_losses.shape == (1,)
    # Loss should be 1.0 because augmentation data is non-zero
    assert jnp.allclose(epoch_losses[0], 1.0)


def test_realistic_nn_training() -> None:
    """Test complete neural network training workflow end-to-end."""

    # Create a concrete trainer
    class SimpleNNTrainer(AbstractScanNNTrainer):
        """Trainer for SimpleNN with Equinox models."""

        def init(
            self, *, key: Array, X: Array, y: Array, learning_rate: float = 1e-2
        ) -> tuple[tuple[Any, Any, Array], tuple[Array, tuple[Array, Array]]]:
            """Initialize model and training data."""
            # Split random key for model init and data prep
            model_key: Array
            data_key: Array
            model_key, data_key = jr.split(key)
            # Create MLP model: 2-input features -> 1-output prediction
            model: eqx.nn.MLP = eqx.nn.MLP(
                in_size=X.shape[1],
                out_size=1,
                width_size=16,
                depth=2,
                activation=jax.nn.relu,
                key=model_key,
            )

            # Initialize Adam optimizer with learning rate
            optimizer: optax.GradientTransformation = optax.adam(learning_rate)
            # Initialize optimizer state with trainable model parameters
            opt_state: Any = optimizer.init(eqx.filter(model, eqx.is_array))

            # Generate epoch-local random key
            carry_key: Array = jr.fold_in(data_key, 0)
            # Tuple of (model, optimizer_state, random_key)
            initial_carry: tuple[Any, Any, Array] = (model, opt_state, carry_key)

            # Mask with all True (include all training samples)
            mask: Array = jnp.ones(len(X), dtype=bool)
            # Epoch data: (mask, (features, labels))
            epoch_data: tuple = (mask, (X, y))

            return initial_carry, epoch_data

        def pack_carry_state(
            self, carry: tuple[Any, Any, Array]
        ) -> tuple[tuple[Any, Any, Array], dict[str, Any]]:
            """Partition model into arrays and static parts for JAX scanning."""
            # Unpack training state
            model: Any
            opt_state: Any
            key: Array
            model, opt_state, key = carry
            # Separate model arrays from static structures
            model_dyn: Any
            model_static: Any
            model_dyn, model_static = eqx.partition(model, eqx.is_array)
            # Return dynamic parts and static dict
            return (model_dyn, opt_state, key), {"model_static": model_static}

        def unpack_carry_state(
            self, carry: tuple[Any, Any, Array], static: dict[str, Any]
        ) -> tuple[Any, Any, Array]:
            """Reconstruct model from partitioned state."""
            # Unpack dynamic parts
            model_dyn: Any
            opt_state: Any
            key: Array
            model_dyn, opt_state, key = carry
            # Retrieve static parts
            model_static: Any = static["model_static"]
            # Combine back into full model
            model: Any = eqx.combine(model_dyn, model_static)
            return (model, opt_state, key)

    # Define optimizer and training step with optimizer captured in closure
    optimizer: optax.GradientTransformation = optax.adam(1e-2)

    def make_step(
        carry: tuple[Any, Any, Array], batch_inputs: tuple
    ) -> tuple[Array, tuple[Any, Any, Array]]:
        """Compute loss and apply optimizer update for one batch."""
        # Unpack carry state
        model: Any
        opt_state: Any
        key: Array
        model, opt_state, key = carry
        # Unpack batch data
        batch_mask: Array
        X_batch: Array
        y_batch: Array
        batch_mask, (X_batch, y_batch) = batch_inputs

        def loss_fn(model: Any) -> Array:
            """Forward pass and MSE computation."""
            # Vectorized model predictions: (batch_size,)
            preds: Array = jax.vmap(model)(X_batch)
            # Mean squared error
            mse: Array = jnp.mean((preds.squeeze() - y_batch) ** 2)
            return mse

        # Compute loss and gradients
        loss: Array
        grads: Any
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        # Compute optimizer updates and new state
        updates: Any
        updates, opt_state = optimizer.update(grads, opt_state)
        # Apply updates to model
        model = eqx.apply_updates(model, updates)

        return loss, (model, opt_state, key)

    # Generate synthetic training data
    key: Array = jr.key(42)
    # Features: 32 samples with 2 features each
    X: Array = jr.normal(key, (32, 2))
    # Target: linear relationship with noise: y = 2*x0 + x1 + noise
    y: Array = 2 * X[:, 0] + X[:, 1] + 0.01 * jr.normal(jr.fold_in(key, 1), (32,))

    # Create trainer instance and initialize
    trainer: SimpleNNTrainer = SimpleNNTrainer(
        make_step=make_step, loss_agg_fn=masked_mean
    )
    # Initial carry and epoch data
    carry: tuple[Any, Any, Array]
    epoch_data: tuple[Array, tuple[Array, Array]]
    carry, epoch_data = trainer.init(key=key, X=X, y=y, learning_rate=1e-2)

    # Verify initial state before training
    model: Any
    opt_state: Any
    model, opt_state, _ = carry
    assert isinstance(model, eqx.nn.MLP)
    assert opt_state is not None

    # Train for 3 epochs with batch size 8
    final_carry: tuple[Any, Any, Array]
    losses: Array
    final_carry, losses = trainer.run(
        carry,
        epoch_data,
        num_epochs=3,
        batch_size=8,
        key=key,
        show_pbar=False,
    )

    # Verify output shapes and types
    assert losses.shape == (3,), f"Expected losses shape (3,), got {losses.shape}"
    # Extract final trained model
    final_model: Any
    final_model, _, _ = final_carry
    assert isinstance(final_model, eqx.nn.MLP)

    # Verify losses are positive and non-zero
    assert losses[0] > 0, "Initial loss should be positive"
    assert losses[-1] > 0, "Final loss should be positive"

    # Test inference on unseen data
    test_X: Array = jr.normal(jr.fold_in(key, 2), (5, 2))
    # Vectorized predictions: (5, 1) shape
    preds: Array = jax.vmap(final_model)(test_X)
    assert preds.shape == (5, 1), f"Expected preds shape (5, 1), got {preds.shape}"
