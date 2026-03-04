"""Copyright (c) 2026 Nathaniel Starkman. All rights reserved.

Neural network training utilities for JAX.
"""

__all__ = ("AbstractScanNNTrainer", "masked_mean", "shuffle_and_batch")

import abc
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Bool, PRNGKeyArray, Real, Shaped

from .optional_deps import OptDeps

# Type variables
CarryT = TypeVar("CarryT")  # Carry type (model, optimizer state, key, etc.)
InputT = TypeVar("InputT")  # Input batch type
OutputT = TypeVar("OutputT")  # Output type (e.g., loss)

# Type aliases
Sz0Like: TypeAlias = Shaped[ArrayLike, " "]  # noqa: F722
RealSz0: TypeAlias = Real[Array, " "]  # noqa: F722
RealSzN: TypeAlias = Real[Array, " N"]  # noqa: F722
RealSzB: TypeAlias = Real[Array, " B"]  # noqa: F722
RealSzE: TypeAlias = Real[Array, " E"]  # noqa: F722
BoolSzN: TypeAlias = Bool[Array, " N"]  # noqa: F722
BoolSzB: TypeAlias = Bool[Array, " B"]  # noqa: F722
SzN: TypeAlias = Shaped[Array, " N"]  # noqa: F722
SzNs: TypeAlias = Shaped[Array, " N ..."]  # noqa: F722
BoolBSzBs: TypeAlias = Bool[Array, "B Bs"]  # noqa: F722
BSzBs: TypeAlias = Shaped[Array, "B Bs ..."]  # noqa: F722
PackedCarry: TypeAlias = tuple[Any, ...]


def masked_mean(arr: RealSzN, mask: BoolSzN) -> RealSz0:
    r"""Compute the mean of an array over only the masked elements.

    Parameters
    ----------
    arr : Array, shape (N,)
        Input array.
    mask : Array, shape (N,)
        Binary mask where True = include in mean, False = exclude.

    Returns
    -------
    mean : Array
        Scalar mean value over masked elements.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxmore.nn import masked_mean
    >>> arr = jnp.array([1.0, 2.0, 3.0, 4.0])
    >>> mask = jnp.array([True, True, False, True])
    >>> masked_mean(arr, mask)  # Mean of [1, 2, 4]
    Array(2.333..., dtype=float32)

    """
    total = jnp.sum(arr * mask)
    count = jnp.sum(mask)
    # When count == 0 (no True entries in mask), return NaN explicitly instead
    # of 0/0.  This preserves previous behavior (NaN) while avoiding an explicit
    # divide-by-zero.
    return jnp.where(count > 0, total / count, jnp.nan)


def shuffle_and_batch(
    mask: BoolSzN,
    /,
    *args: SzNs,
    key: PRNGKeyArray,
    batch_size: int,
    pad_value: Sz0Like = 0,
) -> tuple[BoolBSzBs, tuple[BSzBs, ...]]:
    r"""Shuffle arrays and batch them with padding mask.

    Separates data into usable (True) and ignorable (False) based on the mask.
    Shuffles within each group independently, then batches with usable data first.

    Parameters
    ----------
    mask : Array, shape (N,)
        Binary mask where True = usable data, False = ignorable data.
    *args : Array
        Variable number of arrays with matching first dimension to shuffle and
        batch. All must have shape (N, ...).
    key : PRNGKeyArray
        JAX random key for deterministic shuffling.
    batch_size : int
        Desired batch size.
    pad_value : Sz0Like, optional
        Value to use for padding the arrays. Default is 0.

    Returns
    -------
    combined_mask : Array, shape (n_batches, batch_size)
        Binary mask where True = real usable data, False = padding or ignorable data.
    batched_args : tuple of Array
        Shuffled and batched arrays. Each has shape (n_batches, batch_size, ...).
        Usable data appears first, then ignorable data, with padding at the end.

    Notes
    -----
    This function is useful for training loops where you want to include both
    real data and padding in batches, with a mask to track which samples are
    valid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> from jaxmore.nn import shuffle_and_batch
    >>> key = jr.key(0)
    >>> mask = jnp.array([True, True, False, True])
    >>> x = jnp.arange(4.0)
    >>> batched_mask, (batched_x,) = shuffle_and_batch(
    ...     mask, x, key=key, batch_size=2
    ... )
    >>> batched_mask.shape
    (2, 2)
    >>> batched_x.shape
    (2, 2)

    """
    N = len(mask)  # noqa: N806

    # Step 1: Sort so True comes first, False comes second
    sort_perm = jnp.argsort(~mask)
    sorted_mask = mask[sort_perm]
    sorted_args = tuple(arr[sort_perm] for arr in args)

    # Step 2: Create shuffle permutation that keeps True and False groups
    # separate Generate random values for shuffling
    rand_vals_true = jr.uniform(key, shape=(N,))
    rand_vals_false = jr.uniform(jr.fold_in(key, 1), shape=(N,))

    # Combine random values: True positions get small values, False get large
    # values
    combined_rand = jnp.where(
        sorted_mask,
        rand_vals_true,  # True positions: small random values
        1.0 + rand_vals_false,  # False positions: large random values
    )

    # Permutation that shuffles within groups
    shuffle_perm = jnp.argsort(combined_rand)

    # Apply permutation to mask and args
    shuffled_mask = sorted_mask[shuffle_perm]
    shuffled_args = tuple(arr[shuffle_perm] for arr in sorted_args)

    # Calculate padding needed for constant batch shape
    n_batches = (N + batch_size - 1) // batch_size  # Ceiling division
    total_padded = n_batches * batch_size
    pad_amount = total_padded - N

    # Helper function to pad and batch arrays with custom pad value
    def pad_and_batch_with_value(arr: Array, pad_value: Sz0Like) -> Array:
        # Pad first dimension
        pad_width = [(0, pad_amount)] + [(0, 0)] * (arr.ndim - 1)
        padded = jnp.pad(arr, pad_width, constant_values=pad_value)
        # Reshape to (n_batches, batch_size, ...)
        return padded.reshape((n_batches, batch_size, *arr.shape[1:]))

    # Pad and batch the args
    batched_args = tuple(
        pad_and_batch_with_value(arr, pad_value=pad_value) for arr in shuffled_args
    )

    # Create padding mask: True for real data, False for padding
    padding_mask = jnp.ones((n_batches, batch_size), dtype=bool)
    if pad_amount > 0:
        padding_mask = padding_mask.at[-1, -pad_amount:].set(False)

    # Batch the usable/ignorable mask
    batched_data_mask = pad_and_batch_with_value(shuffled_mask, pad_value=False)

    # Combine masks: True only where data is real AND usable
    combined_mask = padding_mask & batched_data_mask

    return combined_mask, batched_args


@dataclass(frozen=True, slots=True)
class AbstractScanNNTrainer(Generic[CarryT, InputT, OutputT], metaclass=abc.ABCMeta):
    r"""Base class for efficient neural network training with JAX scan.

    This class provides a framework for training neural networks using JAX's
    scan primitive for efficient batching and epoch iteration. It implements
    the full training loop structure, allowing users to focus on:

    - Implementing `make_step`: the per-batch training step
    - Implementing `pack_carry_state` / `unpack_carry_state`: carry
      serialization for efficient scanning

    Attributes
    ----------
    make_step : callable
        Function to execute a single optimization step.  Signature:
        ``make_step(carry, batch_inputs) -> (loss, new_carry)`` Must return loss
        value and updated carry. If the carry contains an RNG key that needs to
        be used in ``make_step``, then ``make_step`` is responsible for
        splitting it and returning the updated key in the new carry.
    loss_agg_fn : callable
        Function to aggregate batch losses into an epoch loss.  Signature:
        ``loss_agg_fn(losses, mask) -> epoch_loss`` where ``losses`` has shape
        ``(num_batches,)`` and ``mask`` has shape ``(num_batches,)`` indicating
        which batches have real data.

    Notes
    -----
    This is an abstract base class. Subclasses must implement:

    - `pack_carry_state(carry) -> (packed_arrays, static_metadata)`: Serialize
      the carry for scanning, separating dynamic arrays from static structure.
    - `unpack_carry_state(packed_arrays, static_metadata) -> carry`: Reconstruct
      the full carry after scanning.

    The `_run` method implements the full training loop:

    1. **Epoch loop** (outer scan): iterates over epochs
    2. **Shuffle and batch** (per epoch): prepares batched data via
       `shuffle_and_batch`
    3. **Batch loop** (inner scan): iterates over batches with conditional
       execution to skip empty batches
    4. **Per-batch training step**: calls `self.make_step` on each batch
    5. **Loss aggregation** (per epoch): aggregates batch losses into epoch
       loss via the provided `loss_agg_fn`

    This pattern matches that used in phasecurvefit's OrderingNet trainer.

    """

    make_step: Callable[..., tuple[RealSz0, CarryT]]
    loss_agg_fn: Callable[[RealSzB, BoolSzB], RealSz0]

    @abc.abstractmethod
    def pack_carry_state(
        self, carry: CarryT
    ) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
        r"""Pack carry and state for JAX scan.

        This method serializes the training carry (containing model, optimizer
        state, RNG key, etc.) into a form suitable for ``jax.lax.scan``. It
        separates dynamic array-like components from static metadata.

        Parameters
        ----------
        carry : CarryT
            The training carry state, typically ``(model_dynamic, opt_state,
            key)``.

        Returns
        -------
        packed_carry : tuple[Any, ...]
            A flat/nested tuple of arrays/pytrees that will be scanned over.
        static_metadata : dict[str, Any] | None
            Optional static metadata (e.g., model static structure).

        Notes
        -----
        For Equinox models when using ``eqx.nn.MLP`` or other Equinox modules,
        you may get an error about ``custom_jvp`` being passed through
        `jax.lax.scan`. In such cases, use ``eqx.partition`` to separate dynamic
        arrays from static structure.

        A common pattern is to use ``eqx.partition`` to separate dynamic (array)
        and static parameters:

            filter_spec = eqx.is_array model_dynamic, model_static =
            eqx.partition(model, filter_spec) return (model_dynamic, opt_state,
            key), {"model_static": model_static}

        This filters out non-array components (methods, activation functions,
        static configuration) that would otherwise break JAX's scan primitive.

        """

    @abc.abstractmethod
    def unpack_carry_state(
        self, carry: tuple[Any, ...], static: dict[str, Any] | None
    ) -> CarryT:
        r"""Unpack carry and state after scan.

        Inverse of `pack_carry_state`. Reconstructs the full carry from packed
        components.

        Parameters
        ----------
        carry : tuple[Any, ...]
            Packed carry from scan.
        static : dict[str, Any] | None
            Static metadata from packing.

        Returns
        -------
        carry : CarryT
            Reconstructed training carry state.

        Notes
        -----
        This is the inverse of ``pack_carry_state``. Use ``eqx.combine`` to
        reconstruct the full model from its partitioned components if:

            model_dynamic, opt_state, key = carry model_static =
            static["model_static"] model = eqx.combine(model_dynamic,
            model_static) return (model, opt_state, key)

        Without proper unpacking, the model will remain in its partitioned state
        and cannot be used for inference or further training.

        """

    @abc.abstractmethod
    def init(self, **kwargs: Any) -> tuple[CarryT, tuple[BoolSzN, tuple[Any, ...]]]:
        r"""Initialize training carry and epoch data.

        Subclasses must implement this to set up the initial training state
        (model, optimizer state, RNG key, etc.) and the training data before
        calling `run()`.

        Parameters
        ----------
        **kwargs : Any
            Arbitrary keyword arguments for initialization (e.g., ``X``, ``y``,
            ``learning_rate``, ``key``, etc.). The exact arguments depend on
            the subclass.

        Returns
        -------
        initial_carry : CarryT
            Initial training state. The structure is user-defined and must be
            compatible with `pack_carry_state()` and `unpack_carry_state()`.
            Typically contains the model, optimizer state, and RNG key.
        epoch_data : tuple[BoolSzN, tuple[Any, ...]]
            Training data structured as ``(mask, (data1, data2, ...))``where:

            - ``mask`` : Boolean array with shape ``(N,)`` indicating usable
              (True) and ignorable/padding (False) samples.
            - ``(data1, data2, ...)`` : Tuple of data arrays, each with first
              dimension ``N``. These are the inputs to `make_step` (e.g.,
              features, targets, etc.).

        Examples
        --------
        >>> class NNTrainer(AbstractScanNNTrainer):
        ...     def init(self, *, key, X, y, learning_rate=1e-3):
        ...         # Initialize model and optimizer
        ...         model_key, data_key = jr.split(key)
        ...         model = MyModel(model_key)
        ...         optimizer = optax.adam(learning_rate)
        ...         opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        ...         carry_key = jr.fold_in(data_key, 0)
        ...         initial_carry = (model, opt_state, carry_key)
        ...
        ...         # Prepare training data (all samples are usable)
        ...         mask = jnp.ones(len(X), dtype=bool)
        ...         epoch_data = (mask, (X, y))
        ...         return initial_carry, epoch_data

        """

    def prepare_data_args(
        self, carry: CarryT, data_args: tuple[Any, ...], /, *, epoch_key: PRNGKeyArray
    ) -> tuple[Any, ...]:
        r"""Prepare per-epoch data arguments before batching.

        This hook is called once per epoch, immediately before
        `shuffle_and_batch`.  Subclasses can override it to generate or modify
        inputs that are fed to `make_step`, such as random negatives sampled
        from epoch-specific RNG.

        The default implementation is a no-op that returns `data_args` unchanged.

        Parameters
        ----------
        carry : CarryT
            Current training carry for the epoch.
        data_args : tuple[Any, ...]
            Data arguments provided via `epoch_data`.
        epoch_key : PRNGKeyArray
            RNG key for this epoch.

        Returns
        -------
        tuple[Any, ...]
            Data arguments to pass to `shuffle_and_batch`.

        """
        _ = (carry, epoch_key)
        return data_args

    def run(
        self,
        initial_carry: CarryT,
        epoch_data: tuple[BoolSzN, tuple[Any, ...]],
        /,
        *,
        num_epochs: int,
        batch_size: int,
        key: PRNGKeyArray,
        step_kw: Mapping[str, Any] | None = None,
        show_pbar: bool = True,
    ) -> tuple[CarryT, RealSzE]:
        r"""Run the training loop over epochs with efficient batching and scanning.

        This method implements the standard training pattern:

        1. **Epoch loop** (outer scan): iterates over epochs
        2. **Shuffle and batch** (per epoch): prepares batched data via
           `shuffle_and_batch`
        3. **Batch loop** (inner scan): iterates over batches with conditional
           execution to skip empty batches
        4. **Per-batch training step**: calls `self.make_step` on each batch
        5. **Loss aggregation** (per epoch): aggregates batch losses into epoch
           loss via `loss_agg_fn`

        This matches the pattern used in phasecurvefit's OrderingNet trainer.

        Parameters
        ----------
        initial_carry : CarryT
            Initial training state (typically ``(model_dynamic, opt_state, key)``
            or similar).
        epoch_data : tuple[Array, tuple[Array, ...]]
            Data for training, structured as ``(mask, (data1, data2, ...))``.

            - ``mask`` : Array with shape ``(N,)`` and dtype ``bool``. True for
              usable data, False for padding/ignorable data.
            - ``(data1, data2, ...)`` : Tuple of data arrays, each with shape
              ``(N, ...)`` matching the first dimension of ``mask``. Typically
              features, targets, random samples, etc. in the order expected by
              `self.make_step`.

        num_epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        key : PRNGKeyArray
            JAX random key for shuffling and other randomness.
        step_kw : Mapping[str, Any] | None, optional
            Keyword arguments forwarded to `self.make_step` on every batch.
            Defaults to None.
        show_pbar : bool, optional
            If True, show a progress bar over epochs via jax_tqdm. Default: True.

        Returns
        -------
        final_carry : CarryT
            Updated training state after all epochs.
        epoch_losses : Array, shape (num_epochs,)
            Loss values aggregated per epoch.

        Notes
        -----
        Before calling `run()`, you must call `init(**kwargs)` to obtain the
        initial carry and epoch data. The typical workflow is:

        1. create an instance
        2. call `init()`
        3. call `run()`.

        The subclass must implement:

        - `init(**kwargs) -> (initial_carry, epoch_data)`: Initialize training
          state and data.
        - `make_step(carry, batch_inputs, **kwargs) -> (loss, new_carry)`:
          Executes one training step on a batch. The `batch_inputs` has format
          ``(batch_mask, (data_arg1, data_arg2, ...))``. If the carry contains
          an RNG key that needs to be used in `make_step`, then `make_step` is
          responsible for splitting it and returning the updated key in the new
          carry.
        - `pack_carry_state(carry) -> (packed_carry, static_meta)`: Prepares
          carry for scanning.
        - `unpack_carry_state(packed_carry, static_meta) -> carry`: Reconstructs
          carry after scanning.

        Subclasses may override:

        - `prepare_data_args(carry, data_args, epoch_key=...) -> tuple[Any,
          ...]`: Optional per-epoch hook to transform or augment `data_args`
          before batching. Default is a no-op.

        The carry is packed and unpacked around each epoch to isolate the
        trainable parameters from static structure, enabling efficient scanning.

        Examples
        --------
        Here is a complete example of training a simple neural network:

        >>> import equinox as eqx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import optax
        >>> from jaxmore.nn import AbstractScanNNTrainer, masked_mean

        >>> class NNTrainer(AbstractScanNNTrainer):
        ...     '''Simple neural network trainer.'''
        ...     def init(self, *, key, X, y, learning_rate=1e-3):
        ...         '''Initialize model and training data.'''
        ...         model_key, data_key = jr.split(key)
        ...         model = eqx.nn.MLP(
        ...             in_size=X.shape[1],
        ...             out_size=1,
        ...             width_size=8,
        ...             depth=2,
        ...             activation=jax.nn.relu,
        ...             key=model_key,
        ...         )
        ...         optimizer = optax.adam(learning_rate)
        ...         opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        ...         carry_key = jr.fold_in(data_key, 0)
        ...         initial_carry = (model, opt_state, carry_key)
        ...
        ...         # Create mask (all True for full dataset)
        ...         mask = jnp.ones(len(X), dtype=bool)
        ...         epoch_data = (mask, (X, y))
        ...         return initial_carry, epoch_data
        ...
        ...     def pack_carry_state(self, carry):
        ...         model, opt_state, key = carry
        ...         model_dyn, model_static = eqx.partition(model, eqx.is_array)
        ...         return (model_dyn, opt_state, key), {'model_static': model_static}
        ...
        ...     def unpack_carry_state(self, carry, static):
        ...         model_dyn, opt_state, key = carry
        ...         model_static = static['model_static']
        ...         model = eqx.combine(model_dyn, model_static)
        ...         return (model, opt_state, key)

        >>> optimizer = optax.adam(1e-3)
        >>> def make_step(carry, batch_inputs, **kw):
        ...     model, opt_state, key = carry
        ...     batch_mask, (X_batch, y_batch) = batch_inputs
        ...     def loss_fn(model):
        ...         preds = jax.vmap(model)(X_batch)
        ...         mse = jnp.mean((preds - y_batch[:, None]) ** 2)
        ...         return mse
        ...     loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        ...     updates, opt_state = optimizer.update(grads, opt_state)
        ...     model = eqx.apply_updates(model, updates)
        ...     return loss, (model, opt_state, key)
        >>> trainer = NNTrainer(make_step=make_step, loss_agg_fn=masked_mean)
        >>> key = jr.key(0)
        >>> X = jr.normal(key, (20, 2))
        >>> y = 2*X[:,0] + X[:,1] + jr.normal(jr.fold_in(key, 1), (20,))
        >>> carry, epoch_data = trainer.init(key=key, X=X, y=y, learning_rate=1e-2)
        >>> final_carry, losses = trainer.run(
        ...     carry, epoch_data, num_epochs=5, batch_size=4, key=key, show_pbar=False
        ... )
        >>> print(losses.shape)
        (5,)

        """
        if step_kw is None:
            step_kw = {}

        # Unpack epoch data
        mask, data_args = epoch_data

        # Pack the initial carry to filter out non-JAX-compatible parts
        packed_carry, carry_static = self.pack_carry_state(initial_carry)

        # Generate epoch-level random keys
        epoch_keys = jr.split(key, num_epochs)

        # -------- Epoch scan function --------

        def epoch_scan_fn(
            packed_carry: PackedCarry, epoch_input: tuple[Any, PRNGKeyArray]
        ) -> tuple[PackedCarry, RealSz0]:
            """Run one epoch: shuffle, batch, then scan over batches."""
            # Unpack epoch input
            _, epoch_key = epoch_input

            # Temporarily unpack carry for prepare_data_args
            carry = self.unpack_carry_state(packed_carry, carry_static)
            epoch_data_args = self.prepare_data_args(
                carry, data_args, epoch_key=epoch_key
            )
            # Repack immediately - we don't keep unpacked carry around
            packed_carry, _ = self.pack_carry_state(carry)

            # Shuffle and batch all data for this epoch
            batch_mask, batched_data = shuffle_and_batch(
                mask, *epoch_data_args, key=epoch_key, batch_size=batch_size
            )
            # batched_data is a tuple of batched arrays, each shape (n_batches,
            # batch_size, ...)

            # Prepare batch inputs: (batch_mask, *batched_data)
            batch_inputs = (batch_mask, *batched_data)

            # Scan over batches - carry stays PACKED throughout
            packed_carry, batch_losses = jax.lax.scan(
                batch_scan_fn, packed_carry, batch_inputs
            )

            # Aggregate batch losses into epoch loss using loss_agg_fn
            # batch_losses shape: (n_batches,), batch_mask shape: (n_batches,
            # batch_size) We need a per-batch mask indicating which batches have
            # real data
            batch_has_data = jnp.any(batch_mask, axis=1)  # shape: (n_batches,)
            epoch_loss = self.loss_agg_fn(batch_losses, batch_has_data)

            return packed_carry, epoch_loss

        # -------- Batch scan functions --------

        def batch_scan_fn(
            packed_carry: PackedCarry, batch_inputs: tuple[Array, ...]
        ) -> tuple[PackedCarry, RealSz0]:
            """Execute one training step on a batch."""
            # Unpack carry for user's make_step
            carry = self.unpack_carry_state(packed_carry, carry_static)

            batch_mask = batch_inputs[0]
            batch_data = batch_inputs[1:]

            # Call user's make_step with batch inputs
            loss, carry = self.make_step(carry, (batch_mask, batch_data), **step_kw)

            # Pack carry before returning to scan
            packed_carry, _ = self.pack_carry_state(carry)

            return packed_carry, loss

        # -------- Run training --------

        # Optionally wrap with progress bar
        if show_pbar:
            if not OptDeps.JAX_TQDM.installed:
                msg = (
                    "jax_tqdm is required for progress bar support. "
                    "Install it, e.g. with `uv add jaxmore --extra tqdm`."
                )
                raise ImportError(msg)
            import jax_tqdm  # type: ignore[import-not-found]  # noqa: PLC0415

            epoch_scan_fn = jax_tqdm.scan_tqdm(
                num_epochs, desc="Training", unit="epoch", dynamic_ncols=True
            )(epoch_scan_fn)

        # Run scan over epochs
        epoch_indices = jnp.arange(num_epochs)
        epoch_inputs = (epoch_indices, epoch_keys)
        packed_final_carry, epoch_losses = jax.lax.scan(
            epoch_scan_fn, packed_carry, epoch_inputs
        )

        # Unpack the final carry
        final_carry = self.unpack_carry_state(packed_final_carry, carry_static)

        return final_carry, epoch_losses
