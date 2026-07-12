"""Copyright (c) 2026 Nathaniel Starkman. All rights reserved.

Neural network training utilities for JAX.
"""

__all__ = ("AbstractScanNNTrainer", "masked_mean", "shuffle_and_batch")

import abc
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar

import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Integer, PRNGKeyArray, Real, Shaped

from .optional_deps import OptDeps

# Type variables
CarryT = TypeVar("CarryT")  # Carry type (model, optimizer state, key, etc.)
InputT = TypeVar("InputT")  # Input batch type
OutputT = TypeVar("OutputT")  # Output type (e.g., loss)

# Type aliases
Sz0Like: TypeAlias = Shaped[ArrayLike, " "]  # noqa: F722
RealSz0: TypeAlias = Real[Array, " "]  # noqa: F722
IntSz0: TypeAlias = Integer[Array, " "]  # noqa: F722
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
        Scalar mean value over masked elements. If `mask` is entirely False,
        returns NaN.

    Notes
    -----
    The division uses a "safe denominator" so that the empty-mask case is
    differentiable. Writing the naive ``jnp.where(count > 0, total / count,
    jnp.nan)`` evaluates ``total / count`` on *both* branches, so when ``count
    == 0`` the unused branch computes ``0 / 0``; the VJP of `jnp.where` then
    multiplies that branch's ``inf`` derivative by zero, yielding a NaN
    gradient. Clamping the denominator to 1 before dividing keeps the forward
    value identical (still NaN when the mask is empty) while making the
    gradient well-defined (zero).

    This matters in practice: `shuffle_and_batch` groups ignorable samples
    together, so a batch can legitimately contain no usable samples at all, and
    a mask-aware loss will then call this function with an all-False mask.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxmore.nn import masked_mean
    >>> arr = jnp.array([1.0, 2.0, 3.0, 4.0])
    >>> mask = jnp.array([True, True, False, True])
    >>> masked_mean(arr, mask)  # Mean of [1, 2, 4]
    Array(2.333..., dtype=float32)

    An empty mask gives NaN, but a *finite* gradient:

    >>> import jax
    >>> empty = jnp.zeros(4, dtype=bool)
    >>> masked_mean(arr, empty)
    Array(nan, dtype=float32)
    >>> jax.grad(masked_mean)(arr, empty)
    Array([0., 0., 0., 0.], dtype=float32)

    """
    total = jnp.sum(arr * mask)
    count = jnp.sum(mask)
    # Clamp the denominator away from zero *before* dividing. See Notes: this
    # keeps the empty-mask value at NaN while avoiding a NaN gradient.
    safe_count = jnp.where(count > 0, count, 1)
    return jnp.where(count > 0, total / safe_count, jnp.nan)


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
        Value to use for padding the arrays. Default is 0. The same value is
        used for *every* array in ``*args``, so it must be a sane filler for all
        of them. Padding rows are always False in ``combined_mask``, so a
        mask-respecting loss never reads them -- but be aware that if `0` is a
        meaningful value in one of your arrays (e.g. a class label), the padded
        entries are indistinguishable from real ones by value alone.

    Returns
    -------
    combined_mask : Array, shape (n_batches, batch_size)
        Binary mask where True = real usable data, False = padding or ignorable data.
    batched_args : tuple of Array
        Shuffled and batched arrays. Each has shape (n_batches, batch_size, ...).
        Usable data appears first, then ignorable data, with padding at the end.

    Raises
    ------
    ValueError
        If ``batch_size`` is not a positive int, or if any array in ``*args``
        does not have the same leading dimension as ``mask``.

    Notes
    -----
    This function is useful for training loops where you want to include both
    real data and padding in batches, with a mask to track which samples are
    valid.

    Because usable samples are sorted *first* and ignorable ones last, the
    ignorable samples cluster together. A batch can therefore contain no usable
    samples at all -- its row of ``combined_mask`` will be entirely False. Any
    loss you compute must tolerate that case; see `masked_mean`.
    `AbstractScanNNTrainer` skips such batches for you.

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

    if not isinstance(batch_size, int) or batch_size <= 0:  # type: ignore[redundant-expr]
        msg = f"batch_size must be a positive int, got {batch_size!r}."
        raise ValueError(msg)

    # Catch the easy mistake of passing an array whose leading dim doesn't line
    # up with the mask: without this the failure surfaces much later, as an
    # opaque shape error from inside `jax.lax.scan`.
    for i, arr in enumerate(args):
        if arr.shape[0] != N:
            msg = (
                f"all arrays must share the mask's leading dimension "
                f"(N={N}), but args[{i}] has shape {arr.shape}."
            )
            raise ValueError(msg)

    # Step 1: Sort so True comes first, False comes second
    sort_perm = jnp.argsort(~mask)
    sorted_mask = mask[sort_perm]
    sorted_args = tuple(arr[sort_perm] for arr in args)

    # Step 2: Build a shuffle permutation that keeps the True and False groups
    # separate. Sorting by a random key shuffles; offsetting the False group's
    # keys by +1.0 puts them in a disjoint, strictly higher range, so the sort
    # shuffles *within* each group without interleaving them.
    rand_vals_true = jr.uniform(key, shape=(N,))
    rand_vals_false = jr.uniform(jr.fold_in(key, 1), shape=(N,))

    combined_rand = jnp.where(
        sorted_mask,
        rand_vals_true,  # True positions -> [0, 1)
        1.0 + rand_vals_false,  # False positions -> [1, 2)
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

    Note that `make_step` and `loss_agg_fn` are *constructor arguments*, not
    methods to override: you pass them in when instantiating the trainer. Only
    `pack_carry_state`, `unpack_carry_state`, and `init` are subclass hooks.

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
        For Equinox models such as ``eqx.nn.MLP``, passing the model straight
        through `jax.lax.scan` raises an error about ``custom_jvp``. Use
        ``eqx.partition`` to split the dynamic (array) leaves from the static
        structure:

        .. code-block:: python

            def pack_carry_state(self, carry):
                model, opt_state, key = carry
                model_dynamic, model_static = eqx.partition(model, eqx.is_array)
                return (model_dynamic, opt_state, key), {"model_static": model_static}

        This filters out non-array components (methods, activation functions,
        static configuration) that would otherwise break JAX's scan primitive.

        The static metadata is captured once, from the *initial* carry, and
        reused for every subsequent pack/unpack. The static structure must
        therefore be invariant across training steps -- which holds for Equinox
        models, whose partitioned structure does not change under gradient
        updates.

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
        This is the inverse of `pack_carry_state`. Use ``eqx.combine`` to
        reconstruct the full model from its partitioned components:

        .. code-block:: python

            def unpack_carry_state(self, carry, static):
                model_dynamic, opt_state, key = carry
                model = eqx.combine(model_dynamic, static["model_static"])
                return (model, opt_state, key)

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
        self,
        carry: CarryT,
        data_args: tuple[Any, ...],
        /,
        *,
        epoch_idx: IntSz0,
        num_epochs: int,
        epoch_key: PRNGKeyArray,
    ) -> tuple[Any, ...]:
        r"""Prepare per-epoch *data* before batching.

        Called once per epoch, immediately before `shuffle_and_batch`. Override
        it to generate or modify the arrays fed to `make_step` -- for example to
        resample random negatives from fresh per-epoch RNG.

        Anything returned here is passed through `shuffle_and_batch`, so every
        element must have leading dimension ``N`` (matching ``mask``). To vary a
        *scalar* hyperparameter across epochs, use `prepare_step_kw` instead.

        The default implementation is a no-op that returns `data_args` unchanged.

        Parameters
        ----------
        carry : CarryT
            Current training carry for the epoch.
        data_args : tuple[Any, ...]
            Data arguments provided via `epoch_data`.
        epoch_idx : Array
            Index of the current epoch, as a traced scalar in ``[0, num_epochs)``.
        num_epochs : int
            Total number of epochs. A static Python int, so it is safe to use in
            Python-level arithmetic and control flow.
        epoch_key : PRNGKeyArray
            RNG key for this epoch.

        Returns
        -------
        tuple[Any, ...]
            Data arguments to pass to `shuffle_and_batch`. Each must have
            leading dimension ``N``.

        Examples
        --------
        Resample uniform "negative" samples every epoch:

        .. code-block:: python

            class MyTrainer(AbstractScanNNTrainer):
                def prepare_data_args(self, carry, data_args, /, *, epoch_key, **kw):
                    (positives,) = data_args
                    negatives = jr.uniform(epoch_key, shape=positives.shape)
                    return (positives, negatives)

        """
        _ = (carry, epoch_idx, num_epochs, epoch_key)
        return data_args

    def prepare_step_kw(
        self,
        /,
        *,
        epoch_idx: IntSz0,
        num_epochs: int,
        epoch_key: PRNGKeyArray,
    ) -> Mapping[str, Any]:
        r"""Prepare per-epoch *keyword arguments* for `make_step`.

        Called once per epoch. Whatever mapping this returns is merged into the
        ``step_kw`` passed to `run`, and forwarded to `make_step` on every batch
        of that epoch. Keys returned here take precedence over ``step_kw``.

        This is the hook for hyperparameters that vary across training: loss-term
        weights, temperature annealing, curriculum thresholds, and similar. The
        returned values may be traced arrays -- they are closed over by the batch
        scan, so they reach `make_step` as ordinary runtime values.

        Use `prepare_data_args` instead for anything with a leading ``N``
        dimension that needs shuffling and batching.

        The default implementation returns an empty mapping.

        Parameters
        ----------
        epoch_idx : Array
            Index of the current epoch, as a traced scalar in ``[0, num_epochs)``.
        num_epochs : int
            Total number of epochs. A static Python int, so ``num_epochs - 1``
            and similar are ordinary Python arithmetic.
        epoch_key : PRNGKeyArray
            RNG key for this epoch.

        Returns
        -------
        Mapping[str, Any]
            Extra keyword arguments for `make_step` during this epoch.

        Examples
        --------
        Linearly ramp a loss weight from ``lam_min`` to ``lam_max`` over training:

        .. code-block:: python

            def prepare_step_kw(self, /, *, epoch_idx, num_epochs, epoch_key):
                frac = epoch_idx / (num_epochs - 1) if num_epochs > 1 else 0.0
                return {"lam": lam_min + (lam_max - lam_min) * frac}

        `make_step` then receives it as a keyword argument::

            def make_step(carry, batch_inputs, *, lam): ...

        """
        _ = (epoch_idx, num_epochs, epoch_key)
        return {}

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
        show_pbar: bool = False,
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
            If True, show a progress bar over epochs via ``jax_tqdm``. Default:
            False. ``jax_tqdm`` is an optional dependency, so this must be opted
            into: install it with the ``tqdm`` extra (e.g. ``pip install
            'jaxmore[tqdm]'``). Passing True without it raises `ImportError`.

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
        - `pack_carry_state(carry) -> (packed_carry, static_meta)`: Prepares
          carry for scanning.
        - `unpack_carry_state(packed_carry, static_meta) -> carry`: Reconstructs
          carry after scanning.

        Passed to the constructor (not overridden):

        - `make_step(carry, batch_inputs, **kwargs) -> (loss, new_carry)`:
          Executes one training step on a batch. `batch_inputs` has the format
          ``(batch_mask, (data_arg1, data_arg2, ...))``, where ``batch_mask``
          marks which rows of the batch are real, usable samples. **Your loss
          should respect ``batch_mask``** -- batches are zero-padded to a
          constant shape, and an unmasked loss will train on that padding. See
          `masked_mean`. If the carry contains an RNG key that `make_step` needs
          to use, `make_step` is responsible for splitting it and returning the
          updated key in the new carry.
        - `loss_agg_fn(losses, batch_has_data) -> epoch_loss`: Aggregates the
          per-batch losses into a single epoch loss.

        Batches with no usable samples are skipped entirely (via `jax.lax.cond`)
        and contribute nothing to `epoch_losses`.

        Subclasses may override two optional per-epoch hooks, both of which
        receive ``epoch_idx`` (a traced scalar), ``num_epochs`` (a static int),
        and ``epoch_key``:

        - `prepare_data_args(carry, data_args, ...) -> tuple[Any, ...]`:
          transform or augment the *data* before batching, e.g. resampling
          random negatives each epoch. Whatever it returns goes through
          `shuffle_and_batch`, so every element needs leading dimension ``N``.
        - `prepare_step_kw(...) -> Mapping[str, Any]`: supply per-epoch *keyword
          arguments* for `make_step`, merged over ``step_kw``. This is how you
          schedule a scalar across training -- a ramped loss weight, an annealed
          temperature, a curriculum threshold. Use it rather than
          `prepare_data_args` for anything that isn't an ``N``-length array.

        Both default to no-ops, and are skipped entirely when not overridden.

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

        Note how `make_step` uses ``batch_mask``: batches are zero-padded to a
        constant shape, so an unmasked loss would train on the padding.

        >>> optimizer = optax.adam(1e-3)
        >>> def make_step(carry, batch_inputs, **kw):
        ...     model, opt_state, key = carry
        ...     batch_mask, (X_batch, y_batch) = batch_inputs
        ...     def loss_fn(model):
        ...         preds = jax.vmap(model)(X_batch).squeeze(-1)
        ...         # Average the squared error over *real* rows only.
        ...         return masked_mean((preds - y_batch) ** 2, batch_mask)
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
        ...     carry, epoch_data, num_epochs=5, batch_size=4, key=key
        ... )
        >>> losses.shape
        (5,)

        Training reduced the loss, and no NaNs leaked in from padded rows:

        >>> bool(losses[-1] < losses[0]), bool(jnp.isfinite(losses).all())
        (True, True)

        """
        if step_kw is None:
            step_kw = {}

        # Unpack epoch data
        mask, data_args = epoch_data

        # Pack the initial carry to filter out non-JAX-compatible parts
        packed_carry, carry_static = self.pack_carry_state(initial_carry)

        # Generate epoch-level random keys
        epoch_keys = jr.split(key, num_epochs)

        # Both per-epoch hooks are opt-in. When `prepare_data_args` is not
        # overridden it is a no-op, so the unpack/repack round-trip needed to
        # call it is pure overhead -- skip it entirely in that (common) case.
        cls = type(self)
        hooks_data_args = (
            cls.prepare_data_args is not AbstractScanNNTrainer.prepare_data_args
        )
        hooks_step_kw = cls.prepare_step_kw is not AbstractScanNNTrainer.prepare_step_kw

        # -------- Epoch scan function --------

        def epoch_scan_fn(
            packed_carry: PackedCarry, epoch_input: tuple[IntSz0, PRNGKeyArray]
        ) -> tuple[PackedCarry, RealSz0]:
            """Run one epoch: shuffle, batch, then scan over batches."""
            epoch_idx, epoch_key = epoch_input

            if hooks_data_args:
                # Temporarily unpack carry for prepare_data_args, then repack
                # immediately -- we don't keep the unpacked carry around.
                carry = self.unpack_carry_state(packed_carry, carry_static)
                epoch_data_args = self.prepare_data_args(
                    carry,
                    data_args,
                    epoch_idx=epoch_idx,
                    num_epochs=num_epochs,
                    epoch_key=epoch_key,
                )
                packed_carry, _ = self.pack_carry_state(carry)
            else:
                epoch_data_args = data_args

            # Per-epoch keyword arguments for `make_step`. Values may be traced
            # (e.g. a schedule computed from `epoch_idx`); the batch scan below
            # closes over them, which is fine -- they arrive at `make_step` as
            # ordinary runtime values. Hook keys win over the static `step_kw`.
            epoch_step_kw: Mapping[str, Any] = step_kw
            if hooks_step_kw:
                epoch_step_kw = {
                    **step_kw,
                    **self.prepare_step_kw(
                        epoch_idx=epoch_idx,
                        num_epochs=num_epochs,
                        epoch_key=epoch_key,
                    ),
                }

            # Shuffle and batch all data for this epoch
            batch_mask, batched_data = shuffle_and_batch(
                mask, *epoch_data_args, key=epoch_key, batch_size=batch_size
            )
            # batched_data is a tuple of batched arrays, each shape (n_batches,
            # batch_size, ...)

            def batch_scan_fn(
                packed: PackedCarry, batch_inputs: tuple[Array, ...]
            ) -> tuple[PackedCarry, RealSz0]:
                """Execute one training step on a batch, skipping empty batches.

                `shuffle_and_batch` sorts usable samples first and ignorable ones
                last, so trailing batches may contain no usable samples at all.
                We guard `make_step` behind a `lax.cond` on ``any(batch_mask)``
                so that such batches neither consume a forward/backward pass nor
                contribute a (NaN) loss.
                """
                batch_mask_i = batch_inputs[0]
                batch_data = batch_inputs[1:]

                def run_step(p: PackedCarry) -> tuple[PackedCarry, RealSz0]:
                    # Unpack carry for the user's make_step
                    carry = self.unpack_carry_state(p, carry_static)
                    loss, carry = self.make_step(
                        carry, (batch_mask_i, batch_data), **epoch_step_kw
                    )
                    # Pack carry before returning to scan
                    new_packed, _ = self.pack_carry_state(carry)
                    return new_packed, jnp.asarray(loss, dtype=float)

                def skip_step(p: PackedCarry) -> tuple[PackedCarry, RealSz0]:
                    # Empty batch: leave the carry untouched and emit a 0.0 loss.
                    # The loss is masked out of the epoch aggregate by
                    # `batch_has_data` below, so its value is never used -- but it
                    # must be finite, since NaN would propagate through `where`.
                    return p, jnp.asarray(0.0)

                return lax.cond(  # type: ignore[no-any-return]
                    jnp.any(batch_mask_i), run_step, skip_step, packed
                )

            # Scan over batches - carry stays PACKED throughout
            batch_inputs = (batch_mask, *batched_data)
            packed_carry, batch_losses = lax.scan(
                batch_scan_fn, packed_carry, batch_inputs
            )

            # Aggregate batch losses into the epoch loss. `batch_has_data` marks
            # which batches actually ran, so skipped batches are excluded rather
            # than diluting the average with their placeholder 0.0.
            batch_has_data = jnp.any(batch_mask, axis=1)  # shape: (n_batches,)
            epoch_loss = self.loss_agg_fn(batch_losses, batch_has_data)

            return packed_carry, epoch_loss

        # -------- Run training --------

        # Optionally wrap with a progress bar. `jax_tqdm` is an optional extra,
        # so this is opt-in: `show_pbar` defaults to False and a bare install
        # never reaches this branch.
        scan_fn: Any = epoch_scan_fn
        if show_pbar:
            if not OptDeps.JAX_TQDM.installed:
                msg = (
                    "jax_tqdm is required for progress bar support (show_pbar=True). "
                    "Install it with the 'tqdm' extra, e.g. "
                    "`pip install 'jaxmore[tqdm]'` or `uv add jaxmore --extra tqdm`."
                )
                raise ImportError(msg)
            # Imported lazily: it is an optional dependency.
            import jax_tqdm  # type: ignore[import-not-found,unused-ignore]  # noqa: PLC0415

            scan_fn = jax_tqdm.scan_tqdm(  # type: ignore[attr-defined,unused-ignore]
                num_epochs, desc="Training", unit="epoch", dynamic_ncols=True
            )(epoch_scan_fn)

        # Run scan over epochs. `jax_tqdm.scan_tqdm` requires the scanned `xs` to
        # carry the epoch index as its first element, hence `epoch_indices`.
        epoch_indices = jnp.arange(num_epochs)
        epoch_inputs = (epoch_indices, epoch_keys)
        packed_final_carry, epoch_losses = lax.scan(scan_fn, packed_carry, epoch_inputs)

        # Unpack the final carry
        final_carry = self.unpack_carry_state(packed_final_carry, carry_static)

        return final_carry, epoch_losses
