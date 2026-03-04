"""Copyright (c) 2026 Nathaniel Starkman. All rights reserved."""

__all__ = ("bounded_while_loop",)

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jax import lax

from .optional_deps import OptDeps

# Conditionally import equinox.error_if if available, otherwise use internal
if OptDeps.EQUINOX.installed:
    import equinox as eqx  # type: ignore[import-not-found]

    _error_if = eqx.error_if
else:
    from .error import error_if as _error_if

# A (very) general PyTree type: any nested structure of JAX arrays/pytrees.
T = TypeVar("T")
_BoolScalar = jax.Array  # convention: shape () boolean array


def bounded_while_loop(
    cond_fn: Callable[[T], Any],
    body_fn: Callable[[T], T],
    init_val: T,
    *,
    max_steps: int,
    check_termination: bool = True,
) -> T:
    r"""Reverse-mode-friendly, bounded `while_loop` implemented via `lax.scan`.

    This function emulates:

    ```
    val = init_val
    i = 0
    while cond_fn(val) and i < max_steps:
        val = body_fn(val)
        i += 1
    return val
    ```

    but with two crucial differences:

    1. **A hard iteration bound**: the loop is unrolled as a fixed-length `scan`
       of length `max_steps`. This is often much friendlier to reverse-mode AD
       than an unbounded `jax.lax.while_loop`.
    2. **Early stop without wasted work**: once the user condition fails
       (i.e. `cond_fn(val)` becomes `False`), we stop applying `body_fn` and
       run only a no-op for the remaining scan steps. This preserves the fixed
       length required by `scan` *without* performing unnecessary computation.

    If the user condition is still `True` after `max_steps` iterations (i.e. the
    loop would continue), an error is raised by default.

    Parameters
    ----------
    cond_fn
        Predicate of the loop, in the same sense as `jax.lax.while_loop`:
        it should return a boolean scalar indicating whether to **continue**
        iterating. The loop halts when this becomes `False`.
    body_fn
        Loop body, mapping the loop carry to a new carry.
    init_val
        Initial loop carry (any PyTree of JAX arrays / scalars / nested containers).
    max_steps
        Maximum number of iterations to attempt. Must be a non-negative Python int.
    check_termination
        Whether to validate after the scan that the loop terminated before `max_steps`.

        - If ``True`` (default), validates termination.
          When ``equinox`` is installed, uses ``equinox.error_if`` for better
          error handling (dead code elimination support, TPU compatibility,
          debugger modes). Otherwise, falls back to an internal ``error_if``
          implementation that uses `jax.debug.callback`.
        - If ``False``, this post-check is skipped and the function returns the
          carry after exactly `max_steps` scan steps.

    Returns
    -------
    T
        Final carry value, either when `cond_fn` first returns `False`, or (if
        that never happens) after `max_steps` iterations.

    Examples
    --------
    Simple loop over a scalar value:

    >>> import jax.numpy as jnp
    >>> from jaxmore import bounded_while_loop
    >>> def cond_fn(x):
    ...     return x < 5
    >>> def body_fn(x):
    ...     return x + 1
    >>> bounded_while_loop(cond_fn, body_fn, jnp.asarray(0), max_steps=10)
    Array(5, dtype=int32, ...)

    Same loop but with a PyTree carry (tuple):

    >>> def cond_fn(state):
    ...     x, _ = state
    ...     return x < 3
    >>> def body_fn(state):
    ...     x, y = state
    ...     return x + 1, y * 2
    >>> bounded_while_loop(cond_fn, body_fn, (jnp.asarray(0), jnp.asarray(1)),
    ...                    max_steps=5)
    (Array(3, dtype=int32, ...), Array(8, dtype=int32, ...))

    Notes
    -----
    Semantics and implementation details:

    We convert the unbounded while loop into a bounded scan by augmenting the
    carry with a boolean flag `done`:

    - `done == False` means we are still logically inside the while loop.
    - `done == True` means the loop has logically terminated; remaining scan
      steps must be no-ops.

    At each scan step we do:

    - If `done` is already `True`: do nothing (no-op).
    - Else (not done):
        - Evaluate `continue_ = cond_fn(val)`.
        - If `continue_` is `True`: apply `body_fn`.
        - If `continue_` is `False`: mark `done = True` and do *not* apply
          `body_fn`.

    After the scan finishes, if `done` is still `False`, then `cond_fn` never
    became false within the allowed steps, meaning the bounded loop
    "overflowed".

    Notes on efficiency:

    - The remaining post-termination scan iterations are routed through a branch
      that returns the carry unchanged. At runtime this avoids executing
      `body_fn` after termination.
    - `body_fn` and `cond_fn` are still traced/compiled as part of the JAX
      program (that is unavoidable), but they are not *executed* once
      `done=True`.

    Error checking implementation:

    - **With equinox** (if installed): Uses ``equinox.error_if`` which provides:

      - Dead code elimination: Error check is optimized away if return value is
        unused (you must assign the result for the check to execute)
      - TPU compatibility: Handles TPU runtime's exception squelching
      - Multiple modes: ``EQX_ON_ERROR`` environment variable controls behavior
        (raise, breakpoint, nan, warn, off)

    - **Without equinox** (fallback): Uses internal and ``error_if`` which:

      - Always executes: Uses ``jax.debug.callback`` (not subject to DCE)
      - CPU overhead: The callback executes on CPU with small host overhead

    - **To disable**: Set ``check_termination=False`` to skip the check
      entirely, regardless of which implementation is used.

    """
    if not isinstance(max_steps, int) or max_steps < 0:  # type: ignore[redundant-expr]
        msg = "max_steps must be a non-negative Python int."
        raise ValueError(msg)

    # Trivial bound: no iterations allowed.
    if max_steps == 0:
        return init_val

    def scan_step(
        carry: tuple[T, _BoolScalar],
        _unused: object,
    ) -> tuple[tuple[T, _BoolScalar], None]:
        """One bounded step.

        carry
            (val, done) where:
            - val: the user loop carry
            - done: whether the loop has already terminated
        """
        val, done = carry

        def already_done(_: object, /) -> tuple[T, _BoolScalar]:
            # No-op: preserve carry; remain done.
            return carry

        def do_body(_: object, /) -> tuple[T, _BoolScalar]:
            # Continue: apply body; still not done.
            return body_fn(val), jnp.asarray(False)  # noqa: FBT003

        def stop_now(_: object, /) -> tuple[T, _BoolScalar]:
            # Stop: mark done, and do not run body.
            return val, jnp.asarray(True)  # noqa: FBT003

        def not_done(_: object, /) -> tuple[T, _BoolScalar]:
            # We are still "in the loop": check whether to continue.
            continue_ = jnp.asarray((cond_fn(val)), dtype=bool)

            # If continue_ is True, run body. Otherwise terminate (done=True).
            return lax.cond(continue_, do_body, stop_now, operand=None)  # type: ignore[no-any-return]

        # If we've already terminated, skip everything. Otherwise, proceed as
        # above.
        new_val, new_done = lax.cond(done, already_done, not_done, operand=None)
        return (new_val, new_done), None

    # Carry includes the termination flag. `done` starts False: we have not
    # terminated.
    init_carry: tuple[T, _BoolScalar] = (init_val, jnp.asarray(False))  # noqa: FBT003

    # Run for exactly max_steps steps. `_` is a dummy scan “sequence” input.
    (final_val, final_done), _ = lax.scan(
        scan_step,
        init_carry,
        xs=None,
        length=max_steps,
    )

    # If final_done is False, then cond_fn never became False within
    # max_steps, meaning the corresponding while-loop would still be
    # continuing.
    final_val = _error_if(
        final_val,
        jnp.logical_not(final_done),
        "bounded_while_loop exceeded max_steps without cond_fn becoming False.",
        on_error="raise" if check_termination else "off",
    )

    return final_val  # noqa: RET504
