"""Copyright (c) 2026 Nathaniel Starkman. All rights reserved."""

__all__ = ("error_if",)

from typing import Any, Literal, TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")


def error_if(
    value: T,
    condition: Any,
    msg: str,
    /,
    *,
    on_error: Literal["raise", "off"] = "raise",
) -> T:
    """Raise an error if a condition is true (JAX-compatible).

    This function is compatible with JAX transformations (jit, vmap, grad) and
    raises an error at runtime if the condition evaluates to True. The raised
    exception is always `jax.errors.JaxRuntimeError`.

    Uses `jax.debug.callback` internally to execute the error-raising logic
    outside of JAX's tracing system. This has important consequences:

    - **Runtime execution**: The callback executes at runtime, not during
      tracing, so it can inspect actual runtime values (not tracer objects).
    - **CPU execution**: The callback always runs on CPU, even if the rest of
      the JAX computation is on GPU/TPU. This is a mild performance cost but
      necessary for error handling.
    - **Not dead code eliminated**: Despite using a callback, the error check is
      NOT eliminated by JAX's optimizations (unlike pure computations that may
      be DCE'd). The callback is considered a side effect and is preserved.
    - **With jit/vmap**: Works seamlessly inside `jax.jit` and `jax.vmap`
      contexts. When vmapped, the callback is called for each mapped instance.
    - **Exception wrapping**: Under `jax.jit`, exceptions raised from
      `jax.debug.callback` are typically wrapped by JAX as
      `jax.errors.JaxRuntimeError` (backend-dependent). Outside jit, the
      `JaxRuntimeError` is raised directly.

    Parameters
    ----------
    value
        The value to return if no error is raised.
    condition
        A boolean or JAX array (scalar or multi-element) indicating whether to
        raise an error. If an array, the error is raised if any element is True.
    msg
        The error message to raise if condition is True.
    on_error
        Controls error handling behavior:

        - ``"raise"`` (default): Raise an error when condition is True.
        - ``"off"``:
            Disable error checking entirely. This is a complete no-op that skips
            even the callback, providing maximum performance when error checks
            are not needed.

    Returns
    -------
    T
        The input value if condition is False, otherwise an error is raised.

    Examples
    --------
    Basic usage (raises error if condition is true):

    .. code-block:: python

        import jax.numpy as jnp
        from jaxmore._src import error_if

        x = jnp.array(5)
        result = error_if(x, x > 10, "x is too large")
        # result = Array(5, dtype=int32)

    Disable error checking for performance:

    .. code-block:: python

        import jax.numpy as jnp
        from jaxmore._src import error_if

        x = jnp.array(15)
        result = error_if(x, x > 10, "ignored", on_error="off")
        # No callback is executed; this is a complete no-op

    Notes
    -----
    For most production code requiring flexible error handling,
    `equinox.error_if` is recommended. The `equinox.error_if` implementation is
    generally more feature-rich and better integrated with the broader
    JAX/Equinox ecosystem:

    - **Multiple error handling modes**: Equinox supports ``"raise"``,
      ``"breakpoint"`` (with configurable debugger frames), ``"nan"`` (replace
      values and continue), ``"warn"`` (emit warning), and ``"off"``.
    - **Better error messages**: Integration with `equinox.filter_jit` provides
      cleaner stack traces and more informative error reporting.
    - **Platform-specific handling**: Special logic for TPU runtime, which
      normally squelches errors.
    - **Automatic differentiation**: Custom JVP rules ensure correct behavior
      under `jax.grad` and other AD transforms.
    - **Dead code elimination awareness**: Uses `lax.cond` to conditionally
      branch, allowing JAX's compiler to optimize away unused checks (though
      this requires the return value to be used in the computation).

    """
    if on_error == "off":
        # Complete no-op: skip the callback entirely for maximum performance.
        return value

    # Reduce array conditions to scalar boolean using 'any' semantics.
    # This ensures the callback can use 'if cond_val:' without ambiguity.
    scalar_condition = jnp.asarray(condition).astype(bool)
    if scalar_condition.ndim > 0:
        scalar_condition = jnp.any(scalar_condition)

    def callback(cond_val: Any) -> None:
        # Callback to check condition and raise error.
        if cond_val:
            raise jax.errors.JaxRuntimeError(str(msg))

    jax.debug.callback(callback, scalar_condition)
    return value
