"""Copyright (c) 2026 Nathaniel Starkman. All rights reserved."""

__all__ = ("filter_jit",)

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, ParamSpec, TypeVar

import jax
import numpy as np

P = ParamSpec("P")
R = TypeVar("R")


_NUMPY_TYPES = (np.ndarray, np.generic)


def _is_array(x: Any, /) -> bool:
    # Ask JAX, so a NumPy value is traced exactly when `jax.jit` accepts it.
    # bfloat16 and int4 are; strings, datetimes, objects, and scalar types like
    # `np.longdouble` aren't, so they stay static.
    return isinstance(x, jax.Array) or (
        isinstance(x, _NUMPY_TYPES) and jax.core.valid_jaxtype(x)
    )


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class _Static:
    """A leafless pytree node holding non-array values, compared by value."""

    value: Any


def _partition(tree: Any, /) -> tuple[list[Any], _Static]:
    leaves, treedef = jax.tree.flatten(tree)
    mask = [_is_array(x) for x in leaves]
    dynamic = [x if m else None for x, m in zip(leaves, mask, strict=True)]
    static = tuple(None if m else x for x, m in zip(leaves, mask, strict=True))
    # The types go in the key too: `1 == 1.0 == True`, but they trace differently.
    # They come before the values, so values of different types are never
    # compared with each other.
    return dynamic, _Static((treedef, tuple(map(type, static)), static))


def _combine(dynamic: list[Any], static: _Static, /) -> Any:
    treedef, _, leaves = static.value
    merged = [s if d is None else d for d, s in zip(dynamic, leaves, strict=True)]
    return jax.tree.unflatten(treedef, merged)


@jax.jit
def _call(dynamic: list[Any], static: _Static, /) -> tuple[list[Any], _Static]:
    # One module-level jit for every wrapped function: `fn` is a static leaf, so
    # it is part of the cache key and re-wrapping the same `fn` hits the cache.
    fn, args, kwargs = _combine(dynamic, static)
    return _partition(fn(*args, **kwargs))


def _hashable(x: Any, /) -> bool:
    try:
        hash(x)
    except TypeError:
        return False
    return True


def _unhashable_error(fn: Any, args: Any, kwargs: Any, /) -> TypeError:
    # A treedef's hash leaves out pytree aux data (its `==` compares it), so if
    # the key won't hash, one of the static leaves won't.
    tree = {"fn": fn, "args": args, "kwargs": kwargs}
    path, leaf = next(
        (p, x)
        for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]
        if not _is_array(x) and not _hashable(x)
    )
    msg = (
        f"filter_jit: the static leaf {path[0].key}"
        f"{jax.tree_util.keystr(path[1:])} of type {type(leaf).__qualname__!r} "
        "is not hashable, so it can't be part of the compile cache key. Make it "
        "hashable, pass it as an array, or use `filter_jit(fn, fallback='eager')`."
    )
    return TypeError(msg)


def filter_jit(
    fn: Callable[P, R], /, *, fallback: Literal["raise", "eager"] = "raise"
) -> Callable[P, R]:
    """Jit-compile ``fn``, tracing array leaves and treating the rest as static.

    All arguments are flattened as pytrees. Array leaves (`jax.Array`, and NumPy
    arrays and scalars that `jax.jit` accepts as arguments) are traced. Every
    other leaf (Python scalars, strings, functions, ...) is static: the compile
    cache is keyed on the pytree structure plus those leaves and their types.
    The leaves are compared with ``==``, as `jax.jit` compares
    ``static_argnums``, so values that compare equal share a compiled program:
    ``0.0`` and ``-0.0`` do, and so do two objects whose ``__eq__``
    ignores a difference that changes the trace. The output may also hold
    non-array leaves; they are returned as-is. Array leaves of the output, NumPy
    scalars included, come back as JAX arrays. The output must be a pytree down
    to its arrays: an array held by an object that isn't a pytree is still a
    tracer when it comes out.

    The wrapped function is part of the cache key, so a module-level function
    reuses its compiled program even when it is wrapped again on every call.
    For a cache hit, every static leaf must compare equal to the one from the
    earlier call. Functions compare by identity, so use a module-level
    function or `equinox.Partial` rather than `functools.partial` or a lambda
    built per call.

    Parameters
    ----------
    fn
        The function to compile.
    fallback
        What to do when a static leaf can't be hashed. ``"raise"`` (default)
        raises a `TypeError` naming the leaf's path and type. ``"eager"``
        calls ``fn`` without jit instead.

    Returns
    -------
    Callable
        The wrapped function, with the same signature as ``fn``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxmore import filter_jit

    ``factor`` and ``label`` are static, ``x`` is traced, and the string comes
    back out unchanged:

    >>> @filter_jit
    ... def scale(x, *, factor, label):
    ...     return {"label": label, "y": x * factor}
    >>> out = scale(jnp.arange(3.0), factor=2, label="double")
    >>> out["label"], out["y"]
    ('double', Array([0., 2., 4.], dtype=float32))

    An unhashable static leaf, such as a `set`, raises and names where it is:

    >>> try:
    ...     scale(jnp.arange(3.0), factor=2, label={"a"})
    ... except TypeError as e:
    ...     print(e)
    filter_jit: the static leaf kwargs['label'] of type 'set' is not hashable, ...

    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        dynamic, static = _partition((fn, args, kwargs))
        try:
            hash(static)
        except TypeError as e:
            if fallback == "eager":
                return fn(*args, **kwargs)
            raise _unhashable_error(fn, args, kwargs) from e
        return _combine(*_call(dynamic, static))  # type: ignore[no-any-return]

    return wrapper
