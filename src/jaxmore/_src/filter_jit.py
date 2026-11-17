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


def _is_array(x: Any, /) -> bool:
    return isinstance(x, jax.Array | np.ndarray | np.generic)


@dataclass(frozen=True, slots=True)
class _Static:
    """A leafless pytree node holding non-array values, compared by value."""

    value: Any


jax.tree_util.register_static(_Static)


def _partition(tree: Any, /) -> tuple[list[Any], _Static]:
    leaves, treedef = jax.tree.flatten(tree)
    dynamic = [x if _is_array(x) else None for x in leaves]
    static = tuple(None if _is_array(x) else x for x in leaves)
    # The types go in the key too: `1 == 1.0 == True`, but they trace differently.
    return dynamic, _Static((treedef, static, tuple(map(type, static))))


def _combine(dynamic: list[Any], static: _Static, /) -> Any:
    treedef, leaves, _ = static.value
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

    All arguments are flattened as pytrees. Array leaves (`jax.Array`,
    `numpy.ndarray`, `numpy.generic`) are traced. Every other leaf (Python
    scalars, strings, functions, ...) is static: the compile cache is keyed on
    the pytree structure plus those leaves, compared by value (and type). The
    output may also hold non-array leaves; they are returned as-is.

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
