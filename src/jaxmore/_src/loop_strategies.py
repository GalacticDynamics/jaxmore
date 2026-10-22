"""Copyright (c) 2026 Nathaniel Starkman. All rights reserved."""

__all__ = (
    "AbstractLoopStrategy",
    "AbstractVMap",
    "Determine",
    "NoLoop",
    "Scan",
    "VMap",
    "Vectorize",
    "loop",
)

import functools as ft
import re
from collections.abc import Callable
from typing import Any, NoReturn

import jax
import jax.numpy as jnp
import jax.tree as jtu

# ===================================================================
# Strategies. These are tags: they are never instantiated, only passed to
# `loop` (or used as dispatch keys by libraries that want their own behavior).


class AbstractLoopStrategy:
    """Abstract class for loop strategies.

    Examples
    --------
    >>> from jaxmore.loop_strategies import AbstractLoopStrategy
    >>> try: AbstractLoopStrategy()
    ... except TypeError as e: print(e)
    Cannot instantiate AbstractLoopStrategy: loop strategies are tags

    """

    def __new__(cls, *_args: Any, **_kw: Any) -> NoReturn:
        """Raise an error when trying to instantiate the class."""
        msg = f"Cannot instantiate {cls.__name__}: loop strategies are tags"
        raise TypeError(msg)


class Determine(AbstractLoopStrategy):
    """Loop strategy that is chosen automatically, by platform."""


class NoLoop(AbstractLoopStrategy):
    """Loop strategy for not performing a loop."""


class AbstractVMap(AbstractLoopStrategy):
    """Abstract loop strategy for vectorizing maps."""


class Vectorize(AbstractVMap):
    """Loop strategy for nested :func:`jax.vmap`, one per batch dimension.

    Like :func:`jax.numpy.vectorize`: batch dimensions of size 1 are not mapped.
    """


class VMap(AbstractVMap):
    """Loop strategy for :func:`jax.vmap`.

    Runs as `Vectorize` does. It is a separate tag so that a library can
    dispatch on it, for example to flatten the batch first.
    """


class Scan(AbstractLoopStrategy):
    """Loop strategy for :func:`jax.lax.map` (a scan) over each batch dimension."""


# ===================================================================
# Implementations. Each takes ``(f, signature)`` and returns a batched function.
#
# Positional arguments may be pytrees: an entry of the signature applies to every
# leaf of its argument. Keyword arguments are passed to ``f`` unmapped. The
# output part of the signature is never needed -- every strategy infers the
# output from ``f``.

_ARG = r"\((?:\w+(?:,\w+)*)?\)"
_ARGS = rf"{_ARG}(?:,{_ARG})*"
_SIGNATURE = re.compile(rf"^{_ARGS}(?:->{_ARGS})?$")
_GROUP = re.compile(r"\(([^)]*)\)")


def _parse_signature(
    signature: str | None, nargs: int, /
) -> tuple[tuple[str, ...], ...]:
    # The core dimensions of each input in a gufunc signature, e.g.
    # "(3),(),(n,m)->()" -> (("3",), (), ("n", "m")). Without a signature, all
    # inputs are scalars.
    if signature is None:
        return ((),) * nargs
    compact = re.sub(r"\s+", "", signature)
    if _SIGNATURE.match(compact) is None:
        msg = f"not a valid gufunc signature: {signature!r}"
        raise ValueError(msg)
    groups = _GROUP.findall(compact.split("->", maxsplit=1)[0])
    dims = tuple(tuple(g.split(",")) if g else () for g in groups)
    if len(dims) != nargs:
        msg = f"signature {signature!r} expects {len(dims)} arguments, got {nargs}"
        raise TypeError(msg)
    return dims


def _flatten_args(
    args: tuple[Any, ...], signature: str | None, /
) -> tuple[list[jax.Array], list[int], Any]:
    # Split pytree arguments into array leaves, each with the core ndim of its
    # argument, plus the treedef to rebuild the arguments. Checks that the
    # leaves agree with the signature, including that a named core dimension has
    # one size throughout.
    arg_dims = _parse_signature(signature, len(args))
    flat, treedef = jtu.flatten(args)
    leaves = [jnp.asarray(leaf) for leaf in flat]
    leaf_dims = [d for a, d in zip(args, arg_dims, strict=True) for _ in jtu.leaves(a)]
    sizes: dict[str, int] = {}
    for arr, dims in zip(leaves, leaf_dims, strict=True):
        if arr.ndim < len(dims):
            msg = f"argument of shape {arr.shape} has fewer dimensions than {dims}"
            raise ValueError(msg)
        for name, size in zip(dims, arr.shape[arr.ndim - len(dims) :], strict=True):
            expected = int(name) if name.isdigit() else sizes.setdefault(name, size)
            if size != expected:
                msg = (
                    f"inconsistent size for core dimension {name!r}: "
                    f"{size} vs {expected}"
                )
                raise ValueError(msg)
    return leaves, [len(d) for d in leaf_dims], treedef


def _vmap_level(
    inner: Callable[..., Any], axes: tuple[int | None, ...], args: tuple[Any, ...], /
) -> Any:
    return jax.vmap(inner, in_axes=axes)(*args)


def _map_level(
    inner: Callable[..., Any], axes: tuple[int | None, ...], args: tuple[Any, ...], /
) -> Any:
    # Sequential: `lax.map` over the mapped arguments, the rest closed over.
    def body(mapped: tuple[Any, ...]) -> Any:
        it = iter(mapped)
        return inner(
            *(next(it) if ax == 0 else a for a, ax in zip(args, axes, strict=True))
        )

    return jax.lax.map(
        body, tuple(a for a, ax in zip(args, axes, strict=True) if ax == 0)
    )


def _nest(
    g: Callable[..., Any], depth: int, level: Callable[..., Any], /
) -> Callable[..., Any]:
    # `g` mapped, by `level`, over `depth` leading batch dimensions. A batch
    # dimension of size 1 is not mapped (like `jnp.vectorize`), so an argument
    # that is shared is never copied.
    if depth == 0:
        return g
    inner = _nest(g, depth - 1, level)

    def step(*args: jax.Array) -> Any:
        axes = tuple(None if a.shape[0] == 1 else 0 for a in args)
        sq = tuple(a[0] if ax is None else a for a, ax in zip(args, axes, strict=True))
        if all(ax is None for ax in axes):  # nothing to map: size-1 batch
            return jtu.map(lambda x: jnp.asarray(x)[None], inner(*sq))
        return level(inner, axes, sq)

    return step


def _make(level: Callable[..., Any], /) -> Callable[..., Callable[..., Any]]:
    # Build an implementation that maps over the batch dimensions with `level`.
    def impl(f: Callable[..., Any], signature: str | None) -> Callable[..., Any]:
        def batched(*args: Any, **kw: Any) -> Any:
            leaves, ndims, treedef = _flatten_args(args, signature)
            if not leaves:
                msg = (
                    "a looped function needs at least one array in its positional "
                    "arguments; keyword arguments are not mapped"
                )
                raise TypeError(msg)
            batch = jnp.broadcast_shapes(
                *(a.shape[: a.ndim - n] for a, n in zip(leaves, ndims, strict=True))
            )
            padded = [  # exactly len(batch) leading batch dimensions, 1 if missing
                a.reshape((1,) * (len(batch) - (a.ndim - n)) + a.shape)
                for a, n in zip(leaves, ndims, strict=True)
            ]

            def g(*flat: jax.Array) -> Any:
                return f(*jtu.unflatten(treedef, flat), **kw)

            return jtu.map(jnp.asarray, _nest(g, len(batch), level)(*padded))

        return batched

    return impl


_vectorize = _make(_vmap_level)
_scan = _make(_map_level)


def _no_loop(f: Callable[..., Any], _signature: str | None) -> Callable[..., Any]:
    return f


def _determine(f: Callable[..., Any], signature: str | None) -> Callable[..., Any]:
    # Both candidates are built once, here. Which one runs is chosen when called,
    # so it follows the backend in use then. `scan` is the cheapest on CPU, where
    # there is no parallel hardware to feed.
    scan, vectorize = _scan(f, signature), _vectorize(f, signature)

    def batched(*args: Any, **kw: Any) -> Any:
        impl = scan if jax.default_backend() == "cpu" else vectorize
        return impl(*args, **kw)

    return batched


_IMPLEMENTATIONS: dict[Any, Callable[..., Callable[..., Any]]] = {
    Determine: _determine,
    NoLoop: _no_loop,
    Scan: _scan,
    Vectorize: _vectorize,
    VMap: _vectorize,
}


def loop(
    strategy: type[AbstractLoopStrategy],
    f: Callable[..., Any],
    /,
    *,
    signature: str | None = None,
) -> Callable[..., Any]:
    r"""Map ``f`` over the batch dimensions of its arguments.

    One call, several ways to run the loop: pick the strategy for the hardware
    and the problem size without rewriting the function.

    Parameters
    ----------
    strategy : type[AbstractLoopStrategy]
        How to run the loop. One of `jaxmore.loop_strategies`:

        - `~jaxmore.loop_strategies.Vectorize`: one `jax.vmap` per batch
          dimension, like `jax.numpy.vectorize`. Batch dimensions of size 1 are
          not mapped.
        - `~jaxmore.loop_strategies.VMap`: the same as ``Vectorize`` here. A
          separate tag, so that libraries can dispatch on it.
        - `~jaxmore.loop_strategies.Scan`: `jax.lax.map`, a scan, over each batch
          dimension. Sequential, so the intermediates are those of a single call;
          only the stacked outputs grow with the batch.

        No strategy broadcasts an argument, so one shared by every batch element
        is never copied.
        - `~jaxmore.loop_strategies.NoLoop`: return ``f`` unchanged, for a
          function that is already batched.
        - `~jaxmore.loop_strategies.Determine`: ``Scan`` on CPU, ``Vectorize``
          elsewhere.

    f : callable
        The function of one batch element. Called with one batch element of
        each positional argument, and with the keyword arguments unchanged.
    signature : str, optional
        Core dimensions of the arguments, as a NumPy generalized ufunc
        signature, e.g. ``"(3),()"`` or ``"(n),(n)->()"``. Dimensions to the
        left of the core dimensions are batch dimensions. A named core
        dimension must have one size across all arguments, and a number fixes
        it. By default all arguments are scalars per batch element. The result
        is inferred from ``f``, so the output part, if given, is checked for
        syntax but otherwise ignored. An entry applies to every leaf of a pytree
        argument, so all leaves must have those core dimensions.

    Returns
    -------
    callable
        ``f`` mapped over the broadcast batch dimensions of its positional
        arguments, which may be pytrees. At least one positional argument must
        contain an array. Keyword arguments are not mapped but passed to every
        call of ``f``. The output may be any pytree of arrays, each leaf with the
        batch dimensions in front. Unlike a decorator, ``loop`` takes the
        strategy first; wrap it with `functools.partial` to use it as one.

    Raises
    ------
    ValueError
        If ``signature`` is malformed, or the arguments do not match it.
    TypeError
        If ``strategy`` is not a loop strategy class with an implementation, or
        no positional argument is mapped.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jaxmore as jm
    >>> from jaxmore import loop_strategies as ls

    >>> def norm_t(x, t):
    ...     return jnp.linalg.norm(x) * t

    >>> x = jnp.array([[3.0, 4.0], [6.0, 8.0]])
    >>> t = jnp.array([1.0, 2.0])

    >>> for strategy in (ls.Vectorize, ls.VMap, ls.Scan, ls.Determine):
    ...     print(jm.loop(strategy, norm_t, signature="(n),()->()")(x, t))
    [ 5. 20.]
    [ 5. 20.]
    [ 5. 20.]
    [ 5. 20.]

    The batch dimensions broadcast, so a scalar ``t`` is shared:

    >>> jm.loop(ls.Scan, norm_t, signature="(n),()->()")(x, 2.0)
    Array([10., 20.], dtype=float32)

    Arguments can be pytrees, and keyword arguments are passed through:

    >>> def f(w, *, scale):
    ...     q, p = w
    ...     return {"energy": scale * (q**2 + p**2)}

    >>> w = (jnp.array([1.0, 2.0]), jnp.array([0.0, 1.0]))
    >>> jm.loop(ls.VMap, f)(w, scale=2.0)
    {'energy': Array([ 2., 10.], dtype=float32)}

    """
    impl = _IMPLEMENTATIONS.get(strategy)
    if impl is None:
        msg = f"no implementation for loop strategy {strategy!r}"
        raise TypeError(msg)

    batched = impl(f, signature)
    # `NoLoop` hands back `f` itself, which must not be rewrapped (that would
    # point `f.__wrapped__` at itself).
    return batched if batched is f else ft.wraps(f)(batched)
