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
import math
import re
from collections.abc import Callable, Sequence
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
    """Loop strategy for flattening the batch, then one :func:`jax.vmap`."""


class Scan(AbstractLoopStrategy):
    """Loop strategy for flattening the batch, then :func:`jax.lax.scan`."""


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
) -> tuple[list[jax.Array], list[int], Callable[[Sequence[Any]], list[Any]]]:
    # Split pytree arguments into array leaves, each with the core ndim of its
    # argument, plus a function to rebuild the arguments from the leaves. Checks
    # that the leaves agree with the signature, including that a named core
    # dimension has one size throughout.
    arg_dims = _parse_signature(signature, len(args))
    treedefs = [jtu.structure(a) for a in args]
    sizes: dict[str, int] = {}
    leaves: list[jax.Array] = []
    ndims: list[int] = []
    counts: list[int] = []
    for a, dims in zip(args, arg_dims, strict=True):
        arrs = [jnp.asarray(leaf) for leaf in jtu.leaves(a)]
        for arr in arrs:
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
        leaves += arrs
        ndims += [len(dims)] * len(arrs)
        counts.append(len(arrs))

    def rebuild(flat: Sequence[Any], /) -> list[Any]:
        out, i = [], 0
        for treedef, count in zip(treedefs, counts, strict=True):
            out.append(jtu.unflatten(treedef, flat[i : i + count]))
            i += count
        return out

    return leaves, ndims, rebuild


def _pad(leaves: list[jax.Array], ndims: list[int], k: int, /) -> list[jax.Array]:
    # Give every leaf exactly `k` leading batch dimensions, of size 1 if missing.
    return [
        a.reshape((1,) * (k - (a.ndim - n)) + a.shape)
        for a, n in zip(leaves, ndims, strict=True)
    ]


def _nest(g: Callable[..., Any], depth: int, /) -> Callable[..., Any]:
    # `g` vmapped over `depth` leading batch dimensions. A batch dimension of
    # size 1 is not mapped (like `jnp.vectorize`), so it is never materialized.
    if depth == 0:
        return g
    inner = _nest(g, depth - 1)

    def step(*args: jax.Array) -> Any:
        axes = tuple(None if a.shape[0] == 1 else 0 for a in args)
        sq = tuple(a[0] if ax is None else a for a, ax in zip(args, axes, strict=True))
        if all(ax is None for ax in axes):  # nothing to map: size-1 batch
            return jtu.map(lambda x: jnp.asarray(x)[None], inner(*sq))
        return jax.vmap(inner, in_axes=axes)(*sq)

    return step


def _element(
    g: Callable[..., Any],
    leaves: list[jax.Array],
    ndims: list[int],
    batch: tuple[int, ...],
    /,
) -> Callable[[jax.Array], Any]:
    # `g` of the `i`-th element of the flattened batch. Each leaf is indexed
    # rather than broadcast: a leaf shared by the whole batch is read at a
    # constant index, so it is never copied (and `vmap` leaves it unmapped).
    padded = _pad(leaves, ndims, len(batch))

    def element(i: jax.Array) -> Any:
        idx = []
        for size in reversed(batch):  # unravel the flat index
            idx.append(i % size)
            i = i // size
        idx.reverse()
        return g(
            *(
                a[tuple(0 if a.shape[d] == 1 else idx[d] for d in range(len(batch)))]
                for a in padded
            )
        )

    return element


def _unflatten(out: Any, batch: tuple[int, ...], /) -> Any:
    return jtu.map(lambda x: x.reshape(*batch, *x.shape[1:]), out)


_Run = Callable[[Callable[..., Any], list[jax.Array], list[int], tuple[int, ...]], Any]


def _run_vectorize(
    g: Callable[..., Any],
    leaves: list[jax.Array],
    ndims: list[int],
    batch: tuple[int, ...],
) -> Any:
    out = _nest(g, len(batch))(*_pad(leaves, ndims, len(batch)))
    return jtu.map(jnp.asarray, out)


def _run_vmap(
    g: Callable[..., Any],
    leaves: list[jax.Array],
    ndims: list[int],
    batch: tuple[int, ...],
) -> Any:
    if 0 in batch:  # nothing to index; the nested form gets the empty shapes
        return _run_vectorize(g, leaves, ndims, batch)
    flat = jnp.arange(math.prod(batch))
    return _unflatten(jax.vmap(_element(g, leaves, ndims, batch))(flat), batch)


def _run_scan(
    g: Callable[..., Any],
    leaves: list[jax.Array],
    ndims: list[int],
    batch: tuple[int, ...],
) -> Any:
    if 0 in batch:  # nothing to index; the nested form gets the empty shapes
        return _run_vectorize(g, leaves, ndims, batch)
    el = _element(g, leaves, ndims, batch)
    flat = jnp.arange(math.prod(batch))
    _, out = jax.lax.scan(lambda c, i: (c, el(i)), None, flat)
    return _unflatten(out, batch)


def _make(run: _Run, /) -> Callable[..., Callable[..., Any]]:
    # Build an implementation from `run(g, leaves, core_ndims, batch_shape)`.
    def impl(f: Callable[..., Any], signature: str | None) -> Callable[..., Any]:
        def batched(*args: Any, **kw: Any) -> Any:
            leaves, ndims, rebuild = _flatten_args(args, signature)
            if not leaves:
                msg = (
                    "a looped function needs at least one array in its positional "
                    "arguments; keyword arguments are not mapped"
                )
                raise TypeError(msg)
            batch = jnp.broadcast_shapes(
                *(a.shape[: a.ndim - n] for a, n in zip(leaves, ndims, strict=True))
            )

            def g(*flat: jax.Array) -> Any:
                return f(*rebuild(flat), **kw)

            return run(g, leaves, ndims, batch)

        return batched

    return impl


_vectorize = _make(_run_vectorize)
_vmap = _make(_run_vmap)
_scan = _make(_run_scan)


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
    VMap: _vmap,
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
        - `~jaxmore.loop_strategies.VMap`: flatten the batch dimensions into one,
          then a single `jax.vmap` over it.
        - `~jaxmore.loop_strategies.Scan`: flatten the batch dimensions into
          one, then `jax.lax.scan` over it. Sequential, so the intermediates are
          those of a single call; only the stacked outputs grow with the batch.

        ``VMap`` and ``Scan`` index each argument instead of broadcasting it, so
        an argument shared by every batch element is never copied.
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
    if not (isinstance(strategy, type) and issubclass(strategy, AbstractLoopStrategy)):  # type: ignore[redundant-expr]
        msg = f"strategy must be a loop strategy class, got {strategy!r}"
        raise TypeError(msg)
    for cls in strategy.__mro__:
        if cls in _IMPLEMENTATIONS:
            impl = _IMPLEMENTATIONS[cls]
            break
    else:
        msg = f"no implementation for loop strategy {strategy.__name__}"
        raise TypeError(msg)

    batched = impl(f, signature)
    # `NoLoop` hands back `f` itself, which must not be rewrapped (that would
    # point `f.__wrapped__` at itself).
    return batched if batched is f else ft.wraps(f)(batched)
