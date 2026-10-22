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
    Cannot instantiate LoopStrategy classes

    """

    def __new__(cls) -> NoReturn:
        """Raise an error when trying to instantiate the class."""
        msg = "Cannot instantiate LoopStrategy classes"
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
    """Loop strategy for flattening then :func:`jax.vmap`."""


class Scan(AbstractLoopStrategy):
    """Loop strategy for flattening then :func:`jax.lax.scan`."""


# ===================================================================
# Implementations. Each takes ``(f, signature)`` and returns a batched function.
#
# Positional arguments may be pytrees: an entry of the signature applies to every
# leaf of its argument. Keyword arguments are passed to ``f`` unmapped. The
# output part of the signature is never needed -- every strategy infers the
# output from ``f``.

_CORE_DIMS = re.compile(r"\(([^()]*)\)")


def _core_ndims(signature: str | None, nargs: int, /) -> tuple[int, ...]:
    # Number of core dimensions of each input in a gufunc signature, e.g.
    # "(3),(),(n,m)->()" -> (1, 0, 2). Without a signature, all are scalars.
    if signature is None:
        return (0,) * nargs
    inputs = signature.split("->", maxsplit=1)[0]
    ndims = tuple(
        len([d for d in group.split(",") if d.strip()])
        for group in _CORE_DIMS.findall(inputs)
    )
    if len(ndims) != nargs:
        msg = f"signature {signature!r} expects {len(ndims)} arguments, got {nargs}"
        raise TypeError(msg)
    return ndims


def _flatten_args(
    args: tuple[Any, ...], signature: str | None, /
) -> tuple[list[jax.Array], list[int], Callable[[Sequence[Any]], list[Any]]]:
    # Split pytree arguments into array leaves, each with the core ndim of its
    # argument, plus a function to rebuild the arguments from the leaves.
    arg_ndims = _core_ndims(signature, len(args))
    treedefs = [jtu.structure(a) for a in args]
    leaves: list[jax.Array] = []
    ndims: list[int] = []
    counts: list[int] = []
    for a, n in zip(args, arg_ndims, strict=True):
        arrs = [jnp.asarray(leaf) for leaf in jtu.leaves(a)]
        if any(arr.ndim < n for arr in arrs):
            msg = f"arguments have fewer dimensions than signature {signature!r}"
            raise ValueError(msg)
        leaves += arrs
        ndims += [n] * len(arrs)
        counts.append(len(arrs))

    def rebuild(flat: Sequence[Any], /) -> list[Any]:
        out, i = [], 0
        for treedef, count in zip(treedefs, counts, strict=True):
            out.append(jtu.unflatten(treedef, flat[i : i + count]))
            i += count
        return out

    return leaves, ndims, rebuild


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


_Run = Callable[[Callable[..., Any], list[jax.Array], list[int], tuple[int, ...]], Any]


def _make(run: _Run, /) -> Callable[..., Callable[..., Any]]:
    # Build an implementation from `run(g, leaves, core_ndims, batch_shape)`.
    def impl(f: Callable[..., Any], signature: str | None) -> Callable[..., Any]:
        def batched(*args: Any, **kw: Any) -> Any:
            leaves, ndims, rebuild = _flatten_args(args, signature)
            batch = jnp.broadcast_shapes(
                *(a.shape[: a.ndim - n] for a, n in zip(leaves, ndims, strict=True))
            )

            def g(*flat: jax.Array) -> Any:
                return f(*rebuild(flat), **kw)

            return run(g, leaves, ndims, batch)

        return batched

    return impl


def _run_vectorize(
    g: Callable[..., Any],
    leaves: list[jax.Array],
    ndims: list[int],
    batch: tuple[int, ...],
) -> Any:
    padded = [
        a.reshape((1,) * (len(batch) - (a.ndim - n)) + a.shape)
        for a, n in zip(leaves, ndims, strict=True)
    ]
    return _nest(g, len(batch))(*padded)


def _flat(
    runner: Callable[[Callable[..., Any], tuple[jax.Array, ...]], Any], /
) -> _Run:
    # Broadcast the batch dimensions, flatten them into one leading axis,
    # `runner` over it, then unflatten the batch dimensions.
    def run(
        g: Callable[..., Any],
        leaves: list[jax.Array],
        ndims: list[int],
        batch: tuple[int, ...],
    ) -> Any:
        flat = tuple(
            jnp.broadcast_to(a, (*batch, *a.shape[a.ndim - n :])).reshape(
                math.prod(batch), *a.shape[a.ndim - n :]
            )
            for a, n in zip(leaves, ndims, strict=True)
        )
        out = runner(g, flat)
        return jtu.map(lambda x: x.reshape(*batch, *x.shape[1:]), out)

    return run


def _scan_runner(g: Callable[..., Any], xs: tuple[jax.Array, ...]) -> Any:
    return jax.lax.scan(lambda c, x: (c, g(*x)), None, xs)[1]


_vectorize = _make(_run_vectorize)
_vmap = _make(_flat(lambda g, xs: jax.vmap(g)(*xs)))
_scan = _make(_flat(_scan_runner))


def _no_loop(f: Callable[..., Any], signature: str | None) -> Callable[..., Any]:  # noqa: ARG001
    return f


def _determine(f: Callable[..., Any], signature: str | None) -> Callable[..., Any]:
    # Chosen when called, not when built, so it follows the backend in use then.
    # `scan` is the cheapest on CPU, where there is no parallel hardware to feed.
    def batched(*args: Any, **kw: Any) -> Any:
        impl = _scan if jax.default_backend() == "cpu" else _vectorize
        return impl(f, signature)(*args, **kw)

    return batched


_IMPLEMENTATIONS: dict[Any, Callable[..., Callable[..., Any]]]  # keys: strategy classes
_IMPLEMENTATIONS = {
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
          never materialized.
        - `~jaxmore.loop_strategies.VMap`: broadcast and flatten the batch
          dimensions, then `jax.vmap`.
        - `~jaxmore.loop_strategies.Scan`: broadcast and flatten the batch
          dimensions, then `jax.lax.scan`. Sequential, so memory use is that of
          a single call.
        - `~jaxmore.loop_strategies.NoLoop`: return ``f`` unchanged, for a
          function that is already batched.
        - `~jaxmore.loop_strategies.Determine`: ``Scan`` on CPU, ``Vectorize``
          elsewhere.

    f : callable
        The function of one batch element. Called with one batch element of
        each positional argument, and with the keyword arguments unchanged.
    signature : str, optional
        Core dimensions of the arguments, as a NumPy generalized ufunc
        signature, e.g. ``"(3),()"`` or ``"(3),()->()"``. Dimensions to the left
        of the core dimensions are batch dimensions. By default all arguments
        are scalars per batch element. The result is inferred from ``f``, so
        the output part, if given, is ignored by every strategy. An entry
        applies to every leaf of a pytree argument.

    Returns
    -------
    callable
        ``f`` mapped over the broadcast batch dimensions of its positional
        arguments, which may be pytrees. Keyword arguments are not mapped but
        passed to every call of ``f``. The output may be any pytree of arrays,
        each leaf with the batch dimensions in front.

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
    for cls in strategy.__mro__:
        if cls in _IMPLEMENTATIONS:
            impl = _IMPLEMENTATIONS[cls]
            break
    else:
        msg = f"no implementation for loop strategy {strategy.__name__}"
        raise TypeError(msg)

    return ft.wraps(f)(impl(f, signature))
