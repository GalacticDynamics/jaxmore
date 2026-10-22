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
    """Loop strategy for :func:`jax.numpy.vectorize`."""


class VMap(AbstractVMap):
    """Loop strategy for flattening then :func:`jax.vmap`."""


class Scan(AbstractLoopStrategy):
    """Loop strategy for flattening then :func:`jax.lax.scan`."""


# ===================================================================
# Implementations. Each takes ``(f, signature)`` and returns a batched function.

_CORE_DIMS = re.compile(r"\(([^()]*)\)")


def _core_ndims(signature: str, /) -> tuple[int, ...]:
    # Number of core dimensions of each input in a gufunc signature, e.g.
    # "(3),(),(n,m)->()" -> (1, 0, 2).
    inputs = signature.split("->", maxsplit=1)[0]
    return tuple(
        len([d for d in group.split(",") if d.strip()])
        for group in _CORE_DIMS.findall(inputs)
    )


def _run_flat(
    f: Callable[..., Any],
    signature: str | None,
    run: Callable[[Callable[..., Any], tuple[jax.Array, ...]], Any],
    args: tuple[Any, ...],
) -> Any:
    # Broadcast the batch dimensions, flatten them into one leading axis,
    # `run` the flattened problem, then unflatten the batch dimensions.
    arrs = tuple(jnp.asarray(a) for a in args)
    ndims = (0,) * len(arrs) if signature is None else _core_ndims(signature)
    if len(ndims) != len(arrs):
        msg = f"signature {signature!r} expects {len(ndims)} arguments, got {len(arrs)}"
        raise TypeError(msg)
    if any(a.ndim < n for a, n in zip(arrs, ndims, strict=True)):
        msg = f"arguments have fewer dimensions than signature {signature!r}"
        raise ValueError(msg)

    cores = tuple(a.shape[a.ndim - n :] for a, n in zip(arrs, ndims, strict=True))
    batch = jnp.broadcast_shapes(
        *(a.shape[: a.ndim - n] for a, n in zip(arrs, ndims, strict=True))
    )
    flat = tuple(
        jnp.broadcast_to(a, (*batch, *core)).reshape(math.prod(batch), *core)
        for a, core in zip(arrs, cores, strict=True)
    )

    out = run(f, flat)
    return jtu.map(lambda x: x.reshape(*batch, *x.shape[1:]), out)


def _vectorize(f: Callable[..., Any], signature: str | None) -> Callable[..., Any]:
    return jnp.vectorize(f, signature=signature)


def _vmap(f: Callable[..., Any], signature: str | None) -> Callable[..., Any]:
    return lambda *args: _run_flat(f, signature, lambda g, xs: jax.vmap(g)(*xs), args)


def _scan(f: Callable[..., Any], signature: str | None) -> Callable[..., Any]:
    def run(g: Callable[..., Any], xs: tuple[jax.Array, ...]) -> Any:
        return jax.lax.scan(lambda c, x: (c, g(*x)), None, xs)[1]

    return lambda *args: _run_flat(f, signature, run, args)


def _no_loop(f: Callable[..., Any], signature: str | None) -> Callable[..., Any]:  # noqa: ARG001
    return f


def _determine(f: Callable[..., Any], signature: str | None) -> Callable[..., Any]:
    # `scan` is the cheapest on CPU, where there is no parallel hardware to feed
    impl = _scan if jax.default_backend() == "cpu" else _vectorize
    return impl(f, signature)


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

        - `~jaxmore.loop_strategies.Vectorize`: `jax.numpy.vectorize`.
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
        The function of one batch element. Called with one array per positional
        argument.
    signature : str, optional
        Core dimensions of the arguments and result, as a NumPy generalized
        ufunc signature, e.g. ``"(3),()->()"``. Dimensions to the left of the
        core dimensions are batch dimensions. By default all arguments are
        scalars per batch element. Only the input part is used by ``VMap`` and
        ``Scan``; ``Vectorize`` (and so ``Determine`` off CPU) also applies the
        output part, to every leaf of the result.

    Returns
    -------
    callable
        ``f`` mapped over the broadcast batch dimensions of its positional
        arguments. Keyword arguments are not mapped; bind them with
        `functools.partial`. The output may be any pytree of arrays.

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

    """
    for cls in strategy.__mro__:
        if cls in _IMPLEMENTATIONS:
            impl = _IMPLEMENTATIONS[cls]
            break
    else:
        msg = f"no implementation for loop strategy {strategy.__name__}"
        raise TypeError(msg)

    return ft.wraps(f)(impl(f, signature))
