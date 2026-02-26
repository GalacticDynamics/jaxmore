"""Copyright (c) 2026 Nathaniel Starkman. All rights reserved."""

__all__ = ("vmap",)

from collections.abc import Callable, Mapping, Sequence
from functools import cache
from operator import itemgetter
from typing import Any, Literal

import jax
import jax.tree as jtu
from jax.tree_util import PyTreeDef


def _make_base_func(
    func: Callable[..., Any],
    static_args: tuple[Any, ...] | None,
    static_kw: dict[str, Any] | None,
) -> Callable[..., Any]:
    # Create the base function with static args/kwargs applied, so that the
    # vmapped wrappers only need to handle the dynamic args/kwargs.
    if static_args is None and static_kw is None:
        return func
    if static_kw is None:  # only static args
        return lambda *args, **kw: func(*static_args, *args, **kw)  # type: ignore[misc]
    if static_args is None:  # only static kwargs
        return lambda *args, **kw: func(*args, **static_kw, **kw)
    return lambda *args, **kw: func(*static_args, *args, **static_kw, **kw)


def _maybe_jit_func(
    func: Callable[..., Any], *, jit: bool | dict[str, Any]
) -> Callable[..., Any]:
    # Optionally jit-compile the base function (outside the vmapped wrapper, so
    # it is not re-jitted for every new axis spec).
    if isinstance(jit, dict):
        return jax.jit(func, **jit)
    if jit:
        return jax.jit(func)
    return func


def _vmap_static(
    func: Callable[..., Any],
    /,
    in_axes: int | None | Sequence[int | None] = 0,
    out_axes: int | None | Sequence[int | None] = 0,
    *,
    static_args: tuple[Any, ...] | None = None,
    static_kw: dict[str, Any] | None = None,
    jit: bool | dict[str, Any] = False,
) -> Callable[..., Any]:
    # Lightweight fast-path for ``vmap(in_kw=False)``: all kwargs are baked
    # into a closure so jax.vmap never sees them (they become static).

    # Create the base function with static args/kwargs applied, so that the
    # vmapped wrappers only need to handle the dynamic args/kwargs.
    base_func = _make_base_func(func, static_args, static_kw)

    # Optionally JIT-compile before vmapping.
    base_func = _maybe_jit_func(base_func, jit=jit)

    # Only positional args are mapped; kwargs pass through the closure.
    return jax.vmap(base_func, in_axes=in_axes, out_axes=out_axes)


def _vmap_kw(
    func: Callable[..., Any],
    /,
    in_axes: int | None | Sequence[int | None] = 0,
    out_axes: int | None | Sequence[int | None] = 0,
    *,
    in_kw: Mapping[str, int | None],
    static_args: tuple[Any, ...] | None = None,
    static_kw: dict[str, Any] | None = None,
    jit: bool | dict[str, Any] = False,
) -> Callable[..., Any]:
    # Fast path for ``vmap`` when ``default_kw_axis == 0`` and ``in_kw`` is a
    # Mapping.  Only the kwargs listed in ``in_kw`` get special axis treatment;
    # any other kwargs passed at call-time flow through to `jax.vmap` as
    # regular kwargs and are mapped along axis 0 (the jax.vmap default).
    #
    # Because the axis spec for the special kwargs is fully determined at
    # vmap-time we pre-build the `jax.vmap` once — no per-call caching.

    # Create the base function with static args/kwargs applied.
    base_func = _make_base_func(func, static_args, static_kw)

    # Optionally JIT-compile before vmapping.
    base_func = _maybe_jit_func(base_func, jit=jit)

    # Wrap so jax.vmap sees (args_tuple, special_kw_dict) as two positional
    # args. Extra kwargs (**rest) are regular kwargs that jax.vmap maps along
    # the default axis 0.
    def packed(
        args: tuple[Any, ...], special_kw: dict[str, Any], /, **rest: Any
    ) -> Any:
        return base_func(*args, **special_kw, **rest)

    # Pre-build the vmapped function once. Build kw_axes from in_kw — fully
    # known at vmap time.
    vmapped = jax.vmap(packed, in_axes=(in_axes, dict(in_kw)), out_axes=out_axes)

    # The set of special kwarg names, for fast splitting at call-time.
    _special_names = frozenset(in_kw)
    _n_special = len(_special_names)

    def wrapper(*args: Any, **kw: Any) -> Any:
        # Fast path: when call-time kwargs are exactly the special set,
        # skip the split entirely and pass kw straight through.
        if kw.keys() == _special_names:
            return vmapped(args, kw)
        # Slow path: split call-time kwargs into special (axis-overridden) and
        # rest (mapped along axis 0 by jax.vmap's default).
        special_kw = {k: kw.pop(k) for k in _special_names if k in kw}
        return vmapped(args, special_kw, **kw)

    return wrapper


# C-level key function for sorting (name, axis) pairs by name.
_sort_by_name = itemgetter(0)


def _vmap_general(
    func: Callable[..., Any],
    /,
    in_axes: int | None | Sequence[int | None] = 0,
    out_axes: int | None | Sequence[int | None] = 0,
    *,
    in_kw: Mapping[str, int | None] | Literal[True],
    default_kw_axis: int | None,
    static_args: tuple[Any, ...] | None = None,
    static_kw: dict[str, Any] | None = None,
    jit: bool | dict[str, Any] = False,
) -> Callable[..., Any]:
    # General path for ``vmap`` when ``default_kw_axis`` is not 0 or the kwarg
    # set is dynamic.  Caches a vmapped wrapper per unique combination of
    # positional-arg tree structure and kwarg-name set.

    # Create the base function with static args/kwargs applied, so that the
    # vmapped wrappers only need to handle the dynamic args/kwargs.
    base_func = _make_base_func(func, static_args, static_kw)

    # Optionally jit-compile the base function (outside the vmapped wrapper, so
    # it is not re-jitted for every new axis spec).
    base_func = _maybe_jit_func(base_func, jit=jit)

    # Wrap base_func so it accepts (args_tuple, kwargs_dict) as two positional
    # arguments — the shape jax.vmap will see — and unpacks them into the real
    # call signature.
    def wrapped_func(args: tuple[Any, ...], kw: dict[str, Any]) -> Any:
        return base_func(*args, **kw)

    # Build and cache a vmapped wrapper for each unique combination of
    # positional-arg axis spec and keyword-arg axis spec.  The arguments are
    # hashable (PyTreeDef + tuples) so that @cache can memoize across calls with
    # the same axis layout, avoiding redundant jax.vmap compilations.
    @cache
    def _get_vmapped(
        args_axes_treedef: PyTreeDef,
        args_axes_leaves: tuple[int | None, ...],
        kw_sorted_items: tuple[tuple[str, int | None], ...],
    ) -> Callable[[tuple[Any, ...], dict[str, Any]], Any]:
        # Reconstruct the positional in_axes pytree from its flattened form.
        args_axes = jtu.unflatten(args_axes_treedef, args_axes_leaves)
        # Convert the sorted (name, axis) pairs back into a dict for jax.vmap.
        kw_axes = dict(kw_sorted_items)
        # Return a vmapped function that expects (args_tuple, kwargs_dict).
        return jax.vmap(wrapped_func, in_axes=(args_axes, kw_axes), out_axes=out_axes)

    # Helper to get the axis for a kwarg, checking in_kw and falling back to
    # default_kw_axis.
    def _axis_for_kw(name: str, /) -> int | None:
        return default_kw_axis if in_kw is True else in_kw.get(name, default_kw_axis)

    # Fast-path cache: maps (args_treedef, kw_names_frozenset) -> vmapped fn.
    # When the caller passes the same arg structure and kwarg names every time
    # (the overwhelmingly common case), this avoids the per-call sort, tuple
    # construction, and _get_vmapped hash lookup entirely.
    _fast_cache: dict[
        tuple[PyTreeDef, frozenset[str]],
        Callable[[tuple[Any, ...], dict[str, Any]], Any],
    ] = {}

    # Pre-compute flattened in_axes when it is a structured sequence, since it
    # doesn't depend on the call-site args.
    if not (isinstance(in_axes, int) or in_axes is None):
        _static_axes_leaves, _static_axes_treedef = jtu.flatten(in_axes)
        _static_axes_leaves_tup = tuple(_static_axes_leaves)
    else:
        _static_axes_leaves_tup = None
        _static_axes_treedef = None

    def wrapped(*args: object, **kw: object) -> object:
        # Compute the args treedef (cheap — no allocation, just introspection).
        args_treedef: PyTreeDef = jtu.structure(args)
        # Fast-path: check if we've seen this exact (treedef, kwarg-names)
        # combination before.  dict.__getitem__ is faster than rebuilding the
        # full cache key every time.
        kw_names = frozenset(kw)
        fast_key = (args_treedef, kw_names)
        vmapped = _fast_cache.get(fast_key)
        if vmapped is not None:
            return vmapped(args, kw)

        # Slow path: first call with this arg-shape / kwarg-name combination.
        # Compute positional axis leaves.
        if _static_axes_leaves_tup is not None:
            axes_treedef = _static_axes_treedef
            axes_leaves = _static_axes_leaves_tup
        else:
            axes_leaves = (in_axes,) * args_treedef.num_leaves
            axes_treedef = args_treedef

        # Resolve and sort kwarg axes.
        if in_kw is True:
            kw_sorted = tuple((k, default_kw_axis) for k in sorted(kw))
        else:
            kw_sorted = tuple(
                sorted(((k, _axis_for_kw(k)) for k in kw), key=_sort_by_name)
            )

        vmapped = _get_vmapped(axes_treedef, axes_leaves, kw_sorted)
        _fast_cache[fast_key] = vmapped
        return vmapped(args, kw)

    return wrapped


def vmap(
    func: Callable[..., Any],
    /,
    in_axes: int | None | Sequence[int | None] = 0,
    out_axes: int | None | Sequence[int | None] = 0,
    *,
    in_kw: Mapping[str, int | None] | bool = False,
    default_kw_axis: int | None = 0,
    static_args: tuple[Any, ...] | None = None,
    static_kw: dict[str, Any] | None = None,
    jit: bool | dict[str, Any] = False,
) -> Callable[..., Any]:
    """Vectorize a function with static-arg support and per-kwarg axis control.

    A drop-in replacement for {func}`jax.vmap` that additionally supports:

    - **Static args/kwargs**: bake constant arguments into a closure so they are
      never flattened or traced by JAX.
    - **Per-kwarg axis control**: map, broadcast, or ignore individual keyword
      arguments independently.
    - **Optional JIT**: JIT-compile the static-folded base function before
      vmapping.

    For best performance, specify any arguments that do not change between calls
    via ``static_args`` and ``static_kw``.  This reduces the number of pytree
    leaves that cross the {func}`jax.jit` dispatch boundary, which can
    significantly lower call-time overhead compared to plain {func}`jax.vmap`.

    Parameters
    ----------
    func:
        Function to vectorize.
    in_axes, out_axes:
        Passed through to {func}`jax.vmap` for positional arguments and outputs.
    in_kw:
        Controls how keyword arguments are mapped.

        - ``Mapping[str, int | None]``: explicit per-kwarg axis mapping.
          Keywords not present fall back to ``default_kw_axis``.
        - `True`: every kwarg uses ``default_kw_axis``.
        - `False` (default): disable kwarg-axis machinery entirely and match the
          default behavior of `jax.vmap`.
    default_kw_axis:
        Default axis for call-time kwargs not listed in ``in_kw``. `None` means
        the kwarg is broadcast (not mapped).
    static_args:
        Positional arguments that are *always* prepended to every call.  These
        are baked into a closure and never seen by {func}`jax.vmap` (i.e. they
        are not mapped). `None` (default) means no static positional arguments.

        .. tip::

           Using ``static_args`` reduces the number of pytree leaves that
           {func}`jax.jit` must flatten and dispatch.  This can significantly
           lower call-time overhead compared to passing the same values as
           regular positional arguments through plain {func}`jax.vmap`.

    static_kw:
        Keyword arguments that are *always* provided.  Merged with call-time
        kwargs as ``**static_kw, **kwargs``. Not included in the kwarg-axis
        pytree.

        .. tip::

           Using ``static_kw`` reduces the number of pytree leaves that
           {func}`jax.jit` must flatten and dispatch.  This can significantly
           lower call-time overhead compared to passing the same values as
           call-time kwargs through plain {func}`jax.vmap`.

    jit:
        If `True`, JIT-compile the wrapped function. If a ``dict``, passed as
        keyword arguments to {func}`jax.jit`.

    Returns
    -------
    Callable[..., Any]
        A vectorized version of ``func`` that maps over the specified axes for
        both positional and keyword arguments.

    Notes
    -----
    **Performance hierarchy**

    The following lists the fast paths from fastest to slowest.  In every case,
    ``static_args`` and ``static_kw`` further improve performance by reducing
    the number of values that cross the {func}`jax.jit` dispatch boundary.

    1. **Static path** — ``in_kw=False`` (default), or ``in_kw=True`` with
       ``default_kw_axis=0``.

       All kwargs are mapped along axis 0, identical to {func}`jax.vmap`. The
       underlying `jax.vmap` is built once at definition time. **This is the
       fastest path** and incurs zero wrapper overhead.

    2. **Kwarg path** — ``in_kw`` is a ``Mapping`` and ``default_kw_axis==0``.

       Only kwargs explicitly listed in ``in_kw`` receive special axis treatment
       (e.g. ``None`` to broadcast); all other call-time kwargs are mapped along
       axis 0 by {func}`jax.vmap`'s default.  The {func}`jax.vmap` is pre-built
       once at definition time.  The only extra overhead is a cheap dict split
       when the caller passes kwargs not in ``in_kw``.

    3. **General path** — any other combination (e.g. ``default_kw_axis=None``).

       Supports arbitrary ``default_kw_axis`` values and dynamic kwarg sets.
       Because the axis spec depends on which kwargs are passed at each call, a
       new `jax.vmap` must potentially be built.  Results are cached per unique
       (arg-treedef, kwarg-names) combination, so repeated calls with the same
       signature incur only a few µs cache-lookup overhead.

    Examples
    --------
    By default `jaxmore.vmap` works the same as `jax.vmap` -- all kwargs are
    mapped along the 0 axis:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxmore import vmap

    >>> def f(x, *, scale):
    ...     return scale * x

    First, let's see `jax.vmap`:

    >>> vf = jax.vmap(f)
    >>> try:
    ...     vf(jnp.arange(3.0), scale=2.0)
    ... except Exception as e:
    ...     print(e)
    vmap was requested to map its argument along axis 0,
    which implies that its rank should be at least 1,
    but is only 0 (its shape is ())

    >>> vf(jnp.arange(3.0), scale=jnp.ones(3))
    Array([0., 1., 2.], dtype=float32)

    Now let's confirm `jaxmore.vmap` works the same:

    >>> vf = vmap(f)
    >>> try:
    ...     vf(jnp.arange(3.0), scale=2.0)
    ... except Exception as e:
    ...     print(e)
    vmap was requested to map its argument along axis 0,
    which implies that its rank should be at least 1,
    but is only 0 (its shape is ())

    >>> vf(jnp.arange(3.0), scale=jnp.ones(3))
    Array([0., 1., 2.], dtype=float32)

    `jaxmore.vmap` offers much more flexibility than `jax.vmap` for dealing with
    function args and kwargs. For example, `jaxmore.vmap` can build a partial
    function with static args and kwargs before applying ``vmap``. Closing over
    static args and kwargs can speed up jitted operations by avoiding flattening
    and unflattening when they pass through the jit boundary.

    >>> def mul(factor, x, *, offset):
    ...     return factor * x + offset

    >>> vmul = vmap(mul, static_args=(3.0,), static_kw={"offset": 1.0})
    >>> vmul(jnp.arange(4.0))
    Array([ 1.,  4.,  7., 10.], dtype=float32)

    `jaxmore.vmap` can also jit-compile the partial function before vmapping,
    which can further speed up the operation when the base function is expensive
    to compile.

    >>> vmul_jit = vmap(mul, static_args=(3.0,), static_kw={"offset": 1.0}, jit=True)
    >>> vmul_jit(jnp.arange(4.0))
    Array([ 1.,  4.,  7., 10.], dtype=float32)

    As a note, the "jit" argument can be a dict of kwargs to pass to `jax.jit`:

    >>> vmul_jit_kw = vmap(
    ...     mul, static_args=(3.0,), static_kw={"offset": 1.0}, jit={"inline": True})
    >>> vmul_jit_kw(jnp.arange(4.0))
    Array([ 1.,  4.,  7., 10.], dtype=float32)

    A major feature of `jaxmore.vmap` is the ability to map over keyword
    arguments with per-kwarg axis control.  This is not possible with `jax.vmap`
    since it maps all kwargs along the 0 axis.

    Map a function over positional args while broadcasting a keyword arg:

    >>> def f(x, *, scale):
    ...     return x * scale

    >>> vf = vmap(f, in_kw=True, default_kw_axis=None)
    >>> vf(jnp.arange(3.0), scale=2.0)
    Array([0., 2., 4.], dtype=float32)

    Map both a positional arg and a keyword arg over axis 0:

    >>> def g(x, *, weights):
    ...     return x * weights

    >>> vg = vmap(g, in_kw={"weights": 0})
    >>> vg(jnp.ones((4, 2)), weights=jnp.arange(4.0))
    Array([[0., 0.],
           [1., 1.],
           [2., 2.],
           [3., 3.]], dtype=float32)

    Mix mapped and broadcast kwargs:

    >>> def h(x, *, a, b):
    ...     return x * a + b

    >>> vh = vmap(h, in_kw={"a": 0, "b": None})
    >>> vh(jnp.ones(3), a=jnp.array([1.0, 2.0, 3.0]), b=10.0)
    Array([11., 12., 13.], dtype=float32)

    When ``in_kw`` is a ``Mapping`` and ``default_kw_axis`` is 0 (the default),
    a kwarg fast path is used since for non-``in_kw`` kwargs {func}`jax.vmap`'s
    default behavior is applied.

    >>> def f(x, *, scale, bias):
    ...     return x * scale + bias

    >>> vf = vmap(f, in_kw={"scale": None})
    >>> vf(jnp.arange(3.0), scale=2.0, bias=jnp.array([10.0, 20.0, 30.0]))
    Array([10., 22., 34.], dtype=float32)

    Here ``scale`` is broadcast (axis ``None``) while ``bias`` — not listed in
    ``in_kw`` — is mapped along axis 0 by {func}`jax.vmap`'s default behavior.

    """
    # Short circuit to a simpler implementation if we have the same behaviour as
    # `jax.vmap`: no kwargs are mapped (`in_kw` is False) or all kwargs are
    # mapped along axis 0 (`in_kw` is True and `default_kw_axis` is 0).
    if in_kw is False or (in_kw is True and default_kw_axis == 0):
        return _vmap_static(
            func,
            in_axes=in_axes,
            out_axes=out_axes,
            static_args=static_args,
            static_kw=static_kw,
            jit=jit,
        )

    # When default_kw_axis == 0 and in_kw is a Mapping, unknown kwargs get
    # axis 0 — the same as jax.vmap's default.  Only the explicitly listed
    # kwargs need special axis treatment, and that set is fully known at
    # vmap-time, so we can pre-build the jax.vmap once.
    if default_kw_axis == 0 and isinstance(in_kw, Mapping):
        return _vmap_kw(
            func,
            in_axes=in_axes,
            out_axes=out_axes,
            in_kw=in_kw,
            static_args=static_args,
            static_kw=static_kw,
            jit=jit,
        )

    # General path: supports arbitrary default_kw_axis values and dynamic
    # kwarg sets.  Caches a vmapped wrapper per unique (treedef, kwarg-names).
    return _vmap_general(
        func,
        in_axes=in_axes,
        out_axes=out_axes,
        in_kw=in_kw,
        default_kw_axis=default_kw_axis,
        static_args=static_args,
        static_kw=static_kw,
        jit=jit,
    )
