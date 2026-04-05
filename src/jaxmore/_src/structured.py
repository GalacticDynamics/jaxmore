"""Structured argument and return value handling.

This module provides the public ``structured`` decorator and internal helpers
for structured processing of function arguments and return values.
"""

__all__ = ("structured",)

import functools as ft
import inspect
from collections.abc import Callable, Iterable
from inspect import Parameter
from typing import Any, ParamSpec, TypeAlias

P = ParamSpec("P")

# Each slot of the `ins` tuple targets one class of function parameter.
ArgF: TypeAlias = Callable[[Any], Any]
PosFs: TypeAlias = tuple[ArgF | None, ...]  # per-positional-param processors
ArgsF: TypeAlias = ArgF | None  # single processor for *args elements
KwFs: TypeAlias = dict[str, ArgF | None]  # per-keyword-only-param processors
KwargsF: TypeAlias = ArgF | None  # single processor for **kwargs values

# The full `ins` type: a 0-to-4-element tuple, where each successive
# length adds the next parameter-class processor.
Ins: TypeAlias = (
    None
    | tuple[()]
    | ArgF
    | tuple[ArgF]
    | tuple[PosFs]
    | tuple[PosFs, ArgsF]
    | tuple[PosFs, ArgsF, KwFs]
    | tuple[PosFs, ArgsF, KwFs, KwargsF]
)

# Type alias for the `outs` parameter of `structured`
Outs: TypeAlias = tuple[ArgF | None, ...] | Iterable[ArgF | None] | ArgF | None

_POS_OR_KW = Parameter.POSITIONAL_OR_KEYWORD
_POS_KINDS = frozenset((Parameter.POSITIONAL_ONLY, _POS_OR_KW))
_VAR_POS = Parameter.VAR_POSITIONAL
_KW_ONLY = Parameter.KEYWORD_ONLY
_VAR_KW = Parameter.VAR_KEYWORD


def _build_call_plan(  # noqa: C901
    sig: inspect.Signature,
    pos_fs: tuple[ArgF | None, ...],
    args_f: ArgF | None,
    kw_fs: dict[str, ArgF | None],
    kwargs_f: ArgF | None,
) -> tuple[
    tuple[tuple[int, ArgF], ...],  # pos_actions
    tuple[int, ArgF] | None,  # var_pos (start_index, processor)
    tuple[tuple[str, ArgF], ...],  # kw_actions
    ArgF | None,  # kwargs_f
    bool,  # has_defaults
    frozenset[str],  # pos_or_kw_names
    int,  # n_pos
    dict[str, Any],  # kw_defaults
]:
    """Build an index-based call plan for the wrapper.

    Returns an 8-tuple::

        (pos_actions, var_pos, kw_actions, kwargs_f,
         has_defaults, pos_or_kw_names, n_pos, kw_defaults)

    pos_actions    — tuple of (arg_index, processor) for POSITIONAL_* params
    var_pos        — tuple[int, ArgF] | None: (start_index, processor) for
                     *args, or None when no *args param or no processor
    kw_actions     — tuple of (name, processor) for KEYWORD_ONLY params
    kwargs_f       — the raw **kwargs processor, or None
    has_defaults   — True iff any processed positional param has a
                     non-empty default
    pos_or_kw_names — frozenset of POSITIONAL_OR_KEYWORD param names that
                      have active processors (for kwarg-fallback detection)
    n_pos          — total count of positional (non-VAR) params in the sig
    kw_defaults    — dict mapping KEYWORD_ONLY param names (with processors)
                     to their default values, for omitted-kwarg processing
    """
    pos_actions: list[tuple[int, ArgF]] = []
    kw_actions: list[tuple[str, ArgF]] = []
    kw_defaults: dict[str, Any] = {}
    var_pos_start: int = -1
    needs_bind = False
    pos_or_kw_names: set[str] = set()

    # Walk the signature once, mapping each parameter to its processor.
    # pos_idx tracks which element of pos_fs we're on (user-supplied tuple),
    # arg_idx tracks the parameter's position in the call-time *args tuple.
    pos_idx = 0
    arg_idx = 0
    for name, param in sig.parameters.items():
        if param.kind in _POS_KINDS:
            f = pos_fs[pos_idx] if pos_idx < len(pos_fs) else None
            if f is not None:
                pos_actions.append((arg_idx, f))
                # Any processed positional param with a default may be
                # omitted by the caller, requiring sig.bind to fill it.
                if param.default is not Parameter.empty:
                    needs_bind = True
                # Track POS_OR_KW params with processors — these can be
                # passed as kwargs at call time, which defeats index-based
                # lookup and requires sig.bind fallback.
                if param.kind is _POS_OR_KW:
                    pos_or_kw_names.add(name)
            pos_idx += 1
            arg_idx += 1
        elif param.kind is _VAR_POS:
            # *args starts at this index; arg_idx is NOT incremented
            # because *args consumes a variable number of positions.
            var_pos_start = arg_idx
        elif param.kind is _KW_ONLY:
            f = kw_fs.get(name)
            if f is not None:
                kw_actions.append((name, f))
                if param.default is not Parameter.empty:
                    kw_defaults[name] = param.default
        elif param.kind is _VAR_KW:
            pass  # kwargs_f is passed through directly
        else:
            continue

    # Validate: args_f supplied but no *args in the signature.
    if args_f is not None and var_pos_start < 0:
        msg = (
            "ins[1] (args_f) provides a *args processor, but the "
            "decorated function has no *args parameter"
        )
        raise TypeError(msg)

    # Validate: kwargs_f supplied but no **kwargs in the signature.
    if kwargs_f is not None and not any(
        p.kind is _VAR_KW for p in sig.parameters.values()
    ):
        msg = (
            "ins[3] (kwargs_f) provides a **kwargs processor, but the "
            "decorated function has no **kwargs parameter"
        )
        raise TypeError(msg)

    return (
        tuple(pos_actions),  # pos_actions
        (var_pos_start, args_f)
        if var_pos_start >= 0 and args_f is not None
        else None,  # var_pos
        tuple(kw_actions),  # kw_actions
        kwargs_f,  # kwargs_f
        needs_bind,  # has_defaults
        frozenset(pos_or_kw_names),  # pos_or_kw_names
        arg_idx,  # n_pos
        kw_defaults,  # kw_defaults
    )


# No-op passthrough used when no output processing is requested.
def _identity(out: Any, /) -> Any:
    return out


def _build_process_out(
    outs: tuple[ArgF | None, ...] | None,
) -> Callable[[Any], Any]:
    """Return a specialised output-processing function.

    Builds a closure tailored to the shape of ``outs``:

    - ``None`` -> identity (no processing).
    - Length-1 tuple -> apply the single processor directly to the return
      value (no sequence unpacking).
    - Length-N tuple -> expect the function to return a sequence of N
      elements; apply each non-None processor at its index and return
      the result as a tuple.

    Pre-building the closure at decoration time avoids per-call branching.
    """
    # No output processors — return the identity passthrough.
    if outs is None:
        return _identity

    n = len(outs)
    # Collect only the (index, processor) pairs that are non-None so the
    # hot loop at call time touches only active slots.
    active = tuple((i, f) for i, f in enumerate(outs) if f is not None)

    if n == 1:
        # Single-value mode: the function returns one value, not a
        # sequence, so we apply the processor directly (no unpacking).
        return active[0][1] if active else _identity

    # Multi-value mode: the function returns a sequence of exactly ``n``
    # elements.  We convert to a mutable list, apply active processors
    # in-place, then freeze back to a tuple.
    def _process_out(out: Iterable[Any], /) -> tuple[Any, ...]:
        out = list(out)
        if len(out) != n:
            raise ValueError(f"structured: outs expects {n} values, got {len(out)}")  # noqa: EM102
        for i, f in active:
            out[i] = f(out[i])
        return tuple(out)

    return _process_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def structured(  # noqa: C901
    ins: Ins = None, outs: Outs = None
) -> Callable[[Callable[P, Any]], Callable[P, Any]]:
    r"""Close (de)structuring I/O from/to arrays.

    In JAX code the fastest functions are those that operate on arrays
    end-to-end. In particular, when a PyTree crosses a JIT boundary, JAX has to
    perform operations like (un)flattening, etc. which can be costly. One way to
    avoid this overhead is to push the structuring logic inside the JIT-compiled
    function, so that the function accepts and returns arrays directly, and the
    structuring is handled by the caller. The ``@structured`` decorator makes it
    easy to write such functions by allowing you to specify how to transform
    function arguments from arrays and return values as arrays at the call
    boundary. The (de)structuring logic is always applied during any tracing
    operations, e.g. JIT compilation, but after that JAX calls the XLA-compiled
    code directly with arrays, so there is no further overhead from the
    structuring logic at runtime.

    Parameters
    ----------
    ins : tuple or callable, optional
        A bare callable ``f`` is sugar for ``((f,),)``, i.e. a single processor
        applied to the first positional parameter.  Likewise, ``(f,)`` is
        equivalent to ``((f,),)``.

        Otherwise, up to 4-element tuple specifying how to process each argument
        class:

        1. ``tuple[ArgF | None, ...]`` - one callable per positional
           (``POSITIONAL_ONLY`` or ``POSITIONAL_OR_KEYWORD``) parameter, in
           declaration order.  A bare callable is sugar for a 1-tuple.  Trailing
           parameters with no entry are skipped.
        2. ``ArgF | None`` - applied element-wise to every ``*args`` value.
        3. ``dict[str, ArgF | None]`` - per-name callables for keyword-only
           parameters.  Unlisted names are skipped.
        4. ``ArgF | None`` - applied to every value in ``**kwargs``.

        ``None`` at any position means "pass through unchanged".

    outs : tuple[ArgF | None, ...] | ArgF | None, optional
        How to process the return value:

        - ``None`` (default) - return unchanged.
        - Single callable - applied to the whole return value.
        - Tuple - zipped element-by-element with the returned sequence; ``None``
          entries pass through unchanged.  A length mismatch raises
          ``ValueError``.

    Returns
    -------
    Callable
        A decorator that transparently wraps *func*, applying the specified
        processors inside the call boundary (i.e. inside a ``jax.jit`` if the
        wrapper is applied before JIT compilation).

    Examples
    --------
    The examples below use trivial processors (dicts, negation, etc.) to
    illustrate the decorator's mechanics.  In practice, ``ins`` and ``outs``
    processors will generally be used to convert between rich domain objects and
    flat arrays at a JIT boundary -- the simple lambdas here just keep the
    examples readable.

    >>> import jax
    >>> import jax.numpy as jnp

    **Bare callable shorthand** — process the first positional argument.
    ``ins=f`` is sugar for ``ins=((f,),)``:

    >>> @structured(ins=lambda x: {"value": x})
    ... def increment(obj):
    ...     return obj["value"] + 1

    >>> increment(3)
    4

    **Multiple positional processors** — one callable per positional param,
    matched left-to-right.  ``None`` skips the corresponding argument:

    >>> to_point = lambda xy: {"x": xy[0], "y": xy[1]}
    >>> to_vec   = lambda xy: {"dx": xy[0], "dy": xy[1]}

    >>> @structured(ins=((to_point, to_vec),))
    ... def translate(pt, v):
    ...     return {"x": pt["x"] + v["dx"], "y": pt["y"] + v["dy"]}

    >>> translate((1, 2), (10, 20))
    {'x': 11, 'y': 22}

    >>> @structured(ins=((None, to_vec),))
    ... def shift(pt, v):
    ...     return {"x": pt["x"] + v["dx"], "y": pt["y"] + v["dy"]}

    >>> shift({"x": 1, "y": 2}, (10, 20))
    {'x': 11, 'y': 22}

    **VAR_POSITIONAL (\\*args)** — a single processor is applied element-wise to
    every value in ``*args``:

    >>> @structured(ins=((), lambda v: {"val": v}))
    ... def collect(*args):
    ...     return tuple(a["val"] for a in args)

    >>> collect(1, 2, 4)
    (1, 2, 4)

    **Keyword-only parameters** — matched by name via the third ``ins`` slot:

    >>> @structured(ins=((), None, {"cfg": lambda d: {**d, "ready": True}}))
    ... def init(x, *, cfg):
    ...     return cfg["ready"], x

    >>> init(5, cfg={"name": "test"})
    (True, 5)

    **VAR_KEYWORD (\\**kwargs)** — a single processor is applied to every value
    in ``**kwargs``:

    >>> @structured(ins=((), None, {}, lambda v: {"val": v}))
    ... def wrap_kw(**kwargs):
    ...     return {k: obj["val"] for k, obj in kwargs.items()}

    >>> wrap_kw(a=1, b=4)
    {'a': 1, 'b': 4}

    **Output processing** — ``outs=f`` applies ``f`` to the whole return value;
    a tuple applies each element independently (``None`` passes through):

    >>> @structured(outs=lambda d: d["result"])
    ... def compute(x):
    ...     return {"result": x + 1, "debug": "ok"}

    >>> compute(4)
    5

    >>> @structured(outs=(lambda d: d["val"], None, lambda d: d["val"]))
    ... def multi_out():
    ...     return ({"val": 10}, 2, {"val": 103})

    >>> multi_out()
    (10, 2, 103)

    **Combined** — processors on inputs and outputs together.  Default parameter
    values are visible to ``ins`` processors:

    >>> @structured(
    ...     ins=((None, lambda v: {"scale": v}),),
    ...     outs=lambda d: d["result"],
    ... )
    ... def apply_scale(x, scale=2):
    ...     return {"result": x * scale["scale"]}

    >>> apply_scale(3)  # scale defaults to 2 -> {"scale": 2}
    6
    >>> apply_scale(3, 4)  # scale=4 -> {"scale": 4}
    12

    **JAX / JIT integration** — processors run *inside* the JIT boundary when
    ``@jax.jit`` is applied outside ``@structured``:

    >>> @jax.jit
    ... @structured(
    ...     ins=(lambda x: {"val": x},),
    ...     outs=lambda d: d["val"],
    ... )
    ... def jit_func(obj):
    ...     return {"val": obj["val"] + jnp.asarray(1)}

    >>> int(jit_func(jnp.asarray(4)))
    5

    """
    # --- Normalise `ins` into its four constituent slots ----------------
    # The user may pass 0-4 elements; we pad with None so downstream code
    # can always destructure into exactly four variables.
    #   slot 0: pos_fs   - tuple of per-positional-param processors
    #   slot 1: args_f   - single processor for *args elements
    #   slot 2: kw_fs    - dict of per-keyword-only-param processors
    #   slot 3: kwargs_f - single processor for **kwargs values
    _ins = ins if ins is not None else ()
    # Allow a bare callable as shorthand for a single positional processor.
    if callable(_ins):
        _ins = ((_ins,),)
    elif len(_ins) > 4:
        raise ValueError("`ins` must be a tuple with at most 4 elements.")
    pos_fs, args_f, kw_fs, kwargs_f = (*_ins, None, None, None, None)[:4]
    # Allow a bare callable as shorthand for a 1-tuple of positionals.
    pos_fs = (pos_fs,) if callable(pos_fs) else (pos_fs or ())
    kw_fs = kw_fs or {}

    # --- Normalise `outs` -----------------------------------------------
    # A bare callable is wrapped in a 1-tuple; any other non-tuple iterable
    # (e.g. list) is coerced to a tuple so _build_process_out always
    # receives None or a tuple.
    if callable(outs):
        _outs: tuple[ArgF | None, ...] | None = (outs,)
    elif outs is None or isinstance(outs, tuple):
        _outs = outs
    else:
        _outs = tuple(outs)

    # --- Pre-check: passthrough shortcut --------------------------------
    # When no input or output processors are configured, skip all wrapper
    # overhead and return the original function unchanged.
    _has_any_ins = (
        bool(pos_fs) or args_f is not None or bool(kw_fs) or kwargs_f is not None
    )
    if not _has_any_ins and _outs is None:
        return _identity_decorator

    def attach_io_processors(func: Callable[P, Any]) -> Callable[P, Any]:  # noqa: C901
        """Attach the configured I/O processors to *func* and return the result.

        This is the decorator returned by :func:`structured`.  It closes over
        the normalised processor slots built by the enclosing call to
        ``structured(ins=..., outs=...)``.

        At decoration time, one of four specialised wrapper strategies is
        selected based on the function's signature and the active processors:

        1. **No-op** — all processors resolve to ``None`` after plan
           construction -> return *func* unchanged.
        2. **Outs-only** — no input processors active, only output processing
           -> minimal wrapper: ``process_out(func(*args, **kwargs))``.
        3. **Bind-free** — active input processors but no
           ``POSITIONAL_OR_KEYWORD`` params have processors -> no ``sig.bind``
           guard needed; index-based mutations only.
        4. **Bind-capable** — at least one ``POSITIONAL_OR_KEYWORD`` param has
           a processor -> guard detects keyword-style calls and falls back to
           ``sig.bind`` + ``apply_defaults``, then reuses the same index-based
           processing on the normalised ``args``/``kwargs``.

        Parameters
        ----------
        func:
            The callable to wrap.  Its signature is inspected once here;
            subsequent calls to the wrapper do not re-inspect it.

        Returns
        -------
        Callable
            A wrapped version of *func* (or *func* itself for the no-op case).

        """
        # --- Decoration-time setup (runs once per decorated function) ---
        sig = inspect.signature(func)
        (
            _pos_actions,
            _var_pos,
            _kw_actions,
            _kwargs_f_val,
            _has_defaults,
            _pos_or_kw_names,
            _n_pos,
            _kw_defaults,
        ) = _build_call_plan(sig, pos_fs, args_f, kw_fs, kwargs_f)
        process_out = _build_process_out(_outs)

        _has_active_ins = (
            bool(_pos_actions)
            or _var_pos is not None
            or bool(_kw_actions)
            or _kwargs_f_val is not None
        )

        # ---- Variant A: No-op ------------------------------------------
        # All ins processors resolved to None and no outs processing.
        if not _has_active_ins and process_out is _identity:
            return func

        # ---- Variant B: Outs-only --------------------------------------
        # No input processors active, only output processing.
        if not _has_active_ins:

            @ft.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return process_out(func(*args, **kwargs))

        # ---- Variant C: Bind-free --------------------------------------
        # No POSITIONAL_OR_KEYWORD params have processors, so callers can
        # never pass a processed arg by keyword — no sig.bind guard needed.

        elif not _pos_or_kw_names:
            # Pre-compute all declared (non-variadic) parameter names so
            # the **kwargs loop only processes true **kwargs extras, never
            # explicit params (POS_OR_KW passed by keyword, or KW_ONLY).
            _declared_names = frozenset(
                p.name
                for p in sig.parameters.values()
                if p.kind is not _VAR_POS and p.kind is not _VAR_KW
            )
            _has_pos = bool(_pos_actions) or _var_pos is not None

            if _has_pos:
                # C1: Has positional/varargs processing -> must copy args
                # to a mutable list for in-place index mutation.
                @ft.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:  # noqa: C901  # pylint: disable=function-redefined
                    # Guard: if processed positional params have defaults
                    # and the caller omitted some, normalise via sig.bind.
                    if _has_defaults and len(args) < _n_pos:
                        ba = sig.bind(*args, **kwargs)
                        ba.apply_defaults()
                        args = ba.args  # type: ignore[assignment]
                        kwargs = ba.kwargs  # type: ignore[assignment]
                    args_l = list(args)
                    # Apply per-positional processors by index.
                    for i, f in _pos_actions:
                        args_l[i] = f(args_l[i])
                    # Apply *args processor element-wise beyond the fixed params.
                    if _var_pos is not None:
                        _vp_start, _vp_f = _var_pos
                        for j in range(_vp_start, len(args_l)):
                            args_l[j] = _vp_f(args_l[j])
                    # Apply per-keyword-only processors by name.
                    for name, f in _kw_actions:
                        if name in kwargs:
                            kwargs[name] = f(kwargs[name])
                        elif name in _kw_defaults:
                            kwargs[name] = f(_kw_defaults[name])
                    # Apply **kwargs processor to remaining keys (skip
                    # declared param names to avoid processing them).
                    if _kwargs_f_val is not None:
                        for k, v in kwargs.items():
                            if k not in _declared_names:
                                kwargs[k] = _kwargs_f_val(v)
                    return process_out(func(*args_l, **kwargs))  # type: ignore[arg-type]
            else:
                # C2: Only kw-only / **kwargs processing -> args tuple is
                # passed through untouched (no list copy needed).
                @ft.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:  # pylint: disable=function-redefined
                    # Apply per-keyword-only processors by name.
                    for name, f in _kw_actions:
                        if name in kwargs:
                            kwargs[name] = f(kwargs[name])
                        elif name in _kw_defaults:
                            kwargs[name] = f(_kw_defaults[name])
                    # Apply **kwargs processor to remaining keys (skip
                    # declared param names to avoid processing them).
                    if _kwargs_f_val is not None:
                        for k, v in kwargs.items():
                            if k not in _declared_names:
                                kwargs[k] = _kwargs_f_val(v)
                    return process_out(func(*args, **kwargs))

        # ---- Variant D: Bind-capable -----------------------------------
        # At least one POSITIONAL_OR_KEYWORD param has a processor.  These
        # params can be passed positionally OR as keywords, so we need a
        # runtime guard to detect the keyword case and normalise via
        # sig.bind before applying index-based processing.

        else:
            _declared_names = frozenset(
                p.name
                for p in sig.parameters.values()
                if p.kind is not _VAR_POS and p.kind is not _VAR_KW
            )
            _has_pos = bool(_pos_actions) or _var_pos is not None

            @ft.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:  # noqa: C901  # pylint: disable=function-redefined
                # Guard: check if any processed POS_OR_KW param was passed as
                # a keyword (intersection with kwargs keys), OR if processed
                # params with defaults were omitted (fewer args than expected).
                # In either case, sig.bind resolves all params into canonical
                # positions and fills defaults.  ba.args / ba.kwargs produce
                # the same index layout as a normal positional call, so the
                # index-based processing below works identically.
                if _pos_or_kw_names & kwargs.keys() or (
                    _has_defaults and len(args) < _n_pos
                ):
                    ba = sig.bind(*args, **kwargs)
                    ba.apply_defaults()
                    args = ba.args  # type: ignore[assignment]
                    kwargs = ba.kwargs  # type: ignore[assignment]

                # Index-based processing — identical to Variant C1 above.
                # After the guard, args/kwargs have the same index layout
                # regardless of whether sig.bind was invoked.
                args_l = list(args)
                # Apply per-positional processors by index.
                for i, f in _pos_actions:
                    args_l[i] = f(args_l[i])
                # Apply *args processor element-wise beyond the fixed params.
                if _var_pos is not None:
                    _vp_start, _vp_f = _var_pos
                    for j in range(_vp_start, len(args_l)):
                        args_l[j] = _vp_f(args_l[j])
                # Apply per-keyword-only processors by name.
                for name, f in _kw_actions:
                    if name in kwargs:
                        kwargs[name] = f(kwargs[name])
                    elif name in _kw_defaults:
                        kwargs[name] = f(_kw_defaults[name])
                # Apply **kwargs processor to remaining keys (skip
                # declared param names to avoid processing them).
                if _kwargs_f_val is not None:
                    for k, v in kwargs.items():
                        if k not in _declared_names:
                            kwargs[k] = _kwargs_f_val(v)
                return process_out(func(*args_l, **kwargs))  # type: ignore[arg-type]

        return wrapper

    return attach_io_processors


def _identity_decorator(func: Callable[P, Any]) -> Callable[P, Any]:
    """Passthrough decorator used when no I/O processors are configured."""
    return func
