"""Unit tests for jaxmore.structured."""

import pytest

from jaxmore import structured

# ============================================================================
# A. Input shorthand normalization
# ============================================================================


class TestInsShorthands:
    """The three equivalent sugar forms for a single positional processor."""

    def test_bare_callable(self) -> None:
        """ins=f is shorthand for ((f,),)."""

        @structured(ins=lambda x: x * 2)
        def f(x):
            return x

        assert f(3) == 6

    def test_single_tuple(self) -> None:
        """ins=(f,) is shorthand for ((f,),)."""

        @structured(ins=(lambda x: x * 2,))
        def f(x):
            return x

        assert f(3) == 6

    def test_nested_explicit(self) -> None:
        """ins=((f,),) is the explicit positional-tuple form."""

        @structured(ins=((lambda x: x * 2,),))
        def f(x):
            return x

        assert f(3) == 6

    def test_empty_tuple_no_processing(self) -> None:
        """ins=((),) with no processors is a passthrough."""

        def original(x):
            return x

        wrapped = structured(ins=((),))(original)
        assert wrapped is original

    def test_ins_tuple_too_long_raises(self) -> None:
        """Ins with more than 4 elements raises ValueError at decoration time."""
        f = lambda x: x
        # Use __wrapped__ when available (e.g. beartype/jaxtyping wraps it) to
        # bypass the external type-check guard and hit the body's ValueError.
        _structured = getattr(structured, "__wrapped__", structured)
        with pytest.raises(
            ValueError, match="`ins` must be a tuple with at most 4 elements"
        ):
            _structured(ins=(f, None, {}, None, f), outs=None)


# ============================================================================
# B. Positional parameter processing
# ============================================================================


class TestPositionalProcessing:
    """Processing POSITIONAL_ONLY and POSITIONAL_OR_KEYWORD parameters."""

    def test_multiple_positionals(self) -> None:
        @structured(ins=((lambda x: x * 2, lambda y: y + 10),))
        def f(x, y):
            return x, y

        assert f(3, 5) == (6, 15)

    def test_none_skips_positional(self) -> None:
        """None in pos_fs leaves the corresponding argument unchanged."""

        @structured(ins=((None, lambda y: y + 10),))
        def f(x, y):
            return x, y

        assert f(3, 5) == (3, 15)

    def test_positional_only_params(self) -> None:
        """POSITIONAL_ONLY parameters (/) are indexed correctly."""

        @structured(ins=((lambda x: x * 2, lambda y: y + 1),))
        def f(x, /, y):
            return x, y

        assert f(3, 4) == (6, 5)

    def test_positional_only_unprocessed_neighbour(self) -> None:
        """POS_ONLY processed, adjacent POS_OR_KW left unprocessed."""

        @structured(ins=((lambda a: a + 100,),))
        def f(a, /, b):
            return a, b

        assert f(1, 2) == (101, 2)

    def test_trailing_extras_ignored(self) -> None:
        """Extra pos_fs entries beyond the parameter count are silently ignored."""

        @structured(ins=((lambda x: x * 2, lambda x: x * 100),))
        def f(x):
            return x

        assert f(3) == 6


# ============================================================================
# C. *args processing
# ============================================================================


class TestVarPositionalProcessing:
    """Processing VAR_POSITIONAL (*args) parameters."""

    def test_element_wise(self) -> None:
        @structured(ins=((), lambda x: x * 2))
        def f(*args):
            return args

        assert f(1, 2, 3) == (2, 4, 6)

    def test_empty_args(self) -> None:
        """No *args passed -- processor is never called, no error."""

        @structured(ins=((), lambda x: x * 2))
        def f(*args):
            return args

        assert f() == ()

    def test_with_fixed_positionals(self) -> None:
        """*args processor only runs on the variadic portion."""

        @structured(ins=((lambda x: x * 2,), lambda v: v + 10))
        def f(x, *args):
            return x, args

        assert f(3, 1, 2) == (6, (11, 12))

    def test_args_f_without_var_positional_raises(self) -> None:
        """Supplying args_f for a function without *args raises at decoration time."""
        with pytest.raises(TypeError, match=r"\*args"):

            @structured(ins=((), lambda x: x * 2))
            def f(x):
                return x


# ============================================================================
# D. Keyword-only processing
# ============================================================================


class TestKeywordOnlyProcessing:
    """Processing KEYWORD_ONLY parameters."""

    def test_by_name(self) -> None:
        @structured(ins=((), None, {"scale": lambda v: v * 2}))
        def f(x, *, scale):
            return x * scale

        assert f(5, scale=3) == 30

    def test_unknown_key_silently_skipped(self) -> None:
        """Keys in kw_fs that do not match any param are silently ignored."""

        @structured(ins=((), None, {"nonexistent": lambda v: v * 100}))
        def f(x):
            return x

        assert f(5) == 5

    def test_multiple_params(self) -> None:
        """Multiple kw-only params each with their own processor."""

        @structured(ins=((), None, {"a": lambda v: v * 2, "b": lambda v: v * 3}))
        def f(*, a, b):
            return a + b

        assert f(a=3, b=4) == 6 + 12

    def test_default_processed_when_omitted(self) -> None:
        """Processor is applied to the default value when caller omits the kwarg."""

        @structured(ins=((), None, {"scale": lambda v: v * 10}))
        def f(*, scale=5):
            return scale

        assert f() == 50  # default 5 → processor → 50
        assert f(scale=3) == 30  # supplied 3 → processor → 30

    def test_default_processed_when_omitted_with_varargs(self) -> None:
        """Default kw-only processing works when the function also has *args."""

        @structured(ins=((), None, {"offset": lambda v: v + 100}))
        def g(*args, offset=1):
            return sum(args) + offset

        assert g(1, 2) == 3 + 101  # offset default 1 → 101
        assert g(1, 2, offset=5) == 3 + 105  # offset supplied 5 → 105


# ============================================================================
# E. **kwargs processing
# ============================================================================


class TestVarKeywordProcessing:
    """Processing VAR_KEYWORD (**kwargs) parameters."""

    def test_values_processed(self) -> None:
        @structured(ins=((), None, {}, lambda v: v * 2))
        def f(**kwargs):
            return kwargs

        assert f(a=1, b=3) == {"a": 2, "b": 6}

    def test_empty_no_error(self) -> None:
        @structured(ins=((), None, {}, lambda v: v * 2))
        def f(x, **kwargs):
            return x, kwargs

        assert f(5) == (5, {})

    def test_kw_only_not_double_processed(self) -> None:
        """Kw-only params are handled by kw_fs, not also by kwargs_f."""

        @structured(ins=((), None, {"k": lambda v: v * 3}, lambda v: v * 2))
        def f(*, k, **kwargs):
            return k, kwargs

        # k=2 -> *3 -> 6 (NOT also *2); a=1 -> *2 -> 2
        assert f(k=2, a=1) == (6, {"a": 2})

    def test_kw_only_without_processor_not_processed_by_kwargs_f(self) -> None:
        """Kw-only params without a processor must not be fed to kwargs_f."""

        @structured(ins=((), None, {}, lambda v: v * 2))
        def f(*, k, **kwargs):
            return k, kwargs

        # k has no processor and must NOT be touched by kwargs_f
        assert f(k=2, a=1) == (2, {"a": 2})

    def test_kwargs_f_without_var_keyword_raises(self) -> None:
        """Giving kwargs_f for a function without **kwargs raises at decoration time."""
        with pytest.raises(TypeError, match=r"\*\*kwargs"):

            @structured(ins=((), None, {}, lambda v: v * 2))
            def f(x):
                return x

    def test_pos_or_kw_passed_as_keyword_not_processed_by_kwargs_f(self) -> None:
        """POS_OR_KW param w/out processor, passed as kwarg, not touched by kwargs_f."""

        @structured(ins=((), None, {}, lambda v: v * 2))
        def f(x, **kw):
            return x, kw

        assert f(5, a=1) == (5, {"a": 2})
        assert f(x=5, a=1) == (5, {"a": 2})  # x passed as keyword

    def test_kw_only_with_explicit_none_processor_not_processed_by_kwargs_f(
        self,
    ) -> None:
        """kw_fs={"k": None} must still exclude k from kwargs_f processing."""

        @structured(ins=((), None, {"k": None}, lambda v: v * 2))
        def f(*, k, **kw):
            return k, kw

        assert f(k=2, a=1) == (2, {"a": 2})


# ============================================================================
# F. Output processing
# ============================================================================


class TestOutsProcessing:
    """Tests for the outs parameter."""

    def test_none_is_identity(self) -> None:
        @structured(outs=None)
        def f(x):
            return x * 3

        assert f(4) == 12

    def test_bare_callable(self) -> None:
        """outs=f applies f to the whole return value."""

        @structured(outs=lambda r: r * 10)
        def f(x):
            return x + 1

        assert f(2) == 30

    def test_single_element_tuple_equivalent(self) -> None:
        """outs=(f,) is equivalent to outs=f."""

        @structured(outs=(lambda r: r * 5,))
        def f(x):
            return x

        assert f(3) == 15

    def test_tuple_multi_value(self) -> None:
        """outs=(f, None, g) applies processors element-wise."""

        @structured(outs=(lambda r: r * 2, None, lambda r: r + 100))
        def f():
            return (3, 5, 7)

        assert f() == (6, 5, 107)

    def test_all_none_passthrough(self) -> None:
        @structured(outs=(None, None))
        def f():
            return (4, 8)

        assert f() == (4, 8)

    def test_length_mismatch_raises(self) -> None:
        @structured(outs=(lambda r: r, lambda r: r))
        def f():
            return (1, 2, 3)

        with pytest.raises(ValueError, match="expects 2 values, got 3"):
            f()

    def test_list_outs_normalized_to_tuple(self) -> None:
        """outs=[f, None, g] (a list) is normalized to a tuple and works correctly."""

        @structured(outs=[lambda r: r * 2, None, lambda r: r + 100])
        def f():
            return (3, 5, 7)

        assert f() == (6, 5, 107)


# ============================================================================
# G. Passthrough / no-op detection
# ============================================================================


class TestPassthrough:
    """When no effective processors exist, structured returns func itself."""

    def test_no_args_returns_func(self) -> None:
        def original(x):
            return x

        assert structured()(original) is original

    def test_all_none_ins_returns_func(self) -> None:
        """Ins with all-None processors returns the original function."""

        def original(x, y):
            return x + y

        assert structured(ins=((None, None),))(original) is original

    def test_outs_only_still_wraps(self) -> None:
        """When ins are all None but outs is set, a wrapper is still created."""

        @structured(ins=((None,),), outs=lambda r: r * 10)
        def f(x):
            return x + 1

        assert f(2) == 30
        assert hasattr(f, "__wrapped__")


# ============================================================================
# H. Default parameter handling
# ============================================================================


class TestDefaultParameters:
    """Default values are filled before processors run."""

    def test_omitted_default_seen_by_processor(self) -> None:
        @structured(ins=((None, lambda v: v * 2),))
        def f(x, y=10):
            return x + y

        assert f(3) == 23  # 3 + (10 * 2)

    def test_supplied_value_overrides_default(self) -> None:
        @structured(ins=((None, lambda v: v * 2),))
        def f(x, y=10):
            return x + y

        assert f(3, 5) == 13  # 3 + (5 * 2)

    def test_omitted_default_with_varargs(self) -> None:
        """Default is filled even when *args follows."""

        @structured(ins=((lambda x: x * 2, lambda y: y + 10),))
        def f(x, y=5, *args):
            return x, y, args

        assert f(3) == (6, 15, ())
        assert f(3, 4, 100) == (6, 14, (100,))

    def test_positional_only_default_omitted(self) -> None:
        """POS_ONLY param with default + processor; caller omits the arg."""

        @structured(ins=(lambda v: v * 2,))
        def f(x=1, /):
            return x

        assert f() == 2  # default 1 * 2
        assert f(5) == 10  # supplied 5 * 2


# ============================================================================
# I. Calling conventions (POS_OR_KW via keyword)
# ============================================================================


class TestCallingConventions:
    """POS_OR_KW params can be passed positionally or as keywords."""

    def test_positional_and_keyword_and_omitted(self) -> None:
        @structured(ins=((lambda x: x * 2, lambda y: y + 10),))
        def f(x, y=5):
            return x, y

        assert f(3, 4) == (6, 14)  # both positional
        assert f(3, y=4) == (6, 14)  # y as keyword
        assert f(3) == (6, 15)  # y omitted -> default 5 -> 15

    def test_all_args_as_keywords(self) -> None:
        @structured(ins=((lambda x: x * 2, lambda y: y + 10),))
        def f(x, y):
            return x, y

        assert f(x=3, y=5) == (6, 15)

    def test_kwarg_with_varargs(self) -> None:
        """POS_OR_KW passed as kwarg when *args is also present."""

        @structured(ins=((lambda x: x * 2,), lambda v: v + 10))
        def f(x, *args):
            return x, args

        assert f(3, 1, 2) == (6, (11, 12))
        assert f(x=3) == (6, ())

    def test_kwarg_with_var_keyword(self) -> None:
        """POS_OR_KW passed as kwarg when **kwargs is also present."""

        @structured(ins=((lambda x: x * 2,), None, {}, lambda v: v + 100))
        def f(x, **kwargs):
            return x, kwargs

        assert f(3, a=1) == (6, {"a": 101})
        assert f(x=3, a=1) == (6, {"a": 101})


# ============================================================================
# J. Mixed parameter kinds
# ============================================================================


class TestMixedParameterKinds:
    """Functions combining multiple parameter kinds in one signature."""

    def test_pos_varargs_kwonly(self) -> None:
        @structured(ins=((lambda x: x * 2,), lambda v: v + 10, {"k": lambda v: v * 3}))
        def f(x, *args, k):
            return x, args, k

        assert f(5, 1, 2, k=4) == (10, (11, 12), 12)

    def test_pos_varargs_varkw(self) -> None:
        @structured(ins=((lambda x: -x,), lambda v: v * 2, {}, lambda v: v + 100))
        def f(x, *args, **kwargs):
            return x, args, kwargs

        assert f(3, 4, 5, a=1, b=2) == (-3, (8, 10), {"a": 101, "b": 102})

    def test_pos_kwonly_varkw_with_defaults(self) -> None:
        """All four ins slots active + POS_OR_KW with default + keyword call."""

        @structured(
            ins=((lambda x: x * 2,), None, {"k": lambda v: v * 3}, lambda v: v + 100)
        )
        def f(x, y=5, *, k, **kw):
            return x, y, k, kw

        assert f(3, 4, k=2, a=1) == (6, 4, 6, {"a": 101})
        assert f(x=3, k=2, a=1) == (6, 5, 6, {"a": 101})

    def test_pos_only_and_pos_or_kw_mixed(self) -> None:
        """POS_ONLY + POS_OR_KW in the same signature, kwarg call supported."""

        @structured(ins=((lambda a: a + 100, lambda b: b * 2),))
        def f(a, /, b):
            return a, b

        assert f(1, 2) == (101, 4)
        assert f(1, b=2) == (101, 4)


# ============================================================================
# K. Combined ins + outs
# ============================================================================


class TestCombinedInsOuts:
    """Processors on both inputs and outputs together."""

    def test_ins_then_outs_ordering(self) -> None:
        """Ins runs first, then the body, then outs."""

        @structured(ins=(lambda x: x * 2,), outs=lambda r: -r)
        def f(x):
            return x + 1

        # x=3 -> ins: 6 -> body: 7 -> outs: -7
        assert f(3) == -7

    def test_multi_in_multi_out(self) -> None:
        @structured(
            ins=((lambda x: x * 2, lambda y: y + 10),),
            outs=(lambda r: -r, None),
        )
        def f(x, y):
            return x + y, x * y

        # x=3->6, y=5->15; body: (21, 90); outs: (-21, 90)
        assert f(3, 5) == (-21, 90)
