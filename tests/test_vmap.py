"""Unit tests for jaxmore.vmap."""

import jax
import jax.numpy as jnp
import pytest

from jaxmore import vmap

# ============================================================================
# Helpers


def scale_fn(x, *, scale):
    """X * scale — simple kwarg function."""
    return x * scale


def mul(factor, x, *, offset):
    """Factor * x + offset — for static_args / static_kw tests."""
    return factor * x + offset


def weighted_sum(x, *, a, b):
    """X * a + b — for multi-kwarg tests."""
    return x * a + b


# ============================================================================
# Static path (in_kw=False — default jax.vmap behaviour)


class TestVmapStatic:
    """Tests for the _vmap_static fast-path (in_kw=False)."""

    def test_basic_positional(self) -> None:
        """Positional-only vmap matches jax.vmap."""
        f = lambda x: x + 1
        x = jnp.arange(4.0)
        result = vmap(f)(x)
        expected = jax.vmap(f)(x)
        assert jnp.allclose(result, expected)

    def test_in_axes_none_broadcasts(self) -> None:
        """in_axes=None broadcasts the argument."""
        f = lambda x, y: x + y
        x, y = jnp.arange(3.0), jnp.array(10.0)
        result = vmap(f, in_axes=(0, None))(x, y)
        expected = jax.vmap(f, in_axes=(0, None))(x, y)
        assert jnp.allclose(result, expected)

    def test_in_axes_sequence(self) -> None:
        """Different axes for each positional arg."""
        # x: shape (3,), map axis 0;  y: shape (1, 3), map axis 1
        f = lambda x, y: x + y
        x = jnp.arange(3.0)
        y = jnp.ones((1, 3)) * jnp.array([[10.0, 20.0, 30.0]])
        result = vmap(f, in_axes=(0, 1))(x, y)
        expected = jax.vmap(f, in_axes=(0, 1))(x, y)
        assert jnp.allclose(result, expected)

    def test_out_axes(self) -> None:
        """out_axes controls output stacking axis."""
        # Each element produces a (2,) vector, so out_axes=1 stacks along dim 1
        f = lambda x: jnp.stack([x, x * 2])
        vf = vmap(f, out_axes=1)
        result = vf(jnp.arange(3.0))
        assert result.shape == (2, 3)
        assert jnp.allclose(result[0], jnp.arange(3.0))
        assert jnp.allclose(result[1], jnp.arange(3.0) * 2)

    def test_in_kw_true_default_axis_0_uses_static_path(self) -> None:
        """in_kw=True + default_kw_axis=0 should use the static path."""
        vf = vmap(scale_fn, in_kw=True, default_kw_axis=0)
        x, scale = jnp.arange(3.0), jnp.array([2.0, 3.0, 4.0])
        result = vf(x, scale=scale)
        expected = jax.vmap(scale_fn)(x, scale=scale)
        assert jnp.allclose(result, expected)


# ============================================================================
# Kwarg path (in_kw=Mapping, default_kw_axis=0)


class TestVmapKw:
    """Tests for the _vmap_kw fast-path."""

    def test_broadcast_kwarg_via_none_axis(self) -> None:
        """Explicitly broadcast a kwarg with axis=None."""
        vf = vmap(scale_fn, in_kw={"scale": None})
        result = vf(jnp.arange(3.0), scale=2.0)
        expected = jnp.array([0.0, 2.0, 4.0])
        assert jnp.allclose(result, expected)

    def test_map_kwarg_axis_0(self) -> None:
        """Map a kwarg along axis 0."""
        vf = vmap(scale_fn, in_kw={"scale": 0})
        result = vf(jnp.ones((4, 2)), scale=jnp.arange(4.0))
        expected = jnp.arange(4.0)[:, None] * jnp.ones((4, 2))
        assert jnp.allclose(result, expected)

    def test_map_kwarg_axis_1(self) -> None:
        """Map a kwarg along a non-zero axis."""
        vf = vmap(scale_fn, in_kw={"scale": 1})
        # x: shape (3,) mapped axis 0 → scalar per call
        # scale: shape (2, 3) mapped axis 1 → (2,) per call
        # Each call: scalar * (2,) → (2,). Stacked over 3 calls → (3, 2).
        x = jnp.array([1.0, 2.0, 3.0])
        scale = jnp.ones((2, 3)) * 10
        result = vf(x, scale=scale)
        assert result.shape == (3, 2)
        assert jnp.allclose(result[0], jnp.array([10.0, 10.0]))
        assert jnp.allclose(result[1], jnp.array([20.0, 20.0]))
        assert jnp.allclose(result[2], jnp.array([30.0, 30.0]))

    def test_mixed_special_and_extra_kwargs(self) -> None:
        """Special kwargs (in in_kw) and extra kwargs (not in in_kw)."""
        vf = vmap(weighted_sum, in_kw={"a": None})
        # a is broadcast (None), b is not in in_kw so mapped axis 0
        result = vf(jnp.arange(3.0), a=2.0, b=jnp.array([10.0, 20.0, 30.0]))
        expected = jnp.array([10.0, 22.0, 34.0])
        assert jnp.allclose(result, expected)

    def test_fast_path_exact_kwarg_match(self) -> None:
        """When call-time kwargs exactly match in_kw keys, skip split."""
        vf = vmap(weighted_sum, in_kw={"a": 0, "b": None})
        result = vf(jnp.ones(3), a=jnp.array([1.0, 2.0, 3.0]), b=10.0)
        expected = jnp.array([11.0, 12.0, 13.0])
        assert jnp.allclose(result, expected)

    def test_kw_path_only_special_kwargs(self) -> None:
        """All call-time kwargs are special kwargs."""
        vf = vmap(scale_fn, in_kw={"scale": None})
        # Only passing "scale" which is exactly the special set
        result = vf(jnp.arange(3.0), scale=2.0)
        expected = jnp.array([0.0, 2.0, 4.0])
        assert jnp.allclose(result, expected)


# ============================================================================
# General path (arbitrary default_kw_axis / dynamic kwarg sets)


class TestVmapGeneral:
    """Tests for the _vmap_general path."""

    def test_default_kw_axis_none_broadcasts_all(self) -> None:
        """default_kw_axis=None broadcasts all kwargs."""
        vf = vmap(scale_fn, in_kw=True, default_kw_axis=None)
        result = vf(jnp.arange(3.0), scale=2.0)
        expected = jnp.array([0.0, 2.0, 4.0])
        assert jnp.allclose(result, expected)

    def test_in_kw_true_default_kw_axis_none(self) -> None:
        """in_kw=True with default_kw_axis=None broadcasts all kwargs."""
        vf = vmap(weighted_sum, in_kw=True, default_kw_axis=None)
        result = vf(jnp.arange(3.0), a=2.0, b=10.0)
        expected = jnp.array([10.0, 12.0, 14.0])
        assert jnp.allclose(result, expected)

    def test_mapping_with_non_zero_default(self) -> None:
        """in_kw Mapping with default_kw_axis != 0 triggers general path."""
        # default_kw_axis=None: kwargs not in in_kw are broadcast
        vf = vmap(weighted_sum, in_kw={"a": 0}, default_kw_axis=None)
        result = vf(jnp.ones(3), a=jnp.array([1.0, 2.0, 3.0]), b=100.0)
        expected = jnp.array([101.0, 102.0, 103.0])
        assert jnp.allclose(result, expected)

    def test_general_path_caching(self) -> None:
        """Repeated calls with same structure use cached vmapped fn."""
        vf = vmap(scale_fn, in_kw=True, default_kw_axis=None)
        x = jnp.arange(3.0)
        # First call populates the cache
        r1 = vf(x, scale=2.0)
        # Second call with same arg structure should hit cache
        r2 = vf(x, scale=3.0)
        assert jnp.allclose(r1, jnp.array([0.0, 2.0, 4.0]))
        assert jnp.allclose(r2, jnp.array([0.0, 3.0, 6.0]))

    def test_general_path_different_kwarg_names(self) -> None:
        """Different kwarg names at each call produce correct results."""

        def f(x, **kw):
            result = x
            for v in kw.values():
                result = result + v
            return result

        vf = vmap(f, in_kw=True, default_kw_axis=None)
        r1 = vf(jnp.arange(3.0), a=1.0)
        r2 = vf(jnp.arange(3.0), b=10.0)
        assert jnp.allclose(r1, jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(r2, jnp.array([10.0, 11.0, 12.0]))

    def test_general_path_structured_in_axes(self) -> None:
        """in_axes as a sequence works in the general path."""

        def f(x, y, *, s):
            return x * s + y

        vf = vmap(f, in_axes=(0, None), in_kw=True, default_kw_axis=None)
        result = vf(jnp.arange(3.0), jnp.array(100.0), s=2.0)
        expected = jnp.array([100.0, 102.0, 104.0])
        assert jnp.allclose(result, expected)


# ============================================================================
# static_args and static_kw


class TestStaticArgsKw:
    """Tests for static_args and static_kw."""

    def test_static_args_prepended(self) -> None:
        """static_args are prepended to every call."""
        vmul = vmap(mul, static_args=(3.0,), static_kw={"offset": 1.0})
        result = vmul(jnp.arange(4.0))
        expected = jnp.array([1.0, 4.0, 7.0, 10.0])
        assert jnp.allclose(result, expected)

    def test_static_args_only(self) -> None:
        """static_args without static_kw."""

        def f(a, x):
            return a * x

        vf = vmap(f, static_args=(5.0,))
        result = vf(jnp.arange(3.0))
        expected = jnp.array([0.0, 5.0, 10.0])
        assert jnp.allclose(result, expected)

    def test_static_kw_only(self) -> None:
        """static_kw without static_args."""
        vmul = vmap(mul, static_kw={"offset": 10.0})
        # mul(factor, x, offset=10.0): both factor and x are positional
        result = vmul(jnp.ones(3) * 2, jnp.arange(3.0))
        expected = jnp.array([10.0, 12.0, 14.0])
        assert jnp.allclose(result, expected)

    def test_static_args_with_kw_path(self) -> None:
        """static_args work with the kwarg fast-path."""

        def f(prefix, x, *, scale):
            return prefix + x * scale

        vf = vmap(f, static_args=(100.0,), in_kw={"scale": None})
        result = vf(jnp.arange(3.0), scale=2.0)
        expected = jnp.array([100.0, 102.0, 104.0])
        assert jnp.allclose(result, expected)

    def test_static_kw_with_general_path(self) -> None:
        """static_kw work with the general path."""

        def f(x, *, offset, scale):
            return (x + offset) * scale

        vf = vmap(f, static_kw={"offset": 10.0}, in_kw=True, default_kw_axis=None)
        result = vf(jnp.arange(3.0), scale=2.0)
        expected = jnp.array([20.0, 22.0, 24.0])
        assert jnp.allclose(result, expected)


# ============================================================================
# JIT integration


class TestJIT:
    """Tests for the jit= option."""

    def test_jit_true(self) -> None:
        """jit=True JIT-compiles the wrapped function."""
        vmul = vmap(mul, static_args=(3.0,), static_kw={"offset": 1.0}, jit=True)
        result = vmul(jnp.arange(4.0))
        expected = jnp.array([1.0, 4.0, 7.0, 10.0])
        assert jnp.allclose(result, expected)

    def test_jit_dict(self) -> None:
        """Jit as a dict is passed to jax.jit."""
        vmul = vmap(
            mul,
            static_args=(3.0,),
            static_kw={"offset": 1.0},
            jit={"inline": True},
        )
        result = vmul(jnp.arange(4.0))
        expected = jnp.array([1.0, 4.0, 7.0, 10.0])
        assert jnp.allclose(result, expected)

    def test_jit_with_kw_path(self) -> None:
        """JIT works with the kwarg path."""
        vf = vmap(scale_fn, in_kw={"scale": None}, jit=True)
        result = vf(jnp.arange(3.0), scale=2.0)
        expected = jnp.array([0.0, 2.0, 4.0])
        assert jnp.allclose(result, expected)

    def test_jit_with_general_path(self) -> None:
        """JIT works with the general path."""
        vf = vmap(scale_fn, in_kw=True, default_kw_axis=None, jit=True)
        result = vf(jnp.arange(3.0), scale=2.0)
        expected = jnp.array([0.0, 2.0, 4.0])
        assert jnp.allclose(result, expected)


# ============================================================================
# JAX transformation compatibility


class TestJAXTransformations:
    """Tests for compatibility with jax.jit, jax.vmap, jax.grad."""

    def test_outer_jit(self) -> None:
        """Vmapped function works inside jax.jit."""
        vf = vmap(lambda x: x**2)

        @jax.jit
        def run(x):
            return vf(x)

        result = run(jnp.arange(4.0))
        expected = jnp.array([0.0, 1.0, 4.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_nested_vmap(self) -> None:
        """Nested vmap works correctly (vmap of vmap)."""
        vf = vmap(vmap(lambda x: x + 1))
        result = vf(jnp.ones((3, 4)))
        expected = jnp.ones((3, 4)) + 1
        assert jnp.allclose(result, expected)

    def test_vmap_then_grad(self) -> None:
        """Vmap composes with jax.grad."""

        def f(x):
            return jnp.sum(x**2)

        # Gradient of sum(x^2) = 2*x
        vf = vmap(jax.grad(f))
        xs = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = vf(xs)
        expected = 2 * xs
        assert jnp.allclose(result, expected)

    def test_jit_with_kwarg_path(self) -> None:
        """Kwarg-path vmap inside jax.jit."""
        vf = vmap(scale_fn, in_kw={"scale": None})

        @jax.jit
        def run(x):
            return vf(x, scale=3.0)

        result = run(jnp.arange(4.0))
        expected = jnp.array([0.0, 3.0, 6.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_jit_with_general_path(self) -> None:
        """General-path vmap inside jax.jit."""
        vf = vmap(scale_fn, in_kw=True, default_kw_axis=None)

        @jax.jit
        def run(x):
            return vf(x, scale=3.0)

        result = run(jnp.arange(4.0))
        expected = jnp.array([0.0, 3.0, 6.0, 9.0])
        assert jnp.allclose(result, expected)


# ============================================================================
# PyTree carries


class TestPyTree:
    """Tests for pytree argument handling."""

    def test_dict_positional_arg(self) -> None:
        """Vmapping over dict pytrees works."""

        def f(d):
            return d["a"] + d["b"]

        vf = vmap(f)
        d = {"a": jnp.arange(3.0), "b": jnp.ones(3) * 10}
        result = vf(d)
        expected = jnp.array([10.0, 11.0, 12.0])
        assert jnp.allclose(result, expected)

    def test_tuple_positional_arg(self) -> None:
        """Vmapping over tuple pytrees works."""
        vf = vmap(lambda t: t[0] + t[1])
        result = vf((jnp.arange(3.0), jnp.ones(3) * 5))
        expected = jnp.array([5.0, 6.0, 7.0])
        assert jnp.allclose(result, expected)

    def test_dict_kwarg_general_path(self) -> None:
        """Vmapping over dict kwargs in general path."""

        def f(x, *, params):
            return x * params["w"] + params["b"]

        vf = vmap(f, in_kw=True, default_kw_axis=None)
        result = vf(jnp.arange(3.0), params={"w": 2.0, "b": 10.0})
        expected = jnp.array([10.0, 12.0, 14.0])
        assert jnp.allclose(result, expected)


# ============================================================================
# Edge cases


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_no_positional_args(self) -> None:
        """Function with only kwargs."""

        def f(*, x, y):
            return x + y

        vf = vmap(f, in_axes=(), in_kw={"x": 0, "y": 0})
        result = vf(x=jnp.arange(3.0), y=jnp.ones(3))
        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(result, expected)

    def test_single_element_batch(self) -> None:
        """Batch size 1 works correctly."""
        vf = vmap(lambda x: x + 1)
        result = vf(jnp.array([5.0]))
        expected = jnp.array([6.0])
        assert jnp.allclose(result, expected)

    def test_high_dimensional_input(self) -> None:
        """Vmapping over rank-3 tensors."""
        vf = vmap(lambda x: jnp.sum(x))
        x = jnp.ones((4, 3, 2))
        result = vf(x)
        expected = jnp.ones(4) * 6
        assert jnp.allclose(result, expected)

    def test_in_axes_none_all(self) -> None:
        """in_axes=None broadcasts all positional args."""

        # With in_axes=None nothing is mapped, so the output will have no
        # batch dimension unless out_axes adds one.  But actually jax.vmap
        # with in_axes=None and no mapped args is an error — it needs at
        # least one mapped arg.  So we test that it works correctly when
        # kwargs provide the mapped axis.
        def f(x, *, w):
            return x + w

        vf = vmap(f, in_axes=None, in_kw={"w": 0})
        result = vf(jnp.array(1.0), w=jnp.arange(3.0))
        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(result, expected)

    def test_empty_in_kw_mapping(self) -> None:
        """Empty in_kw mapping behaves like plain vmap for kwargs."""
        vf = vmap(scale_fn, in_kw={})
        # Empty mapping with default_kw_axis=0: falls to _vmap_kw path,
        # but all kwargs get default axis 0 treatment from jax.vmap
        result = vf(jnp.arange(3.0), scale=jnp.ones(3) * 2)
        expected = jnp.array([0.0, 2.0, 4.0])
        assert jnp.allclose(result, expected)

    def test_scalar_output(self) -> None:
        """Function returning a scalar from each batch element."""
        vf = vmap(jnp.sum)
        x = jnp.ones((3, 4))
        result = vf(x)
        expected = jnp.ones(3) * 4
        assert jnp.allclose(result, expected)

    def test_multiple_outputs(self) -> None:
        """Function returning a tuple of arrays."""

        def f(x):
            return x, x**2

        vf = vmap(f)
        x = jnp.arange(3.0)
        a, b = vf(x)
        assert jnp.allclose(a, x)
        assert jnp.allclose(b, x**2)

    def test_out_axes_tuple(self) -> None:
        """out_axes as a tuple for multiple outputs."""

        def f(x):
            return jnp.stack([x, x]), jnp.stack([x * 2, x * 2])

        vf = vmap(f, out_axes=(0, 1))
        x = jnp.arange(3.0)
        a, b = vf(x)
        assert a.shape == (3, 2)
        assert b.shape == (2, 3)

    def test_kw_path_all_special_kwargs_broadcast(self) -> None:
        """Kwarg path works when all special kwargs are broadcast (None)."""

        def f(x, *, scale, bias):
            return x * scale + bias

        vf = vmap(f, in_kw={"scale": None, "bias": None})
        result = vf(jnp.arange(3.0), scale=2.0, bias=10.0)
        expected = jnp.array([10.0, 12.0, 14.0])
        assert jnp.allclose(result, expected)

    def test_general_path_no_kwargs(self) -> None:
        """General path with no kwargs at call-time."""

        def f(x):
            return x + 1

        vf = vmap(f, in_kw=True, default_kw_axis=None)
        result = vf(jnp.arange(3.0))
        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(result, expected)


# ============================================================================
# Correctness: match jax.vmap behaviour


class TestMatchJaxVmap:
    """Verify that vmap matches jax.vmap when used in its default mode."""

    def test_positional_only_matches_jax(self) -> None:
        """Default vmap is identical to jax.vmap for positional args."""

        def f(x, y):
            return x * y + jnp.sum(y)

        x = jnp.arange(3.0)
        y = jnp.ones((3, 2))
        expected = jax.vmap(f)(x, y)
        result = vmap(f)(x, y)
        assert jnp.allclose(result, expected)

    def test_kwargs_mapped_axis0_matches_jax(self) -> None:
        """Default kwarg handling matches jax.vmap (axis 0 for all kwargs)."""
        result_jax = jax.vmap(scale_fn)(jnp.arange(3.0), scale=jnp.ones(3) * 2)
        result_ours = vmap(scale_fn)(jnp.arange(3.0), scale=jnp.ones(3) * 2)
        assert jnp.allclose(result_jax, result_ours)

    @pytest.mark.parametrize("in_axes", [0, None, (0, None), (None, 0)])
    def test_in_axes_parametrized(self, in_axes) -> None:
        """Various in_axes specs produce same result as jax.vmap."""

        def f(x, y):
            return x + y

        # Set up shapes based on which axes are mapped
        if in_axes == 0:
            x, y = jnp.arange(3.0), jnp.ones(3)
        elif in_axes is None:
            # jax.vmap with in_axes=None on all args raises an error, skip
            pytest.skip("in_axes=None for all args is not valid for jax.vmap")
        elif in_axes == (0, None):
            x, y = jnp.arange(3.0), jnp.array(10.0)
        else:  # (None, 0)
            x, y = jnp.array(10.0), jnp.arange(3.0)

        expected = jax.vmap(f, in_axes=in_axes)(x, y)
        result = vmap(f, in_axes=in_axes)(x, y)
        assert jnp.allclose(result, expected)
