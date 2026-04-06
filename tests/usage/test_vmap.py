"""Usage tests for jaxmore.vmap — JAX transform composition, PyTree, parity."""

import jax
import jax.numpy as jnp
import pytest

from jaxmore import vmap

# ============================================================================
# Helpers


def scale_fn(x, *, scale):
    """X * scale — simple kwarg function."""
    return x * scale


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

        if in_axes == 0:
            x, y = jnp.arange(3.0), jnp.ones(3)
        elif in_axes is None:
            pytest.skip("in_axes=None for all args is not valid for jax.vmap")
        elif in_axes == (0, None):
            x, y = jnp.arange(3.0), jnp.array(10.0)
        else:  # (None, 0)
            x, y = jnp.array(10.0), jnp.arange(3.0)

        expected = jax.vmap(f, in_axes=in_axes)(x, y)
        result = vmap(f, in_axes=in_axes)(x, y)
        assert jnp.allclose(result, expected)
