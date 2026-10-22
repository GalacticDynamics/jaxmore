"""Tests for jaxmore.loop and jaxmore.loop_strategies."""

import jax
import jax.numpy as jnp
import pytest

import jaxmore as jm
from jaxmore import loop_strategies as ls

MAPPED = [ls.Vectorize, ls.VMap, ls.Scan, ls.Determine]


def norm_t(x, t):
    """Scalar-per-element function with a vector argument."""
    return jnp.linalg.norm(x) * t


def pytree_fn(x, t):
    """Return a pytree to check outputs are unflattened leaf-by-leaf."""
    return {"a": jnp.sum(x) * t, "b": (jnp.sum(x), t)}


X = jnp.arange(24.0).reshape(2, 3, 4)
T = jnp.array([1.0, 2.0, 3.0])


def test_strategies_cannot_be_instantiated() -> None:
    """Strategies are tags."""
    with pytest.raises(TypeError, match="Cannot instantiate"):
        ls.Scan()


@pytest.mark.parametrize("strategy", MAPPED)
def test_matches_reference(strategy) -> None:
    """Every strategy gives the broadcast result, batch shape (2, 3)."""
    out = jm.loop(strategy, norm_t, signature="(n),()->()")(X, T)
    expected = jnp.linalg.norm(X, axis=-1) * T
    assert out.shape == (2, 3)
    assert jnp.allclose(out, expected)


@pytest.mark.parametrize("strategy", MAPPED)
def test_no_signature_is_elementwise(strategy) -> None:
    """Without a signature every argument is a scalar per element."""
    out = jm.loop(strategy, lambda a, b: a * b + 1)(jnp.arange(3.0), 2.0)
    assert jnp.allclose(out, jnp.arange(3.0) * 2 + 1)


@pytest.mark.parametrize("strategy", MAPPED)
def test_scalar_batch(strategy) -> None:
    """No batch dimensions at all."""
    out = jm.loop(strategy, norm_t, signature="(n),()->()")(X[0, 0], 2.0)
    assert out.shape == ()
    assert jnp.allclose(out, jnp.linalg.norm(X[0, 0]) * 2.0)


@pytest.mark.parametrize("strategy", MAPPED)
def test_pytree_output(strategy) -> None:
    """Outputs may be pytrees; each leaf gets the batch shape back."""
    out = jm.loop(strategy, pytree_fn, signature="(n),()->()")(X, T)
    assert out["a"].shape == out["b"][0].shape == out["b"][1].shape == (2, 3)
    assert jnp.allclose(out["a"], jnp.sum(X, axis=-1) * T)


@pytest.mark.parametrize("strategy", MAPPED)
def test_jit(strategy) -> None:
    """The result is traceable."""
    f = jax.jit(jm.loop(strategy, norm_t, signature="(n),()->()"))
    assert jnp.allclose(f(X, T), jnp.linalg.norm(X, axis=-1) * T)


def test_no_loop_is_identity() -> None:
    """NoLoop returns the function itself."""
    assert jm.loop(ls.NoLoop, norm_t) is norm_t


def test_determine_on_cpu_is_scan() -> None:
    """Determine matches Scan on CPU."""
    if jax.default_backend() != "cpu":
        pytest.skip("CPU only")
    a = jm.loop(ls.Determine, norm_t, signature="(n),()->()")(X, T)
    b = jm.loop(ls.Scan, norm_t, signature="(n),()->()")(X, T)
    assert jnp.array_equal(a, b)


def test_subclass_uses_parent_implementation() -> None:
    """A user subclass of a strategy runs the parent's implementation."""

    class MyScan(ls.Scan):
        pass

    out = jm.loop(MyScan, norm_t, signature="(n),()->()")(X, T)
    assert jnp.allclose(out, jnp.linalg.norm(X, axis=-1) * T)


def test_abstract_strategy_raises() -> None:
    """Abstract strategies have no implementation."""
    with pytest.raises(TypeError, match="no implementation"):
        jm.loop(ls.AbstractVMap, norm_t)


@pytest.mark.parametrize("strategy", [ls.VMap, ls.Scan])
def test_bad_signature(strategy) -> None:
    """Argument count and rank are checked against the signature."""
    f = jm.loop(strategy, norm_t, signature="(n),()->()")
    with pytest.raises(TypeError, match="expects 2 arguments"):
        f(X)
    with pytest.raises(ValueError, match="fewer dimensions"):
        f(jnp.float32(1), T)


@pytest.mark.parametrize("strategy", [ls.VMap, ls.Scan])
def test_grad(strategy) -> None:
    """Gradients flow through the loop."""
    f = jm.loop(strategy, lambda x: x**2)
    g = jax.grad(lambda x: f(x).sum())(jnp.arange(3.0))
    assert jnp.allclose(g, 2 * jnp.arange(3.0))
