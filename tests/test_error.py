"""Tests for error_if function."""

import jax
import jax.numpy as jnp
import pytest

from jaxmore._src import error_if


def test_error_if_no_error_when_false() -> None:
    """error_if returns value unchanged when condition is False."""
    x = jnp.array(5)
    result = error_if(x, jnp.array(False), "should not raise")
    assert jnp.array_equal(result, x)


def test_error_if_raises_when_true() -> None:
    """error_if raises JaxRuntimeError when condition is True."""
    x = jnp.array(5)
    msg = "test error message"
    with pytest.raises(jax.errors.JaxRuntimeError, match=msg):
        error_if(x, jnp.array(True), msg)


def test_error_if_under_jit_no_error() -> None:
    """error_if works correctly under jit when condition is False."""

    @jax.jit
    def f(x):
        return error_if(x, x > 10, "x exceeds 10")

    result = f(jnp.array(5))
    assert int(result) == 5


def test_error_if_under_jit_with_error() -> None:
    """error_if raises JaxRuntimeError under jit when condition is True."""

    @jax.jit
    def f(x):
        return error_if(x, x > 10, "x exceeds 10")

    msg = "x exceeds 10"
    with pytest.raises(jax.errors.JaxRuntimeError, match=msg):
        f(jnp.array(15))


def test_error_if_under_jit_with_traced_condition() -> None:
    """error_if works with traced conditions inside jit."""

    @jax.jit
    def f(x):
        condition = x < 0
        return error_if(x, condition, "x must be non-negative")

    # Should work fine for non-negative
    result = f(jnp.array(5.0))
    assert float(result) == 5.0

    # Should raise for negative
    with pytest.raises(jax.errors.JaxRuntimeError, match="x must be non-negative"):
        f(jnp.array(-3.0))


def test_error_if_under_vmap() -> None:
    """error_if works correctly when vmapped."""

    def check_positive(x):
        return error_if(x, x <= 0, "value must be positive")

    # Vmap over the function
    vmapped_check = jax.vmap(check_positive)

    # All positive values should pass through
    result = vmapped_check(jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))


def test_error_if_under_vmap_with_error() -> None:
    """error_if raises for vmapped input when any element violates condition."""

    def check_positive(x):
        return error_if(x, x <= 0, "value must be positive")

    vmapped_check = jax.vmap(check_positive)

    # When one element fails, the error should be raised
    msg = "value must be positive"
    with pytest.raises(jax.errors.JaxRuntimeError, match=msg):
        vmapped_check(jnp.array([1.0, -2.0, 3.0]))


def test_error_if_pytree_carry() -> None:
    """error_if works with PyTree values under jit."""

    @jax.jit
    def f(state):
        x, y = state
        checked_state = error_if(state, x > 10, "x exceeds 10")
        return checked_state

    result = f((jnp.array(5), jnp.array(3)))
    assert jnp.array_equal(result[0], jnp.array(5))
    assert jnp.array_equal(result[1], jnp.array(3))


def test_error_if_pytree_carry_with_error() -> None:
    """error_if raises error for PyTree values under jit."""

    @jax.jit
    def f(state):
        x, y = state
        checked_state = error_if(state, x > 10, "x exceeds 10")
        return checked_state

    msg = "x exceeds 10"
    with pytest.raises(jax.errors.JaxRuntimeError, match=msg):
        f((jnp.array(15), jnp.array(3)))


def test_error_if_multiple_calls_in_jit() -> None:
    """error_if works correctly with multiple error checks in jit."""

    @jax.jit
    def f(x):
        x = error_if(x, x < 0, "x must be non-negative")
        x = error_if(x, x > 100, "x must be <= 100")
        return x

    # Should pass both checks
    result = f(jnp.array(50))
    assert int(result) == 50

    # Should fail first check
    with pytest.raises(jax.errors.JaxRuntimeError, match="x must be non-negative"):
        f(jnp.array(-5))

    # Should fail second check
    with pytest.raises(jax.errors.JaxRuntimeError, match="x must be <= 100"):
        f(jnp.array(150))


def test_error_if_with_array_condition_all_false() -> None:
    """error_if handles multi-element array conditions (all False)."""
    x = jnp.array([1.0, 2.0, 3.0])
    # Array condition where all elements are False
    condition = x < 0
    result = error_if(x, condition, "should not raise")
    assert jnp.allclose(result, x)


def test_error_if_with_array_condition_some_true() -> None:
    """error_if raises when any element of array condition is True."""
    x = jnp.array([1.0, -2.0, 3.0])
    # Array condition where some elements are True
    condition = x < 0
    msg = "found negative value"
    with pytest.raises(jax.errors.JaxRuntimeError, match=msg):
        error_if(x, condition, msg)


def test_error_if_with_array_condition_under_jit() -> None:
    """error_if works with array conditions under jit."""

    @jax.jit
    def f(x):
        condition = x < 0
        return error_if(x, condition, "x has negative values")

    # All positive should pass
    result = f(jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))

    # With any negative should fail
    msg = "x has negative values"
    with pytest.raises(jax.errors.JaxRuntimeError, match=msg):
        f(jnp.array([1.0, -2.0, 3.0]))
