"""Unit tests for bounded_while_loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxmore import bounded_while_loop


def test_scalar_loop_stops_on_condition() -> None:
    """Stop when `cond_fn` first becomes False for a scalar carry."""

    def cond_fn(x):
        return x < 5

    def body_fn(x):
        return x + 1

    result = bounded_while_loop(cond_fn, body_fn, jnp.asarray(0), max_steps=10)
    assert int(result) == 5


def test_pytree_carry_tuple() -> None:
    """Handle tuple PyTree carries correctly across iterations."""

    def cond_fn(state):
        x, _ = state
        return x < 3

    def body_fn(state):
        x, y = state
        return x + 1, y * 2

    result = bounded_while_loop(
        cond_fn,
        body_fn,
        (jnp.asarray(0), jnp.asarray(1)),
        max_steps=5,
    )
    assert int(result[0]) == 3
    assert int(result[1]) == 8


def test_early_stop_no_extra_body_calls() -> None:
    """Avoid extra body calls after termination by keeping a counter."""

    def cond_fn(state):
        x, _ = state
        return x < 2

    def body_fn(state):
        x, count = state
        return x + 1, count + 1

    result = bounded_while_loop(
        cond_fn,
        body_fn,
        (jnp.asarray(0), jnp.asarray(0)),
        max_steps=10,
    )
    assert int(result[0]) == 2
    assert int(result[1]) == 2


def test_max_steps_zero_returns_init_without_calling_fns() -> None:
    """Return `init_val` immediately when `max_steps` is zero."""

    def cond_fn(_: object) -> bool:  # pragma: no cover - should never run
        msg = "cond_fn should not be called when max_steps=0"
        raise AssertionError(msg)

    def body_fn(_: object):  # pragma: no cover - should never run
        msg = "body_fn should not be called when max_steps=0"
        raise AssertionError(msg)

    init = {"x": jnp.asarray(1.0)}
    result = bounded_while_loop(cond_fn, body_fn, init, max_steps=0)
    assert float(result["x"]) == 1.0


@pytest.mark.parametrize("max_steps", [-1, 1.5, "3"])  # type: ignore[list-item]
def test_invalid_max_steps_raises(max_steps) -> None:
    """Reject non-integer or negative values for `max_steps`."""

    def cond_fn(x):
        return x < 1

    def body_fn(x):
        return x + 1

    expected_error: type[Exception]
    expected_match: str
    if isinstance(max_steps, int):
        expected_error = ValueError
        expected_match = "max_steps must be a non-negative Python int"
    else:
        expected_error = Exception
        expected_match = "max_steps"

    with pytest.raises(expected_error, match=expected_match):
        bounded_while_loop(cond_fn, body_fn, jnp.asarray(0), max_steps=max_steps)


def test_raises_when_condition_never_false() -> None:
    """Raise a runtime error if the loop never terminates within the bound."""

    def cond_fn(_: jax.Array):
        return True

    def body_fn(x):
        return x + 1

    with pytest.raises(RuntimeError, match="bounded_while_loop exceeded max_steps"):
        bounded_while_loop(cond_fn, body_fn, jnp.asarray(0), max_steps=3)


def test_jit_compilation() -> None:
    """Work correctly under `jax.jit` compilation."""

    def cond_fn(x):
        return x < 4

    def body_fn(x):
        return x + 2

    @jax.jit
    def run(x):
        return bounded_while_loop(cond_fn, body_fn, x, max_steps=5)

    result = run(jnp.asarray(0))
    assert int(result) == 4
