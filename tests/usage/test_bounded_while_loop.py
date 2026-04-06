"""Usage tests for bounded_while_loop — JAX integration."""

import jax
import jax.numpy as jnp

from jaxmore import bounded_while_loop


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
