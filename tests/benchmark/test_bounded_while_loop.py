"""Benchmarks for bounded_while_loop."""

import jax
import jax.numpy as jnp
import pytest

from jaxmore import bounded_while_loop


@pytest.mark.benchmark
def test_bench_scalar_loop(benchmark) -> None:
    """Benchmark bounded_while_loop with a scalar carry."""

    def cond_fn(x):
        return x < 100

    def body_fn(x):
        return x + 1

    init = jnp.asarray(0)

    # Warm up (trigger JAX tracing)
    bounded_while_loop(cond_fn, body_fn, init, max_steps=200).block_until_ready()

    benchmark(
        lambda: bounded_while_loop(
            cond_fn, body_fn, init, max_steps=200
        ).block_until_ready()
    )


@pytest.mark.benchmark
def test_bench_scalar_loop_jit(benchmark) -> None:
    """Benchmark JIT-compiled bounded_while_loop."""

    def cond_fn(x):
        return x < 100

    def body_fn(x):
        return x + 1

    @jax.jit
    def run(x):
        return bounded_while_loop(cond_fn, body_fn, x, max_steps=200)

    init = jnp.asarray(0)

    # Warm up (JIT compile)
    run(init).block_until_ready()

    benchmark(lambda: run(init).block_until_ready())


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [2_000, 128_000])
def test_bench_large_carry_jit(benchmark, n) -> None:
    """Per-step cost should not scale with carry size (GH #29)."""
    steps = 2_000

    def cond_fn(s):
        return s[1] < steps

    def body_fn(s):
        path, step = s
        return path.at[step].set(step), step + 1

    @jax.jit
    def run(x):
        return bounded_while_loop(cond_fn, body_fn, x, max_steps=steps + 1)

    init = (-jnp.ones(n, int), jnp.asarray(0))

    # Warm up (JIT compile)
    jax.block_until_ready(run(init))

    benchmark(lambda: jax.block_until_ready(run(init)))
