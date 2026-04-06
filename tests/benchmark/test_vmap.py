"""Benchmarks for jaxmore.vmap."""

import jax.numpy as jnp
import pytest

from jaxmore import vmap


def _scale_fn(x, *, scale):
    return x * scale


@pytest.mark.benchmark
def test_bench_static_path(benchmark) -> None:
    """Benchmark vmap static path (in_kw=False, default behaviour)."""
    f = lambda x: x + 1
    vf = vmap(f)
    x = jnp.arange(1000.0)

    # Warm up
    vf(x).block_until_ready()

    benchmark(lambda: vf(x).block_until_ready())


@pytest.mark.benchmark
def test_bench_kw_path(benchmark) -> None:
    """Benchmark vmap kwarg fast-path (in_kw=Mapping, default_kw_axis=0)."""
    vf = vmap(_scale_fn, in_kw={"scale": 0})
    x = jnp.ones(1000)
    scale = jnp.arange(1000.0)

    # Warm up
    vf(x, scale=scale).block_until_ready()

    benchmark(lambda: vf(x, scale=scale).block_until_ready())


@pytest.mark.benchmark
def test_bench_general_path(benchmark) -> None:
    """Benchmark vmap general path (in_kw=True, default_kw_axis=None)."""
    vf = vmap(_scale_fn, in_kw=True, default_kw_axis=None)
    x = jnp.arange(1000.0)
    scale = 2.0

    # Warm up
    vf(x, scale=scale).block_until_ready()

    benchmark(lambda: vf(x, scale=scale).block_until_ready())
