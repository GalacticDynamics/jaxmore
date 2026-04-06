"""Benchmarks for jaxmore.structured."""

import pytest

from jaxmore import structured


@pytest.mark.benchmark
def test_bench_fast_path_single_positional(benchmark) -> None:
    """Benchmark fast path: single positional processor, positional call."""

    @structured(ins=((lambda x: x + 1,),))
    def f(x, y=0):
        return x + y

    # Warm up
    f(42, 0)

    benchmark(lambda: f(42, 0))


@pytest.mark.benchmark
def test_bench_fast_path_two_positionals(benchmark) -> None:
    """Benchmark fast path: two positional processors, positional call."""

    @structured(ins=((lambda x: x + 1, lambda y: y * 2),))
    def f(x, y=0):
        return x + y

    # Warm up
    f(10, 20)

    benchmark(lambda: f(10, 20))


@pytest.mark.benchmark
def test_bench_fast_path_with_kwonly(benchmark) -> None:
    """Benchmark fast path: positional + kw-only processors, positional call."""

    @structured(ins=((lambda x: x + 1,), None, {"k": lambda v: v * 3}))
    def f(x, y=0, *, k):
        return x + k + y

    # Warm up
    f(1, 0, k=4)

    benchmark(lambda: f(1, 0, k=4))


@pytest.mark.benchmark
def test_bench_outs_only(benchmark) -> None:
    """Benchmark outs-only wrapper (no input processing)."""

    @structured(outs=lambda r: -r)
    def f(x):
        return x + 1

    # Warm up
    f(42)

    benchmark(lambda: f(42))


@pytest.mark.benchmark
def test_bench_bind_free_pos_only(benchmark) -> None:
    """Benchmark bind-free path: POS_ONLY parameters (no bind guard needed)."""

    @structured(ins=((lambda x: x + 1, lambda y: y * 2),))
    def f(x, y, /):
        return x + y

    # Warm up
    f(10, 20)

    benchmark(lambda: f(10, 20))


@pytest.mark.benchmark
def test_bench_varargs_bind_free(benchmark) -> None:
    """Benchmark bind-free path: POS_ONLY + *args."""

    @structured(ins=((lambda x: x + 1,), lambda v: v * 2))
    def f(x, /, *args):
        return x, args

    # Warm up
    f(10, 20, 30)

    benchmark(lambda: f(10, 20, 30))


@pytest.mark.benchmark
def test_bench_pos_only_default_omitted(benchmark) -> None:
    """Benchmark bind path: POS_ONLY with default, caller omits the arg."""

    @structured(ins=(lambda v: v + 1,))
    def f(x=1, /):
        return x

    # Warm up
    f()

    benchmark(lambda: f())
