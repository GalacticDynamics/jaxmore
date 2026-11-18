"""Unit tests for `jaxmore.filter_jit`."""

from collections.abc import Callable
from contextlib import AbstractContextManager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmore import filter_jit

CountCompiles = Callable[[], AbstractContextManager[list[int]]]


def _affine(x: jax.Array, a: float, *, op: str = "add") -> jax.Array:
    return x * a + 1 if op == "add" else x * a - 1


def test_matches_jax_jit() -> None:
    """On a plain array function it agrees with `jax.jit`."""
    x = jnp.linspace(0.0, 1.0, 5)
    fn: Callable[[jax.Array], jax.Array] = lambda x: jnp.sin(x) ** 2 + x
    assert jnp.allclose(filter_jit(fn)(x), jax.jit(fn)(x))


def test_second_call_does_not_compile(count_compiles: CountCompiles) -> None:
    """A repeat call with the same shapes and static values reuses the program."""
    f = filter_jit(_affine)
    x = jnp.arange(4.0)
    x2 = x + 1
    f(x, 2.0, op="sub")
    with count_compiles() as n:
        out = f(x2, 2.0, op="sub")
    assert n[0] == 0
    assert jnp.allclose(out, x2 * 2.0 - 1)


def test_rewrapping_same_function_hits_cache(count_compiles: CountCompiles) -> None:
    """The cache is keyed on `fn`, so wrapping it again still hits."""
    x = jnp.arange(3.0)
    filter_jit(_affine)(x, 3.0)
    with count_compiles() as n:
        filter_jit(_affine)(x, 3.0)
    assert n[0] == 0


def test_static_scalars_and_strings(count_compiles: CountCompiles) -> None:
    """Python scalars and strings are static: new values recompile and are used."""
    f = filter_jit(_affine)
    x = jnp.arange(3.0)
    assert jnp.allclose(f(x, 2.0), x * 2 + 1)
    assert jnp.allclose(f(x, 3.0, op="sub"), x * 3 - 1)
    with count_compiles() as n:
        f(x, 4.0)
    assert n[0] == 1


def test_static_type_is_part_of_key() -> None:
    """`1 == 1.0` but they trace differently, so they must not share a program."""
    f = filter_jit(lambda a: jnp.asarray(a))
    assert f(1).dtype == jnp.int32
    assert f(1.0).dtype == jnp.float32


def test_numpy_leaves_are_traced(count_compiles: CountCompiles) -> None:
    """NumPy arrays and scalars are traced, not static."""
    f = filter_jit(lambda x, s: x + s)
    f(np.ones(3), np.float32(1.0))
    with count_compiles() as n:
        out = f(np.zeros(3), np.float32(2.0))
    assert n[0] == 0
    assert jnp.allclose(out, 2.0)


def test_numpy_string_scalars_are_static() -> None:
    """`np.str_` can't be traced, so it is static like a Python string."""
    f = filter_jit(lambda x, label: x + len(label))
    assert jnp.allclose(f(jnp.zeros(2), np.str_("abc")), 3.0)


def test_numpy_string_array_is_static() -> None:
    """A NumPy array JAX can't trace is static, so it must be hashable."""
    f = filter_jit(lambda x, labels: x + len(labels))
    with pytest.raises(TypeError, match=r"args\[1\] of type 'ndarray'"):
        f(jnp.zeros(2), np.array(["a", "b"]))


def test_numpy_bfloat16_is_traced(count_compiles: CountCompiles) -> None:
    """JAX's extra dtypes (NumPy kind "V") are traced, like `jax.jit` does."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    f = filter_jit(lambda x: x * 2)
    x = np.ones(2, ml_dtypes.bfloat16)
    y = np.full(2, 3, ml_dtypes.bfloat16)
    f(x)
    with count_compiles() as n:
        out = f(y)
    assert n[0] == 0
    assert jnp.allclose(out.astype(jnp.float32), 6.0)


class _Picky:
    """Hashable, but comparing it with another type raises."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Picky):
            msg = "compared with a different type"
            raise TypeError(msg)
        return True

    def __hash__(self) -> int:
        return 0


def test_static_types_are_compared_before_values() -> None:
    """A static slot that changes type compiles again; values aren't compared."""
    f = filter_jit(lambda x, tag: x + bool(tag))
    f(jnp.zeros(1), "a")
    assert jnp.allclose(f(jnp.zeros(1), _Picky()), 1.0)


@pytest.mark.parametrize("scalar", [np.longdouble(1.5), np.ulonglong(3)])
def test_numpy_scalars_jit_rejects_are_static(scalar: np.generic) -> None:
    """NumPy scalar types `jax.jit` won't take as arguments stay static."""
    if jax.core.valid_jaxtype(scalar):  # e.g. `np.ulonglong` is `np.uint64` on Windows
        pytest.skip(f"jax.jit accepts {type(scalar).__name__} on this platform")
    f = filter_jit(lambda x, s: (x * float(s), s))
    y, s = f(jnp.ones(()), scalar)
    assert jnp.allclose(y, float(scalar))
    assert type(s) is type(scalar)


@pytest.mark.parametrize(
    "value",
    [np.timedelta64(5), np.ma.array([1.0], mask=[True])],
    ids=["hash-raises-ValueError", "masked-array"],
)
def test_odd_static_values_fall_back_or_raise_typeerror(value: object) -> None:
    """Any failure to trace or hash takes the fallback, or raises TypeError."""
    f = filter_jit(lambda x, *_: x, fallback="eager")
    assert jnp.allclose(f(jnp.ones(2), value), 1.0)
    with pytest.raises(TypeError, match="is not hashable"):
        filter_jit(lambda x, *_: x)(jnp.ones(2), value)


def test_unflattenable_arguments() -> None:
    """Arguments that can't be flattened take the fallback, or raise TypeError."""
    unsortable = {1: 1.0, "b": 2.0}
    assert filter_jit(lambda d: len(d), fallback="eager")(unsortable) == 2
    with pytest.raises(TypeError, match="can't be flattened"):
        filter_jit(lambda d: len(d))(unsortable)


def test_non_array_outputs(count_compiles: CountCompiles) -> None:
    """Non-array leaves in the output come back unchanged."""
    f = filter_jit(lambda x, name: (name, {"n": 3, "y": x + 1}, None))
    name, d, none = f(jnp.zeros(2), "hello")
    assert name == "hello"
    assert d["n"] == 3
    assert none is None
    assert jnp.allclose(d["y"], 1.0)
    # And on a cache hit too.
    with count_compiles() as n:
        name, d, _ = f(jnp.ones(2), "hello")
    assert n[0] == 0
    assert name == "hello"
    assert d["n"] == 3


def test_nested_containers() -> None:
    """Arrays nested in dicts, lists and tuples are traced; the rest is static."""

    def fn(tree: dict[str, object]) -> jax.Array:
        a, (b, scale) = tree["pair"]  # type: ignore[misc]
        return (a + b) * scale if tree["flag"] else a - b  # type: ignore[operator]

    f = filter_jit(fn)
    tree = {"pair": [jnp.ones(2), (jnp.full(2, 2.0), 10)], "flag": True}
    assert jnp.allclose(f(tree), 30.0)
    tree["flag"] = False
    assert jnp.allclose(f(tree), -1.0)


def test_unhashable_static_leaf_raises() -> None:
    """The error names the offending leaf's path and type."""
    f = filter_jit(lambda x, **kw: x + len(kw))
    with pytest.raises(TypeError, match=r"kwargs\['cfg'\]\['s'\] of type 'set'"):
        f(jnp.zeros(1), cfg={"s": {1, 2}})


def test_unhashable_static_leaf_eager_fallback() -> None:
    """With ``fallback="eager"`` the function runs without jit."""
    seen: list[bool] = []

    def fn(x: jax.Array, s: set[int]) -> jax.Array:
        seen.append(isinstance(x, jax.core.Tracer))
        return x + len(s)

    out = filter_jit(fn, fallback="eager")(jnp.zeros(2), {1, 2})
    assert seen == [False]
    assert jnp.allclose(out, 2.0)
