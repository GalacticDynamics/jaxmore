"""Tests for jaxmore.loop and jaxmore.loop_strategies."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxmore as jm
import jaxmore._src.loop_strategies as impl
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


ALL_TAGS = [getattr(ls, name) for name in ls.__all__]


@pytest.mark.parametrize("tag", ALL_TAGS)
def test_strategies_cannot_be_instantiated(tag) -> None:
    """Strategies are tags: no instance, with or without arguments."""
    with pytest.raises(TypeError, match=f"Cannot instantiate {tag.__name__}"):
        tag()
    with pytest.raises(TypeError, match="Cannot instantiate"):
        tag(1)


def test_strategy_hierarchy_and_identity() -> None:
    """The public tags are the private classes, with the documented bases."""
    assert all(getattr(ls, n) is getattr(impl, n) for n in ls.__all__)
    for tag in ALL_TAGS:
        assert issubclass(tag, ls.AbstractLoopStrategy)
    assert issubclass(ls.Vectorize, ls.AbstractVMap)
    assert issubclass(ls.VMap, ls.AbstractVMap)
    assert not issubclass(ls.Scan, ls.AbstractVMap)


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


@pytest.mark.parametrize("strategy", MAPPED)
def test_output_signature_is_ignored(strategy) -> None:
    """Output core dims of any shape are inferred, never validated."""
    f = jm.loop(
        strategy, lambda x, t: {"a": x * t, "b": jnp.sum(x)}, signature="(n),()"
    )
    out = f(X, T)
    assert out["a"].shape == X.shape
    assert out["b"].shape == (2, 3)
    assert jnp.allclose(out["a"], X * T[:, None])


@pytest.mark.parametrize("strategy", MAPPED)
def test_pytree_args(strategy) -> None:
    """A signature entry applies to every leaf of a pytree argument."""

    def f(w, t):
        q, p = w
        return jnp.sum(q * p) * t

    w = (X, X + 1)
    out = jm.loop(strategy, f, signature="(n),()")(w, T)
    assert jnp.allclose(out, jnp.sum(X * (X + 1), axis=-1) * T)


@pytest.mark.parametrize("strategy", MAPPED)
def test_kwargs_are_not_mapped(strategy) -> None:
    """Keyword arguments reach every call whole."""
    k = jnp.arange(5.0)  # not a batch dimension
    out = jm.loop(strategy, lambda x, *, k: jnp.sum(x * k))(jnp.arange(3.0), k=k)
    assert out.shape == (3,)
    assert jnp.allclose(out, jnp.arange(3.0) * jnp.sum(k))


@pytest.mark.parametrize("strategy", MAPPED)
def test_size_one_batch_dims(strategy) -> None:
    """Size-1 batch dimensions broadcast, including when every one is size 1."""
    f = jm.loop(strategy, lambda a, b: a * b)
    a, b = jnp.arange(3.0)[:, None], jnp.arange(4.0)[None, :]
    assert jnp.allclose(f(a, b), a * b)
    assert jnp.allclose(f(a[:1], b[:, :1]), a[:1] * b[:, :1])


def test_vectorize_matches_jnp_vectorize() -> None:
    """Vectorize reproduces jnp.vectorize, incl. mixed batch ranks."""
    ref = jnp.vectorize(norm_t, signature="(n),()->()")
    out = jm.loop(ls.Vectorize, norm_t, signature="(n),()->()")(X, T[:, None, None])
    assert jnp.allclose(out, ref(X, T[:, None, None]))


@pytest.mark.parametrize(
    ("backend", "expected"), [("cpu", "scan"), ("gpu", "vectorize")]
)
def test_determine_resolves_at_call_time(monkeypatch, backend, expected) -> None:
    """Determine picks Scan on CPU and Vectorize elsewhere, when called."""
    monkeypatch.setattr(impl, "_scan", lambda *_: lambda *_a, **_k: "scan")
    monkeypatch.setattr(impl, "_vectorize", lambda *_: lambda *_a, **_k: "vectorize")
    f = jm.loop(ls.Determine, norm_t)  # built before the backend is known
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    assert f(1.0, 1.0) == expected


@pytest.mark.parametrize("strategy", [ls.Vectorize, ls.VMap, ls.Scan])
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


# ---------------------------------------------------------------------------
# NoLoop must not touch the function


def test_no_loop_leaves_function_intact() -> None:
    """`loop(NoLoop, f)` returns `f` without rewrapping it onto itself."""

    def f(x):
        return x * 2

    assert jm.loop(ls.NoLoop, f) is f
    assert not hasattr(f, "__wrapped__")
    assert str(inspect.signature(f)) == "(x)"
    assert jnp.allclose(jax.jit(f)(jnp.ones(2)), 2.0)


def test_no_loop_accepts_any_callable() -> None:
    """Builtins, ufuncs and bound methods are returned as they are."""

    class A:
        def m(self, x):
            return x

    assert jm.loop(ls.NoLoop, np.sin) is np.sin
    assert jm.loop(ls.NoLoop, len) is len
    a = A()
    assert jm.loop(ls.NoLoop, a.m) == a.m


@pytest.mark.parametrize("strategy", [ls.Vectorize, ls.VMap, ls.Scan])
def test_ufunc_is_accepted(strategy) -> None:
    """A callable without a writable `__name__` can still be looped."""
    out = jm.loop(strategy, jnp.sin)(jnp.arange(3.0))
    assert jnp.allclose(out, jnp.sin(jnp.arange(3.0)))


# ---------------------------------------------------------------------------
# Shared arguments are not copied


@pytest.mark.parametrize("strategy", [ls.Vectorize, ls.VMap, ls.Scan])
def test_shared_argument_is_not_broadcast(strategy) -> None:
    """A 40 kB argument shared by 2000 elements must not become 80 MB."""
    table, t = jnp.ones((10_000,)), jnp.arange(2000.0)
    f = jax.jit(jm.loop(strategy, lambda tab, t: jnp.sum(tab) * t, signature="(n),()"))
    mem = f.lower(table, t).compile().memory_analysis()
    if mem is None:
        pytest.skip("backend reports no memory analysis")
    assert mem.temp_size_in_bytes < 1_000_000
    assert jnp.allclose(f(table, t), 10_000.0 * t)


@pytest.mark.parametrize("strategy", MAPPED)
def test_partially_broadcast_arguments(strategy) -> None:
    """Arguments broadcast against each other along different batch dims."""
    a, b = jnp.arange(6.0).reshape(2, 1, 3), jnp.arange(4.0).reshape(1, 4)
    out = jm.loop(strategy, lambda x, y: jnp.sum(x) * y, signature="(n),()")(a, b)
    assert out.shape == (2, 4)
    assert jnp.allclose(out, jnp.sum(a, axis=-1) * b)


@pytest.mark.parametrize("strategy", MAPPED)
@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 3)])
def test_empty_batch(strategy, shape) -> None:
    """A zero-size batch gives a zero-size result of the right shape."""
    x = jnp.ones((*shape, 3))
    out = jm.loop(strategy, jnp.sum, signature="(n)")(x)
    assert out.shape == shape


@pytest.mark.parametrize("strategy", MAPPED)
def test_scalar_batch_gives_arrays(strategy) -> None:
    """Outputs are arrays even when there is no batch dimension."""
    out = jm.loop(strategy, lambda _x: 2.0)(1.0)
    assert isinstance(out, jax.Array)
    assert out.shape == ()


# ---------------------------------------------------------------------------
# Signatures


@pytest.mark.parametrize(
    "signature",
    [
        "foo",
        "",
        "(n)()",
        "((n))",
        "(,n)",
        "(n),(->()",
        "(n)->()extra",
        "(...)",
    ],
)
@pytest.mark.parametrize("strategy", [ls.Vectorize, ls.VMap, ls.Scan])
def test_malformed_signature(strategy, signature) -> None:
    """A string that is not a gufunc signature is rejected as such."""
    with pytest.raises(ValueError, match="not a valid gufunc signature"):
        jm.loop(strategy, lambda *_a: 0.0, signature=signature)(jnp.ones(3))


@pytest.mark.parametrize(
    "signature", ["(n),(m)", " ( n ) , ( m ) ", "(n),(m)->()", "(3),(4)->(3,4)"]
)
def test_valid_signature_spellings(signature) -> None:
    """Whitespace, an optional output part and fixed sizes are fine."""
    out = jm.loop(ls.Vectorize, lambda a, b: a[0] + b[0], signature=signature)(
        jnp.ones(3), jnp.ones(4)
    )
    assert out == 2.0


@pytest.mark.parametrize("strategy", [ls.Vectorize, ls.VMap, ls.Scan])
def test_inconsistent_core_size(strategy) -> None:
    """A repeated core dimension name must have one size."""
    f = jm.loop(strategy, lambda a, b: a @ b, signature="(n),(n)")
    with pytest.raises(ValueError, match="inconsistent size for core dimension 'n'"):
        f(jnp.ones((2, 3)), jnp.ones((2, 4)))


@pytest.mark.parametrize("strategy", [ls.Vectorize, ls.VMap, ls.Scan])
def test_fixed_core_size(strategy) -> None:
    """A number in the signature fixes that core dimension."""
    f = jm.loop(strategy, jnp.sum, signature="(5)")
    with pytest.raises(ValueError, match="inconsistent size"):
        f(jnp.ones((2, 3)))
    assert f(jnp.ones((2, 5))).shape == (2,)


@pytest.mark.parametrize("strategy", [ls.Vectorize, ls.VMap, ls.Scan])
def test_pytree_leaves_must_share_core_dims(strategy) -> None:
    """Leaves of one argument that disagree on its core dims are an error."""
    w = (jnp.ones((5, 3)), jnp.ones((5,)))
    f = jm.loop(strategy, lambda w: jnp.sum(w[0]) * w[1], signature="(n)")
    with pytest.raises(ValueError, match="inconsistent size"):
        f(w)


# ---------------------------------------------------------------------------
# Calling and strategy errors


@pytest.mark.parametrize("strategy", MAPPED)
def test_no_positional_array(strategy) -> None:
    """Without a mapped array there is nothing to loop over."""
    f = jm.loop(strategy, lambda _x=None: 1.0)
    for call in (f, lambda: f(_x=jnp.ones(3)), lambda: f(None)):
        with pytest.raises(TypeError, match="at least one array"):
            call()


@pytest.mark.parametrize("bad", ["vmap", None, 3, ls.Scan.__name__, int])
def test_strategy_must_be_a_strategy_class(bad) -> None:
    """Anything but a strategy class is a TypeError."""
    with pytest.raises(TypeError, match=r"(?i)strategy"):
        jm.loop(bad, norm_t)
