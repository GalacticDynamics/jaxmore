<h1 align='center'> jaxmore </h1>
<h3 align="center">There's more to JAX.</h3>

<p align="center">
    <a href="https://pypi.org/project/jaxmore/"> <img alt="PyPI version" src="https://img.shields.io/pypi/v/jaxmore" /> </a>
    <a href="https://pypi.org/project/jaxmore/"> <img alt="PyPI platforms" src="https://img.shields.io/pypi/pyversions/jaxmore" /> </a>
    <a href="https://github.com/GalacticDynamics/jaxmore/actions"> <img alt="Actions status" src="https://github.com/GalacticDynamics/jaxmore/workflows/CI/badge.svg" /> </a>
</p>

This package provides some useful functionality that is missing in base `JAX`.
Major features include:

- `vmap` — a drop-in replacement for `jax.vmap` with static-arg/kwarg support
  and per-kwarg axis control.
- `bounded_while_loop` — a reverse-mode-friendly, bounded `while_loop`
  implemented via `lax.scan`.
- `structured` — a decorator that applies per-argument and per-return-value
  transformations at call time, in a structured, declarative way.

## Installation

```bash
pip install jaxmore
```

## Examples

### `vmap` — static arguments and per-kwarg axis mapping

`jaxmore.vmap` is a drop-in replacement for `jax.vmap`. By default it behaves
identically:

```python
import jax.numpy as jnp
from jaxmore import vmap


def f(x, *, scale):
    return x * scale


vf = vmap(f)
vf(jnp.arange(3.0), scale=jnp.ones(3))  # Array([0., 1., 2.], dtype=float32)
```

**Static args & kwargs** — bake constants into a closure so they never cross the
`jax.jit` boundary, reducing dispatch overhead:

```python
import jax.numpy as jnp
from jaxmore import vmap


def mul(factor, x, *, offset):
    return factor * x + offset


vmul = vmap(mul, static_args=(3.0,), static_kw={"offset": 1.0})
print(vmul(jnp.arange(4.0)))  # Array([ 1.,  4.,  7., 10.], dtype=float32)
```

**Per-kwarg axis control** — map, broadcast, or ignore individual keyword
arguments independently (not possible with `jax.vmap`):

```python
import jax.numpy as jnp
from jaxmore import vmap


def h(x, *, a, b):
    return x * a + b


# 'a' is mapped along axis 0, 'b' is broadcast (not mapped)
vh = vmap(h, in_kw={"a": 0, "b": None})
print(vh(jnp.ones(3), a=jnp.array([1.0, 2.0, 3.0]), b=10.0))
# Array([11., 12., 13.], dtype=float32)
```

**Broadcast a kwarg while mapping positional args:**

```python
import jax.numpy as jnp
from jaxmore import vmap


def f(x, *, scale):
    return x * scale


vf = vmap(f, in_kw=True, default_kw_axis=None)
print(vf(jnp.arange(3.0), scale=2.0))  # Array([0., 2., 4.], dtype=float32)
```

**Optional JIT** — JIT-compile the static-folded function before vmapping:

```python
import jax.numpy as jnp
from jaxmore import vmap


def mul(factor, x, *, offset):
    return factor * x + offset


vmul = vmap(mul, static_args=(3.0,), static_kw={"offset": 1.0}, jit=True)
print(vmul(jnp.arange(4.0)))  # Array([ 1.,  4.,  7., 10.], dtype=float32)
```

### `bounded_while_loop`

Simple loop over a scalar:

```python
import jax.numpy as jnp
from jaxmore import bounded_while_loop


def cond_fn(x):
    return x < 5


def body_fn(x):
    return x + 1


result = bounded_while_loop(cond_fn, body_fn, jnp.asarray(0), max_steps=10)
print(result)  # Array(5, dtype=int32)
```

PyTree carry (tuple):

```python
import jax.numpy as jnp
from jaxmore import bounded_while_loop


def cond_fn(state):
    x, _ = state
    return x < 3


def body_fn(state):
    x, y = state
    return x + 1, y * 2


result = bounded_while_loop(
    cond_fn, body_fn, (jnp.asarray(0), jnp.asarray(1)), max_steps=5
)
print(result)  # (Array(3, dtype=int32), Array(8, dtype=int32))
```

### `structured` — per-argument and per-return-value transformations

`structured` is a decorator factory that applies user-supplied callables to
function arguments and return values at call time. It is useful for converting
between raw JAX arrays and richer Python objects (e.g. dataclasses or dicts) at
the boundary of a `jax.jit`-compiled region.

The examples below use trivial processors (dicts, tuples, etc.) to illustrate
the decorator's mechanics. In practice, you should use `structured` to convert
between rich domain objects and flat arrays at a JIT boundary.

**Bare callable shorthand** — process the first positional argument. `ins=f` is
sugar for `ins=((f,),)`:

```python
from jaxmore import structured


@structured(ins=lambda x: {"value": x})
def increment(obj):
    return obj["value"] + 1


print(increment(3))  # 4
```

**Multiple positional processors** — one callable per positional param, matched
left-to-right. `None` skips the corresponding argument:

```python
from jaxmore import structured

to_point = lambda xy: {"x": xy[0], "y": xy[1]}
to_vec = lambda xy: {"dx": xy[0], "dy": xy[1]}


@structured(ins=((to_point, to_vec),))
def translate(pt, v):
    return {"x": pt["x"] + v["dx"], "y": pt["y"] + v["dy"]}


print(translate((1, 2), (10, 20)))  # {'x': 11, 'y': 22}
```

**VAR_POSITIONAL (`*args`)** — a single processor is applied element-wise to
every value passed via `*args`:

```python
from jaxmore import structured


@structured(ins=((), lambda v: {"val": v}))
def collect(*args):
    return tuple(a["val"] for a in args)


print(collect(1, 2, 4))  # (1, 2, 4)
```

**Keyword-only parameters** — matched by name via the third `ins` slot:

```python
from jaxmore import structured


@structured(ins=((), None, {"cfg": lambda d: {**d, "ready": True}}))
def init(x, *, cfg):
    return cfg["ready"], x


print(init(5, cfg={"name": "test"}))  # (True, 5)
```

**VAR_KEYWORD
(`**kwargs`)** — a single processor is applied to every value in `\*\*kwargs`:

```python
from jaxmore import structured


@structured(ins=((), None, {}, lambda v: {"val": v}))
def wrap_kw(**kwargs):
    return {k: obj["val"] for k, obj in kwargs.items()}


print(wrap_kw(a=1, b=4))  # {'a': 1, 'b': 4}
```

**Output processing** — `outs=f` applies `f` to the whole return value. A tuple
applies each processor element-wise; `None` entries pass through:

```python
from jaxmore import structured


@structured(outs=lambda d: d["result"])
def compute(x):
    return {"result": x + 1, "debug": "ok"}


print(compute(4))  # 5


@structured(outs=(lambda d: d["val"], None, lambda d: d["val"]))
def multi_out():
    return ({"val": 10}, 2, {"val": 103})


print(multi_out())  # (10, 2, 103)
```

**Combined with JAX / JIT** — processors run _inside_ the JIT boundary when
`@jax.jit` is applied _outside_ `@structured`. Default parameter values are
filled before processors run:

```python
import jax
import jax.numpy as jnp
from jaxmore import structured


@jax.jit
@structured(
    ins=(lambda x: {"val": x},),
    outs=lambda d: d["val"],
)
def jit_func(obj):
    return {"val": obj["val"] + jnp.asarray(1)}


print(int(jit_func(jnp.asarray(4))))  # 5
```
