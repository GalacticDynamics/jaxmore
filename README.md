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

If you know a loop will terminate before `n` steps then `bounded_while_loop` is better than a normal while loop.
`bounded_while_loop` uses `jax.lax.scan` under the hood, enabling both forward-mode and backward-mode differentiation.
Speed and efficiency are maintained by evaluating the termination condition and, when satisfied, switching to a no-op function that jax compiles away under jit for all remaining steps.

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
