<h1 align='center'> jax-bounded-while </h1>
<h3 align="center">Bounded while loop in JAX.</h3>

<p align="center">
    <a href="https://pypi.org/project/jax-bounded-while/"> <img alt="PyPI version" src="https://img.shields.io/pypi/v/jax-bounded-while" /> </a>
    <a href="https://pypi.org/project/jax-bounded-while/"> <img alt="PyPI platforms" src="https://img.shields.io/pypi/pyversions/jax-bounded-while" /> </a>
    <a href="https://github.com/GalacticDynamics/jax-bounded-while/actions"> <img alt="Actions status" src="https://github.com/GalacticDynamics/jax-bounded-while/workflows/CI/badge.svg" /> </a>
</p>

This is a micro-package, containing the single function `bounded_while_loop`.
</br> Reverse-mode-friendly, bounded `while_loop` implemented via `lax.scan`.

> **Note:** This library is being renamed to **`jaxmore`** and expanded in
> scope. In addition to `bounded_while_loop`, it will include more JAX-related
> functionality — such as a `vmap` that supports keyword arguments. This will be
> the last release for `jax-bounded-while`.

## Installation

```bash
pip install jax-bounded-while
```

## Examples

Simple loop over a scalar:

```python
import jax.numpy as jnp
from jax_bounded_while import bounded_while_loop


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
from jax_bounded_while import bounded_while_loop


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
