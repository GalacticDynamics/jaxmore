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

### `jaxmore.nn` — efficient neural network training with JAX scan

The `AbstractScanNNTrainer` class provides a foundation for building efficient
training loops using
[`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html)
to scan over batches and epochs. It handles shuffling, batching, and loss
aggregation automatically, while you focus on defining `make_step()`.

> **Note**: The neural network training utilities in `jaxmore.nn` are more
> experimental than other components of this library. The code is adapted from
> [phasecurvefit](https://phasecurvefit.readthedocs.io/en/latest/). If you have
> interestingly shaped data or encounter errors, please
> [submit an issue](https://github.com/GalacticDynamics/jaxmore/issues) so we
> can continue to generalize this training code.

Here's a complete example of training a simple feed-forward network:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxmore.nn import AbstractScanNNTrainer, masked_mean


# Define a simple neural network using eqx.nn.MLP
model = eqx.nn.MLP(
    in_size=2,
    out_size=1,
    width_size=32,
    depth=2,
    activation=jax.nn.relu,
    key=jr.key(0),
)


# Create a trainer subclass
class NNTrainer(AbstractScanNNTrainer):
    """Concrete trainer implementation for SimpleNN."""

    def init(self, *, key, X, y, learning_rate=1e-2):
        """Initialize model and training data."""
        model_key, data_key = jr.split(key)
        model = eqx.nn.MLP(
            in_size=X.shape[1],
            out_size=1,
            width_size=32,
            depth=2,
            activation=jax.nn.relu,
            key=model_key,
        )

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        # Create training carry: (model, optimizer_state, rng_key)
        carry_key = jr.fold_in(data_key, 0)
        initial_carry = (model, opt_state, carry_key)

        # All samples are usable (True), none are padding (False)
        mask = jnp.ones(len(X), dtype=bool)
        epoch_data = (mask, (X, y))

        return initial_carry, epoch_data

    def pack_carry_state(self, carry):
        """Partition model into arrays and static structure."""
        model, opt_state, key = carry
        model_dyn, model_static = eqx.partition(model, eqx.is_array)
        return (model_dyn, opt_state, key), {"model_static": model_static}

    def unpack_carry_state(self, carry, static):
        """Reconstruct full model from partitioned state."""
        model_dyn, opt_state, key = carry
        model_static = static["model_static"]
        model = eqx.combine(model_dyn, model_static)
        return (model, opt_state, key)


# Define optimizer
optimizer = optax.adam(1e-2)


# Define the per-batch training step
def make_step(carry, batch_inputs):
    """Execute one batch of training."""
    model, opt_state, key = carry
    batch_mask, (X_batch, y_batch) = batch_inputs

    def loss_fn(model):
        preds = jax.vmap(model)(X_batch)
        mse = jnp.mean((preds.squeeze() - y_batch) ** 2)
        return mse

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, (model, opt_state, key)


# Generate synthetic training data
key = jr.key(0)
X = jr.normal(key, (100, 2))
y = 2 * X[:, 0] + X[:, 1] + 0.1 * jr.normal(jr.fold_in(key, 1), (100,))

# Create trainer and initialize
trainer = NNTrainer(make_step=make_step, loss_agg_fn=masked_mean)
carry, epoch_data = trainer.init(key=key, X=X, y=y, learning_rate=1e-2)

# Train for 10 epochs with batch size 16
final_carry, losses = trainer.run(
    carry,
    epoch_data,
    num_epochs=10,
    batch_size=16,
    key=key,
    show_pbar=False,
)

print(f"Final epoch loss: {losses[-1]:.6f}")  # doctest: +SKIP
```

Key features:

- **Automatic batching** — `shuffle_and_batch()` handles shuffling and padding
- **Efficient scanning** — Uses `jax.lax.scan` for epochs and batches
  (JAX-friendly)
- **Model partitioning** — Separate dynamic arrays (model weights) from static
  structure for efficient JIT compilation
- **Loss aggregation** — Customize how per-batch losses combine into epoch
  losses

#### ⚠️ Sharp Edge: Equinox Models Require `eqx.partition`

**CRITICAL**: When using Equinox models (`eqx.nn.MLP`, `eqx.nn.Linear`, etc.),
you **MUST** implement `pack_carry_state()` and `unpack_carry_state()` using
`eqx.partition` and `eqx.combine`. This is not optional!

**Why?** Equinox modules contain methods decorated with `@jax.custom_jvp`, which
JAX cannot scan over directly. Without partitioning, you'll encounter:

```
TypeError: Argument '...' is not a valid JAX type
```

**Correct pattern** (as shown above):

```python
def pack_carry_state(self, carry):
    model, opt_state, key = carry
    # Separate arrays from static structure (methods, activations, etc.)
    model_dyn, model_static = eqx.partition(model, eqx.is_array)
    return (model_dyn, opt_state, key), {"model_static": model_static}


def unpack_carry_state(self, carry, static):
    model_dyn, opt_state, key = carry
    model_static = static["model_static"]
    # Reconstruct full model
    model = eqx.combine(model_dyn, model_static)
    return (model, opt_state, key)
```

This separates trainable arrays (which JAX can scan over) from static Python
objects (which it cannot), enabling the training loop to work correctly.
