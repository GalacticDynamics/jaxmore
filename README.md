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
- `nn` - additions for `jax.nn`
  - `AbstractScanNNTrainer` - A base class for efficient neural network training
    by `jax.lax.scan` over both epochs and batches within epochs.
  - `masked_mean` - compute the mean of an array over only the masked elements
  - `shuffle_and_batch` - shuffle arrays and batch them with padding mask, for
    use in training.

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

**VAR_KEYWORD (`**kwargs`)** — a single processor is applied to every value in
`\*\*kwargs`:

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

#### Optimizer Passing Strategies: Three Patterns

When using `AbstractScanNNTrainer`, you have three options for passing the
optimizer to your `make_step()` function. Each has different tradeoffs regarding
flexibility, reusability, and code organization.

##### **Option 1: Optimizer in Closure** (captured in `make_step`)

The optimizer is captured in a closure inside the `make_step` function:

```python
# Create optimizer ONCE outside
optimizer = optax.adam(1e-2)


# Define make_step with optimizer in closure
def make_step(carry, batch_inputs):
    model, opt_state, key = carry
    # optimizer is captured from outer scope
    updates, opt_state = optimizer.update(grads, opt_state)
    return loss, (model, opt_state, key)


trainer = NNTrainer(make_step=make_step, loss_agg_fn=masked_mean)
```

**Pros:**

- Simple and straightforward
- Minimal boilerplate

**Cons:**

- ⚠️ **Each optimizer change requires creating a new `make_step` function**
- Limited flexibility for experimenting with different optimizers
- The optimizer is "hidden" in the closure, not explicitly visible

**When to use:** Simple scripts with a single, fixed optimizer configuration.

**Limitation example:**

```python
# Want to try a different optimizer? You must create a new make_step:
optimizer_sgd = optax.sgd(1e-2)


def make_step_sgd(carry, batch_inputs):  # NEW function!
    model, opt_state, key = carry
    updates, opt_state = optimizer_sgd.update(grads, opt_state)
    return loss, (model, opt_state, key)


trainer_sgd = NNTrainer(make_step=make_step_sgd, ...)  # NEW trainer!
```

**Option 1b: Closure via Lambda Wrapping** (more flexible variant)

A more flexible approach: define a single `make_step` that accepts `optimizer`
as a kwarg, then wrap it in a lambda to capture the optimizer:

```python
# Define make_step that takes optimizer as a kwarg
def make_step(carry, batch_inputs, *, optimizer):
    model, opt_state, key = carry
    # ... loss computation ...
    updates, opt_state = optimizer.update(grads, opt_state)
    return loss, (model, opt_state, key)


# Create closures for different optimizers using lambda
optimizer_adam = optax.adam(1e-2)
optimizer_sgd = optax.sgd(1e-2)

# Wrap with lambda to capture each optimizer
make_step_adam = lambda carry, batch_inputs: make_step(
    carry, batch_inputs, optimizer=optimizer_adam
)
make_step_sgd = lambda carry, batch_inputs: make_step(
    carry, batch_inputs, optimizer=optimizer_sgd
)

# Use whichever closure you want
trainer_adam = NNTrainer(make_step=make_step_adam, loss_agg_fn=masked_mean)
trainer_sgd = NNTrainer(make_step=make_step_sgd, loss_agg_fn=masked_mean)
```

**Pros:**

- ✅ **Fastest option** (5-10% faster than other approaches)
- More flexible than bare Option 1 — easy to swap optimizers
- Single `make_step` function definition
- Still a closure-based approach

**Cons:**

- Requires lambda wrapping boilerplate
- Still creates a new trainer instance for each optimizer
- Less clean than Option 2 for many optimizer experiments

**When to use:** When you need both speed and flexibility for testing a few
different optimizers.

##### **Option 2: Optimizer in Carry State** (✅ **Recommended**)

Pack the optimizer into the carry state tuple alongside model, opt_state, and
key:

```python
class NNTrainer(AbstractScanNNTrainer):
    def init(self, *, key, X, y, learning_rate=1e-2):
        """Initialize model, optimizer, and training data."""
        model_key, data_key = jr.split(key)
        model = eqx.nn.MLP(...)

        # Create optimizer instance
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        # Carry now includes optimizer
        initial_carry = (model, opt_state, optimizer, key)

        mask = jnp.ones(len(X), dtype=bool)
        epoch_data = (mask, (X, y))
        return initial_carry, epoch_data

    def pack_carry_state(self, carry):
        model, opt_state, optimizer, key = carry
        model_dyn, model_static = eqx.partition(model, eqx.is_array)
        return (
            (model_dyn, opt_state, key),
            {"model_static": model_static, "optimizer": optimizer},
        )

    def unpack_carry_state(self, carry, static):
        model_dyn, opt_state, key = carry
        model = eqx.combine(model_dyn, static["model_static"])
        optimizer = static["optimizer"]
        return (model, opt_state, optimizer, key)


def make_step(carry, batch_inputs):
    model, opt_state, optimizer, key = carry  # optimizer is explicit
    # ... loss computation ...
    updates, opt_state = optimizer.update(grads, opt_state)
    return loss, (model, opt_state, optimizer, key)


trainer = NNTrainer(make_step=make_step, loss_agg_fn=masked_mean)
```

**Pros:**

- ✅ **Trainer can be reused with different optimizers**
- Optimizer is explicit and visible in training state
- No need to create new `make_step` functions
- Very flexible for hyperparameter searches

**Cons:**

- Slightly larger carry tuple
- Must implement explicit packing/unpacking logic

**When to use:** Production code, hyperparameter sweeps, reusable trainer
implementations.

**Flexibility example:**

```python
# Same trainer works with any optimizer!
trainer = NNTrainer(make_step=make_step, loss_agg_fn=masked_mean)

# Try Adam
carry_adam = trainer.init(key=key, X=X, y=y, learning_rate=1e-2)
final, losses = trainer.run(carry_adam, epoch_data, num_epochs=10, ...)

# Try SGD with same trainer (no new make_step needed!)
carry_sgd = trainer.init(key=key, X=X, y=y, learning_rate=1e-3)
final, losses = trainer.run(carry_sgd, epoch_data, num_epochs=10, ...)
# ^^ Only init() changes, not the trainer
```

##### **Option 3: Optimizer via `step_kw` Parameter**

Pass the optimizer as a keyword argument via the `step_kw` parameter in
`trainer.run()`:

```python
def make_step(carry, batch_inputs, *, optimizer):  # optimizer as kwarg
    model, opt_state, key = carry
    # ... loss computation ...
    updates, opt_state = optimizer.update(grads, opt_state)
    return loss, (model, opt_state, key)


optimizer = optax.adam(1e-2)
trainer = NNTrainer(make_step=make_step, loss_agg_fn=masked_mean)
carry, epoch_data = trainer.init(...)

final_carry, losses = trainer.run(
    carry,
    epoch_data,
    num_epochs=10,
    batch_size=16,
    key=key,
    step_kw={"optimizer": optimizer},  # pass optimizer here
)
```

**Pros:**

- Clean separation of hyper-parameter config from carry state
- Simple implementation
- Works for any optimizer via `step_kw`

**Cons:**

- Optimizer not explicitly part of the training state
- Less common pattern in JAX training code

**When to use:** When you want optimizer as a configuration parameter rather
than training state.

##### **Recommendation: Use Option 2 (Optimizer in Carry)**

**Option 2 is the recommended approach** for most use cases because:

1. **Flexibility** — The same trainer works with any optimizer configuration
2. **Explicitness** — Optimizer is a visible, integral part of training state
3. **Reusability** — No need to re-create functions for different optimizers
4. **Clarity** — Training state is complete and self-contained in carry
5. **Performance** — Within 1% of the fastest option; differences are
   **negligible**

The extra complexity of packing/unpacking is minimal and well worth the
flexibility gains.

**Benchmark Results** (Multi-trial with warmup, 5 epochs, 128 samples,
batch_size=16):

| Option                       | Mean Time | Overhead           | Std Dev |
| ---------------------------- | --------- | ------------------ | ------- |
| **1: Direct Closure**        | 0.279s    | Baseline (fastest) | ±0.005s |
| **2: Optimizer in Carry** ✅ | 0.282s    | +1.0%              | ±0.019s |
| **3: step_kw**               | 0.287s    | +2.9%              | ±0.021s |
| **1b: Lambda Wrapper**       | 0.294s    | +5.3%              | ±0.003s |

**Key insight**: All strategies perform within ~5% of each other. Performance is
**negligible** relative to actual training costs (forward/backward pass, data
loading), so choose based on code clarity and maintainability, not speed.

**When to choose each option:**

- **Option 1 (bare closure)**: Quick scripts with exactly one fixed optimizer
- **Option 1b (lambda wrapping)**: Flexible single-function approach; middle
  ground between 1 and 3
- **Option 3 (step_kw)**: Optimizer as pure hyperparameter, not training state
- **Option 2 (Optimizer in Carry)** ✅ **Recommended**: Production code,
  hyperparameter sweeps, reusable trainers

See `tests/test_nn_trainer.py` for complete examples and benchmarks of all
patterns.
