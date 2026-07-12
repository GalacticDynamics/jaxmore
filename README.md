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


# Create a trainer subclass. You implement three hooks:
#   init()               -- build the initial carry and the epoch data
#   pack_carry_state()   -- split the carry into arrays + static structure
#   unpack_carry_state() -- put it back together
class NNTrainer(AbstractScanNNTrainer):
    """Trainer for a small feed-forward regressor."""

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

        # Training carry: (model, optimizer_state, rng_key)
        carry_key = jr.fold_in(data_key, 0)
        initial_carry = (model, opt_state, carry_key)

        # Every sample is usable here. Mark a sample False to have the trainer
        # carry it along but never train on it.
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
        model = eqx.combine(model_dyn, static["model_static"])
        return (model, opt_state, key)


# The optimizer is captured in the closure below. See "Optimizer Passing
# Strategies" for the alternatives.
optimizer = optax.adam(1e-2)


# The per-batch training step. `make_step` is passed to the constructor -- it is
# NOT a method you override.
def make_step(carry, batch_inputs):
    """Execute one batch of training."""
    model, opt_state, key = carry
    batch_mask, (X_batch, y_batch) = batch_inputs

    def loss_fn(model):
        preds = jax.vmap(model)(X_batch).squeeze(-1)
        # NOTE: average over *real* rows only. Batches are zero-padded to a
        # constant shape, so `jnp.mean` here would train on the padding.
        return masked_mean((preds - y_batch) ** 2, batch_mask)

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
)

trained_model, _, _ = final_carry
assert losses.shape == (10,)
assert losses[-1] < losses[0]  # the loss went down
```

Key features:

- **Automatic batching** — `shuffle_and_batch()` handles shuffling and padding
- **Efficient scanning** — Uses `jax.lax.scan` for epochs and batches
  (JAX-friendly)
- **Model partitioning** — Separate dynamic arrays (model weights) from static
  structure for efficient JIT compilation
- **Loss aggregation** — Customize how per-batch losses combine into epoch
  losses

#### ⚠️ Sharp Edge: Your Loss Must Respect `batch_mask`

`shuffle_and_batch` pads the final batch so that every batch has the same shape
— `jax.lax.scan` requires it. Those padded rows are **fake data** (zeros by
default), and they are handed to your `make_step` along with a `batch_mask` that
marks them False.

If you ignore `batch_mask` and write the obvious thing:

```python
# WRONG -- trains on the zero-padded rows
def loss_fn(model):
    preds = jax.vmap(model)(X_batch).squeeze(-1)
    return jnp.mean((preds - y_batch) ** 2)
```

…then with `N=100` and `batch_size=16` you have 112 slots, so **12 fabricated
`(0, 0) -> 0` samples contribute a gradient every single epoch**. Nothing
errors; your model just quietly fits noise.

Use the mask:

```python
# RIGHT -- averages over real rows only
def loss_fn(model):
    preds = jax.vmap(model)(X_batch).squeeze(-1)
    return masked_mean((preds - y_batch) ** 2, batch_mask)
```

The same applies to the `mask` you return from `init()`. Samples you mark False
are sorted to the _end_ of the dataset, so they cluster into whole batches — a
batch can legitimately contain zero usable samples. `masked_mean` returns `NaN`
there (with a well-defined, zero gradient), and `AbstractScanNNTrainer` skips
those batches entirely, so they never reach your `make_step`.

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

#### Varying things across epochs

Two optional hooks run once per epoch. Both receive `epoch_idx` (a traced
scalar), `num_epochs` (a static Python int), and `epoch_key`. Override whichever
you need; by default both are no-ops and are skipped entirely.

| Hook                | Returns                          | Use it for                                                                    |
| ------------------- | -------------------------------- | ----------------------------------------------------------------------------- |
| `prepare_step_kw`   | a mapping, merged over `step_kw` | scalars: scheduled loss weights, annealed temperatures, curriculum thresholds |
| `prepare_data_args` | the data tuple, before batching  | arrays with leading dim `N`: resampled negatives, augmentations               |

The split matters: anything `prepare_data_args` returns goes through
`shuffle_and_batch`, so it must be `N`-long. A scalar can't ride along — that's
what `prepare_step_kw` is for.

##### Scheduling a hyperparameter — `prepare_step_kw`

Ramp a loss weight linearly from `1.0` to `5.0` over training:

```python
LAM_MIN, LAM_MAX = 1.0, 5.0


class RampTrainer(NNTrainer):
    def prepare_step_kw(self, /, *, epoch_idx, num_epochs, epoch_key):
        frac = epoch_idx / (num_epochs - 1) if num_epochs > 1 else 0.0
        return {"lam": LAM_MIN + (LAM_MAX - LAM_MIN) * frac}


def make_step_ramp(carry, batch_inputs, *, lam):  # <- arrives as a kwarg
    model, opt_state, key = carry
    batch_mask, (X_batch, y_batch) = batch_inputs

    def loss_fn(m):
        preds = jax.vmap(m)(X_batch).squeeze(-1)
        return lam * masked_mean((preds - y_batch) ** 2, batch_mask)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    return loss, (eqx.apply_updates(model, updates), opt_state, key)


trainer = RampTrainer(make_step=make_step_ramp, loss_agg_fn=masked_mean)
carry, epoch_data = trainer.init(key=key, X=X, y=y)
_, losses = trainer.run(carry, epoch_data, num_epochs=6, batch_size=16, key=key)
assert losses.shape == (6,)
```

Keys returned by the hook take precedence over the static `step_kw` passed to
`run()`. The returned values may be traced arrays — the batch scan closes over
them, so they reach `make_step` as ordinary runtime values.

##### Resampling data each epoch — `prepare_data_args`

Draw fresh random "negatives" every epoch, batched alongside the real samples:

```python
lo, hi = X.min(axis=0), X.max(axis=0)


class NegativesTrainer(NNTrainer):
    def prepare_data_args(
        self, carry, data_args, /, *, epoch_idx, num_epochs, epoch_key
    ):
        X_real, y_real = data_args
        negatives = jr.uniform(epoch_key, shape=X_real.shape, minval=lo, maxval=hi)
        return (X_real, y_real, negatives)  # make_step now sees three arrays


def make_step_neg(carry, batch_inputs):
    model, opt_state, key = carry
    batch_mask, (X_batch, y_batch, X_neg) = batch_inputs

    def loss_fn(m):
        preds = jax.vmap(m)(X_batch).squeeze(-1)
        fit = masked_mean((preds - y_batch) ** 2, batch_mask)
        neg = masked_mean(jax.vmap(m)(X_neg).squeeze(-1) ** 2, batch_mask)
        return fit + 0.01 * neg

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    return loss, (eqx.apply_updates(model, updates), opt_state, key)


trainer = NegativesTrainer(make_step=make_step_neg, loss_agg_fn=masked_mean)
carry, epoch_data = trainer.init(key=key, X=X, y=y)
_, losses = trainer.run(carry, epoch_data, num_epochs=10, batch_size=16, key=key)
assert losses[-1] < losses[0]
```

#### Freezing part of a model

`eqx.is_array` makes _every_ array trainable. To freeze a submodule, build the
filter spec by hand and zero out that subtree — frozen arrays then land in the
static half of the partition, where the optimizer never sees them and `scan`
carries them as constants.

```python
class Encoder(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(2, 4, 8, 1, activation=jax.nn.relu, key=key)

    def __call__(self, x):
        return self.mlp(x)


class AutoEncoder(eqx.Module):
    encoder: Encoder
    decoder: eqx.nn.MLP

    def __init__(self, key):
        ekey, dkey = jr.split(key)
        self.encoder = Encoder(ekey)
        self.decoder = eqx.nn.MLP(4, 1, 8, 1, activation=jax.nn.relu, key=dkey)

    def __call__(self, x):
        return self.decoder(self.encoder(x))


def trainable_spec(model):
    """Every array is trainable EXCEPT those under `.encoder`."""
    spec = jax.tree_util.tree_map(eqx.is_array, model)
    frozen = jax.tree_util.tree_map(lambda _: False, model.encoder)
    return eqx.tree_at(lambda m: m.encoder, spec, frozen)


class FrozenEncoderTrainer(AbstractScanNNTrainer):
    def init(self, *, key, X, y):
        model_key, data_key = jr.split(key)
        model = AutoEncoder(model_key)
        # Initialize the optimizer on the TRAINABLE leaves only, so its state
        # matches the gradients make_step will hand it.
        trainable, _ = eqx.partition(model, trainable_spec(model))
        opt_state = optax.adam(1e-2).init(trainable)
        return (model, opt_state, jr.fold_in(data_key, 0)), (
            jnp.ones(len(X), dtype=bool),
            (X, y),
        )

    def pack_carry_state(self, carry):
        model, opt_state, key = carry
        model_dyn, model_static = eqx.partition(model, trainable_spec(model))
        return (model_dyn, opt_state, key), {"model_static": model_static}

    def unpack_carry_state(self, carry, static):
        model_dyn, opt_state, key = carry
        return (eqx.combine(model_dyn, static["model_static"]), opt_state, key)


def make_step_frozen(carry, batch_inputs):
    model, opt_state, key = carry
    batch_mask, (X_batch, y_batch) = batch_inputs

    # Differentiate w.r.t. the trainable half only. Grading the whole model
    # would produce encoder gradients that opt_state has no slot for.
    trainable, frozen = eqx.partition(model, trainable_spec(model))

    def loss_fn(t):
        preds = jax.vmap(eqx.combine(t, frozen))(X_batch).squeeze(-1)
        return masked_mean((preds - y_batch) ** 2, batch_mask)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
    updates, opt_state = optimizer.update(grads, opt_state)
    trainable = eqx.apply_updates(trainable, updates)
    return loss, (eqx.combine(trainable, frozen), opt_state, key)


trainer = FrozenEncoderTrainer(make_step=make_step_frozen, loss_agg_fn=masked_mean)
carry, epoch_data = trainer.init(key=key, X=X, y=y)
before = carry[0]
final_carry, losses = trainer.run(
    carry, epoch_data, num_epochs=10, batch_size=16, key=key
)
after = final_carry[0]

# The encoder came out bit-for-bit unchanged; the decoder trained.
enc_b = jax.tree_util.tree_leaves(eqx.filter(before.encoder, eqx.is_array))
enc_a = jax.tree_util.tree_leaves(eqx.filter(after.encoder, eqx.is_array))
assert all(jnp.array_equal(b, a) for b, a in zip(enc_b, enc_a))
assert losses[-1] < losses[0]
```

#### Optimizer Passing Strategies

`make_step` needs the optimizer, but `make_step` is just a function you hand to
the constructor — so how does it get one? There are three patterns. Every block
below runs.

They perform the same. Pick on clarity, not speed: the optimizer plumbing is
noise next to the forward/backward pass, and the differences we measured were
within run-to-run variance. **If you don't want to think about it, use
Option 2.**

##### Option 1: Optimizer in a closure

The simplest thing that works. `make_step` closes over an optimizer defined
outside it — this is what the example above does.

```python
optimizer_adam = optax.adam(1e-2)


def make_step_closure(carry, batch_inputs):
    model, opt_state, key = carry
    batch_mask, (X_batch, y_batch) = batch_inputs

    def loss_fn(m):
        preds = jax.vmap(m)(X_batch).squeeze(-1)
        return masked_mean((preds - y_batch) ** 2, batch_mask)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer_adam.update(grads, opt_state)
    return loss, (eqx.apply_updates(model, updates), opt_state, key)


trainer = NNTrainer(make_step=make_step_closure, loss_agg_fn=masked_mean)
carry, epoch_data = trainer.init(key=key, X=X, y=y)
_, losses = trainer.run(carry, epoch_data, num_epochs=5, batch_size=16, key=key)
assert losses[-1] < losses[0]
```

**Trade-off:** the optimizer is baked in. Swapping it means writing another
`make_step`. Fine for a script with one optimizer; annoying for a sweep.

**Use it when:** you have exactly one fixed optimizer.

##### Option 2: Optimizer in the carry ✅ _recommended_

Put the optimizer in the carry and stash it in the _static_ metadata during
packing. An `optax.GradientTransformation` is a pair of functions, not arrays,
so it belongs on the static side — the same reasoning as `eqx.partition`.

```python
class TrainerWithOptimizer(AbstractScanNNTrainer):
    """Carries the optimizer as part of the training state."""

    def init(self, *, key, X, y, optimizer):
        model_key, data_key = jr.split(key)
        model = eqx.nn.MLP(
            in_size=X.shape[1],
            out_size=1,
            width_size=32,
            depth=2,
            activation=jax.nn.relu,
            key=model_key,
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        initial_carry = (model, opt_state, optimizer, jr.fold_in(data_key, 0))
        return initial_carry, (jnp.ones(len(X), dtype=bool), (X, y))

    def pack_carry_state(self, carry):
        model, opt_state, optimizer, key = carry
        model_dyn, model_static = eqx.partition(model, eqx.is_array)
        # The optimizer is functions, not arrays -> static side.
        return (
            (model_dyn, opt_state, key),
            {"model_static": model_static, "optimizer": optimizer},
        )

    def unpack_carry_state(self, carry, static):
        model_dyn, opt_state, key = carry
        model = eqx.combine(model_dyn, static["model_static"])
        return (model, opt_state, static["optimizer"], key)


def make_step_carry(carry, batch_inputs):
    model, opt_state, optimizer, key = carry  # optimizer is explicit
    batch_mask, (X_batch, y_batch) = batch_inputs

    def loss_fn(m):
        preds = jax.vmap(m)(X_batch).squeeze(-1)
        return masked_mean((preds - y_batch) ** 2, batch_mask)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    # Pass `params` too: optimizers like `adamw` need the current parameters
    # (for decoupled weight decay). Optimizers that don't need them ignore them,
    # so passing `params` always is the safe default.
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return loss, (eqx.apply_updates(model, updates), opt_state, optimizer, key)


# One trainer, any optimizer -- only `init()` changes.
trainer = TrainerWithOptimizer(make_step=make_step_carry, loss_agg_fn=masked_mean)

for opt in [optax.adam(1e-2), optax.sgd(1e-2), optax.adamw(1e-2)]:
    carry, epoch_data = trainer.init(key=key, X=X, y=y, optimizer=opt)
    _, losses = trainer.run(carry, epoch_data, num_epochs=5, batch_size=16, key=key)
    assert losses[-1] < losses[0]
```

**Trade-off:** a slightly bigger carry and explicit packing logic.

**Use it when:** production code, hyperparameter sweeps, reusable trainers —
i.e. most of the time.

##### Option 3: Optimizer via `step_kw`

`run(step_kw=...)` forwards keyword arguments to `make_step` on every batch.
This keeps the optimizer out of the training state and treats it as
configuration.

```python
def make_step_kw(carry, batch_inputs, *, optimizer):
    model, opt_state, key = carry
    batch_mask, (X_batch, y_batch) = batch_inputs

    def loss_fn(m):
        preds = jax.vmap(m)(X_batch).squeeze(-1)
        return masked_mean((preds - y_batch) ** 2, batch_mask)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    return loss, (eqx.apply_updates(model, updates), opt_state, key)


trainer = NNTrainer(make_step=make_step_kw, loss_agg_fn=masked_mean)
carry, epoch_data = trainer.init(key=key, X=X, y=y)
_, losses = trainer.run(
    carry,
    epoch_data,
    num_epochs=5,
    batch_size=16,
    key=key,
    step_kw={"optimizer": optax.adam(1e-2)},  # <- forwarded to make_step
)
assert losses[-1] < losses[0]
```

**Trade-off:** the optimizer isn't part of the training state, which is a little
unusual for JAX training code. `step_kw` values are captured at trace time, so
they must be static across the whole run — you can't schedule them per-epoch
this way.

**Use it when:** the optimizer is genuinely configuration rather than state.

##### Summary

| Pattern        | Swap optimizers without a new `make_step`? | Optimizer visible in state? |
| -------------- | ------------------------------------------ | --------------------------- |
| 1: Closure     | ❌                                         | ❌                          |
| 2: In carry ✅ | ✅                                         | ✅                          |
| 3: `step_kw`   | ✅                                         | ❌                          |

See `tests/usage/test_nn.py` for these patterns as end-to-end tests.
