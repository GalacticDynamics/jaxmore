"""Usage tests for jaxmore.structured — functools integration, JAX transforms."""

import jax
import jax.numpy as jnp

from jaxmore import structured

# ============================================================================
# Integration with functools and JAX


class TestIntegration:
    """Integration with functools and JAX."""

    def test_functools_wraps_preserved(self) -> None:
        def original(x):
            """My docstring."""
            return x

        wrapped = structured(ins=lambda v: v)(original)

        assert wrapped.__name__ == "original"
        assert wrapped.__doc__ == "My docstring."
        assert wrapped.__wrapped__ is original

    def test_jit_compatibility(self) -> None:
        @jax.jit
        @structured(
            ins=(lambda x: {"val": x},),
            outs=lambda d: d["val"],
        )
        def f(obj):
            return {"val": obj["val"] + jnp.asarray(1)}

        assert int(f(jnp.asarray(4))) == 5
