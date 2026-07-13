"""Copyright (c) 2026 Nathaniel Starkman. All rights reserved.

Optional dependencies for jaxmore.
"""

__all__ = ("OptDeps",)

from optional_dependencies import OptionalDependencyEnum, auto


class OptDeps(OptionalDependencyEnum):
    """Optional dependencies for jaxmore."""

    EQUINOX = auto()
    JAX_TQDM = auto()
