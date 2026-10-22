"""Copyright (c) 2026 Nathaniel Starkman. All rights reserved.

Strategies for :func:`jaxmore.loop`.
"""

# pylint: disable=duplicate-code

__all__ = (
    "AbstractLoopStrategy",
    "AbstractVMap",
    "Determine",
    "NoLoop",
    "Scan",
    "VMap",
    "Vectorize",
)

from ._src.loop_strategies import (
    AbstractLoopStrategy,
    AbstractVMap,
    Determine,
    NoLoop,
    Scan,
    Vectorize,
    VMap,
)
