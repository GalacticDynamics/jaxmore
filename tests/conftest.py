"""Shared test fixtures."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import jax
import pytest


@contextmanager
def _count_compiles() -> Iterator[list[int]]:
    count = [0]

    def listener(event: str, _: float, **__: object) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            count[0] += 1

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        yield count
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)


@pytest.fixture
def count_compiles() -> Callable[[], AbstractContextManager[list[int]]]:
    """Return a context manager that counts XLA backend compiles in its block."""
    return _count_compiles
