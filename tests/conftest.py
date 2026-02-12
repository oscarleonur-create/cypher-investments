"""Shared test fixtures."""

from __future__ import annotations

import pytest

from advisor.strategies.registry import StrategyRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the strategy registry before each test."""
    StrategyRegistry.reset()
    yield
    StrategyRegistry.reset()
