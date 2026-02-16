"""Tests for strategy registry."""

from typing import ClassVar

import pytest
from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


def test_register_strategy():
    @StrategyRegistry.register
    class TestStrat(StrategyBase):
        strategy_name: ClassVar[str] = "test_strat"
        strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
        description: ClassVar[str] = "A test strategy"

        def next(self):
            pass

    registry = StrategyRegistry()
    assert "test_strat" in registry.names
    assert registry.get("test_strat") is TestStrat


def test_register_requires_name():
    with pytest.raises(ValueError, match="must define 'strategy_name'"):

        @StrategyRegistry.register
        class BadStrat(StrategyBase):
            def next(self):
                pass


def test_get_strategy_not_found():
    registry = StrategyRegistry()
    with pytest.raises(KeyError, match="not found"):
        registry.get_strategy("nonexistent")


def test_list_strategies():
    @StrategyRegistry.register
    class Strat1(StrategyBase):
        strategy_name: ClassVar[str] = "strat_1"
        strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
        description: ClassVar[str] = "Strategy 1"

        def next(self):
            pass

    registry = StrategyRegistry()
    strategies = registry.list_strategies()
    assert len(strategies) == 1
    assert strategies[0]["name"] == "strat_1"


def test_get_metadata():
    @StrategyRegistry.register
    class MetaStrat(StrategyBase):
        strategy_name: ClassVar[str] = "meta_strat"
        strategy_type: ClassVar[StrategyType] = StrategyType.OPTIONS
        description: ClassVar[str] = "Meta strategy"
        params: ClassVar[tuple] = (("threshold", 0.5),)

        def next(self):
            pass

    meta = MetaStrat.get_metadata()
    assert meta["name"] == "meta_strat"
    assert meta["type"] == "options"
    assert meta["params"]["threshold"] == 0.5


def test_discover():
    registry = StrategyRegistry()
    registry.discover()
    assert "buy_hold" in registry.names
    assert "covered_call" in registry.names
