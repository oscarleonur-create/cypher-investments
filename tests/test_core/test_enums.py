"""Tests for core enums."""

from advisor.core.enums import OptionType, OrderSide, StrategyType


def test_option_type_values():
    assert OptionType.CALL == "call"
    assert OptionType.PUT == "put"


def test_order_side_values():
    assert OrderSide.BUY == "buy"
    assert OrderSide.SELL == "sell"


def test_strategy_type_values():
    assert StrategyType.EQUITY == "equity"
    assert StrategyType.OPTIONS == "options"
    assert StrategyType.MIXED == "mixed"
