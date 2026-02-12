"""Tests for core Pydantic models."""

from datetime import date, timedelta

from advisor.core.enums import OptionType, OrderSide
from advisor.core.models import Greeks, OptionContract, Position, Trade


def test_greeks_defaults():
    g = Greeks()
    assert g.delta == 0.0
    assert g.gamma == 0.0
    assert g.theta == 0.0
    assert g.vega == 0.0
    assert g.rho == 0.0


def test_option_contract():
    exp = date.today() + timedelta(days=30)
    contract = OptionContract(
        symbol="SPY240315C500",
        underlying="SPY",
        option_type=OptionType.CALL,
        strike=500.0,
        expiration=exp,
        premium=5.50,
    )
    assert contract.underlying == "SPY"
    assert contract.strike == 500.0
    assert not contract.is_expired
    assert contract.days_to_expiry > 0


def test_option_contract_expired():
    contract = OptionContract(
        symbol="SPY240101C500",
        underlying="SPY",
        option_type=OptionType.CALL,
        strike=500.0,
        expiration=date(2024, 1, 1),
    )
    assert contract.is_expired
    assert contract.days_to_expiry == 0


def test_position_equity():
    pos = Position(symbol="SPY", quantity=100, avg_price=450.0, side=OrderSide.BUY)
    assert not pos.is_option
    assert pos.market_value == 100 * 450.0


def test_position_option():
    exp = date.today() + timedelta(days=30)
    contract = OptionContract(
        symbol="SPY240315C500",
        underlying="SPY",
        option_type=OptionType.CALL,
        strike=500.0,
        expiration=exp,
        premium=5.0,
    )
    pos = Position(
        symbol="SPY240315C500",
        quantity=1,
        avg_price=5.0,
        side=OrderSide.BUY,
        option_contract=contract,
    )
    assert pos.is_option
    assert pos.market_value == 1 * 5.0 * 100


def test_trade():
    trade = Trade(
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=100,
        price=450.0,
        commission=1.0,
    )
    assert not trade.is_option
    assert trade.total_cost == 100 * 450.0 + 1.0
