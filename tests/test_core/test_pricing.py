"""Tests for BSM pricing utilities."""

import math

from advisor.core.enums import OptionType
from advisor.core.pricing import bsm_price


def test_call_price_basic():
    # S=100, K=100, T=1yr, r=5%, sigma=20%
    result = bsm_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    # ATM call should be around $10
    assert 8.0 < result.price < 12.0
    # Delta should be around 0.5+ for ATM call
    assert 0.5 < result.delta < 0.8


def test_put_price_basic():
    result = bsm_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type=OptionType.PUT)
    # ATM put should be less than call (due to r > 0)
    assert 5.0 < result.price < 10.0
    # Put delta should be negative (ATM ~= -0.36 for r=5%)
    assert -0.2 > result.delta > -0.8


def test_put_call_parity():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    call = bsm_price(S, K, T, r, sigma, OptionType.CALL)
    put = bsm_price(S, K, T, r, sigma, OptionType.PUT)
    # C - P = S - K*e^(-rT)
    lhs = call.price - put.price
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 0.01


def test_deep_itm_call():
    result = bsm_price(S=150, K=100, T=0.5, r=0.05, sigma=0.20)
    # Deep ITM call: price should be close to intrinsic
    assert result.price > 49.0
    assert result.delta > 0.95


def test_expired_option():
    result = bsm_price(S=110, K=100, T=0.0, r=0.05, sigma=0.20)
    assert result.price == 10.0  # Max(110-100, 0)
    assert result.delta == 1.0


def test_greeks_signs():
    result = bsm_price(S=100, K=100, T=0.5, r=0.05, sigma=0.20)
    assert result.delta > 0  # Call delta positive
    assert result.gamma > 0  # Gamma always positive
    assert result.theta < 0  # Theta negative (time decay)
    assert result.vega > 0  # Vega positive
