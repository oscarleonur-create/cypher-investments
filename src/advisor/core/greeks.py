"""Greeks calculation utilities using BSM pricing."""

from __future__ import annotations

from advisor.core.enums import OptionType
from advisor.core.models import Greeks
from advisor.core.pricing import bsm_price


def calculate_greeks(
    underlying_price: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType = OptionType.CALL,
) -> Greeks:
    """Calculate option Greeks using Black-Scholes-Merton model."""
    result = bsm_price(
        S=underlying_price,
        K=strike,
        T=time_to_expiry_years,
        r=risk_free_rate,
        sigma=volatility,
        option_type=option_type,
    )
    return Greeks(
        delta=result.delta,
        gamma=result.gamma,
        theta=result.theta,
        vega=result.vega,
        rho=result.rho,
    )
