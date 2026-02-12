"""Black-Scholes-Merton pricing utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from advisor.core.enums import OptionType


@dataclass
class BSMResult:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = OptionType.CALL,
) -> BSMResult:
    """Calculate Black-Scholes option price and Greeks.

    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility (annualized)
        option_type: CALL or PUT
    """
    if T <= 0:
        # At expiration
        if option_type == OptionType.CALL:
            intrinsic = max(S - K, 0.0)
            delta = 1.0 if S > K else 0.0
        else:
            intrinsic = max(K - S, 0.0)
            delta = -1.0 if S < K else 0.0
        return BSMResult(price=intrinsic, delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)

    nd1 = norm.cdf(_d1)
    nd2 = norm.cdf(_d2)
    npd1 = norm.pdf(_d1)

    if option_type == OptionType.CALL:
        price = S * nd1 - K * math.exp(-r * T) * nd2
        delta = nd1
        theta = (
            -(S * npd1 * sigma) / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * nd2
        ) / 365.0
        rho = K * T * math.exp(-r * T) * nd2 / 100.0
    else:
        price = K * math.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)
        delta = nd1 - 1.0
        theta = (
            -(S * npd1 * sigma) / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * norm.cdf(-_d2)
        ) / 365.0
        rho = -K * T * math.exp(-r * T) * norm.cdf(-_d2) / 100.0

    gamma = npd1 / (S * sigma * sqrt_T)
    vega = S * npd1 * sqrt_T / 100.0

    return BSMResult(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
