"""Reverse-DCF: solve for the revenue growth rate implied by the current price.

Uses bisection on the DCF model from dcf.py. Holds FCF margin and WACC at
their base-case values, and solves for the single constant growth rate (yr1-10)
that makes the implied price equal to the current market price.
"""

from __future__ import annotations

import logging

from advisor.research.models import DcfResult

logger = logging.getLogger(__name__)

_TOL = 0.001  # convergence: growth rates within 0.1% pp
_MAX_ITER = 60


def solve_implied_growth(dcf: DcfResult) -> float | None:
    """Return the constant growth rate (yr1-10) that equates DCF price to market price.

    Returns None if the model can't converge (e.g. current price > even 50% growth implies).
    """
    if dcf.base is None or dcf.current_price <= 0:
        return None

    assump = dcf.base.assumptions
    target_price = dcf.current_price

    def _price_at_growth(g: float) -> float:
        from advisor.research.models import DcfAssumptions
        from advisor.research.valuation.dcf import _run_scenario

        a = DcfAssumptions(
            scenario="implied",
            revenue_growth_yr1_3=g,
            revenue_growth_yr4_10=g,  # constant for simplicity
            target_fcf_margin=assump.target_fcf_margin,
            capex_intensity=assump.capex_intensity,
            terminal_growth_rate=assump.terminal_growth_rate,
            terminal_exit_multiple=assump.terminal_exit_multiple,
            wacc=assump.wacc,
        )
        scenario = _run_scenario(
            a,
            base_revenue=dcf.base.projected_fcf[0] / max(assump.target_fcf_margin, 0.001),
            seed_fcf=dcf.base.projected_fcf[0],
            net_debt=dcf.net_debt,
            shares=dcf.shares_outstanding,
            current_price=target_price,
        )
        return scenario.implied_price

    lo, hi = -0.10, 0.50  # search from -10% to +50% growth

    # Bracket check
    if _price_at_growth(lo) > target_price:
        logger.debug("Market price below even -10%% growth DCF — very cheap or broken model")
        return lo
    if _price_at_growth(hi) < target_price:
        logger.debug("Market price above even 50%% growth DCF — very expensive or broken model")
        return hi

    for _ in range(_MAX_ITER):
        mid = (lo + hi) / 2
        price_mid = _price_at_growth(mid)
        if abs(price_mid - target_price) / max(target_price, 1) < _TOL:
            return mid
        if price_mid < target_price:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2
