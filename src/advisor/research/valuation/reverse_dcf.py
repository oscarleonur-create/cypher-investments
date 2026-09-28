"""Reverse DCF: the constant revenue growth the current price implies.

The engine's backward solve (``valuation.dcf.implied_growth``) at the base
scenario's margin and discount rate, from the base revenue and today's margin
the DCF was built on.

It used to rebuild base revenue as year-one FCF over the *target* margin —
neither the base year nor the margin that FCF was made at — and to return the
bracket's edge (-10% or +50%) as if it were an answer when the price fell
outside it. It returns None now; the workstation shows nothing rather than a
bound dressed as a result.
"""

from __future__ import annotations

import logging

from advisor.research.models import DcfResult
from advisor.valuation import dcf as engine

logger = logging.getLogger(__name__)


def solve_implied_growth(dcf: DcfResult) -> float | None:
    """The constant growth (years 1–10) at which the DCF equals the market price.

    None when there is no base scenario, no price, no base revenue, or the
    price lies outside what -50% to +150% a year could justify.
    """
    if dcf.base is None or dcf.current_price <= 0 or dcf.shares_outstanding <= 0:
        return None
    if not dcf.base_revenue or dcf.base_revenue <= 0 or dcf.seed_fcf is None:
        return None
    a = dcf.base.assumptions
    ev = engine.market_ev(dcf.current_price, dcf.shares_outstanding, -dcf.net_debt)
    if ev is None:
        return None
    solved = engine.implied_growth(
        ev,
        dcf.base_revenue,
        dcf.seed_fcf / dcf.base_revenue,
        a.target_fcf_margin,
        discount_rate=a.wacc,
        terminal_growth=a.terminal_growth_rate,
    )
    if solved.value is None:
        logger.debug("reverse DCF for %s: %s", dcf.symbol, solved.describe())
    return solved.value
