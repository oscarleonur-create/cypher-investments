"""What a price requires, rather than what a business is worth.

A discounted cash flow asks "what is this worth" and answers with whatever the
author's assumptions imply — which is why two analysts produce two numbers and
neither is falsifiable. This runs the arithmetic backwards: given the price the
market is charging today, what growth would the business have to deliver to
justify it?

That number is a fact about arithmetic, not a judgment. It is also
*falsifiable*: "the price requires 26% revenue growth for a decade" becomes
true or false quarter by quarter, which is exactly what a thesis claim needs
and what a valuation opinion can never be.

Nothing here recommends anything. It reports what would have to happen.
"""

from __future__ import annotations

import logging
from datetime import date

from advisor.valuation.models import Fundamentals, ImpliedExpectations, ValuationSnapshot

logger = logging.getLogger(__name__)

DEFAULT_YEARS = 10

# Three readings of the same price, spanning plausible steady states rather
# than one false-precision answer. A mature software business converts ~30% of
# revenue to free cash flow; an infrastructure-heavy one, far less.
DEFAULT_SCENARIOS: tuple[tuple[float, float], ...] = (
    (30.0, 0.30),  # generous: high terminal multiple, software-like margin
    (25.0, 0.25),  # middle
    (20.0, 0.20),  # demanding: capital-intensive, modest multiple
)


def implied_expectations(
    enterprise_value: float,
    revenue_runrate: float,
    *,
    terminal_multiple: float,
    fcf_margin: float,
    years: int = DEFAULT_YEARS,
) -> ImpliedExpectations | None:
    """The revenue CAGR the price requires under one set of assumptions."""
    if enterprise_value <= 0 or revenue_runrate <= 0:
        return None
    if terminal_multiple <= 0 or not (0 < fcf_margin <= 1) or years <= 0:
        return None

    required_fcf = enterprise_value / terminal_multiple
    required_revenue = required_fcf / fcf_margin
    cagr = (required_revenue / revenue_runrate) ** (1 / years) - 1
    return ImpliedExpectations(
        terminal_multiple=terminal_multiple,
        fcf_margin=fcf_margin,
        years=years,
        required_fcf=required_fcf,
        required_revenue=required_revenue,
        implied_cagr=cagr,
    )


def build_snapshot(
    fundamentals: Fundamentals,
    price: float,
    *,
    asof: date | None = None,
    scenarios: tuple[tuple[float, float], ...] = DEFAULT_SCENARIOS,
    years: int = DEFAULT_YEARS,
) -> ValuationSnapshot | None:
    """Price plus one filing into a full valuation snapshot.

    Refuses rather than approximates: a filing missing revenue, cash, shares
    or debt cannot support this, and a snapshot built on a defaulted input
    would be confidently wrong in the flattering direction.
    """
    if not fundamentals.complete or price <= 0:
        logger.info(
            "valuation: cannot value %s — missing %s",
            fundamentals.symbol,
            ", ".join(fundamentals.missing) or "a valid price",
        )
        return None

    shares = fundamentals.shares_outstanding or 0.0
    market_cap = shares * price
    net_cash = fundamentals.net_cash or 0.0
    enterprise_value = market_cap - net_cash
    runrate = fundamentals.revenue_runrate

    built = [
        implied_expectations(
            enterprise_value,
            runrate or 0.0,
            terminal_multiple=multiple,
            fcf_margin=margin,
            years=years,
        )
        for multiple, margin in scenarios
    ]

    return ValuationSnapshot(
        symbol=fundamentals.symbol,
        asof=asof or date.today(),
        price=price,
        shares_outstanding=shares,
        market_cap=market_cap,
        net_cash=net_cash,
        enterprise_value=enterprise_value,
        revenue_runrate=runrate,
        ev_to_revenue=(enterprise_value / runrate) if runrate else None,
        source_accession=fundamentals.source_accession,
        period_end=fundamentals.period_end,
        scenarios=[s for s in built if s is not None],
    )
