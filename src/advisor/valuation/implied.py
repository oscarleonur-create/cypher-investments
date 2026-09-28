"""What a price requires, and — beside it — what the filings say it is worth.

The headline runs the engine backwards: given the price the market is
charging today, what growth would the business have to deliver to justify it?
That number is arithmetic, not a judgment, and it is *falsifiable*: "the price
requires 18% revenue growth a year for a decade" becomes true or false quarter
by quarter, which is exactly what a thesis claim needs.

It is now discounted. The first version asked what revenue would make the
business worth today's enterprise value *in ten years* — a 0% return, with the
cash in between ignored — and at the generic 25% margin had AMZN "requiring"
revenue to shrink 6% a year. Both directions now run through ``valuation.dcf``
at a stated 10% discount rate and 3% terminal growth.

The generic margins stay the snapshot's scenarios, so a thesis rule tests the
same kind of quantity it was written against (user decision, 2026-09-25): the
base case is the 25% margin. The company's own margins are recorded beside
them for the range, and the value range (bear/base/bull per share) is built
from those own margins only.

Nothing here recommends anything.
"""

from __future__ import annotations

import logging
from datetime import date

from advisor.valuation.dcf import (
    DISCOUNT_RATE,
    GROWTH_BOUNDS,
    TERMINAL_GROWTH,
    YEARS,
    clamp,
    faded_growth,
    gordon_multiple,
    implied_margin,
    required_path,
    value_range,
)
from advisor.valuation.figures import Figures, build_figures
from advisor.valuation.models import (
    Fundamentals,
    ImpliedExpectations,
    ScenarioValue,
    ValuationSnapshot,
)

logger = logging.getLogger(__name__)

DEFAULT_YEARS = YEARS

# The generic steady-state margins the thesis quantity is read at. A mature
# software business converts ~30% of revenue to free cash flow; an
# infrastructure-heavy one, far less. The middle one is the base case.
GENERIC_MARGINS: tuple[float, ...] = (0.30, 0.25, 0.20)


def implied_expectations(
    enterprise_value: float,
    base_revenue: float,
    *,
    fcf_margin: float,
    start_margin: float | None = None,
    discount_rate: float = DISCOUNT_RATE,
    terminal_growth: float = TERMINAL_GROWTH,
) -> ImpliedExpectations | None:
    """The constant revenue growth the price requires at one steady-state margin.

    Today's margin fades to ``fcf_margin`` over the decade; without a known
    start the steady state applies from year one. None when the price is
    beyond what any growth in the solver's bracket could justify, or the
    inputs cannot describe a business.
    """
    if base_revenue is None or base_revenue <= 0 or enterprise_value <= 0 or fcf_margin <= 0:
        return None
    start = fcf_margin if start_margin is None else start_margin
    solved, projection = required_path(
        enterprise_value,
        base_revenue,
        start,
        fcf_margin,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
    )
    if solved.value is None or projection is None:
        return None
    return ImpliedExpectations(
        terminal_multiple=round(gordon_multiple(discount_rate, terminal_growth) or 0.0, 2),
        fcf_margin=fcf_margin,
        years=YEARS,
        required_fcf=projection.fcf[-1],
        required_revenue=projection.revenue[-1],
        implied_cagr=solved.value,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
    )


def undiscounted_expectations(
    enterprise_value: float,
    revenue_runrate: float,
    *,
    terminal_multiple: float,
    fcf_margin: float,
    years: int = YEARS,
) -> ImpliedExpectations | None:
    """The reading every snapshot stored before 2026-09-27 carries.

    Required revenue in year ten is today's enterprise value over a terminal
    multiple and margin — what would make the business worth today's price in
    ten years, a 0% return with the cash in between ignored. Kept only so an
    old row can be read the way it was computed; nothing new is built on it.
    """
    if enterprise_value <= 0 or revenue_runrate <= 0:
        return None
    if terminal_multiple <= 0 or not (0 < fcf_margin <= 1) or years <= 0:
        return None
    required_fcf = enterprise_value / terminal_multiple
    required_revenue = required_fcf / fcf_margin
    return ImpliedExpectations(
        terminal_multiple=terminal_multiple,
        fcf_margin=fcf_margin,
        years=years,
        required_fcf=required_fcf,
        required_revenue=required_revenue,
        implied_cagr=(required_revenue / revenue_runrate) ** (1 / years) - 1,
    )


def build_snapshot(
    source: Figures | Fundamentals,
    price: float | None = None,
    *,
    asof: date | None = None,
    margins: tuple[float, ...] = GENERIC_MARGINS,
    discount_rate: float = DISCOUNT_RATE,
    terminal_growth: float = TERMINAL_GROWTH,
) -> ValuationSnapshot | None:
    """Figures (or one filing plus a price) into a full valuation snapshot.

    Refuses rather than approximates: without a price, a share count, a
    balance sheet and revenue there is nothing to compute, and a snapshot
    built on a defaulted input would be confidently wrong.
    """
    if isinstance(source, Fundamentals):
        if not source.complete:
            logger.info(
                "valuation: cannot value %s — missing %s",
                source.symbol,
                ", ".join(source.missing),
            )
            return None
        figures = build_figures(source.symbol, price, source)
    else:
        figures = source
        if price is not None:
            figures = figures.model_copy(update={"price": price})

    ev = figures.enterprise_value
    base = figures.revenue_base
    if (
        figures.price is None
        or figures.price <= 0
        or not figures.shares
        or figures.shares <= 0
        or ev is None
        or not base
        or base <= 0
    ):
        logger.info(
            "valuation: cannot value %s — no price, shares, balance or revenue", figures.symbol
        )
        return None

    start = figures.start_margin
    built = [
        implied_expectations(
            ev,
            base,
            fcf_margin=m,
            start_margin=start,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
        )
        for m in margins
    ]

    reading = value_range(
        price=figures.price,
        shares=figures.shares,
        net_cash=figures.net_cash,
        base_revenue=base,
        start_margin=start,
        current_growth=figures.revenue_growth,
        margins=figures.margin_readings(),
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
    )

    needed_margin, needed_label = None, ""
    if figures.revenue_growth is not None:
        # The same bounded year-one growth the value range starts from: CRDO's
        # +165% faded from there needs almost no margin at all, which says
        # more about the fade than the business.
        growth = clamp(figures.revenue_growth, GROWTH_BOUNDS)
        early, late = faded_growth(growth, terminal_growth)
        solved = implied_margin(
            ev,
            base,
            start,
            early,
            late,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
        )
        needed_margin = solved.value
        needed_label = (
            f"with growth fading from {growth:+.1%} to " f"{terminal_growth:.0%} over {YEARS} years"
            if solved.value is not None
            else f"no margin up to 100% justifies the price at that growth ({solved.beyond})"
        )

    trailing = figures.margin("fcf") or figures.margin("yahoo")
    median = figures.margin("median")
    return ValuationSnapshot(
        symbol=figures.symbol,
        asof=asof or date.today(),
        price=figures.price,
        price_source=figures.price_source,
        shares_outstanding=figures.shares,
        market_cap=figures.market_cap,
        net_cash=figures.net_cash,
        enterprise_value=ev,
        revenue_runrate=figures.revenue_runrate,
        ev_to_revenue=ev / base,
        source_accession=figures.source_accession,
        period_end=figures.balance_asof or asof or date.today(),
        revenue_yoy=figures.revenue_growth,
        scenarios=[s for s in built if s is not None],
        margin_trailing=trailing.value if trailing else None,
        margin_trailing_label=trailing.label if trailing else "",
        margin_median=median.value if median else None,
        margin_median_label=median.label if median else "",
        method="dcf",
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        revenue_base=base,
        revenue_base_label=figures.revenue_base_label,
        revenue_growth_label=figures.revenue_growth_label,
        start_margin=start,
        start_margin_label=figures.start_margin_label,
        own_margins=list(figures.margins),
        value=[
            ScenarioValue(
                name=s.name,
                value_per_share=s.value_per_share,
                upside=s.upside,
                growth_start=s.current_growth,
                growth_early=s.path.growth_early,
                growth_late=s.path.growth_late,
                target_margin=s.path.target_margin,
                margin_label=s.margin_label,
                discount_rate=s.path.discount_rate,
                terminal_growth=s.path.terminal_growth,
                enterprise_value=s.projection.enterprise_value,
                terminal_share=s.projection.terminal_share,
            )
            for s in reading.scenarios
        ],
        value_refused=reading.refused,
        implied_margin=needed_margin,
        implied_margin_label=needed_label,
        notes=list(figures.notes) + list(reading.notes),
    )
