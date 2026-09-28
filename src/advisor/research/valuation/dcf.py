"""The research workstation's DCF, on the one engine the daemon uses.

Every input comes from ``valuation.figures`` — the SEC's filings for revenue,
cash flow, margins, balance sheet and shares, and the broker's feed for the
price — and every projection from ``valuation.dcf``. This module only shapes
the answer into the ``DcfResult`` the workstation, the Bayesian engine and the
research agent already read.

It used to take price, shares, net debt and revenue from ``yfinance.info``.
On 2026-09-27 Yahoo rate-limited every request; shares defaulted to 1.0, the
price to 0, and JBL's DCF was published at $0.00 a share. It also valued its
exit multiple on an EBITDA of 25% of revenue for every company. Both are gone:
a missing input now produces a ``DcfResult`` with no scenarios and a ``note``
saying which input, and the terminal value is a Gordon perpetuity at the
stated discount rate.

Public API used by the research workstation's what-if sliders:
- ``compute_dcf_scenario(assumptions, base_revenue, seed_fcf, net_debt, shares, current_price)``
- ``dcf_inputs_from_report(report)`` returns the five values the function above needs.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from advisor.research.models import (
    DcfAssumptions,
    DcfResult,
    DcfScenario,
    ResearchReport,
    StatementBundle,
)
from advisor.valuation import dcf as engine

logger = logging.getLogger(__name__)

_PROJECTION_YEARS = engine.YEARS


def build_dcf(
    symbol: str,
    statements: StatementBundle | None = None,  # noqa: ARG001 — kept for callers
    explicit_wacc: float | None = None,
    *,
    price: float | None = None,
    figures=None,
) -> DcfResult:
    """Bear / base / bull DCF for ``symbol`` from the company's own filings.

    ``figures`` may be passed to value without the network (tests, or a caller
    that already loaded them). ``explicit_wacc`` replaces the stated discount
    rate. Never raises for missing data: the result says what is missing.
    """
    from advisor.valuation.figures import load_figures

    if figures is None:
        figures = load_figures(symbol, price)
    elif price is not None:
        figures = figures.model_copy(update={"price": price})

    rate = explicit_wacc or engine.DISCOUNT_RATE
    shares = figures.shares or 0.0
    net_cash = figures.net_cash
    result = DcfResult(
        symbol=symbol.upper(),
        current_price=figures.price or 0.0,
        shares_outstanding=shares,
        net_debt=-(net_cash or 0.0),
        wacc=rate,
        base_revenue=figures.revenue_base,
        seed_fcf=(
            figures.start_margin * figures.revenue_base
            if figures.start_margin is not None and figures.revenue_base
            else None
        ),
        source=_source(figures),
    )

    reading = engine.value_range(
        price=figures.price,
        shares=figures.shares,
        net_cash=net_cash,
        base_revenue=figures.revenue_base,
        start_margin=figures.start_margin,
        current_growth=figures.revenue_growth,
        margins=figures.margin_readings(),
        discount_rate=rate,
    )
    if reading.refused:
        result.note = f"No value range: {reading.refused}."
        logger.info("dcf: %s — %s", symbol.upper(), reading.refused)
    else:
        result.note = "; ".join(reading.notes)
        # A value range was computed, so price, shares and revenue exist.
        if result.seed_fcf is None:
            # No trailing cash flow: the steady state applies from year one,
            # and the sliders must replay that, so the seed carries it.
            result.seed_fcf = reading.get("base").path.target_margin * figures.revenue_base
        # Informational only: the projection runs on FCF margins, which are
        # already net of capex.
        capex = figures.capex_intensity or 0.0
        for s in reading.scenarios:
            assumptions = DcfAssumptions(
                scenario=s.name,
                revenue_growth_yr1_3=s.path.growth_early,
                revenue_growth_yr4_10=s.path.growth_late,
                target_fcf_margin=s.path.target_margin,
                capex_intensity=capex,
                terminal_growth_rate=s.path.terminal_growth,
                terminal_exit_multiple=None,
                wacc=s.path.discount_rate,
                revenue_growth_path=list(s.path.growth),
                growth_held_years=s.held_years,
            )
            setattr(
                result,
                s.name,
                compute_dcf_scenario(
                    assumptions,
                    result.base_revenue,
                    result.seed_fcf,
                    result.net_debt,
                    shares,
                    figures.price,
                ),
            )

    ev = figures.enterprise_value
    if ev is not None and ev > 0 and figures.revenue_base and figures.revenue_growth is not None:
        result.implied_margin = engine.implied_margin(
            ev,
            figures.revenue_base,
            figures.start_margin,
            figures.revenue_growth,
            discount_rate=rate,
        ).value
    return result


def _source(figures) -> str:
    parts = []
    if figures.price_source:
        parts.append(f"price: {figures.price_source}")
    if figures.revenue_base_label:
        parts.append(f"revenue: {figures.revenue_base_label}")
    if figures.revenue_growth_label:
        parts.append(f"growth: {figures.revenue_growth_label}")
    if figures.source_accession:
        parts.append(
            f"balance sheet and shares: {figures.source_accession} ({figures.balance_asof})"
        )
    if figures.margins:
        parts.append("margins: " + "; ".join(f"{m.value:+.1%} {m.label}" for m in figures.margins))
    parts.extend(figures.notes)
    return " · ".join(parts)


# ── Scenario engine (public — used by the dashboard's what-if sliders) ──────


class DcfInputs(NamedTuple):
    """The five values `compute_dcf_scenario` needs.

    Returned by `dcf_inputs_from_report` so callers don't have to know which
    fields of a cached ResearchReport carry these.
    """

    base_revenue: float
    seed_fcf: float
    net_debt: float
    shares: float
    current_price: float


def dcf_inputs_from_report(report: ResearchReport) -> DcfInputs | None:
    """The inputs a cached report needs to recompute a DCF scenario, or None.

    None when the report has no usable DCF: no price, no shares or no base
    revenue. A report cached before 2026-09-27 may carry ``shares=1.0`` and
    ``current_price=0`` from the old yfinance defaults; those are refused
    here rather than replayed.
    """
    dcf = report.dcf
    if dcf is None or dcf.base is None:
        return None
    if not dcf.base_revenue or dcf.base_revenue <= 0 or dcf.seed_fcf is None:
        return None
    if dcf.current_price <= 0 or dcf.shares_outstanding <= 1.0:
        return None
    return DcfInputs(
        base_revenue=float(dcf.base_revenue),
        seed_fcf=float(dcf.seed_fcf),
        net_debt=float(dcf.net_debt),
        shares=float(dcf.shares_outstanding),
        current_price=float(dcf.current_price),
    )


def compute_dcf_scenario(
    assump: DcfAssumptions,
    base_revenue: float,
    seed_fcf: float,
    net_debt: float,
    shares: float,
    current_price: float,
) -> DcfScenario:
    """Project 10 years of FCF under ``assump``, discount, and return a DcfScenario.

    Pure — no I/O. The dashboard's sliders call this directly to update the
    implied price live as the user moves a knob. The projection is the
    engine's (``valuation.dcf.project``); ``terminal_exit_multiple`` is not
    used, because the only EBITDA this module ever had was an assumed 25% of
    revenue.
    """
    start_margin = seed_fcf / base_revenue if base_revenue else assump.target_fcf_margin
    path = engine.Path(
        growth_early=assump.revenue_growth_yr1_3,
        growth_late=assump.revenue_growth_yr4_10,
        target_margin=assump.target_fcf_margin,
        discount_rate=assump.wacc,
        terminal_growth=assump.terminal_growth_rate,
        growth=_path_if_unmoved(assump),
    )
    projection = engine.project(base_revenue, start_margin, path)
    if projection is None:
        return DcfScenario(
            assumptions=assump,
            enterprise_value=0.0,
            equity_value=0.0,
            implied_price=0.0,
            upside_pct=0.0,
        )
    enterprise_value = projection.enterprise_value
    equity_value = max(enterprise_value - net_debt, 0.0)
    implied_price = engine.per_share(enterprise_value, -net_debt, shares) or 0.0
    upside = (implied_price / current_price - 1) if current_price and current_price > 0 else 0.0
    return DcfScenario(
        assumptions=assump,
        projected_fcf=list(projection.fcf),
        terminal_value_gordon=projection.pv_terminal,
        terminal_value_exit=None,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        implied_price=implied_price,
        upside_pct=upside,
    )


def _path_if_unmoved(assump: DcfAssumptions) -> tuple[float, ...]:
    """The stored yearly path, while its averages are still the two steps.

    The path is what the engine built; the steps are what a slider moves. Once
    a step no longer matches the path's average, the user has moved it, and
    the steps are the assumption.
    """
    path = tuple(assump.revenue_growth_path or ())
    if len(path) != engine.YEARS:
        return ()
    early, late = engine.step_averages(path)
    if (
        abs(early - assump.revenue_growth_yr1_3) > 1e-9
        or abs(late - assump.revenue_growth_yr4_10) > 1e-9
    ):
        return ()
    return path
