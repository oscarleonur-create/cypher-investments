"""DCF valuation engine.

Produces base / bull / bear scenarios from explicit assumption structs.
WACC is derived from CAPM (beta × ERP + Rf) + after-tax cost of debt (weighted).

Inputs come from yfinance (price, beta, shares, net debt) and the
StatementBundle (latest FCF margin, capex intensity).
Risk-free rate is approximated from the 10-yr Treasury via yfinance (^TNX).

Public API used by the research workstation's what-if sliders:
- `compute_dcf_scenario(assumptions, base_revenue, seed_fcf, net_debt, shares, current_price)`
- `dcf_inputs_from_report(report)` returns the five values the function above needs.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

from advisor.research.models import (
    DcfAssumptions,
    DcfResult,
    DcfScenario,
    ResearchReport,
    StatementBundle,
)

logger = logging.getLogger(__name__)

_EQUITY_RISK_PREMIUM = 0.055  # Damodaran long-run ERP
_DEFAULT_RISK_FREE = 0.043  # fallback if TNX fetch fails
_DEFAULT_TAX_RATE = 0.21
_PROJECTION_YEARS = 10


def build_dcf(
    symbol: str,
    statements: StatementBundle | None = None,
    explicit_wacc: float | None = None,
) -> DcfResult:
    """Compute base / bull / bear DCF for `symbol`."""
    info = _yf_info(symbol)

    current_price = _f(info, "currentPrice") or _f(info, "regularMarketPrice") or 0.0
    shares = _f(info, "sharesOutstanding") or _f(info, "floatShares") or 1.0
    market_cap = current_price * shares
    total_debt = _f(info, "totalDebt") or 0.0
    cash = _f(info, "totalCash") or 0.0
    net_debt = total_debt - cash
    beta = _f(info, "beta") or 1.0

    rf = _risk_free_rate()
    erp = _EQUITY_RISK_PREMIUM
    cost_of_equity = rf + beta * erp

    debt_cost_pretax = _f(info, "averageInterestRate") or 0.04
    tax_rate = _f(info, "effectiveTaxRate") or _DEFAULT_TAX_RATE
    cost_of_debt = debt_cost_pretax * (1 - tax_rate)

    ev = market_cap + net_debt
    if ev > 0:
        w_equity = market_cap / ev
        w_debt = net_debt / ev
    else:
        w_equity, w_debt = 1.0, 0.0

    wacc = explicit_wacc or (w_equity * cost_of_equity + max(w_debt, 0) * cost_of_debt)
    wacc = max(wacc, 0.06)  # floor at 6%

    # Seed FCF from statements or yfinance
    seed_fcf = _seed_fcf(statements, info)
    seed_fcf_margin = _seed_fcf_margin(statements, info)
    base_revenue = _f(info, "totalRevenue") or 0.0
    seed_growth = _f(info, "revenueGrowth") or 0.05
    exit_multiple = _f(info, "enterpriseToEbitda")

    base_assump = DcfAssumptions(
        scenario="base",
        revenue_growth_yr1_3=seed_growth,
        revenue_growth_yr4_10=max(seed_growth * 0.5, 0.03),
        target_fcf_margin=seed_fcf_margin,
        capex_intensity=_seed_capex_intensity(statements, info),
        terminal_growth_rate=0.025,
        terminal_exit_multiple=exit_multiple,
        wacc=wacc,
    )
    bull_assump = DcfAssumptions(
        scenario="bull",
        revenue_growth_yr1_3=seed_growth * 1.3,
        revenue_growth_yr4_10=max(seed_growth * 0.7, 0.04),
        target_fcf_margin=seed_fcf_margin * 1.15,
        capex_intensity=base_assump.capex_intensity,
        terminal_growth_rate=0.030,
        terminal_exit_multiple=exit_multiple,
        wacc=max(wacc - 0.01, 0.06),
    )
    bear_assump = DcfAssumptions(
        scenario="bear",
        revenue_growth_yr1_3=max(seed_growth * 0.5, 0.0),
        revenue_growth_yr4_10=max(seed_growth * 0.2, 0.01),
        target_fcf_margin=seed_fcf_margin * 0.80,
        capex_intensity=base_assump.capex_intensity,
        terminal_growth_rate=0.020,
        terminal_exit_multiple=exit_multiple,
        wacc=wacc + 0.01,
    )

    return DcfResult(
        symbol=symbol.upper(),
        current_price=current_price,
        shares_outstanding=shares,
        net_debt=net_debt,
        wacc=wacc,
        risk_free_rate=rf,
        equity_risk_premium=erp,
        beta=beta,
        base=compute_dcf_scenario(
            base_assump, base_revenue, seed_fcf, net_debt, shares, current_price
        ),
        bull=compute_dcf_scenario(
            bull_assump, base_revenue, seed_fcf, net_debt, shares, current_price
        ),
        bear=compute_dcf_scenario(
            bear_assump, base_revenue, seed_fcf, net_debt, shares, current_price
        ),
        base_revenue=base_revenue,
        seed_fcf=seed_fcf,
    )


# ── Scenario engine (public — used by the dashboard's what-if sliders) ──────


class DcfInputs(NamedTuple):
    """The five values `compute_dcf_scenario` needs.

    Returned by `dcf_inputs_from_report` so callers don't have to know which
    fields of a cached ResearchReport carry these (some live on `dcf`, some on
    `statements`).
    """

    base_revenue: float
    seed_fcf: float
    net_debt: float
    shares: float
    current_price: float


def dcf_inputs_from_report(report: ResearchReport) -> DcfInputs | None:
    """Pull the inputs a cached report needs to recompute a DCF scenario.

    Prefers the inputs persisted on `DcfResult` (`base_revenue`, `seed_fcf`)
    so the what-if sliders replay the base scenario exactly. Falls back to
    deriving them from `statements` for old reports built before those
    fields existed.
    """
    if report.dcf is None:
        return None

    base_revenue = report.dcf.base_revenue
    seed_fcf = report.dcf.seed_fcf

    if base_revenue is None or seed_fcf is None:
        # Older cached report — derive from statements
        if report.statements is None:
            return None
        inc = report.statements.latest_income()
        cf = report.statements.latest_cashflow()
        if inc is None or inc.revenue is None:
            return None
        base_revenue = float(inc.revenue)
        seed_fcf = 0.0
        if cf is not None:
            seed_fcf = cf.free_cash_flow or ((cf.operating_cash_flow or 0.0) - abs(cf.capex or 0.0))

    return DcfInputs(
        base_revenue=float(base_revenue),
        seed_fcf=float(seed_fcf),
        net_debt=float(report.dcf.net_debt),
        shares=float(report.dcf.shares_outstanding),
        current_price=float(report.dcf.current_price),
    )


def compute_dcf_scenario(
    assump: DcfAssumptions,
    base_revenue: float,
    seed_fcf: float,
    net_debt: float,
    shares: float,
    current_price: float,
) -> DcfScenario:
    """Project 10 years of FCF under `assump`, discount, and return a DcfScenario.

    Pure function — no I/O. The dashboard's sliders call this directly to
    update the implied price live as the user moves a knob.
    """
    wacc = assump.wacc
    g_term = assump.terminal_growth_rate

    projected_fcf: list[float] = []
    revenue = base_revenue

    for year in range(1, _PROJECTION_YEARS + 1):
        g = assump.revenue_growth_yr1_3 if year <= 3 else assump.revenue_growth_yr4_10
        # Linearly fade margin from seed toward target over 10 years
        margin_progress = year / _PROJECTION_YEARS
        fcf_margin = (
            (
                (1 - margin_progress) * (seed_fcf / max(base_revenue, 1))
                + margin_progress * assump.target_fcf_margin
            )
            if base_revenue
            else assump.target_fcf_margin
        )
        revenue *= 1 + g
        projected_fcf.append(revenue * fcf_margin)

    # PV of FCF
    pv_fcf = sum(cf / (1 + wacc) ** t for t, cf in enumerate(projected_fcf, 1))

    # Terminal value — Gordon growth
    terminal_fcf = projected_fcf[-1] * (1 + g_term)
    tv_gordon = terminal_fcf / max(wacc - g_term, 0.001)
    pv_gordon = tv_gordon / (1 + wacc) ** _PROJECTION_YEARS

    # Terminal value — exit multiple (if available)
    pv_exit: float | None = None
    if assump.terminal_exit_multiple and projected_fcf:
        terminal_ebitda_proxy = projected_fcf[-1] / max(assump.target_fcf_margin, 0.01) * 0.25
        tv_exit = terminal_ebitda_proxy * assump.terminal_exit_multiple
        pv_exit = tv_exit / (1 + wacc) ** _PROJECTION_YEARS

    pv_terminal = pv_exit if pv_exit is not None else pv_gordon
    enterprise_value = pv_fcf + pv_terminal
    equity_value = max(enterprise_value - net_debt, 0)
    implied_price = equity_value / max(shares, 1)
    upside = (implied_price / current_price - 1) if current_price else 0.0

    return DcfScenario(
        assumptions=assump,
        projected_fcf=projected_fcf,
        terminal_value_gordon=pv_gordon,
        terminal_value_exit=pv_exit,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        implied_price=implied_price,
        upside_pct=upside,
    )


# ── Data helpers ─────────────────────────────────────────────────────────────


def _yf_info(symbol: str) -> dict[str, Any]:
    import yfinance as yf

    try:
        return yf.Ticker(symbol).info or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance info failed for %s: %s", symbol, exc)
        return {}


def _risk_free_rate() -> float:
    """Fetch 10-yr Treasury yield (^TNX) as a proxy for Rf."""
    import yfinance as yf

    try:
        tnx = yf.Ticker("^TNX").fast_info
        rate = getattr(tnx, "last_price", None)
        if rate and 0 < rate < 20:
            return rate / 100  # ^TNX is in percentage points
    except Exception:  # noqa: BLE001
        pass
    return _DEFAULT_RISK_FREE


def _seed_fcf(statements: StatementBundle | None, info: dict) -> float:
    if statements:
        cf = statements.latest_cashflow()
        if cf:
            fcf = cf.free_cash_flow
            if fcf is not None:
                return fcf
            if cf.operating_cash_flow is not None:
                return cf.operating_cash_flow - abs(cf.capex or 0)
    return _f(info, "freeCashflow") or 0.0


def _seed_fcf_margin(statements: StatementBundle | None, info: dict) -> float:
    if statements:
        inc = statements.latest_income()
        cf = statements.latest_cashflow()
        if inc and cf and inc.revenue:
            fcf = cf.free_cash_flow or ((cf.operating_cash_flow or 0) - abs(cf.capex or 0))
            if fcf:
                return fcf / inc.revenue
    revenue = _f(info, "totalRevenue") or 1.0
    fcf = _f(info, "freeCashflow") or 0.0
    return fcf / revenue if revenue else 0.10


def _seed_capex_intensity(statements: StatementBundle | None, info: dict) -> float:
    if statements:
        inc = statements.latest_income()
        cf = statements.latest_cashflow()
        if inc and cf and inc.revenue and cf.capex:
            return abs(cf.capex) / inc.revenue
    return 0.05  # default 5% of revenue


def _f(info: dict[str, Any], key: str) -> float | None:
    v = info.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None
