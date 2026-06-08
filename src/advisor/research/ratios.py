"""Pure ratio computations over a `StatementBundle`.

All functions are total: missing inputs return None instead of raising.
Ordering convention: bundle lists are newest-first, so periods[i] aligns
with the same i across income / balance / cashflow.
"""

from __future__ import annotations

from advisor.research.models import (
    BalanceSheetPeriod,
    CashFlowPeriod,
    IncomeStatementPeriod,
    RatioBundle,
    RatioPeriod,
    StatementBundle,
)

# Default tax rate used for NOPAT when an effective rate can't be derived
_DEFAULT_TAX_RATE = 0.21


def compute_ratios(bundle: StatementBundle) -> RatioBundle:
    """Compute a 5-yr RatioPeriod sequence aligned with the bundle's periods."""
    periods: list[RatioPeriod] = []

    # Align by index (yfinance & edgartools both return periods newest-first)
    n = min(len(bundle.income), len(bundle.balance), len(bundle.cashflow))
    for i in range(n):
        inc = bundle.income[i]
        bs = bundle.balance[i]
        cf = bundle.cashflow[i]
        # For averaged ratios we need the prior period if present
        prior_bs = bundle.balance[i + 1] if i + 1 < len(bundle.balance) else None
        periods.append(_compute_period_ratios(inc, bs, cf, prior_bs))

    return RatioBundle(
        symbol=bundle.symbol,
        periods=periods,
        share_count_cagr_3y=_share_count_cagr(bundle.income, years=3),
    )


# ── Per-period ratios ────────────────────────────────────────────────────────


def _compute_period_ratios(
    inc: IncomeStatementPeriod,
    bs: BalanceSheetPeriod,
    cf: CashFlowPeriod,
    prior_bs: BalanceSheetPeriod | None,
) -> RatioPeriod:
    revenue = inc.revenue
    net_income = inc.net_income

    # ── Profitability ────────────────────────────────────────────────────
    gross_margin = _safe_div(inc.gross_profit, revenue)
    operating_margin = _safe_div(inc.operating_income, revenue)
    net_margin = _safe_div(net_income, revenue)

    avg_assets = _avg(bs.total_assets, prior_bs.total_assets if prior_bs else None)
    avg_equity = _avg(bs.total_equity, prior_bs.total_equity if prior_bs else None)
    roa = _safe_div(net_income, avg_assets)
    roe = _safe_div(net_income, avg_equity)

    # NOPAT / invested capital
    nopat = _nopat(inc)
    invested = _invested_capital(bs)
    roic = _safe_div(nopat, invested)

    # ── Liquidity & leverage ─────────────────────────────────────────────
    current_ratio = _safe_div(bs.current_assets, bs.current_liabilities)
    quick_assets = _sub(
        bs.current_assets,
        bs.inventory,
    )
    quick_ratio = _safe_div(quick_assets, bs.current_liabilities)

    total_debt = _sum_optional(bs.short_term_debt, bs.long_term_debt)
    debt_to_equity = _safe_div(total_debt, bs.total_equity)
    debt_to_ebitda = _safe_div(total_debt, inc.ebitda)
    interest_coverage = _safe_div(inc.operating_income, _abs_or_none(inc.interest_expense))

    # ── Efficiency ───────────────────────────────────────────────────────
    asset_turnover = _safe_div(revenue, avg_assets)
    avg_inventory = _avg(bs.inventory, prior_bs.inventory if prior_bs else None)
    inventory_turns = _safe_div(inc.cost_of_revenue, avg_inventory)
    dso = _dso(revenue, bs.accounts_receivable)

    # ── Cash quality ─────────────────────────────────────────────────────
    fcf = _free_cash_flow(cf)
    fcf_margin = _safe_div(fcf, revenue)
    capex_intensity = _safe_div(_abs_or_none(cf.capex), revenue)
    fcf_to_net_income = _safe_div(fcf, net_income)

    return RatioPeriod(
        period_end=inc.period_end,
        fiscal_year=inc.fiscal_year,
        is_ttm=inc.is_ttm,
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        net_margin=net_margin,
        roa=roa,
        roe=roe,
        roic=roic,
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        debt_to_equity=debt_to_equity,
        debt_to_ebitda=debt_to_ebitda,
        interest_coverage=interest_coverage,
        asset_turnover=asset_turnover,
        inventory_turns=inventory_turns,
        dso=dso,
        fcf_margin=fcf_margin,
        capex_intensity=capex_intensity,
        fcf_to_net_income=fcf_to_net_income,
    )


# ── Building blocks ──────────────────────────────────────────────────────────


def _nopat(inc: IncomeStatementPeriod) -> float | None:
    op_income = inc.operating_income
    if op_income is None:
        return None
    tax_rate = _effective_tax_rate(inc)
    return op_income * (1 - tax_rate)


def _effective_tax_rate(inc: IncomeStatementPeriod) -> float:
    if inc.pretax_income and inc.pretax_income > 0 and inc.income_tax is not None:
        rate = inc.income_tax / inc.pretax_income
        # clamp to a sane range (negative pretax or tax credits can produce noise)
        if 0 <= rate <= 0.5:
            return rate
    return _DEFAULT_TAX_RATE


def _invested_capital(bs: BalanceSheetPeriod) -> float | None:
    equity = bs.total_equity
    debt = _sum_optional(bs.short_term_debt, bs.long_term_debt)
    if equity is None and debt is None:
        return None
    return (equity or 0) + (debt or 0)


def _free_cash_flow(cf: CashFlowPeriod) -> float | None:
    if cf.free_cash_flow is not None:
        return cf.free_cash_flow
    if cf.operating_cash_flow is None:
        return None
    capex = cf.capex
    if capex is None:
        return cf.operating_cash_flow
    # capex may be reported as negative (outflow) or positive — normalise to subtract magnitude
    return cf.operating_cash_flow - abs(capex)


def _dso(revenue: float | None, ar: float | None) -> float | None:
    if revenue is None or revenue == 0 or ar is None:
        return None
    return ar / revenue * 365.0


def _share_count_cagr(income: list[IncomeStatementPeriod], years: int) -> float | None:
    """CAGR of diluted share count over `years` (positive = dilution)."""
    if len(income) <= years:
        return None
    latest = income[0].shares_diluted
    base = income[years].shares_diluted
    if not latest or not base or base <= 0:
        return None
    return (latest / base) ** (1 / years) - 1


# ── Numeric utilities ────────────────────────────────────────────────────────


def _safe_div(num: float | None, denom: float | None) -> float | None:
    if num is None or denom is None or denom == 0:
        return None
    return num / denom


def _avg(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2


def _sub(a: float | None, b: float | None) -> float | None:
    if a is None:
        return None
    return a - (b or 0)


def _sum_optional(*values: float | None) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present)


def _abs_or_none(v: float | None) -> float | None:
    return abs(v) if v is not None else None
