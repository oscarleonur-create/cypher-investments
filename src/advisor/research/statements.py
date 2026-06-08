"""5-year financial statement extraction.

Strategy:
1. Try EDGAR (via `EdgarClient.get_financials`). XBRL is the authoritative source.
2. Fall back to yfinance when EDGAR is sparse or unavailable. yfinance label
   coverage is broad enough for the ratios in Phase 1.
3. Always tag the source on the returned bundle.

The yfinance path is tested directly; the EDGAR path is exercised in
integration tests (gated on the package being installed + network).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

from advisor.research.models import (
    BalanceSheetPeriod,
    CashFlowPeriod,
    IncomeStatementPeriod,
    StatementBundle,
)

logger = logging.getLogger(__name__)


# yfinance row labels → our model field name. Multiple aliases per field; first
# present wins. Keeping these as constants makes it easy to extend.

_INCOME_MAP: dict[str, list[str]] = {
    "revenue": ["Total Revenue", "Revenue", "Operating Revenue"],
    "cost_of_revenue": ["Cost Of Revenue", "Cost of Revenue", "Reconciled Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_expenses": ["Operating Expense", "Operating Expenses"],
    "operating_income": [
        "Operating Income",
        "Total Operating Income As Reported",
        "Operating Income (Loss)",  # EDGAR XBRL label
    ],
    "interest_expense": [
        "Interest Expense",
        "Interest Expense Non Operating",
        "InterestExpenseNonoperating",  # EDGAR XBRL label
    ],
    "pretax_income": [
        "Pretax Income",
        "Pre-Tax Income",
        # EDGAR XBRL full label
        "Income (Loss) from Continuing Operations before Income Taxes, Noncontrolling Interest",
    ],
    "income_tax": ["Tax Provision", "Income Tax Expense", "Income Tax Expense (Benefit)"],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income (Loss) Attributable to Parent",  # EDGAR XBRL label
    ],
    "eps_basic": ["Basic EPS"],
    "eps_diluted": ["Diluted EPS"],
    "shares_basic": ["Basic Average Shares"],
    "shares_diluted": ["Diluted Average Shares"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
}

_BALANCE_MAP: dict[str, list[str]] = {
    "cash_and_equivalents": [
        "Cash And Cash Equivalents",
        "Cash",
        "Cash and Cash Equivalents, at Carrying Value",  # EDGAR XBRL label
    ],
    "short_term_investments": ["Other Short Term Investments", "Short Term Investments"],
    "accounts_receivable": [
        "Accounts Receivable",
        "Receivables",
        "Accounts Receivable, after Allowance for Credit Loss, Current",  # EDGAR XBRL label
    ],
    "inventory": ["Inventory", "Inventory, Net"],
    "current_assets": ["Current Assets", "Total Current Assets", "Assets, Current"],
    "goodwill": ["Goodwill"],
    "intangibles": ["Other Intangible Assets", "Intangible Assets"],
    "total_assets": ["Total Assets", "Assets"],  # EDGAR uses "Assets" for the total
    "accounts_payable": ["Accounts Payable", "Payables", "Accounts Payable, Current"],
    "short_term_debt": [
        "Current Debt",
        "Short Long Term Debt",
        "Long-term Debt, Current Maturities",  # EDGAR XBRL label
    ],
    "current_liabilities": [
        "Current Liabilities",
        "Total Current Liabilities",
        "Liabilities, Current",
    ],
    "long_term_debt": ["Long Term Debt", "Long-term Debt, Excluding Current Maturities"],
    "total_liabilities": [
        "Total Liabilities Net Minority Interest",
        "Total Liab",
        "Liabilities",  # EDGAR uses "Liabilities" for total
    ],
    "total_equity": [
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Common Stock Equity",
        "Stockholders' Equity Attributable to Parent",  # EDGAR XBRL label
    ],
    "shares_outstanding": ["Share Issued", "Ordinary Shares Number"],
}

_CASHFLOW_MAP: dict[str, list[str]] = {
    "operating_cash_flow": [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Net Cash Provided by (Used in) Operating Activities",  # EDGAR XBRL label
    ],
    "capex": [
        "Capital Expenditure",
        "Capital Expenditures",
        "Payments to Acquire Property, Plant, and Equipment",  # EDGAR XBRL label
    ],
    "free_cash_flow": ["Free Cash Flow"],
    "investing_cash_flow": [
        "Investing Cash Flow",
        "Total Cashflows From Investing Activities",
        "Net Cash Provided by (Used in) Investing Activities",  # EDGAR XBRL label
    ],
    "financing_cash_flow": [
        "Financing Cash Flow",
        "Total Cash From Financing Activities",
        "Net Cash Provided by (Used in) Financing Activities",  # EDGAR XBRL label
    ],
    "dividends_paid": ["Cash Dividends Paid", "Dividends Paid"],
    "share_repurchases": [
        "Repurchase Of Capital Stock",
        "Common Stock Repurchase",
        "Payment, Tax Withholding, Share-based Payment Arrangement",  # EDGAR XBRL label
    ],
    "net_change_in_cash": [
        "Changes In Cash",
        "Change In Cash",
        # EDGAR XBRL full label
        "Cash, Cash Equivalents, Restricted Cash and Restricted Cash Equivalents, "
        "Period Increase (Decrease), Including Exchange Rate Effect",
    ],
}


# ── Public entry point ───────────────────────────────────────────────────────


def extract_statements(
    symbol: str,
    edgar_client: Any | None = None,
    n_years: int = 5,
    fallback_to_yfinance: bool = True,
) -> StatementBundle:
    """Return a 5-yr StatementBundle for `symbol`.

    Tries EDGAR first if a client is supplied; falls back to yfinance when
    EDGAR returns empty.
    """
    bundle = StatementBundle(symbol=symbol.upper())

    if edgar_client is not None:
        try:
            bundle = _from_edgar(symbol, edgar_client, n_years)
            if _is_populated(bundle):
                return bundle
            logger.info("EDGAR returned sparse data for %s, falling back", symbol)
        except ModuleNotFoundError:
            logger.debug("edgartools not installed, falling back to yfinance for %s", symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("EDGAR fetch failed for %s: %s", symbol, exc)

    if fallback_to_yfinance:
        return _from_yfinance(symbol, n_years)

    return bundle


# ── EDGAR path ───────────────────────────────────────────────────────────────


def _from_edgar(symbol: str, edgar_client: Any, n_years: int) -> StatementBundle:
    """Map EdgarClient.get_financials raw frames into typed periods."""
    raw = edgar_client.get_financials(symbol, n_years=n_years)

    income = [_row_to_income_period(row) for row in raw.get("income", [])]
    balance = [_row_to_balance_period(row) for row in raw.get("balance", [])]
    cashflow = [_row_to_cashflow_period(row) for row in raw.get("cashflow", [])]

    return StatementBundle(
        symbol=symbol.upper(),
        income=[p for p in income if p is not None],
        balance=[p for p in balance if p is not None],
        cashflow=[p for p in cashflow if p is not None],
        source="edgar",
    )


def _row_to_income_period(row: dict[str, Any]) -> IncomeStatementPeriod | None:
    period = _extract_period(row)
    if period is None:
        return None
    return IncomeStatementPeriod(
        period_end=period,
        fiscal_year=period.year,
        revenue=_first_present(row, _INCOME_MAP["revenue"]),
        cost_of_revenue=_first_present(row, _INCOME_MAP["cost_of_revenue"]),
        gross_profit=_first_present(row, _INCOME_MAP["gross_profit"]),
        operating_expenses=_first_present(row, _INCOME_MAP["operating_expenses"]),
        operating_income=_first_present(row, _INCOME_MAP["operating_income"]),
        interest_expense=_first_present(row, _INCOME_MAP["interest_expense"]),
        pretax_income=_first_present(row, _INCOME_MAP["pretax_income"]),
        income_tax=_first_present(row, _INCOME_MAP["income_tax"]),
        net_income=_first_present(row, _INCOME_MAP["net_income"]),
        eps_basic=_first_present(row, _INCOME_MAP["eps_basic"]),
        eps_diluted=_first_present(row, _INCOME_MAP["eps_diluted"]),
        shares_basic=_first_present(row, _INCOME_MAP["shares_basic"]),
        shares_diluted=_first_present(row, _INCOME_MAP["shares_diluted"]),
        ebitda=_first_present(row, _INCOME_MAP["ebitda"]),
    )


def _row_to_balance_period(row: dict[str, Any]) -> BalanceSheetPeriod | None:
    period = _extract_period(row)
    if period is None:
        return None
    return BalanceSheetPeriod(
        period_end=period,
        fiscal_year=period.year,
        **{k: _first_present(row, v) for k, v in _BALANCE_MAP.items()},
    )


def _row_to_cashflow_period(row: dict[str, Any]) -> CashFlowPeriod | None:
    period = _extract_period(row)
    if period is None:
        return None
    return CashFlowPeriod(
        period_end=period,
        fiscal_year=period.year,
        **{k: _first_present(row, v) for k, v in _CASHFLOW_MAP.items()},
    )


# ── yfinance path ────────────────────────────────────────────────────────────


def _from_yfinance(symbol: str, n_years: int) -> StatementBundle:
    import yfinance as yf

    ticker = yf.Ticker(symbol.upper())

    income_df = _safe_df(ticker, "income_stmt")
    balance_df = _safe_df(ticker, "balance_sheet")
    cashflow_df = _safe_df(ticker, "cashflow")

    return StatementBundle(
        symbol=symbol.upper(),
        income=_income_from_yf(income_df, n_years),
        balance=_balance_from_yf(balance_df, n_years),
        cashflow=_cashflow_from_yf(cashflow_df, n_years),
        source="yfinance",
    )


def _safe_df(ticker: Any, attr: str) -> pd.DataFrame | None:
    try:
        df = getattr(ticker, attr, None)
        if df is None or df.empty:
            return None
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance %s fetch failed: %s", attr, exc)
        return None


def _income_from_yf(df: pd.DataFrame | None, n_years: int) -> list[IncomeStatementPeriod]:
    return list(_periods_from_yf(df, n_years, _INCOME_MAP, IncomeStatementPeriod))


def _balance_from_yf(df: pd.DataFrame | None, n_years: int) -> list[BalanceSheetPeriod]:
    return list(_periods_from_yf(df, n_years, _BALANCE_MAP, BalanceSheetPeriod))


def _cashflow_from_yf(df: pd.DataFrame | None, n_years: int) -> list[CashFlowPeriod]:
    return list(_periods_from_yf(df, n_years, _CASHFLOW_MAP, CashFlowPeriod))


def _periods_from_yf(
    df: pd.DataFrame | None,
    n_years: int,
    field_map: dict[str, list[str]],
    cls: type,
) -> Iterable[Any]:
    """Convert a yfinance statement DataFrame (rows=labels, cols=periods) into typed periods."""
    if df is None:
        return []

    # yfinance returns columns as Timestamps (period-end dates), newest first
    periods = list(df.columns)[:n_years]
    out: list[Any] = []
    for col in periods:
        period_end = _to_date(col)
        if period_end is None:
            continue
        kwargs: dict[str, Any] = {
            "period_end": period_end,
            "fiscal_year": period_end.year,
        }
        for field, aliases in field_map.items():
            kwargs[field] = _first_present_in_df(df, col, aliases)
        out.append(cls(**kwargs))
    return out


# ── Shared helpers ───────────────────────────────────────────────────────────


def _first_present(row: dict[str, Any], aliases: list[str]) -> float | None:
    for alias in aliases:
        if alias in row and row[alias] is not None:
            return _to_float(row[alias])
    return None


def _first_present_in_df(df: pd.DataFrame, col: Any, aliases: list[str]) -> float | None:
    for alias in aliases:
        if alias in df.index:
            value = df.at[alias, col]
            return _to_float(value)
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # pandas/numpy NaN check
    if f != f:
        return None
    return f


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "date"):
        try:
            return value.date()
        except Exception:  # noqa: BLE001
            pass
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _extract_period(row: dict[str, Any]) -> date | None:
    """Find the period-end date in a row dict by trying common keys."""
    for key in ("index", "period", "period_end", "fiscal_period_end", "date"):
        if key in row:
            d = _to_date(row[key])
            if d is not None:
                return d
    return None


def _is_populated(bundle: StatementBundle) -> bool:
    """True if at least one period has revenue + total_assets + operating_cash_flow."""
    has_rev = any(p.revenue for p in bundle.income)
    has_assets = any(p.total_assets for p in bundle.balance)
    has_ocf = any(p.operating_cash_flow for p in bundle.cashflow)
    return has_rev and has_assets and has_ocf
