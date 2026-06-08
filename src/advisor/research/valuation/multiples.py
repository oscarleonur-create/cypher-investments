"""Multiples-based valuation — peer comp table + own 5-yr history.

Data source: yfinance (free, no API key needed).
All multiples use trailing (LTM) figures from yfinance's `.info` dict.
Historical multiples for the subject are approximated from StatementBundle +
year-end closing prices from yfinance price history.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from advisor.research.models import (
    HistoricalMultiple,
    MultiplesResult,
    PeerSnapshot,
    StatementBundle,
)

logger = logging.getLogger(__name__)


def build_multiples(
    symbol: str,
    peers: list[str],
    statements: StatementBundle | None = None,
) -> MultiplesResult:
    """Return peer comp table + own historical multiples."""
    subject = _snapshot(symbol)
    peer_snaps = [s for s in (_snapshot(p) for p in peers) if s is not None]
    history = _own_history(symbol, statements) if statements else []

    return MultiplesResult(
        subject=subject,
        peers=peer_snaps,
        own_history=history,
    )


# ── Snapshot fetcher ─────────────────────────────────────────────────────────


def _snapshot(symbol: str) -> PeerSnapshot:
    import yfinance as yf

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance info failed for %s: %s", symbol, exc)
        info = {}

    market_cap = _f(info, "marketCap")
    ev = _f(info, "enterpriseValue")
    price = _f(info, "currentPrice") or _f(info, "regularMarketPrice")
    revenue = _f(info, "totalRevenue")
    ebitda = _f(info, "ebitda")
    fcf = _f(info, "freeCashflow")
    net_income = _f(info, "netIncomeToCommon")
    eps = _f(info, "trailingEps")

    ev_sales = _safe_div(ev, revenue)
    ev_ebitda = _safe_div(ev, ebitda)
    pe_trail = _f(info, "trailingPE")
    pe_fwd = _f(info, "forwardPE")
    p_fcf = _safe_div(market_cap, fcf)
    peg = _f(info, "pegRatio")

    return PeerSnapshot(
        symbol=symbol.upper(),
        name=info.get("shortName", ""),
        sector=info.get("sector", ""),
        industry=info.get("industry", ""),
        market_cap=market_cap,
        enterprise_value=ev,
        price=price,
        revenue_ttm=revenue,
        ebitda_ttm=ebitda,
        fcf_ttm=fcf,
        net_income_ttm=net_income,
        eps_ttm=eps,
        revenue_growth_yoy=_f(info, "revenueGrowth"),
        gross_margin=_f(info, "grossMargins"),
        operating_margin=_f(info, "operatingMargins"),
        ev_to_sales=ev_sales,
        ev_to_ebitda=ev_ebitda,
        pe_trailing=pe_trail,
        pe_forward=pe_fwd,
        p_to_fcf=p_fcf,
        peg=peg,
        beta=_f(info, "beta"),
    )


# ── Historical multiples for subject ────────────────────────────────────────


def _own_history(symbol: str, bundle: StatementBundle) -> list[HistoricalMultiple]:
    """Approximate 5-yr multiples using year-end prices + statement data."""
    import yfinance as yf

    if not bundle.income:
        return []

    # Fetch 6 years of price history (extra buffer)
    end = date.today()
    start = date(end.year - 6, 1, 1)
    try:
        df = yf.download(symbol, start=start.isoformat(), end=end.isoformat(), progress=False)
        if df.empty:
            return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Price history failed for %s: %s", symbol, exc)
        return []

    # Flatten multi-level columns if present
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    close_col = "Close" if "Close" in df.columns else df.columns[0]

    history: list[HistoricalMultiple] = []
    for i, inc in enumerate(bundle.income):
        if i >= len(bundle.balance) or i >= len(bundle.cashflow):
            break
        bs = bundle.balance[i]
        cf = bundle.cashflow[i]

        # Find the closing price nearest to fiscal year-end
        price = _price_near(df, close_col, inc.period_end)
        if price is None:
            continue

        shares = bs.shares_outstanding or inc.shares_diluted
        if not shares:
            continue

        market_cap = price * shares
        net_debt = _net_debt(bs)
        ev = market_cap + (net_debt or 0)

        fcf = cf.free_cash_flow or ((cf.operating_cash_flow or 0) - abs(cf.capex or 0))
        net_income = inc.net_income

        history.append(
            HistoricalMultiple(
                period_end=inc.period_end,
                fiscal_year=inc.fiscal_year,
                ev_to_sales=_safe_div(ev, inc.revenue),
                ev_to_ebitda=_safe_div(ev, inc.ebitda),
                pe=_safe_div(price, _safe_div(net_income, shares)) if net_income else None,
                p_to_fcf=_safe_div(market_cap, fcf) if fcf and fcf > 0 else None,
            )
        )

    return history


# ── Helpers ─────────────────────────────────────────────────────────────────


def _f(info: dict[str, Any], key: str) -> float | None:
    v = info.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN guard
    except (TypeError, ValueError):
        return None


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _net_debt(bs: Any) -> float | None:
    debt = (bs.short_term_debt or 0) + (bs.long_term_debt or 0)
    cash = bs.cash_and_equivalents or 0
    return debt - cash


def _price_near(df: Any, col: str, target: date, window_days: int = 10) -> float | None:
    """Return closing price closest to `target`, within `window_days`."""
    import pandas as pd

    target_ts = pd.Timestamp(target)
    window_start = pd.Timestamp(target - timedelta(days=window_days))

    subset = df.loc[window_start:target_ts]
    if subset.empty:
        # Try a small lookahead (e.g. fiscal year ends on a holiday)
        subset = df.loc[target_ts : pd.Timestamp(target + timedelta(days=window_days))]
    if subset.empty:
        return None

    price = subset[col].iloc[-1]
    try:
        return float(price)
    except (TypeError, ValueError):
        return None
