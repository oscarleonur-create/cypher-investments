"""Daily price history + fundamentals overlay for the ticker chart.

Pulls ~5 years of daily closes from yfinance and aligns the fundamentals
already cached on the `ResearchReport` (income-statement trend, ratios) onto
that timeline:

- earnings markers at each reported period (revenue / diluted EPS + YoY),
- a trailing-P/E series computed per bar (close ÷ most-recent trailing EPS),
- revenue / EPS / net-margin points for the sub-panel.

The yfinance fetch is factored behind `fetch_daily_bars` so `build_price_history`
stays a pure, testable assembly function (tests inject fake bars).
"""

from __future__ import annotations

import logging
from datetime import date

from advisor.research.models import (
    EarningsMarker,
    FundamentalPoint,
    PePoint,
    PriceBar,
    PriceHistoryResult,
    ResearchReport,
)

logger = logging.getLogger(__name__)

_HISTORY_YEARS = 5


def _period_eps(p, shares_fallback: float | None = None) -> float | None:
    """Diluted (or basic) EPS for a period, deriving it from net income ÷ shares
    when the provider didn't map an EPS line (common with EDGAR XBRL). Uses the
    period's own share count when present, else `shares_fallback` (the current
    share count from the DCF) — an approximation that ignores historical dilution."""
    if p.eps_diluted is not None:
        return float(p.eps_diluted)
    if p.eps_basic is not None:
        return float(p.eps_basic)
    shares = p.shares_diluted or p.shares_basic or shares_fallback
    if p.net_income is not None and shares:
        return float(p.net_income) / float(shares)
    return None


def fetch_daily_bars(symbol: str, years: int = _HISTORY_YEARS) -> list[PriceBar]:
    """Fetch daily close/volume bars from yfinance.

    Mirrors the download + multi-level-column flatten already used in
    `valuation/multiples.py:_own_history`.
    """
    import yfinance as yf

    end = date.today()
    start = date(end.year - years, end.month, end.day)
    try:
        df = yf.download(
            symbol, start=start.isoformat(), end=end.isoformat(), progress=False, auto_adjust=True
        )
        if df is None or df.empty:
            return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Price history download failed for %s: %s", symbol, exc)
        return []

    # Flatten multi-level columns (yfinance returns them for some symbols).
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    close_col = "Close" if "Close" in df.columns else df.columns[0]
    has_volume = "Volume" in df.columns

    bars: list[PriceBar] = []
    for idx, row in df.iterrows():
        close = row[close_col]
        if close is None or close != close:  # NaN guard
            continue
        bars.append(
            PriceBar(
                date=idx.date() if hasattr(idx, "date") else idx,
                close=round(float(close), 4),
                volume=(
                    float(row["Volume"]) if has_volume and row["Volume"] == row["Volume"] else None
                ),
            )
        )
    return bars


def _trailing_eps_steps(
    report: ResearchReport, shares_fallback: float | None
) -> list[tuple[date, float]]:
    """(period_end, trailing-EPS) steps, ascending by date.

    Prefers a TTM period's EPS where present; otherwise the period's diluted
    (or basic) EPS. Used as a step function: each price bar takes the most
    recent step whose period_end is on or before the bar date.
    """
    income = report.statements.income if report.statements else []
    steps: list[tuple[date, float]] = []
    for p in income:
        eps = _period_eps(p, shares_fallback)
        if eps is None:
            continue
        steps.append((p.period_end, eps))
    steps.sort(key=lambda s: s[0])
    return steps


def _build_pe_series(bars: list[PriceBar], steps: list[tuple[date, float]]) -> list[PePoint]:
    """Close ÷ most-recent trailing EPS per bar; null where EPS ≤ 0 or unknown."""
    if not steps:
        return []
    out: list[PePoint] = []
    i = 0
    n = len(steps)
    for bar in bars:  # bars are ascending by date
        while i + 1 < n and steps[i + 1][0] <= bar.date:
            i += 1
        eff_date, eps = steps[i]
        pe = round(bar.close / eps, 2) if (eff_date <= bar.date and eps > 0) else None
        out.append(PePoint(date=bar.date, pe=pe))
    return out


def _build_earnings(report: ResearchReport, shares_fallback: float | None) -> list[EarningsMarker]:
    """One marker per income period; YoY EPS growth vs the same period a year prior."""
    income = list(report.statements.income) if report.statements else []
    if not income:
        return []
    # Ascending by date so YoY look-back is straightforward.
    income = sorted(income, key=lambda p: p.period_end)
    markers: list[EarningsMarker] = []
    for idx, p in enumerate(income):
        eps = _period_eps(p, shares_fallback)
        yoy = None
        if eps is not None and idx > 0:
            prev_eps = _period_eps(income[idx - 1], shares_fallback)
            if prev_eps not in (None, 0):
                yoy = round((eps - prev_eps) / abs(prev_eps), 4)
        markers.append(
            EarningsMarker(
                date=p.period_end,
                revenue=p.revenue,
                eps=eps,
                yoy_eps_growth=yoy,
            )
        )
    return markers


def _build_fundamentals(
    report: ResearchReport, shares_fallback: float | None
) -> list[FundamentalPoint]:
    """Revenue / EPS / net-margin per income period for the sub-panel bars."""
    income = list(report.statements.income) if report.statements else []
    if not income:
        return []
    # Map net margin by period_end from the ratio bundle, if available.
    margins: dict[date, float] = {}
    if report.ratios:
        for rp in report.ratios.periods:
            if rp.net_margin is not None:
                margins[rp.period_end] = float(rp.net_margin)

    points: list[FundamentalPoint] = []
    for p in sorted(income, key=lambda x: x.period_end):
        eps = _period_eps(p, shares_fallback)
        points.append(
            FundamentalPoint(
                date=p.period_end,
                fiscal_year=p.fiscal_year,
                revenue=p.revenue,
                eps=eps,
                net_margin=margins.get(p.period_end),
            )
        )
    return points


def build_price_history(
    symbol: str,
    report: ResearchReport,
    *,
    bars: list[PriceBar] | None = None,
) -> PriceHistoryResult:
    """Assemble the chart payload. `bars` is injectable for tests."""
    sym = symbol.upper()
    if bars is None:
        bars = fetch_daily_bars(sym)
    bars = sorted(bars, key=lambda b: b.date)

    # Current share count from the DCF — used to derive EPS for EDGAR periods
    # that carry net income but no per-share / share-count line.
    shares_fallback = (
        float(report.dcf.shares_outstanding)
        if report.dcf and report.dcf.shares_outstanding
        else None
    )

    steps = _trailing_eps_steps(report, shares_fallback)
    return PriceHistoryResult(
        symbol=sym,
        bars=bars,
        earnings=_build_earnings(report, shares_fallback),
        pe_series=_build_pe_series(bars, steps),
        fundamentals=_build_fundamentals(report, shares_fallback),
    )
