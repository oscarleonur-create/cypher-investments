"""What analysts expect revenue to be, as a dated third-party figure.

The implied-expectations number says what the price *requires*: SPCX at
$152.64 requires 25.9% a year for a decade. Alone, that reads as a warning —
it trips the holder's 25% invalidation line. Beside the business's actual
growth (+91.9% year on year) and the consensus for next year (+142%), it reads
the other way: if the consensus is right, the rest of the decade needs about
14% a year. The reading reached "at risk" because it had the first number and
not the other two.

These are estimates, not facts about the company, and they are shown as such:
with the number of analysts and the range, never as a single point, and never
turned into a value for the business. They come from yfinance (free tier) and
are cached for a day.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from pydantic import BaseModel, Field

from advisor.daemon.market_calendar import now_et

logger = logging.getLogger(__name__)

CACHE_DAYS = 1


class RevenueEstimate(BaseModel):
    label: str  # "FY2026"
    fiscal_year_end: date
    avg: float
    low: float | None = None
    high: float | None = None
    analysts: int | None = None
    growth: float | None = None  # vs the prior fiscal year, as the provider states it


class Consensus(BaseModel):
    symbol: str
    asof: datetime = Field(default_factory=now_et)
    years: list[RevenueEstimate] = Field(default_factory=list)
    source: str = "yfinance"


def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN is absent


def fetch_consensus(symbol: str) -> Consensus | None:
    """Current- and next-fiscal-year revenue consensus, or None when unavailable."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol.upper())
        frame = ticker.revenue_estimate
        fye = (ticker.info or {}).get("nextFiscalYearEnd")
    except Exception as exc:  # noqa: BLE001
        logger.info("consensus: unavailable for %s: %s", symbol, exc)
        return None
    if frame is None or frame.empty or not fye:
        return None

    current_end = datetime.fromtimestamp(float(fye), tz=timezone.utc).date()
    years = []
    for period, offset in (("0y", 0), ("+1y", 1)):
        if period not in frame.index:
            continue
        row = frame.loc[period]
        avg = _num(row.get("avg"))
        if avg is None or avg <= 0:
            continue
        end = current_end.replace(year=current_end.year + offset)
        analysts = _num(row.get("numberOfAnalysts"))
        years.append(
            RevenueEstimate(
                label=f"FY{end.year}",
                fiscal_year_end=end,
                avg=avg,
                low=_num(row.get("low")),
                high=_num(row.get("high")),
                analysts=int(analysts) if analysts is not None else None,
                growth=_num(row.get("growth")),
            )
        )
    return Consensus(symbol=symbol.upper(), years=years) if years else None


def load_consensus(store, symbol: str, *, fetch=fetch_consensus) -> Consensus | None:
    """The cached consensus if under a day old, else a fresh one (cached)."""
    raw = store.load_consensus(symbol)
    if raw is not None:
        cached = Consensus.model_validate_json(raw)
        if now_et() - cached.asof < timedelta(days=CACHE_DAYS):
            return cached
    fresh = fetch(symbol)
    if fresh is not None:
        store.save_consensus(symbol, fresh.model_dump_json())
        return fresh
    # A stale estimate is still an estimate with a date on it; better than none.
    return Consensus.model_validate_json(raw) if raw is not None else None


def remaining_cagr(
    required_revenue: float,
    horizon_end: date,
    estimate: RevenueEstimate,
) -> tuple[float, float] | None:
    """(growth a year, years) still needed after ``estimate`` to reach
    ``required_revenue`` by ``horizon_end``. None when it cannot be computed.

    Arithmetic on two stated numbers, not a forecast: "if the consensus for
    FY2027 is met, this is what the remaining years must deliver".
    """
    years = (horizon_end - estimate.fiscal_year_end).days / 365.25
    if years <= 0 or estimate.avg <= 0 or required_revenue <= 0:
        return None
    return (required_revenue / estimate.avg) ** (1 / years) - 1, years
