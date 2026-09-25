"""The entry zone: is the price cheaper than this name usually is?

**The zone is relative.** Today's price-to-sales against the name's own last
two years, and "in the zone" means at or below its median — the user's choice
on 2026-09-25. Revenue and share counts are those known on each past day
(``valuation.history``), so the comparison is like for like.

It is relative because the absolute alternative failed live. The first
version drew the zone at the price where the 10-year revenue growth the price
requires (under the generic 25x FCF, 25% margin scenario) fell to the growth
the company delivers. Every Swing name came out "in zone", with tops of
$1,407 for AMZN and $10,269 for AMD: AMZN "required" -6.0% growth because the
scenario assumed a 25% free-cash-flow margin against a real one of -0.3%
(AI capex), INTC 25% against 5.0%. A comparison of a name with itself cancels
that assumption, because it sits on both sides.

What the relative zone cannot say: whether the name is cheap *for a reason*.
A business that got worse deserves a lower multiple; the zone flags the
multiple, the reading and the events have to judge the reason.

**The absolute requirement stays as context**, recomputed with the company's
own trailing free-cash-flow margin instead of the generic one, and printed
with that assumption. When the margin is negative the requirement is
undefined and the sheet says so rather than showing a number.
"""

from __future__ import annotations

import logging
import statistics
from datetime import date, timedelta

from pydantic import BaseModel, Field

from advisor.valuation.history import Series, as_of
from advisor.valuation.models import ValuationSnapshot

logger = logging.getLogger(__name__)

WINDOW_DAYS = 730  # two years
MIN_OBSERVATIONS = 250  # about a year of sessions; less is not a median worth trusting


class RelativeZone(BaseModel):
    price: float
    ps_now: float
    median: float
    percentile: float  # share of the window's sessions with P/S at or below today's
    top: float  # the price at which today's P/S equals the window median
    window_start: date
    window_end: date
    observations: int
    basis: str = "price / sales on trailing-12-month revenue and diluted shares, as known each day"

    @property
    def in_zone(self) -> bool:
        return self.ps_now <= self.median

    @property
    def distance(self) -> float:
        """Price vs the zone top: +8% means 8% above it, negative means inside."""
        return self.price / self.top - 1


def relative_zone(
    closes: list[tuple[date, float]], series: Series | None, today: date, price: float
) -> RelativeZone | None:
    """Today's P/S against its own two-year history. None when history is too thin."""
    if series is None or not price or price <= 0:
        return None
    start = today - timedelta(days=WINDOW_DAYS)
    if series.broken_after is not None:
        # A different capital structure before this: its prices are not
        # comparable. What remains must still clear MIN_OBSERVATIONS.
        start = max(start, series.broken_after + timedelta(days=1))
    history: list[float] = []
    for day, px in closes:
        if day < start or day > today or not px or px <= 0:
            continue
        rev, sh = as_of(series.revenue_ttm, day), as_of(series.shares, day)
        if rev is None or sh is None or rev.value <= 0:
            continue
        history.append(px * sh.value / rev.value)
    rev_now, sh_now = as_of(series.revenue_ttm, today), as_of(series.shares, today)
    if len(history) < MIN_OBSERVATIONS or rev_now is None or sh_now is None or rev_now.value <= 0:
        return None
    ps_now = price * sh_now.value / rev_now.value
    median = statistics.median(history)
    return RelativeZone(
        price=price,
        ps_now=ps_now,
        median=median,
        percentile=sum(v <= ps_now for v in history) / len(history),
        top=median * rev_now.value / sh_now.value,
        window_start=start,
        window_end=today,
        observations=len(history),
    )


# ── Absolute context ──────────────────────────────────────────────────────


class Scenario(BaseModel):
    terminal_multiple: float
    fcf_margin: float
    years: int


class Absolute(BaseModel):
    """What the price requires, at the generic and at the company's own margin."""

    price: float
    generic: Scenario
    required_generic: float | None = None
    own_margin: float | None = None  # trailing free-cash-flow margin
    required_own: float | None = None
    delivered: float | None = None  # revenue YoY, latest filing
    consensus: float | None = None  # annualised growth, run-rate -> last consensus year
    consensus_label: str = ""
    stale: bool = False
    notes: list[str] = Field(default_factory=list)


def required_at(snapshot: ValuationSnapshot, price: float, scenario: Scenario) -> float | None:
    """Revenue CAGR a price requires, from a stored snapshot's balance sheet."""
    runrate = snapshot.revenue_runrate
    if not runrate or runrate <= 0 or price <= 0 or snapshot.shares_outstanding <= 0:
        return None
    if scenario.fcf_margin <= 0 or scenario.terminal_multiple <= 0:
        return None
    ev = snapshot.shares_outstanding * price - (snapshot.net_cash or 0.0)
    if ev <= 0:
        return None
    revenue = ev / (scenario.terminal_multiple * scenario.fcf_margin)
    return (revenue / runrate) ** (1 / scenario.years) - 1


def consensus_growth(snapshot: ValuationSnapshot, consensus) -> tuple[float | None, str]:
    """Annualised growth from the run-rate to the furthest consensus year."""
    if consensus is None or not consensus.years or not snapshot.revenue_runrate:
        return None, ""
    last = consensus.years[-1]
    years = (last.fiscal_year_end - snapshot.period_end) / timedelta(days=365.25)
    if years <= 0.25 or last.avg <= 0:
        return None, ""
    growth = (last.avg / snapshot.revenue_runrate) ** (1 / years) - 1
    analysts = f", {last.analysts} analysts" if last.analysts else ""
    return growth, f"consensus {last.label} (${last.avg / 1e9:,.1f}bn{analysts})"


def absolute_context(
    snapshot: ValuationSnapshot | None,
    price: float | None,
    *,
    own_margin: float | None = None,
    consensus=None,
) -> Absolute | None:
    if snapshot is None or not price or price <= 0:
        return None
    base = snapshot.base_case()
    if base is None:
        return None
    generic = Scenario(
        terminal_multiple=base.terminal_multiple, fcf_margin=base.fcf_margin, years=base.years
    )
    ctx = Absolute(price=price, generic=generic, stale=snapshot.is_stale())
    ctx.required_generic = required_at(snapshot, price, generic)
    ctx.own_margin = own_margin
    if own_margin is None:
        ctx.notes.append("own free-cash-flow margin unavailable")
    elif own_margin <= 0:
        ctx.notes.append(
            f"trailing free cash flow is negative ({own_margin:+.1%} of revenue): "
            "a requirement at the company's own margin is undefined"
        )
    else:
        own = generic.model_copy(update={"fcf_margin": own_margin})
        ctx.required_own = required_at(snapshot, price, own)
    ctx.delivered = snapshot.revenue_yoy
    ctx.consensus, ctx.consensus_label = consensus_growth(snapshot, consensus)
    if ctx.stale:
        ctx.notes.append(f"the filing behind this is stale (period ending {snapshot.period_end})")
    return ctx


def ttm_fcf_margin(symbol: str) -> float | None:
    """Trailing four quarters of free cash flow over revenue, from yfinance. None on failure."""
    try:
        import yfinance as yf

        t = yf.Ticker(symbol)
        fcf = t.quarterly_cashflow.loc["Free Cash Flow"].dropna().sort_index().iloc[-4:]
        rev = t.quarterly_income_stmt.loc["Total Revenue"].dropna().sort_index().iloc[-4:]
    except Exception as exc:  # noqa: BLE001
        logger.info("zone: no FCF margin for %s: %s", symbol, exc)
        return None
    if len(fcf) < 4 or len(rev) < 4 or float(rev.sum()) <= 0:
        return None
    return float(fcf.sum()) / float(rev.sum())
