"""What the price did after a candidate was seen.

Every return is measured from the detection price, long, because both setups
are traded long: A with the move, C against it. Horizons match how the user
actually holds these trades — minutes to the next session — not the 5/10/20
day horizons of the swing plan.

A horizon is written once it is knowable and never rewritten. A horizon that
cannot exist (+120m for a 15:00 detection) is written as None, so it is final
rather than "pending forever".
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from advisor.daemon import market_calendar as mc
from advisor.scanner import sources
from advisor.scanner.models import Candidate
from advisor.scanner.store import ScannerStore

logger = logging.getLogger(__name__)

MINUTE_HORIZONS = (30, 60, 120)
KEYS = ("r30", "r60", "r120", "close", "mfe", "mae", "next_open", "next_close")
# yfinance serves 5-minute bars for ~60 days; past that a gap stays a gap.
MAX_AGE_DAYS = 55
_SETTLE = timedelta(minutes=15)


def next_trading_day(day: date) -> date:
    cursor = day + timedelta(days=1)
    while not mc.is_trading_day(cursor):
        cursor += timedelta(days=1)
    return cursor


def _at(day: date, t) -> datetime:
    return datetime.combine(day, t, tzinfo=mc.MARKET_TZ)


def compute(c: Candidate, bars, now: datetime) -> dict[str, float | None]:
    """Every outcome knowable at ``now`` from ``bars``; absent keys are pending."""
    out: dict[str, float | None] = {}
    if bars is None or len(bars) == 0 or not c.price or c.price <= 0:
        return out
    now = mc.to_et(now)
    detected = mc.to_et(c.detected_at)
    close_dt = _at(c.session, mc.session_close(c.session))
    day = bars[bars.index.date == c.session]
    day = day[day.index < close_dt]

    def ret(px) -> float | None:
        return None if px is None or px != px or px <= 0 else float(px) / c.price - 1

    for minutes in MINUTE_HORIZONS:
        target = detected + timedelta(minutes=minutes)
        if target >= close_dt:
            out[f"r{minutes}"] = None  # the session ends first: final
            continue
        if now < target + timedelta(minutes=5):
            continue
        later = day[day.index >= target]
        if len(later):
            out[f"r{minutes}"] = ret(later["Open"].iloc[0])

    if now >= close_dt + _SETTLE and len(day):
        out["close"] = ret(day["Close"].iloc[-1])
        # Bars still open at detection onward: a bar starting 09:35 covers a
        # 09:37 detection, one starting 09:30 does not.
        held = day[day.index > detected - timedelta(minutes=5)]
        if len(held):
            out["mfe"] = ret(held["High"].max())
            out["mae"] = ret(held["Low"].min())

    nxt = next_trading_day(c.session)
    nday = bars[bars.index.date == nxt]
    if now >= _at(nxt, mc.REGULAR_OPEN) + timedelta(minutes=5) and len(nday):
        out["next_open"] = ret(nday["Open"].iloc[0])
    if now >= _at(nxt, mc.session_close(nxt)) + _SETTLE and len(nday):
        out["next_close"] = ret(nday["Close"].iloc[-1])
    return out


@dataclass
class OutcomeResult:
    pending: int = 0
    updated: int = 0
    fields: int = 0
    no_bars: list[str] | None = None

    def summary(self) -> str:
        text = f"{self.pending} pending, {self.updated} updated ({self.fields} fields)"
        if self.no_bars:
            text += f"; no bars for {', '.join(self.no_bars[:5])}"
        return text


def fill_outcomes(
    store: ScannerStore,
    now: datetime,
    *,
    bars_fn: Callable = sources.intraday_bars,
    max_age_days: int = MAX_AGE_DAYS,
) -> OutcomeResult:
    """Fill whatever outcomes have become knowable. Safe to re-run any time."""
    now = mc.to_et(now)
    since = now.date() - timedelta(days=max_age_days)
    result = OutcomeResult(no_bars=[])
    for c in store.list(since=since, limit=5000):
        if all(k in c.outcomes for k in KEYS):
            continue
        result.pending += 1
        end = next_trading_day(c.session) + timedelta(days=1)
        bars = bars_fn(c.symbol, c.session, end)
        if bars is None or len(bars) == 0:
            result.no_bars.append(c.symbol)
            continue
        added = store.merge_outcomes(c, compute(c, bars, now))
        if added:
            result.updated += 1
            result.fields += added
    return result
