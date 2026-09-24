"""What the price did after a candidate was seen.

Every return is long, because all three setups are traded long: A and B with
the move, C against it. It is measured from the candidate's **entry**:

- a session candidate enters at the price when it was detected;
- a premarket candidate enters at the official open (see ``Phase``). Its
  ``open`` outcome is how far the open was from the premarket reference —
  the part of the move already gone before an order could fill.

Horizons span how the user holds these trades — minutes to the next session —
and the swing horizons of 5, 10 and 20 sessions, so the same record answers
"would holding longer have paid?".

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
from advisor.scanner.models import Candidate, Phase
from advisor.scanner.store import ScannerStore

logger = logging.getLogger(__name__)

MINUTE_HORIZONS = (30, 60, 120)
SESSION_HORIZONS = (5, 10, 20)
INTRADAY_KEYS = ("r30", "r60", "r120", "close", "mfe", "mae", "next_open", "next_close")
DAILY_KEYS = tuple(f"d{n}" for n in SESSION_HORIZONS)
KEYS = INTRADAY_KEYS + DAILY_KEYS  # a session candidate is complete with all of these
PREMARKET_KEYS = ("open", *KEYS)
# yfinance serves 5-minute bars for ~60 days; past that a gap stays a gap.
MAX_INTRADAY_AGE_DAYS = 55
# d20 is ~28 calendar days out; leave room for holidays and a sleeping laptop.
MAX_AGE_DAYS = 60
_SETTLE = timedelta(minutes=15)


def keys_for(c: Candidate) -> tuple[str, ...]:
    return PREMARKET_KEYS if c.phase is Phase.PREMARKET else KEYS


def next_trading_day(day: date) -> date:
    cursor = day + timedelta(days=1)
    while not mc.is_trading_day(cursor):
        cursor += timedelta(days=1)
    return cursor


def nth_trading_day(day: date, n: int) -> date:
    for _ in range(n):
        day = next_trading_day(day)
    return day


def _at(day: date, t) -> datetime:
    return datetime.combine(day, t, tzinfo=mc.MARKET_TZ)


def _entry(c: Candidate, day) -> tuple[float | None, datetime | None]:
    """(entry price, entry time), or (None, None) while not yet knowable."""
    if c.phase is Phase.PREMARKET:
        if not len(day):
            return None, None
        first = day.iloc[0]
        px = float(first["Open"])
        return (px if px > 0 else None), _at(c.session, mc.REGULAR_OPEN)
    return (c.price if c.price and c.price > 0 else None), mc.to_et(c.detected_at)


def compute(c: Candidate, bars, now: datetime, daily=None) -> dict[str, float | None]:
    """Every outcome knowable at ``now``; absent keys are pending.

    ``bars`` are 5-minute regular-session bars in ET; ``daily`` is a daily
    close series indexed by session date. Either may be None.
    """
    out: dict[str, float | None] = {}
    now = mc.to_et(now)
    close_dt = _at(c.session, mc.session_close(c.session))
    day = None
    if bars is not None and len(bars):
        day = bars[bars.index.date == c.session]
        day = day[day.index < close_dt]

    base, anchor = _entry(c, day if day is not None else [])

    def ret(px) -> float | None:
        if base is None or px is None or px != px or px <= 0:
            return None
        return float(px) / base - 1

    if day is not None and base is not None:
        if c.phase is Phase.PREMARKET and c.price and c.price > 0:
            out["open"] = base / c.price - 1

        for minutes in MINUTE_HORIZONS:
            target = anchor + timedelta(minutes=minutes)
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
            # Bars still open at entry onward: a bar starting 09:35 covers a
            # 09:37 detection, one starting 09:30 does not.
            held = day[day.index > anchor - timedelta(minutes=5)]
            if len(held):
                out["mfe"] = ret(held["High"].max())
                out["mae"] = ret(held["Low"].min())

        nxt = next_trading_day(c.session)
        nday = bars[bars.index.date == nxt]
        if now >= _at(nxt, mc.REGULAR_OPEN) + timedelta(minutes=5) and len(nday):
            out["next_open"] = ret(nday["Open"].iloc[0])
        if now >= _at(nxt, mc.session_close(nxt)) + _SETTLE and len(nday):
            out["next_close"] = ret(nday["Close"].iloc[-1])

    if daily is not None and base is not None:
        for n in SESSION_HORIZONS:
            target = nth_trading_day(c.session, n)
            if now < _at(target, mc.session_close(target)) + _SETTLE:
                continue
            px = daily.get(target)
            if px is not None and px == px and px > 0:
                out[f"d{n}"] = ret(px)
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
    daily_fn: Callable = sources.daily_closes,
    max_age_days: int = MAX_AGE_DAYS,
) -> OutcomeResult:
    """Fill whatever outcomes have become knowable. Safe to re-run any time."""
    now = mc.to_et(now)
    since = now.date() - timedelta(days=max_age_days)
    result = OutcomeResult(no_bars=[])
    for c in store.list(since=since, limit=5000):
        missing = [k for k in keys_for(c) if k not in c.outcomes]
        if not missing:
            continue
        result.pending += 1
        age = (now.date() - c.session).days
        # Premarket rows need the open for every horizon, daily ones included.
        needs_bars = age <= MAX_INTRADAY_AGE_DAYS and (
            any(k not in DAILY_KEYS for k in missing) or c.phase is Phase.PREMARKET
        )
        bars = None
        if needs_bars:
            bars = bars_fn(c.symbol, c.session, next_trading_day(c.session) + timedelta(days=1))
            if bars is None or len(bars) == 0:
                result.no_bars.append(c.symbol)
                bars = None
        daily = None
        if any(k in DAILY_KEYS for k in missing) and now.date() > next_trading_day(c.session):
            daily = daily_fn(
                c.symbol, c.session, nth_trading_day(c.session, 20) + timedelta(days=1)
            )
        if bars is None and daily is None:
            continue
        added = store.merge_outcomes(c, compute(c, bars, now, daily))
        if added:
            result.updated += 1
            result.fields += added
    return result
