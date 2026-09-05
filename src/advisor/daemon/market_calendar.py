"""US equity market sessions.

The daemon runs on a laptop that sleeps, so every scheduling decision is made
against wall-clock time in ``America/New_York`` rather than elapsed time. This
module answers three questions: is the market open right now, was a given day a
trading day, and when does a given session end.

Holidays and early closes are hardcoded because the NYSE publishes them years
ahead and they effectively never move — a hand-maintained table is more
reliable here than a scraper. **Extend ``_HOLIDAYS`` and ``_EARLY_CLOSES`` once
a year.**
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

# NYSE full-day closures.
_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2026
        date(2026, 1, 1),  # New Year's Day
        date(2026, 1, 19),  # MLK Jr. Day
        date(2026, 2, 16),  # Presidents' Day
        date(2026, 4, 3),  # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),  # Independence Day (observed)
        date(2026, 9, 7),  # Labor Day
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas
        # 2027
        date(2027, 1, 1),  # New Year's Day
        date(2027, 1, 18),  # MLK Jr. Day
        date(2027, 2, 15),  # Presidents' Day
        date(2027, 3, 26),  # Good Friday
        date(2027, 5, 31),  # Memorial Day
        date(2027, 6, 18),  # Juneteenth (observed)
        date(2027, 7, 5),  # Independence Day (observed)
        date(2027, 9, 6),  # Labor Day
        date(2027, 11, 25),  # Thanksgiving
        date(2027, 12, 24),  # Christmas (observed)
    }
)

# Sessions that close at 13:00 ET instead of 16:00 ET.
_EARLY_CLOSES: frozenset[date] = frozenset(
    {
        date(2026, 11, 27),  # day after Thanksgiving
        date(2026, 12, 24),  # Christmas Eve
        date(2027, 7, 2),  # day before Independence Day (observed)
        date(2027, 11, 26),  # day after Thanksgiving
    }
)


def now_et() -> datetime:
    """Current wall-clock time in market timezone."""
    return datetime.now(MARKET_TZ)


def to_et(moment: datetime) -> datetime:
    """Convert ``moment`` to market timezone, assuming naive input is already ET."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=MARKET_TZ)
    return moment.astimezone(MARKET_TZ)


def is_trading_day(day: date) -> bool:
    """True when the market holds a session on ``day``."""
    return day.weekday() < 5 and day not in _HOLIDAYS


def session_close(day: date) -> time:
    """Closing bell for ``day`` — 13:00 ET on early-close days, else 16:00 ET."""
    return EARLY_CLOSE if day in _EARLY_CLOSES else REGULAR_CLOSE


def is_market_open(moment: datetime | None = None) -> bool:
    """True when ``moment`` falls inside a regular trading session."""
    et = to_et(moment) if moment else now_et()
    if not is_trading_day(et.date()):
        return False
    return REGULAR_OPEN <= et.time() < session_close(et.date())


def previous_trading_day(day: date) -> date:
    """The most recent trading day strictly before ``day``."""
    from datetime import timedelta

    cursor = day - timedelta(days=1)
    while not is_trading_day(cursor):
        cursor -= timedelta(days=1)
    return cursor
