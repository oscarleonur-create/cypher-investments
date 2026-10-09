"""Dated macro events the book should be decided before.

Hand-maintained, like the market calendar's holidays, and for the same
reason: the free sources do not publish them in a form worth parsing. Only
dates that are fixed by law or announced well ahead go here — never a
guess. Refresh it each year beside ``daemon/market_calendar.py``.
"""

from __future__ import annotations

from datetime import date

# (day, what it is) — the text reads after "through": "through the US midterm elections".
MACRO_EVENTS: tuple[tuple[date, str], ...] = (
    # First Tuesday after the first Monday of November (2 U.S.C. §7).
    (date(2026, 11, 3), "the US midterm elections"),
)


def upcoming(today: date, within: int) -> list[tuple[date, str, int]]:
    """Events from ``today`` to ``within`` trading sessions ahead: (day, label, sessions). Pure."""
    from advisor.entry.sheet import sessions_until

    out = []
    for day, label in MACRO_EVENTS:
        if day < today:
            continue
        n = sessions_until(today, day)
        if n <= within:
            out.append((day, label, n))
    return sorted(out)
