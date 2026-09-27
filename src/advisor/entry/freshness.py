"""How old each input to a proposal is, and which ones are too old to act on.

User decision (2026-09-27): "we cannot have old hypotheses making new
decisions". A proposal is only as current as its oldest input, and several
inputs fail silently: when the pre-open brief does not run, no filing lands,
so the tier-A blocker has nothing to fire on and an ENTER goes out on the day
of an 8-K nobody read. An empty event list and an unread one look the same
to the proposal; this module tells them apart.

Each check names the input, when it was last current, and the limit. A
stale input that could open a position turns the entry into WAIT with the
reason — never an exit: a stop or an EXIT still goes out, because the
cautious direction does not need fresh data to be right.

| Input | Current when | Stale blocks |
|---|---|---|
| price | its last bar is the proposal's session | entry |
| book | taken within ``BOOK_MAX_AGE_MINUTES`` in session, else since the last close | entry |
| filings | the brief ingested them since the session before | entry |
| distress news (held) | a sweep read it within ``DISTRESS_MAX_AGE_HOURS`` | adding |
| valuation (thesis) | computed within ``VALUATION_MAX_AGE_DAYS`` | the thesis bonus |

The limits are the user's (``decided``): the learning loop may never loosen a
guard on its own inputs.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from pydantic import BaseModel

from advisor.daemon import market_calendar as mc

# In session the book moves with every fill; the watch job stores it every 15 minutes.
BOOK_MAX_AGE_MINUTES = 60
# The distress sweep runs at 08:15 and 12:30 every day, weekends included.
DISTRESS_MAX_AGE_HOURS = 24
# The valuation job runs weekly; one missed run is tolerated, two are not.
VALUATION_MAX_AGE_DAYS = 10
# The jobs whose success means filings were ingested, and distress read.
FILINGS_JOBS = ("brief",)
DISTRESS_JOBS = ("distress_premarket", "distress_midday")

ENTRY = "entry"  # a stale input blocks opening or adding
ADDING = "adding"  # blocks an ADD on a held name only
BONUS = "bonus"  # withholds the thesis bonus


class Stale(BaseModel):
    input: str
    asof: str | None  # when it was last current (ISO), None = never
    limit: str  # what current means, in words
    blocks: str  # ENTRY | ADDING | BONUS

    @property
    def text(self) -> str:
        when = f"last {self.asof[:16].replace('T', ' ')}" if self.asof else "never"
        return f"{self.input} is stale ({when}; needs {self.limit})"


def book_reference(now: datetime) -> datetime:
    """The oldest a book may be at ``now``: an hour in session, else the last close."""
    now = mc.to_et(now)
    if mc.is_market_open(now):
        return now - timedelta(minutes=BOOK_MAX_AGE_MINUTES)
    day = mc.session_of(now)
    close = datetime.combine(day, mc.session_close(day), tzinfo=mc.MARKET_TZ)
    return close if close <= now else now - timedelta(minutes=BOOK_MAX_AGE_MINUTES)


def filings_reference(session: date) -> datetime:
    """Filings are current when ingested after the close of the session before this one."""
    prev = mc.previous_trading_day(session)
    return datetime.combine(prev, mc.session_close(prev), tzinfo=mc.MARKET_TZ)


def latest_ok(store, jobs) -> datetime | None:
    stamps = [store.get_heartbeat(j).last_ok_at for j in jobs]
    stamps = [mc.to_et(s) for s in stamps if s is not None]
    return max(stamps) if stamps else None


def _iso(moment) -> str | None:
    return moment.isoformat() if moment is not None else None


def assess(
    now: datetime,
    *,
    price_asof: date | None,
    book_asof: datetime | None,
    filings_ok: datetime | None,
    held: bool,
    distress_ok: datetime | None,
    thesis: str | None,
    valuation_asof: date | None,
) -> list[Stale]:
    """Every stale input for a proposal built at ``now``. Pure."""
    now = mc.to_et(now)
    session = mc.session_of(now)
    out: list[Stale] = []
    if price_asof is not None and price_asof < session:
        out.append(
            Stale(
                input="price",
                asof=price_asof.isoformat(),
                limit=f"a bar for {session.isoformat()}",
                blocks=ENTRY,
            )
        )
    ref = book_reference(now)
    if book_asof is None or mc.to_et(book_asof) < ref:
        out.append(
            Stale(
                input="book",
                asof=_iso(book_asof),
                limit=f"a snapshot since {ref.strftime('%Y-%m-%d %H:%M')}",
                blocks=ENTRY,
            )
        )
    ref = filings_reference(session)
    if filings_ok is None or filings_ok < ref:
        out.append(
            Stale(
                input="filings",
                asof=_iso(filings_ok),
                limit=f"ingested since {ref.strftime('%Y-%m-%d %H:%M')}",
                blocks=ENTRY,
            )
        )
    distress_ref = now - timedelta(hours=DISTRESS_MAX_AGE_HOURS)
    if held and (distress_ok is None or distress_ok < distress_ref):
        out.append(
            Stale(
                input="distress news",
                asof=_iso(distress_ok),
                limit=f"a sweep within {DISTRESS_MAX_AGE_HOURS}h",
                blocks=ADDING,
            )
        )
    if thesis == "intact" and (
        valuation_asof is None or (now.date() - valuation_asof).days > VALUATION_MAX_AGE_DAYS
    ):
        out.append(
            Stale(
                input="valuation",
                asof=_iso(valuation_asof),
                limit=f"computed within {VALUATION_MAX_AGE_DAYS} days",
                blocks=BONUS,
            )
        )
    return out
