"""What the scanner sees, what it records, and what it later learns."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Setup(StrEnum):
    CATALYST_GAP = "A"  # gap up on fresh news, heavy volume — with the move
    DAY_TWO = "B"  # continuation the day after a strong up day — with the move
    NEWS_DIP = "C"  # large cap falling hard on news — against the move


class Phase(StrEnum):
    """When a candidate was seen, which decides what its entry price is.

    A session candidate's entry is the price at detection. A premarket
    candidate cannot be entered at its premarket quote in any size the user
    trades, so its entry is the official open; the premarket price is kept
    to measure how much of the move the open had already given away.
    """

    SESSION = "session"
    PREMARKET = "premarket"


class Mover(BaseModel):
    """One symbol's session so far, as a screener or quote reports it.

    Every numeric field is optional because free data sources omit fields
    freely; the detector decides what it can and cannot judge without them.
    """

    symbol: str
    name: str | None = None
    price: float | None = None
    prev_close: float | None = None
    open: float | None = None
    volume: float | None = None
    avg_volume: float | None = None  # 3-month average daily volume
    market_cap: float | None = None
    source: str = "screener"


class CatalystItem(BaseModel):
    """A news item or filing that could explain the move. Never scored."""

    kind: str  # "news" or the filing form, e.g. "8-K"
    title: str
    published_at: datetime
    provider: str = ""
    url: str = ""


def candidate_id(session: date, setup: Setup, symbol: str, phase: Phase = Phase.SESSION) -> str:
    """Stable id. Session ids keep their original form so existing rows match.

    Premarket and session are separate candidates on purpose: META flagged at
    08:40 (entry at the open) and again at 10:05 (entry at 10:05) are two
    different trades, and each outcome must be measured from its own entry.
    """
    base = f"{session.isoformat()}:{setup.value}"
    if phase is Phase.PREMARKET:
        base += "@pre"
    return f"{base}:{symbol.upper()}"


class Candidate(BaseModel):
    """A setup detected at one moment, plus the outcomes filled in later."""

    session: date
    setup: Setup
    symbol: str
    detected_at: datetime  # tz-aware ET
    phase: Phase = Phase.SESSION  # rows written before premarket existed are session
    source: str = ""  # where the symbol came from, e.g. "tastytrade:Swing"
    name: str | None = None
    price: float  # the price an entry at detection would have paid
    prev_close: float
    open: float | None = None
    # Price vs prev close. None only for a premarket B with no premarket
    # prints: silence before the bell is not a 0% move.
    change: float | None
    gap: float | None = None  # open vs prev close
    rvol: float | None = None  # volume vs the pace expected by now
    sigma: float | None = None  # |change| in units of 60-session daily vol
    market_cap: float | None = None
    peers: list[str] = Field(default_factory=list)  # C: most-correlated industry peers
    peer_move: float | None = None  # C: their median move today; None = not measured
    prev_day_change: float | None = None  # B: yesterday's close vs the day before
    prev_day_rvol: float | None = None  # B: yesterday's volume vs its 20-day average
    premarket_volume: float | None = None  # premarket only: shares traded 04:00-now
    catalysts: list[CatalystItem] = Field(default_factory=list)
    news_checked: bool = False  # False: nobody looked, which is not "no news"
    outcomes: dict[str, float | None] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return candidate_id(self.session, self.setup, self.symbol, self.phase)

    @property
    def has_catalyst(self) -> bool | None:
        """True/False once news was checked; None when it never was."""
        if not self.news_checked:
            return None
        return bool(self.catalysts)


# ── Journal: what happened to each candidate ─────────────────────────────


class DecisionSource(StrEnum):
    BROKER = "broker"  # a fill in the account matched the candidate
    USER = "user"  # the user said so


class SkipReason(StrEnum):
    """Why a candidate was not taken. Coarse on purpose: the review groups by it.

    The point is to learn which reasons cost money. "late" that keeps missing
    winners is a different lesson from "news" that keeps dodging losers.
    """

    LATE = "late"  # saw it after the move, or the entry had run
    NEWS = "news"  # the catalyst looked like it changes the business
    SIZE = "size"  # no cash, or the size would have been wrong
    EXPOSED = "exposed"  # already exposed to the name or its sector
    MISSED = "missed"  # never saw it
    RULES = "rules"  # outside a rule the user trades by
    OTHER = "other"


class TradeDecision(BaseModel):
    """Taken or not, and why. One candidate can collect several over time."""

    candidate_id: str
    taken: bool
    source: DecisionSource
    reason: SkipReason | None = None  # only for not-taken
    note: str = ""
    # From the broker when taken: the first opening fill on the candidate's
    # session. Kept so a review can compare the user's entry with the setup's.
    fill_price: float | None = None
    fill_at: datetime | None = None
    quantity: float | None = None
    account: str | None = None
    decided_at: datetime = Field(default_factory=lambda: _now_et())


def _now_et() -> datetime:
    from advisor.daemon.market_calendar import now_et

    return now_et()
