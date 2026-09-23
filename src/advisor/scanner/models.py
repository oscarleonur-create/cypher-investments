"""What the scanner sees, what it records, and what it later learns."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Setup(StrEnum):
    CATALYST_GAP = "A"  # gap up on fresh news, heavy volume — with the move
    NEWS_DIP = "C"  # large cap falling hard on news — against the move


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


class Candidate(BaseModel):
    """A setup detected at one moment, plus the outcomes filled in later."""

    session: date
    setup: Setup
    symbol: str
    detected_at: datetime  # tz-aware ET
    name: str | None = None
    price: float  # the price an entry at detection would have paid
    prev_close: float
    open: float | None = None
    change: float  # price vs prev close
    gap: float | None = None  # open vs prev close
    rvol: float | None = None  # volume vs the pace expected by now
    sigma: float | None = None  # |change| in units of 60-session daily vol
    market_cap: float | None = None
    catalysts: list[CatalystItem] = Field(default_factory=list)
    news_checked: bool = False  # False: nobody looked, which is not "no news"
    outcomes: dict[str, float | None] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.session.isoformat()}:{self.setup.value}:{self.symbol}"

    @property
    def has_catalyst(self) -> bool | None:
        """True/False once news was checked; None when it never was."""
        if not self.news_checked:
            return None
        return bool(self.catalysts)
