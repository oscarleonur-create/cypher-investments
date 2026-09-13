"""The story contract: slots, and the honesty rules that govern them.

A story is one anchor event plus everything needed to judge it — the position
held at the time, the session that priced it, how much of the move macro can
account for, and what independent sources said. It is assembled from stored
rows only: no network, no model.

Two rules run through every slot here, both learned the hard way:

**A slot that cannot be filled says so.** There is no such thing as a
defaulted position or an assumed price. `PositionAtEvent` carries the date of
the snapshot it used and whether that snapshot actually covers the event; a
first prototype reported today's position against a three-week-old event and
would have described a loss on shares that might since have been sold.

**A slot never claims more than its statistic supports.** The same prototype
called AAOI's -13.77% session "idiosyncratic" from a residual z of -1.12,
when the alert threshold is 2.0 and that name's residual vol makes such a day
unremarkable. `Attribution.verdict` is derived from z by a fixed ladder, and
the numbers travel with it so the reader can disagree.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from advisor.daemon.models import EventTier


class Confidence(StrEnum):
    """How firmly a slot's claim is supported."""

    MEASURED = "MEASURED"  # from a primary source or an exact calculation
    ESTIMATED = "ESTIMATED"  # from a model, with its uncertainty stated
    UNAVAILABLE = "UNAVAILABLE"  # the slot could not be filled; say so


class Anchor(BaseModel):
    """The event the story is about."""

    event_id: str
    kind: str
    tier: EventTier
    symbol: str
    occurred_at: datetime  # when the thing happened, not when we noticed
    ingested_at: datetime
    headline: str
    source: str
    url: str | None = None
    quote: str | None = None  # verbatim, from the document itself
    facts: dict = Field(default_factory=dict)  # sized amounts, item codes, z

    @property
    def backfilled(self) -> bool:
        """True when we learned about this later than it happened."""
        return self.occurred_at.date() != self.ingested_at.date()


class PositionAtEvent(BaseModel):
    """What you held when it happened — or why that cannot be established."""

    confidence: Confidence
    quantity: float | None = None
    avg_open_price: float | None = None
    price_then: float | None = None
    weight_of_net_liq: float | None = None
    net_liq: float | None = None
    snapshot_asof: datetime | None = None
    covers_event: bool = False  # snapshot taken at or before the event
    note: str = ""

    @property
    def held(self) -> bool:
        return bool(self.quantity)


class PriceReaction(BaseModel):
    """The session that priced the event, and what it did to your shares."""

    confidence: Confidence
    session: date | None = None
    before: float | None = None
    after: float | None = None
    pct_move: float | None = None
    dollars: float | None = None  # on the position actually held
    pct_of_book: float | None = None
    priced_next_session: bool = False  # event landed after the close
    note: str = ""


class Verdict(StrEnum):
    """Calibrated readings of a residual, never stronger than the z supports."""

    NOT_MARKET = "NOT_MARKET"  # |z| >= 3
    UNEXPLAINED = "UNEXPLAINED"  # |z| >= 2 — the alerting threshold
    PARTLY_SPECIFIC = "PARTLY_SPECIFIC"  # |z| >= 1, inside this name's noise
    CONSISTENT = "CONSISTENT"  # |z| < 1
    UNKNOWN = "UNKNOWN"  # no usable estimate


VERDICT_TEXT: dict[Verdict, str] = {
    Verdict.NOT_MARKET: "the market did not do this",
    Verdict.UNEXPLAINED: "macro cannot explain this move",
    Verdict.PARTLY_SPECIFIC: ("partly company-specific, but inside this name's normal daily range"),
    Verdict.CONSISTENT: "consistent with what macro did that day",
    Verdict.UNKNOWN: "no usable factor estimate for this name",
}


class Attribution(BaseModel):
    """How much of the move the factor model accounts for."""

    confidence: Confidence
    verdict: Verdict = Verdict.UNKNOWN
    actual_return: float | None = None
    expected_return: float | None = None
    residual_z: float | None = None
    resid_vol: float | None = None  # the yardstick the z is measured in
    r2: float | None = None
    note: str = ""

    @property
    def would_have_fired(self) -> bool:
        """Whether this alone would have raised a RESIDUAL_DIVERGENCE."""
        return self.residual_z is not None and abs(self.residual_z) >= 2.0


class Corroboration(BaseModel):
    """Independent items around the event, with their provenance intact."""

    window_days: int
    items: list[dict] = Field(default_factory=list)  # tier, title, url, published_at
    primary_count: int = 0
    aggregator_count: int = 0
    untagged_count: int = 0


class ThesisLink(BaseModel):
    """What you said you believed, and which parts this event tests.

    ``exists`` and ``substantive`` are different questions. A row in the
    theses table is not a view: CRDO's document is the blank template, and
    reporting it as a thesis made the advisor claim the user held an opinion
    they had never written down.
    """

    confidence: Confidence
    exists: bool = False
    substantive: bool = False
    title: str | None = None
    conviction: str | None = None
    status: str | None = None
    claims_total: int = 0
    claims_monitored: int = 0
    evaluations: list[dict] = Field(default_factory=list)  # Evaluation, serialised
    note: str = ""

    @property
    def tripped(self) -> list[dict]:
        """Claims this event actually broke — the reason the slot exists."""
        return [e for e in self.evaluations if e.get("tripped")]

    @property
    def invalidated(self) -> bool:
        return any(e.get("kind") == "INVALIDATION" for e in self.tripped)


class Story(BaseModel):
    """One event, assembled. Complete without a model."""

    symbol: str
    assembled_at: datetime
    anchor: Anchor
    position: PositionAtEvent
    reaction: PriceReaction
    attribution: Attribution
    corroboration: Corroboration
    thesis: ThesisLink

    @property
    def unavailable_slots(self) -> list[str]:
        """Slots that could not be filled — surfaced, never hidden."""
        missing = []
        for name in ("position", "reaction", "attribution", "thesis"):
            slot = getattr(self, name)
            if slot.confidence is Confidence.UNAVAILABLE:
                missing.append(name)
        return missing
