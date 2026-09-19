"""A thesis as a set of testable claims, not an essay.

The repo already had a thesis feature: a nine-section markdown template. Both
theses in the live database are still mostly that template — CRDO's is the
blank form verbatim, PENG's has one real paragraph and eight placeholder
sections. That is the strongest available evidence about what gets filled in.
A prose template is a chore that produces nothing; a claim that the daemon
checks against every filing pays for itself the first time it fires.

So a claim carries a **trigger**: which event kinds test it, which payload
field to read, and the threshold that decides. "Dilution above 5% of market
cap kills this" is a sentence a person can write in ten seconds and the
machine can evaluate forever.

Claims are deliberately allowed to be untriggered. "Management is honest" is
a real part of a thesis and no event will ever test it; it is recorded, shown,
and never claimed to be monitored.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from advisor.daemon.market_calendar import now_et


class ClaimKind(StrEnum):
    DRIVER = "DRIVER"  # what has to go right
    INVALIDATION = "INVALIDATION"  # what kills the thesis
    KPI = "KPI"  # a number to keep an eye on
    MACRO_DRIVER = "MACRO_DRIVER"  # a factor the thesis leans on
    CATALYST = "CATALYST"  # a dated event to wait for
    RISK = "RISK"  # acknowledged, not necessarily testable


class Comparator(StrEnum):
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    HAPPENS = "HAPPENS"  # the event occurring at all is the test


class Trigger(BaseModel):
    """What in the event stream would test a claim.

    ``field`` reads a payload key from the matching event. When it is None the
    claim trips on the event happening at all — which is right for
    "the auditor resigns" and wrong for "dilution above 5%".
    """

    event_kinds: list[str] = Field(default_factory=list)
    field: str | None = None
    comparator: Comparator = Comparator.HAPPENS
    threshold: float | None = None
    factor: str | None = None  # MACRO_DRIVER: the factor panel name

    @property
    def is_testable(self) -> bool:
        return bool(self.event_kinds) or bool(self.factor)

    def describe(self) -> str:
        if self.factor:
            direction = "positive" if self.comparator is Comparator.ABOVE else "negative"
            return f"{self.factor} loading turns {direction}"
        if not self.event_kinds:
            return "not machine-testable"
        kinds = " or ".join(
            k.replace("FILING_", "").replace("_", " ").lower() for k in self.event_kinds
        )
        if self.field is None or self.threshold is None:
            return f"any {kinds}"
        word = "above" if self.comparator is Comparator.ABOVE else "below"
        return f"{kinds} with {self.field} {word} {self.threshold:g}"


class Claim(BaseModel):
    """One falsifiable statement inside a thesis."""

    id: str | None = None
    kind: ClaimKind
    text: str
    trigger: Trigger = Field(default_factory=Trigger)
    # What the user decided to do if this trips, written at the time they were
    # calm. Quoted back verbatim when it fires — the system proposes nothing of
    # its own, it returns the decision to the person who made it.
    response: str = ""
    due: date | None = None  # CATALYST only
    created_at: datetime = Field(default_factory=now_et)

    @property
    def monitored(self) -> bool:
        """True when the daemon can actually check this."""
        return self.trigger.is_testable


class Evaluation(BaseModel):
    """The result of testing one claim against one event."""

    claim_id: str | None
    kind: ClaimKind
    text: str
    tripped: bool
    observed: float | None = None
    threshold: float | None = None
    note: str = ""


class StructuredThesis(BaseModel):
    """A thesis with its claims attached.

    ``substantive`` is the distinction phase 3 was missing: a row existing in
    the theses table is not the same as having written anything down. A story
    that says "you have a thesis" about an untouched template is worse than
    one that says nothing.
    """

    symbol: str
    title: str
    conviction: str | None = None
    status: str | None = None
    substantive: bool = False
    claims: list[Claim] = Field(default_factory=list)
    prose_note: str = ""
    # claim id -> why its trigger can never fire for this symbol. Empty when
    # everything testable is also reachable.
    blocked: dict[str, str] = Field(default_factory=dict)

    def of_kind(self, kind: ClaimKind) -> list[Claim]:
        return [c for c in self.claims if c.kind is kind]

    @property
    def monitored_claims(self) -> list[Claim]:
        """Claims that are testable *and* reachable for this symbol.

        A claim with a trigger no event can produce here is not monitored,
        however well-formed it looks. SPCX's residual-divergence claim is the
        case that forced the distinction: valid trigger, no factor estimate,
        never fires.
        """
        return [c for c in self.claims if c.monitored and c.id not in self.blocked]

    @property
    def coverage(self) -> float:
        """Share of claims the daemon can actually test."""
        return len(self.monitored_claims) / len(self.claims) if self.claims else 0.0
