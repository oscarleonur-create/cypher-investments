"""What the user decided, so the card stops saying the same thing forever.

The loop the rest of this was built for ends here. A thesis is written, a
claim breaks, a card says which rule broke and why — and then says it again
tomorrow, and the day after, identically, until the channel stops being read.
Every alerting system dies this way. The missing half is the user's answer.

A decision attaches to **what was decided about**, never to a card: cards are
reassembled from scratch on every read and have no identity. A claim and an
event do — `claim.id` and the event's dedup key — and those are what a
decision points at.

It also records **the value at the time**. Acknowledging a 14% concentration
is not acknowledging a 25% one, and a decision that suppressed regardless of
where the number went would be worse than no decision at all. This is the same
distinction that separated BROKEN from STANDING: a level-triggered condition
can only be answered relative to a level.

The four verdicts describe what the system should do next, because that is
all it can honestly act on. It cannot verify that a position was trimmed or a
thesis rethought; it can only record what the user asserted and behave
accordingly.

    ACKNOWLEDGED    seen, nothing changes — quiet until it gets materially worse
    ACTED           the user says they acted — quiet briefly, then check
    DISMISSED       not relevant to how they hold this — quiet until the rule changes
    THESIS_REVISED  the rule was rewritten — this one is superseded

`ACTED` is the only one that comes back on its own. If someone marks a broken
stop as acted on and the position is in the same state three sessions later,
that is worth saying, and it is the one thing here the stored facts can
actually check.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from advisor.daemon.market_calendar import now_et
from advisor.thesis.models import Claim, Comparator

# How far a number must move past what was acknowledged before it is a new
# situation rather than the same one. Relative, because the same absolute step
# means different things to a 3% weight and a 30% one.
MATERIAL_WORSENING = 0.10

# How long an action gets to show up in the stored facts before the condition
# it was meant to resolve is raised again.
ACTED_GRACE_DAYS = 3


class Verdict(StrEnum):
    ACKNOWLEDGED = "ACKNOWLEDGED"
    ACTED = "ACTED"
    DISMISSED = "DISMISSED"
    THESIS_REVISED = "THESIS_REVISED"


VERDICT_TEXT: dict[Verdict, str] = {
    Verdict.ACKNOWLEDGED: "acknowledged",
    Verdict.ACTED: "acted on",
    Verdict.DISMISSED: "dismissed",
    Verdict.THESIS_REVISED: "thesis revised",
}


class SubjectKind(StrEnum):
    CLAIM = "CLAIM"
    EVENT = "EVENT"


class Direction(StrEnum):
    """Which way the observed number has to move to be worse than it was."""

    UP = "UP"
    DOWN = "DOWN"
    NEITHER = "NEITHER"


class Decision(BaseModel):
    """One answer, to one thing, at one value."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    symbol: str
    subject_kind: SubjectKind
    subject_id: str
    verdict: Verdict
    note: str = ""
    # The reading when the call was made. None for a subject that carries no
    # number — a filing arriving is not a quantity.
    observed: float | None = None
    # Stored rather than re-derived: the decision records the situation as it
    # stood, and must not change meaning because the rule was later edited.
    worse_is: Direction = Direction.NEITHER
    decided_at: datetime = Field(default_factory=now_et)

    def describe(self) -> str:
        when = self.decided_at.date().isoformat()
        if self.observed is None:
            return f"{VERDICT_TEXT[self.verdict]} on {when}"
        return f"{VERDICT_TEXT[self.verdict]} on {when} at {self.observed:,.4g}"


def worse_direction(claim: Claim) -> Direction:
    """Which way a claim's observed value goes when the situation deteriorates.

    A claim written "trips above X" is worse as the number rises. One written
    "trips below X" is worse as it falls. A claim that trips on an event
    happening at all has no number and therefore no direction.
    """
    if claim.trigger.threshold is None or claim.trigger.field is None:
        return Direction.NEITHER
    if claim.trigger.comparator is Comparator.ABOVE:
        return Direction.UP
    if claim.trigger.comparator is Comparator.BELOW:
        return Direction.DOWN
    return Direction.NEITHER


def materially_worse(decision: Decision, current: float | None) -> bool:
    """Whether the number has moved far enough to be a new situation."""
    if decision.observed is None or current is None:
        return False
    if decision.worse_is is Direction.NEITHER:
        return False
    moved = current - decision.observed
    if decision.worse_is is Direction.UP and moved <= 0:
        return False
    if decision.worse_is is Direction.DOWN and moved >= 0:
        return False
    base = abs(decision.observed)
    if base == 0:
        # Anything off zero is the whole of the move; there is no proportion
        # to take and reporting it is right.
        return True
    return abs(moved) / base >= MATERIAL_WORSENING


class Suppression(BaseModel):
    """Whether a decision still answers the situation, and why."""

    quiet: bool
    reason: str


def evaluate(
    decision: Decision, *, current: float | None, today: date | None = None
) -> Suppression:
    """Does this decision still stand, given where the number is now?"""
    today = today or date.today()
    stood_for = (today - decision.decided_at.date()).days

    if decision.verdict is Verdict.DISMISSED:
        return Suppression(quiet=True, reason=f"you dismissed this on {decision.decided_at.date()}")

    if decision.verdict is Verdict.THESIS_REVISED:
        # A rewritten claim is a new row with a new id, so it never reaches
        # this decision. What does reach it is the old rule, and the user has
        # already said what they think of it.
        return Suppression(
            quiet=True, reason=f"you revised this thesis on {decision.decided_at.date()}"
        )

    if decision.verdict is Verdict.ACTED:
        if stood_for < ACTED_GRACE_DAYS:
            return Suppression(
                quiet=True, reason=f"you acted on this on {decision.decided_at.date()}"
            )
        if materially_worse(decision, current) or current is None:
            return Suppression(quiet=False, reason="")
        return Suppression(
            quiet=False,
            reason=(
                f"you marked this acted on {decision.decided_at.date()} at "
                f"{decision.observed:,.4g} and it reads {current:,.4g} now"
                if decision.observed is not None
                else f"you marked this acted on {decision.decided_at.date()}"
            ),
        )

    # ACKNOWLEDGED
    if materially_worse(decision, current):
        return Suppression(
            quiet=False,
            reason=(
                f"you acknowledged this at {decision.observed:,.4g}; " f"it is {current:,.4g} now"
            ),
        )
    return Suppression(quiet=True, reason=f"you acknowledged this on {decision.decided_at.date()}")
