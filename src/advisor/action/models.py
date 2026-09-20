"""What the system may propose, and what it must refuse to.

This module never forms a view. It has no opinion about whether a company is
good, a price is fair, or a position should be larger — and it cannot acquire
one, because nothing here reads anything but stored facts and the user's own
written rules.

What it does is close the loop the rest of the system was built for. The user
writes "an equity raise above 5% of market cap breaks this"; EDGAR delivers a
424B5 at 6.7%; the claim trips; and this says **the rule you wrote has
broken**, with the filing attached. The judgement was made in advance, by the
user, in writing. This only reports that the condition arrived.

So the actions are about attention, never about trades:

    REVIEW_NOW      a rule you wrote has broken, and there is a deadline
    REVIEW          something material happened that no rule covers
    WRITE_THESIS    nothing is written down, so nothing can be tested
    HOLD            rules intact, nothing fired
    CANNOT_SAY      the stored data is not good enough to say any of the above

`CANNOT_SAY` is the one that makes the rest trustworthy. A card that always
produces an answer is a card that will confidently produce a wrong one; the
evidence is graded first and blocks the verdict when it is too thin.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from advisor.daemon.market_calendar import now_et


class ActionKind(StrEnum):
    REVIEW_NOW = "REVIEW_NOW"
    REVIEW = "REVIEW"
    WRITE_THESIS = "WRITE_THESIS"
    HOLD = "HOLD"
    CANNOT_SAY = "CANNOT_SAY"


ACTION_TEXT: dict[ActionKind, str] = {
    ActionKind.REVIEW_NOW: "review now — a rule you wrote has broken",
    ActionKind.REVIEW: "worth reading — something material happened",
    ActionKind.WRITE_THESIS: "write down what would make this wrong",
    ActionKind.HOLD: "nothing to do — your rules are intact",
    ActionKind.CANNOT_SAY: "cannot say — the stored data is not good enough",
}

# Ordered by urgency; used when several considerations apply at once.
ACTION_RANK: dict[ActionKind, int] = {
    ActionKind.CANNOT_SAY: 0,
    ActionKind.REVIEW_NOW: 1,
    ActionKind.REVIEW: 2,
    ActionKind.WRITE_THESIS: 3,
    ActionKind.HOLD: 4,
}


class EvidenceItem(BaseModel):
    """One input, when it was established, and whether it can be relied on."""

    name: str
    asof: date | None = None
    detail: str = ""
    ok: bool = True
    # A gap that is merely a limitation (no factor model for a recent listing)
    # narrows the answer. A gap that is a fault (prices disagree across
    # sources) blocks it entirely.
    blocking: bool = False


class Evidence(BaseModel):
    """Everything the card rests on, graded."""

    items: list[EvidenceItem] = Field(default_factory=list)

    @property
    def blockers(self) -> list[EvidenceItem]:
        return [i for i in self.items if i.blocking]

    @property
    def gaps(self) -> list[EvidenceItem]:
        return [i for i in self.items if not i.ok and not i.blocking]

    @property
    def trustworthy(self) -> bool:
        return not self.blockers


class TriggerRef(BaseModel):
    """One thing that happened, with its own numbers."""

    kind: str
    tier: str
    when: datetime
    detail: str = ""
    url: str | None = None


class ClaimVerdict(BaseModel):
    """One of the user's written rules, and where it stands."""

    text: str
    kind: str
    status: str  # BROKEN | INTACT | UNTESTED | UNREACHABLE
    note: str = ""


class ActionCard(BaseModel):
    """What is known about one ticker, and what follows from it."""

    symbol: str
    assembled_at: datetime = Field(default_factory=now_et)
    action: ActionKind
    headline: str
    # Each line traceable to a stored fact. This is the whole justification;
    # if a claim cannot be put here with its number, it does not belong.
    because: list[str] = Field(default_factory=list)
    deadline: date | None = None
    position_note: str = ""
    triggers: list[TriggerRef] = Field(default_factory=list)
    claims: list[ClaimVerdict] = Field(default_factory=list)
    evidence: Evidence = Field(default_factory=Evidence)
    # The chain of stored facts behind the verdict, and the next step it
    # implies. `rationale.proposed` always names where it came from.
    rationale: dict = Field(default_factory=dict)
    # What the user could do that would let the system say more next time.
    what_would_sharpen_this: list[str] = Field(default_factory=list)

    @property
    def broken(self) -> list[ClaimVerdict]:
        return [c for c in self.claims if c.status == "BROKEN"]

    @property
    def urgent(self) -> bool:
        return self.action is ActionKind.REVIEW_NOW
