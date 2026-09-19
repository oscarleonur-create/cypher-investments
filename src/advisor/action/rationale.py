"""The reasoning chain behind a card, and where a proposed action comes from.

Two rules, and the second is the one that keeps this honest.

**Every step is a stored fact.** The chain is assembled from what the layers
below already established — the position, the events, the user's rules, the
factor model, the valuation, the data checks. A step that cannot carry its
number does not go in.

**Every proposed action declares its provenance.** There are exactly two
places an action may come from: a response the user wrote down in advance, or
arithmetic. Nothing else. The system has no third source, because a third
source would be an opinion it is not entitled to hold.

    YOUR_RULE   you wrote "if dilution is above 5%, exit half"; it tripped at
                6.7%, and here is what that is in shares and dollars
    ARITHMETIC  this is 22.1% of the book against the 20% limit you set;
                returning to the limit is 2 shares
    NONE        nothing you wrote covers this, so there is nothing to return
                to you — and that absence is itself the finding
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum

from pydantic import BaseModel, Field

from advisor.daemon.book import BookSnapshot, Position
from advisor.daemon.store import DaemonStore

logger = logging.getLogger(__name__)

# The concentration limit the mechanics layer already enforces. Kept in step
# with it rather than restated, so the two cannot drift.
from advisor.daemon.mechanics import MechanicsLimits  # noqa: E402


class Bearing(StrEnum):
    """What a fact does to the picture, never whether it is good or bad."""

    SUPPORTS = "SUPPORTS"  # consistent with holding as-is
    AGAINST = "AGAINST"  # a condition the user named, or a limit exceeded
    CONTEXT = "CONTEXT"  # true and relevant, decides nothing
    BLIND = "BLIND"  # the system cannot see this, and says so


class ReasonStep(BaseModel):
    """One link in the chain."""

    label: str
    fact: str
    bearing: Bearing = Bearing.CONTEXT


class ActionSource(StrEnum):
    YOUR_RULE = "YOUR_RULE"
    ARITHMETIC = "ARITHMETIC"
    NONE = "NONE"


class ProposedAction(BaseModel):
    """A next step, and the only two places it may have come from."""

    source: ActionSource
    text: str
    # The arithmetic, when there is any: what the action is in shares and
    # dollars at the current price.
    size: str = ""
    quoted_from: str = ""  # the claim whose response this is

    @property
    def is_the_users_own(self) -> bool:
        return self.source is ActionSource.YOUR_RULE


class Rationale(BaseModel):
    steps: list[ReasonStep] = Field(default_factory=list)
    proposed: ProposedAction | None = None

    @property
    def against(self) -> list[ReasonStep]:
        return [s for s in self.steps if s.bearing is Bearing.AGAINST]

    @property
    def blind_spots(self) -> list[ReasonStep]:
        return [s for s in self.steps if s.bearing is Bearing.BLIND]


def _sizing(position: Position, fraction: float) -> str:
    """A fraction of a position, in shares and dollars at the current price."""
    shares = abs(position.quantity) * fraction
    value = shares * position.price
    unit = "share" if round(shares) == 1 else "shares"
    return f"{shares:,.0f} {unit} ≈ ${value:,.0f} at {position.price:,.2f}"


def build_rationale(
    store: DaemonStore,
    symbol: str,
    book: BookSnapshot,
    card,
) -> Rationale:
    """The chain of stored facts behind a card, and what follows from it."""

    rationale = Rationale()
    position = next((p for p in book.positions if p.underlying.upper() == symbol), None)
    limits = MechanicsLimits()

    # 1. What is held.
    if position is not None:
        weight = position.signed_notional / book.net_liq if book.net_liq else 0.0
        rationale.steps.append(
            ReasonStep(
                label="position",
                fact=(
                    f"{position.quantity:+,.0f} at {position.avg_open_price:,.2f}, now "
                    f"{position.price:,.2f} — {position.unrealized_pct * 100:+.1f}%, "
                    f"{weight * 100:.1f}% of the book"
                ),
                bearing=(
                    Bearing.AGAINST if abs(weight) > limits.concentration_pct else Bearing.CONTEXT
                ),
            )
        )

    # 2. What happened. One line per kind: four concentration warnings in a
    # week are one standing condition observed four times, not four facts,
    # and repeating it buries the things that only happened once.
    seen: set[str] = set()
    distinct = []
    for trigger in card.triggers:
        if trigger.kind in seen:
            continue
        seen.add(trigger.kind)
        distinct.append(trigger)
    for trigger in distinct[:4]:
        rationale.steps.append(
            ReasonStep(
                label=trigger.kind.replace("_", " ").lower(),
                fact=trigger.detail or f"tier {trigger.tier}",
                bearing=Bearing.AGAINST if trigger.tier == "A" else Bearing.CONTEXT,
            )
        )

    # 3. What the user said.
    broken = [c for c in card.claims if c.status == "BROKEN"]
    intact = [c for c in card.claims if c.status == "INTACT"]
    if broken:
        for claim in broken:
            rationale.steps.append(
                ReasonStep(
                    label="your rule broke",
                    fact=f"{claim.text} — {claim.note}",
                    bearing=Bearing.AGAINST,
                )
            )
    elif intact:
        for claim in intact[:2]:
            rationale.steps.append(
                ReasonStep(
                    label="your rule held",
                    fact=f"{claim.text} — {claim.note}",
                    bearing=Bearing.SUPPORTS,
                )
            )
    elif not card.claims:
        rationale.steps.append(
            ReasonStep(
                label="your rules",
                fact="nothing written for this name, so none of the above was judged",
                bearing=Bearing.BLIND,
            )
        )

    # 4. What macro can say.
    sensitivity = store.load_sensitivity(symbol)
    if sensitivity is None:
        rationale.steps.append(
            ReasonStep(
                label="macro",
                fact="no factor estimate — a move here can be neither explained nor "
                "exonerated by the market",
                bearing=Bearing.BLIND,
            )
        )
    else:
        rationale.steps.append(
            ReasonStep(
                label="macro",
                fact=f"factors explain {sensitivity.r2:.0%} of this name's daily variance; "
                f"residual {sensitivity.resid_vol * 100:.1f}%/day",
                bearing=Bearing.CONTEXT,
            )
        )

    # 5. What the price requires.
    valuation = store.load_latest_valuation(symbol)
    base = valuation.base_case() if valuation else None
    if base is not None:
        rationale.steps.append(
            ReasonStep(
                label="price requires",
                fact=f"{base.implied_cagr * 100:.1f}% revenue growth a year for "
                f"{base.years} years, at {valuation.ev_to_revenue:,.1f}x revenue"
                + (" — the figures are stale" if valuation.is_stale() else ""),
                bearing=Bearing.CONTEXT,
            )
        )

    rationale.proposed = _propose(store, symbol, position, card, limits)
    return rationale


def _propose(store, symbol, position, card, limits) -> ProposedAction | None:
    """The next step, from a written response or from arithmetic. Never else."""
    from advisor.action.models import ActionKind

    if card.action is ActionKind.CANNOT_SAY:
        return None

    # First source: something the user already decided.
    broken = [c for c in card.claims if c.status == "BROKEN"]
    if broken:
        from advisor.thesis.repo import ThesisReadError, load_thesis

        try:
            thesis = load_thesis(store, symbol)
        except ThesisReadError:
            thesis = None
        responses = {c.text: c.response for c in thesis.claims if c.response} if thesis else {}
        for verdict in broken:
            written = responses.get(verdict.text)
            if written:
                return ProposedAction(
                    source=ActionSource.YOUR_RULE,
                    text=written,
                    size=_size_hint(written, position),
                    quoted_from=verdict.text,
                )
        # A rule broke and carries no response: say so rather than inventing one.
        return ProposedAction(
            source=ActionSource.NONE,
            text=(
                "you wrote what would break this but not what you would do about it — "
                "add `--then` so the next time it fires the card can return your own "
                "decision instead of asking you to make it again"
            ),
        )

    # Second source: arithmetic the user's own limit implies.
    if position is not None:
        weight = _weight(position, card)
        if weight is not None and abs(weight) > limits.concentration_pct:
            excess = (abs(weight) - limits.concentration_pct) / abs(weight)
            return ProposedAction(
                source=ActionSource.ARITHMETIC,
                text=(
                    f"this is {abs(weight) * 100:.1f}% of the book against the "
                    f"{limits.concentration_pct * 100:.0f}% limit in your settings; "
                    "returning to the limit is the trim below"
                ),
                size=_sizing(position, excess),
            )

    if card.action is ActionKind.WRITE_THESIS:
        return ProposedAction(
            source=ActionSource.NONE,
            text=(
                "nothing you wrote covers this name, so there is nothing to return to "
                "you — write one invalidation and the next event will be judged"
            ),
        )
    return None


def _weight(position: Position, card) -> float | None:
    """The position's share of the book, read back from the card's evidence."""
    for item in card.evidence.items:
        if item.name == "position" and "% of the book" in item.detail:
            try:
                return float(item.detail.rsplit(",", 1)[-1].strip().split("%")[0]) / 100
            except (ValueError, IndexError):
                return None
    return None


# Word boundaries, not substrings. "call the IR line and reassess" contains
# "all" and would otherwise be read as selling the entire position — which is
# precisely the failure this module exists to prevent.
_FRACTIONS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\bhalf\b|\bmitad\b"), 0.5),
    (re.compile(r"\ba third\b|\bun tercio\b"), 1 / 3),
    (re.compile(r"\ba quarter\b|\bun cuarto\b"), 0.25),
    (re.compile(r"\ball of it\b|\beverything\b|\btodo\b|\bla totalidad\b"), 1.0),
)


def _size_hint(response: str, position: Position | None) -> str:
    """Arithmetic for a response that names a fraction, when it plainly does.

    Deliberately narrow: it reads "half", "a third", "all" and nothing more.
    Guessing at a quantity the user did not state would put a number in their
    mouth, which is the one thing this module exists not to do.
    """
    if position is None:
        return ""
    lowered = response.lower()
    for pattern, fraction in _FRACTIONS:
        if pattern.search(lowered):
            return _sizing(position, fraction)
    return ""
