"""Testing a claim against what is true now, not only against what happened.

Claims were evaluated against the event stream alone, and the event stream is
deliberately edge-triggered: it reports *changes*. That is right for alerts —
phase 2a learned it expensively when level-triggered thresholds fired nine
interrupts at once — and it is wrong for the question a thesis actually asks,
which is "is this still true?"

The gap, live: the user wrote "required growth must not exceed 25% a year for
a decade" on SPCX. The stored valuation says 25.8%. No
`IMPLIED_EXPECTATIONS_SHIFT` had fired, because the number had not *moved* two
points — so the card reported "nothing this week tested it" about a rule that
was being violated the whole time.

So both readings are kept, and they are not the same thing:

    BROKEN     an event tripped it — something happened, this is news
    STANDING   the current state violates it — this is a condition, and it
               was true yesterday too

A standing violation must never be urgent. It would be urgent every day
forever, which is how an alert channel stops being read.

Not every claim has a standing form. Dilution is inherently an event: there
is no "current dilution" to read. Concentration, drawdown and implied growth
are inherently states. A claim with no standing reading returns None rather
than a pass, exactly as the event evaluator does.
"""

from __future__ import annotations

import logging

from advisor.daemon.book import BookSnapshot
from advisor.daemon.store import DaemonStore
from advisor.thesis.models import Claim, Comparator, Evaluation

logger = logging.getLogger(__name__)

# Event kinds whose underlying quantity exists as a standing value, and where
# to read it from. A kind absent here has no standing form by nature.
STATEFUL_KINDS: dict[str, str] = {
    "IMPLIED_EXPECTATIONS_SHIFT": "implied_cagr",
    "CONCENTRATION_WARNING": "weight",
    "STOP_BREACHED": "unrealized_pct",
    "PROFIT_TARGET_HIT": "unrealized_pct",
    "DEEP_DRAWDOWN": "unrealized_pct",
    "RESIDUAL_DIVERGENCE": None,  # a residual is a single session, never a state
}


def _current(store: DaemonStore, symbol: str, book: BookSnapshot, field: str) -> float | None:
    """Read one quantity from stored state, or None when it is unavailable."""
    if field in {"unrealized_pct", "weight"}:
        held = [p for p in book.positions if p.underlying.upper() == symbol]
        if not held:
            return None
        if field == "unrealized_pct":
            return held[0].unrealized_pct
        if not book.net_liq:
            return None
        return sum(p.signed_notional for p in held) / book.net_liq

    if field == "implied_cagr":
        valuation = store.load_latest_valuation(symbol)
        base = valuation.base_case() if valuation else None
        return base.implied_cagr if base else None

    return None


def evaluate_against_state(
    store: DaemonStore, symbol: str, claim: Claim, book: BookSnapshot
) -> Evaluation | None:
    """Whether the current state violates this claim, or None if it has no
    standing form."""
    trigger = claim.trigger
    if not trigger.event_kinds or trigger.field is None or trigger.threshold is None:
        return None

    fields = {
        STATEFUL_KINDS.get(kind)
        for kind in trigger.event_kinds
        if STATEFUL_KINDS.get(kind) is not None
    }
    if len(fields) != 1:
        # Either no kind has a standing reading, or they disagree about which
        # quantity to read. Both are reasons to say nothing.
        return None

    field = fields.pop()
    if field != trigger.field:
        logger.debug(
            "thesis: %s watches %s but its kinds carry %s; no standing reading",
            symbol,
            trigger.field,
            field,
        )
        return None

    observed = _current(store, symbol, book, field)
    if observed is None:
        return None

    tripped = (
        observed > trigger.threshold
        if trigger.comparator is Comparator.ABOVE
        else observed < trigger.threshold
    )
    word = "above" if trigger.comparator is Comparator.ABOVE else "below"
    return Evaluation(
        claim_id=claim.id,
        kind=claim.kind,
        text=claim.text,
        tripped=tripped,
        observed=observed,
        threshold=trigger.threshold,
        note=(
            f"{field} is {observed:.4g} right now, "
            f"{'' if tripped else 'not '}{word} {trigger.threshold:g}"
        ),
    )
