"""Testing claims against the event stream.

This is what makes structure worth writing. "Dilution above 5% of market cap
kills this" takes ten seconds to type and is then checked against every filing
forever — and when AAOI's 424B5 lands at 6.7%, the story stops saying "you
have a thesis" and starts saying *which line of it just broke*.

A claim with no trigger is never evaluated and never silently passes. The
difference between "checked and fine" and "cannot be checked" is the whole
point, and both are reported.
"""

from __future__ import annotations

import logging

from advisor.daemon.models import Event
from advisor.thesis.models import Claim, ClaimKind, Comparator, Evaluation, StructuredThesis

logger = logging.getLogger(__name__)


def _observed(event: Event, field: str | None) -> float | None:
    if field is None:
        return None
    raw = (event.payload or {}).get(field)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.debug("thesis: %s is not numeric on %s", field, event.kind)
        return None


def evaluate_claim(claim: Claim, event: Event) -> Evaluation | None:
    """Test one claim against one event, or None when it does not apply.

    Returning None rather than a false evaluation matters: a claim about
    dilution is not "fine" because an earnings filing arrived, it is simply
    untested by it.
    """
    trigger = claim.trigger
    base = dict(claim_id=claim.id, kind=claim.kind, text=claim.text)

    if not trigger.event_kinds or event.kind not in trigger.event_kinds:
        return None

    if trigger.field is None or trigger.threshold is None:
        # The event happening at all is the test — an auditor resigning needs
        # no threshold.
        return Evaluation(**base, tripped=True, note=f"{event.kind} occurred")

    observed = _observed(event, trigger.field)
    if observed is None:
        return Evaluation(
            **base,
            tripped=False,
            threshold=trigger.threshold,
            note=f"{event.kind} carried no readable {trigger.field}",
        )

    tripped = (
        observed > trigger.threshold
        if trigger.comparator is Comparator.ABOVE
        else observed < trigger.threshold
    )
    word = "above" if trigger.comparator is Comparator.ABOVE else "below"
    return Evaluation(
        **base,
        tripped=tripped,
        observed=observed,
        threshold=trigger.threshold,
        note=(
            f"{trigger.field} {observed:g} is "
            f"{'' if tripped else 'not '}{word} {trigger.threshold:g}"
        ),
    )


def evaluate_thesis(thesis: StructuredThesis, event: Event) -> list[Evaluation]:
    """Every claim this event has something to say about, tripped first."""
    results = [
        result
        for result in (evaluate_claim(claim, event) for claim in thesis.claims)
        if result is not None
    ]
    results.sort(key=lambda r: (not r.tripped, r.kind is not ClaimKind.INVALIDATION))
    return results


def macro_conflicts(thesis: StructuredThesis, sensitivity) -> list[Evaluation]:
    """Macro drivers whose sign the measured loadings contradict.

    A thesis that leans on falling rates while the position carries a negative
    duration loading is not wrong — but it is a disagreement between what you
    said and what the position actually does, and you should see it.
    """
    if sensitivity is None:
        return []

    out: list[Evaluation] = []
    for claim in thesis.of_kind(ClaimKind.MACRO_DRIVER):
        factor = claim.trigger.factor
        if not factor:
            continue
        loading = sensitivity.loading(factor)
        wants_positive = claim.trigger.comparator is Comparator.ABOVE
        # Only a materially opposite loading counts as a contradiction; noise
        # around zero is not a disagreement worth reporting.
        contradicts = (loading < -0.3) if wants_positive else (loading > 0.3)
        out.append(
            Evaluation(
                claim_id=claim.id,
                kind=claim.kind,
                text=claim.text,
                tripped=contradicts,
                observed=loading,
                note=(
                    f"the thesis wants {factor} exposure to be "
                    f"{'positive' if wants_positive else 'negative'}; "
                    f"the measured loading is {loading:+.2f}"
                ),
            )
        )
    return out
