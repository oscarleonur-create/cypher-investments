"""Whether a decision still answers the situation it was made about.

This is the half the loop was missing. A card said "1 rule you wrote is
violated right now" about SPCX's implied growth, and said it again the next
day, and would have said it identically forever. Every alerting channel dies
that way.

The rule that makes a decision safe is that it is made **at a value**.
Acknowledging a 25.9% required growth against a 25% limit is not
acknowledging a 35% one, and a verdict that silenced both would be worse than
no verdict at all.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from advisor.action.decisions import (
    ACTED_GRACE_DAYS,
    MATERIAL_WORSENING,
    Decision,
    Direction,
    SubjectKind,
    Verdict,
    evaluate,
    materially_worse,
    worse_direction,
)
from advisor.daemon.market_calendar import now_et
from advisor.thesis.models import Claim, ClaimKind, Comparator, Trigger

TODAY = date(2026, 9, 20)


def decision(
    *,
    verdict: Verdict = Verdict.ACKNOWLEDGED,
    observed: float | None = 0.2588,
    worse_is: Direction = Direction.UP,
    days_ago: int = 0,
) -> Decision:
    return Decision(
        symbol="SPCX",
        subject_kind=SubjectKind.CLAIM,
        subject_id="23b2d9fb63be",
        verdict=verdict,
        note="rounding noise against a 25% limit",
        observed=observed,
        worse_is=worse_is,
        decided_at=now_et().replace(year=TODAY.year, month=TODAY.month, day=TODAY.day)
        - timedelta(days=days_ago),
    )


# --- which way is worse -----------------------------------------------------


def claim(comparator: Comparator, *, field: str | None = "implied_cagr", threshold=0.25) -> Claim:
    return Claim(
        symbol="SPCX",
        text="a rule",
        kind=ClaimKind.INVALIDATION,
        trigger=Trigger(
            event_kinds=["IMPLIED_EXPECTATIONS_SHIFT"],
            field=field,
            comparator=comparator,
            threshold=threshold,
        ),
    )


def test_a_rule_that_trips_above_is_worse_as_the_number_rises():
    assert worse_direction(claim(Comparator.ABOVE)) is Direction.UP


def test_a_rule_that_trips_below_is_worse_as_the_number_falls():
    assert worse_direction(claim(Comparator.BELOW)) is Direction.DOWN


def test_a_rule_with_no_number_has_no_direction():
    assert worse_direction(claim(Comparator.HAPPENS, field=None, threshold=None)) is (
        Direction.NEITHER
    )


# --- materially worse -------------------------------------------------------


def test_moving_the_wrong_way_is_not_worse():
    assert not materially_worse(decision(worse_is=Direction.UP), 0.20)
    assert not materially_worse(decision(worse_is=Direction.DOWN), 0.30)


def test_a_small_move_the_bad_way_is_still_the_same_situation():
    """0.2588 to 0.27 is 4.3% — the number wobbled, nothing changed."""
    assert not materially_worse(decision(), 0.27)


def test_a_move_past_the_threshold_is_a_new_situation():
    """0.2588 to 0.30 is 15.9%."""
    assert materially_worse(decision(), 0.30)


def test_exactly_at_the_worsening_threshold_counts():
    d = decision(observed=0.10)
    assert materially_worse(d, 0.10 * (1 + MATERIAL_WORSENING))


def test_just_under_the_worsening_threshold_does_not():
    d = decision(observed=0.10)
    assert not materially_worse(d, 0.10 * (1 + MATERIAL_WORSENING) - 1e-9)


def test_a_negative_base_worsens_downward():
    """A -20% drawdown acknowledged; -25% is a fifth worse."""
    d = decision(observed=-0.20, worse_is=Direction.DOWN)
    assert materially_worse(d, -0.25)
    assert not materially_worse(d, -0.21)


def test_any_move_off_a_zero_base_is_material():
    """There is no proportion to take, and the move is the whole of it."""
    d = decision(observed=0.0, worse_is=Direction.UP)
    assert materially_worse(d, 0.0001)


def test_nothing_to_compare_against_is_never_worse():
    assert not materially_worse(decision(observed=None), 0.9)
    assert not materially_worse(decision(), None)
    assert not materially_worse(decision(worse_is=Direction.NEITHER), 99.0)


# --- does the decision still stand ------------------------------------------


def test_an_acknowledgement_holds_while_the_number_holds():
    outcome = evaluate(decision(), current=0.26, today=TODAY)
    assert outcome.quiet
    assert "acknowledged" in outcome.reason


def test_an_acknowledgement_breaks_when_the_number_moves_past_it():
    outcome = evaluate(decision(), current=0.30, today=TODAY)
    assert not outcome.quiet
    # Both numbers, because that is what makes this stronger than the first
    # alert rather than a repeat of it.
    assert "0.2588" in outcome.reason and "0.3" in outcome.reason


def test_a_dismissal_holds_regardless_of_the_number():
    outcome = evaluate(decision(verdict=Verdict.DISMISSED), current=99.0, today=TODAY)
    assert outcome.quiet
    assert "dismissed" in outcome.reason


def test_a_revised_thesis_holds():
    """A rewritten claim is a new row with a new id, so it never reaches this
    decision. What reaches it is the old rule, already answered."""
    outcome = evaluate(decision(verdict=Verdict.THESIS_REVISED), current=99.0, today=TODAY)
    assert outcome.quiet


@pytest.mark.parametrize("days", [0, 1, ACTED_GRACE_DAYS - 1])
def test_an_action_is_given_time_to_show(days):
    outcome = evaluate(decision(verdict=Verdict.ACTED, days_ago=days), current=0.2588, today=TODAY)
    assert outcome.quiet


def test_an_action_that_changed_nothing_comes_back_saying_so():
    """The one verdict that returns on its own, and the only thing here the
    stored facts can actually check."""
    outcome = evaluate(
        decision(verdict=Verdict.ACTED, days_ago=ACTED_GRACE_DAYS), current=0.2588, today=TODAY
    )
    assert not outcome.quiet
    assert "acted" in outcome.reason and "0.2588" in outcome.reason


def test_an_action_followed_by_improvement_still_reports_its_state():
    outcome = evaluate(
        decision(verdict=Verdict.ACTED, days_ago=ACTED_GRACE_DAYS + 5), current=0.10, today=TODAY
    )
    assert not outcome.quiet
    assert "0.1" in outcome.reason


def test_a_decision_on_something_with_no_number_still_reads():
    d = decision(observed=None, worse_is=Direction.NEITHER)
    assert "at" not in d.describe()
    assert evaluate(d, current=None, today=TODAY).quiet
