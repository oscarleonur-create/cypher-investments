"""A card that has been answered stops saying the same thing.

The live case: SPCX's card said "1 rule you wrote is violated right now"
about implied growth of 25.9% against a 25% limit, and would have said it
identically every day the condition lasted. The user's answer — "rounding
noise, I'll look again past 28%" — is the missing half of the loop.

Two properties matter and both are tested here. A decided item is **shown,
not dropped**: hiding it would make the card quieter and less complete at the
same time. And a decision **stops answering** when the number moves past
where it was made, coming back with both readings, because "you acknowledged
this at 0.2588 and it is 0.30 now" is a stronger sentence than the original
alert rather than a repeat of it.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from advisor.action.assemble import build_card
from advisor.action.decisions import (
    ACTED_GRACE_DAYS,
    Decision,
    Direction,
    SubjectKind,
    Verdict,
)
from advisor.action.models import ActionKind
from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.daemon.market_calendar import now_et
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.thesis.models import Claim, ClaimKind, Comparator, Trigger

TODAY = date(2026, 9, 19)


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def book(weight_pct=0.16) -> BookSnapshot:
    net_liq = 10_000.0
    price = net_liq * weight_pct / 10
    return BookSnapshot(
        as_of=now_et(),
        positions=[
            Position(
                account="5WI30382",
                symbol="CBRS",
                underlying="CBRS",
                instrument=EQUITY,
                quantity=10,
                multiplier=1,
                avg_open_price=200.0,
                close_price=price,
            )
        ],
        net_liq=net_liq,
    )


def dilution_claim() -> Claim:
    return Claim(
        symbol="CBRS",
        kind=ClaimKind.INVALIDATION,
        text="An equity raise above 5% of market cap breaks this",
        trigger=Trigger(
            event_kinds=["FILING_DILUTION"],
            field="dilution_pct",
            comparator=Comparator.ABOVE,
            threshold=0.05,
        ),
    )


def dilution_event(pct=0.067) -> Event:
    return Event(
        source=EventSource.EDGAR,
        kind="FILING_DILUTION",
        tier=EventTier.A,
        symbol="CBRS",
        dedup_key="dil-1",
        payload={"dilution_pct": pct, "offering_usd": 6e8, "form": "424B5"},
    )


def setup(store: DaemonStore, *, pct=0.067) -> Claim:
    claim = dilution_claim()
    store.save_claim("CBRS", claim)
    store.emit(dilution_event(pct))
    return claim


def decide(
    store: DaemonStore,
    claim: Claim,
    *,
    verdict=Verdict.ACKNOWLEDGED,
    observed=0.067,
    days_ago=0,
) -> Decision:
    d = Decision(
        symbol="CBRS",
        subject_kind=SubjectKind.CLAIM,
        subject_id=claim.id or "",
        verdict=verdict,
        note="sized small enough that I can live with it",
        observed=observed,
        worse_is=Direction.UP,
        decided_at=now_et().replace(year=TODAY.year, month=TODAY.month, day=TODAY.day)
        - timedelta(days=days_ago),
    )
    store.record_decision(d)
    return d


# --- the card before and after an answer ------------------------------------


def test_an_unanswered_broken_rule_drives_the_card(store):
    setup(store)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is ActionKind.REVIEW_NOW
    assert card.decided == []


def test_an_answered_rule_no_longer_drives_the_card(store):
    claim = setup(store)
    decide(store, claim)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is not ActionKind.REVIEW_NOW
    assert card.deadline is None


def test_the_answered_rule_is_shown_not_dropped(store):
    claim = setup(store)
    decide(store, claim)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert len(card.decided) == 1
    ref = card.decided[0]
    assert ref.subject_id == claim.id
    assert ref.verdict == "ACKNOWLEDGED"
    assert "live with it" in ref.note
    assert ref.decided_at == TODAY


def test_the_rule_still_reads_as_broken_under_your_rules(store):
    """Answered is not intact. The card must not launder one into the other."""
    claim = setup(store)
    decide(store, claim)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert [c.status for c in card.claims if c.claim_id == claim.id] == ["BROKEN"]


def test_the_headline_says_it_was_answered_rather_than_intact(store):
    claim = setup(store)
    decide(store, claim)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert "already answered" in card.headline
    assert "intact" not in card.headline


# --- coming back ------------------------------------------------------------


def test_a_worse_number_reopens_the_rule(store):
    """Acknowledged at 6.7%; a 12% raise is a different situation."""
    claim = setup(store, pct=0.12)
    decide(store, claim, observed=0.067)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is ActionKind.REVIEW_NOW
    assert len(card.reopened) == 1
    assert card.decided == []


def test_a_reopened_rule_carries_both_readings(store):
    claim = setup(store, pct=0.12)
    decide(store, claim, observed=0.067)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert any("0.067" in r and "0.12" in r for r in card.because)


def test_a_slightly_worse_number_is_the_same_situation(store):
    """6.7% to 7.0% is under 5% of the way; the number wobbled."""
    claim = setup(store, pct=0.070)
    decide(store, claim, observed=0.067)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is not ActionKind.REVIEW_NOW
    assert len(card.decided) == 1


def test_a_dismissal_survives_any_number(store):
    claim = setup(store, pct=0.40)
    decide(store, claim, verdict=Verdict.DISMISSED, observed=0.067)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is not ActionKind.REVIEW_NOW
    assert card.decided[0].verdict == "DISMISSED"


def test_an_action_comes_back_after_its_grace_period(store):
    claim = setup(store)
    decide(store, claim, verdict=Verdict.ACTED, days_ago=ACTED_GRACE_DAYS)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is ActionKind.REVIEW_NOW
    assert len(card.reopened) == 1
    assert "acted" in card.reopened[0].reason


def test_an_action_is_quiet_inside_its_grace_period(store):
    claim = setup(store)
    decide(store, claim, verdict=Verdict.ACTED, days_ago=ACTED_GRACE_DAYS - 1)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is not ActionKind.REVIEW_NOW


# --- the record -------------------------------------------------------------


def test_a_change_of_mind_is_a_new_row_and_the_newest_one_answers(store):
    claim = setup(store, pct=0.40)
    decide(store, claim, verdict=Verdict.ACKNOWLEDGED, observed=0.40)
    decide(store, claim, verdict=Verdict.DISMISSED, observed=0.40)
    assert len(store.decision_history("CBRS")) == 2
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.decided[0].verdict == "DISMISSED"


def test_a_decision_on_another_symbol_does_not_answer_this_one(store):
    claim = setup(store)
    other = Decision(
        symbol="AAOI",
        subject_kind=SubjectKind.CLAIM,
        subject_id=claim.id or "",
        verdict=Verdict.DISMISSED,
        worse_is=Direction.UP,
    )
    store.record_decision(other)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is ActionKind.REVIEW_NOW
    assert card.decided == []


def test_no_decisions_leaves_the_card_exactly_as_it_was(store):
    setup(store)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.decided == [] and card.reopened == []
    assert card.action is ActionKind.REVIEW_NOW


# --- what "no rule covers this" has to mean ---------------------------------


def test_an_event_a_rule_watches_is_not_reported_as_uncovered(store):
    """Found by answering a broken rule. With the rule answered the card fell
    through to "1 material event, no written rule covers them" about the very
    filing the answered rule was written for. The sentence was false in
    exactly the case the thesis layer exists to handle."""
    claim = setup(store)
    decide(store, claim)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert "no written rule covers" not in card.headline


def test_an_event_no_rule_watches_is_still_reported(store):
    """The other side: a rule about dilution says nothing about a merger."""
    store.save_claim("CBRS", dilution_claim())
    store.emit(
        Event(
            source=EventSource.EDGAR,
            kind="FILING_MERGER",
            tier=EventTier.A,
            symbol="CBRS",
            dedup_key="merger-1",
            payload={"form": "8-K", "label": "merger agreement"},
        )
    )
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert "no written rule covers" in card.headline


def test_a_rule_that_looked_and_found_nothing_counts_as_cover(store):
    """A 2% raise against a 5% rule is covered and fine, not uncovered."""
    setup(store, pct=0.02)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is not ActionKind.REVIEW_NOW
    assert "no written rule covers" not in card.headline
