"""A rule broken weeks ago stays broken on the card until the user answers it.

The live case (2026-09-27): AAOI's 6.7% at-the-market raise on 2026-09-06
broke the user's 5% dilution rule. A week later the event aged out of the
card's window and Signals said "nothing to do — your rules are intact",
while the ticker's reading and the entry proposal both showed the rule broken.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from advisor.action.assemble import build_card
from advisor.action.decisions import Decision, Direction, SubjectKind, Verdict
from advisor.action.models import ActionKind
from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.daemon.market_calendar import now_et
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.entry.sheet import THESIS_LOOKBACK_DAYS
from advisor.thesis.models import Claim, ClaimKind, Comparator, Trigger

TODAY = now_et().date()


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def book() -> BookSnapshot:
    return BookSnapshot(
        as_of=now_et(),
        positions=[
            Position(
                account="A",
                symbol="CBRS",
                underlying="CBRS",
                instrument=EQUITY,
                quantity=10,
                multiplier=1,
                avg_open_price=200.0,
                close_price=160.0,
            )
        ],  # fmt: skip
        net_liq=10_000.0,
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


def decide(store, claim, *, observed: float) -> None:
    store.record_decision(
        Decision(
            symbol="CBRS",
            subject_kind=SubjectKind.CLAIM,
            subject_id=claim.id or "",
            verdict=Verdict.ACKNOWLEDGED,
            note="small enough to live with",
            observed=observed,
            worse_is=Direction.UP,
            decided_at=now_et() - timedelta(days=15),
        )
    )


def old_dilution(store, *, days_ago=21, pct=0.067, key="dil-old"):
    store.emit(
        Event(
            ts=now_et() - timedelta(days=days_ago),
            source=EventSource.EDGAR,
            kind="FILING_DILUTION",
            tier=EventTier.A,
            symbol="CBRS",
            dedup_key=key,
            payload={"dilution_pct": pct, "offering_usd": 6e8, "form": "424B5"},
        )
    )


def with_claim(store):
    claim = dilution_claim()
    store.save_claim("CBRS", claim)
    return claim


def test_a_rule_broken_three_weeks_ago_still_asks(store):
    with_claim(store)
    old_dilution(store, days_ago=21)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is ActionKind.REVIEW
    assert "violated right now" in card.headline
    assert "not answered since" in card.because[0]
    (claim,) = card.claims
    assert claim.status == "STANDING" and claim.observed == 0.067


def test_it_is_a_review_not_an_urgent_deadline(store):
    """Standing, not news: urgent every day would train the user to ignore it."""
    with_claim(store)
    old_dilution(store)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is not ActionKind.REVIEW_NOW and card.deadline is None


def test_answered_it_goes_quiet(store):
    claim = with_claim(store)
    old_dilution(store)
    decide(store, claim, observed=0.067)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is ActionKind.HOLD
    assert "you already answered" in card.headline


def test_a_worse_raise_after_the_answer_reopens_it(store):
    claim = with_claim(store)
    old_dilution(store, days_ago=30, pct=0.067, key="first")
    decide(store, claim, observed=0.067)
    old_dilution(store, days_ago=10, pct=0.12, key="second")
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is ActionKind.REVIEW
    assert card.claims[0].observed == 0.12  # the newest breach is the one shown


def test_past_the_lookback_it_is_history(store):
    with_claim(store)
    old_dilution(store, days_ago=THESIS_LOOKBACK_DAYS + 5)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.claims[0].status == "UNTESTED"


def test_a_raise_within_the_rule_does_not_stand(store):
    with_claim(store)
    old_dilution(store, pct=0.03)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.claims[0].status == "UNTESTED"
    assert card.action is ActionKind.HOLD


def test_a_breach_this_week_is_still_urgent(store):
    """The recent path is unchanged: news this week is REVIEW_NOW."""
    with_claim(store)
    old_dilution(store, days_ago=1)
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.action is ActionKind.REVIEW_NOW


def test_an_old_breach_and_a_small_recent_raise(store):
    """A small raise this week does not undo a large one three weeks ago."""
    with_claim(store)
    old_dilution(store, days_ago=21, pct=0.067, key="big")
    old_dilution(store, days_ago=1, pct=0.02, key="small")
    card = build_card(store, "CBRS", book(), today=TODAY)
    assert card.claims[0].status == "STANDING"
