"""Action cards: what the system may propose, and what it must refuse to.

The rule that shapes every test here: the system never forms a view. It
reports that a condition the user defined in advance has or has not arrived.
A card that always produces an answer is a card that will eventually produce
a confident wrong one, so evidence is graded before anything else and blocks
the verdict when it is too thin.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from advisor.action.assemble import POSITION_STALE_DAYS, build_all, build_card
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


def pos(symbol="CBRS", *, qty=2, entry=217.09, price=198.89) -> Position:
    return Position(
        account="5WI30382",
        symbol=symbol,
        underlying=symbol,
        instrument=EQUITY,
        quantity=qty,
        multiplier=1,
        avg_open_price=entry,
        close_price=price,
    )


def book(*positions, net_liq=7_890.0, as_of=None) -> BookSnapshot:
    return BookSnapshot(
        as_of=as_of or now_et(), positions=list(positions) or [pos()], net_liq=net_liq
    )


def stop_event(**kw) -> Event:
    base = dict(
        source=EventSource.COMPUTED,
        kind="STOP_BREACHED",
        tier=EventTier.A,
        symbol="CBRS",
        dedup_key="stop-1",
        payload={
            "entry": 217.09,
            "price": 199.66,
            "unrealized_pct": -0.0803,
            "unrealized_usd": -34.86,
            "threshold": -0.08,
        },
    )
    return Event(**{**base, **kw})


def dilution_event(pct=0.067, **kw) -> Event:
    base = dict(
        source=EventSource.EDGAR,
        kind="FILING_DILUTION",
        tier=EventTier.A,
        symbol="CBRS",
        dedup_key="dil-1",
        payload={"dilution_pct": pct, "offering_usd": 6e8, "form": "424B5"},
    )
    return Event(**{**base, **kw})


def claim(threshold=0.05) -> Claim:
    return Claim(
        kind=ClaimKind.INVALIDATION,
        text="An equity raise above 5% of market cap breaks this",
        trigger=Trigger(
            event_kinds=["FILING_DILUTION"],
            field="dilution_pct",
            comparator=Comparator.ABOVE,
            threshold=threshold,
        ),
    )


class TestEvidenceBlocksFirst:
    """A flag the system raised itself cannot then be ignored."""

    def test_a_symbol_not_held_cannot_be_advised_on(self, store):
        card = build_card(store, "NVDA", book(), today=TODAY)
        assert card.action is ActionKind.CANNOT_SAY
        assert "not held" in card.because[0]

    def test_a_stale_snapshot_blocks_the_verdict(self, store):
        old = book(as_of=now_et() - timedelta(days=POSITION_STALE_DAYS + 7))
        card = build_card(store, "CBRS", old, today=TODAY)
        assert card.action is ActionKind.CANNOT_SAY
        assert "days old" in card.because[0]

    def test_a_blocked_card_computes_nothing_further(self, store):
        """Refusing means refusing, not refusing while still doing the work."""
        store.emit(stop_event())
        old = book(as_of=now_et() - timedelta(days=30))
        card = build_card(store, "CBRS", old, today=TODAY)
        assert card.triggers == []
        assert card.claims == []

    def test_a_failed_data_check_blocks_the_verdict(self, store):
        """The advisor said it could not trust these numbers; acting on them
        anyway would make that warning decorative."""
        store.emit(
            Event(
                source=EventSource.DAEMON,
                kind="DATA_QUALITY_FAILURE",
                tier=EventTier.A,
                symbol=None,
                dedup_key="dq",
                payload={"check": "price_agreement", "failed": 1, "symbols": ["CBRS"]},
            )
        )
        card = build_card(store, "CBRS", book(), today=TODAY)
        assert card.action is ActionKind.CANNOT_SAY
        assert "price_agreement" in card.because[0]

    def test_a_failure_for_another_symbol_does_not_block_this_one(self, store):
        store.emit(
            Event(
                source=EventSource.DAEMON,
                kind="DATA_QUALITY_FAILURE",
                tier=EventTier.A,
                symbol=None,
                dedup_key="dq",
                payload={"check": "price_agreement", "failed": 1, "symbols": ["AAOI"]},
            )
        )
        assert build_card(store, "CBRS", book(), today=TODAY).action is not ActionKind.CANNOT_SAY

    def test_a_missing_factor_model_narrows_but_does_not_block(self, store):
        """CBRS has 88 bars against a 120 floor — a limitation, not a fault."""
        card = build_card(store, "CBRS", book(), today=TODAY)
        assert card.action is not ActionKind.CANNOT_SAY
        names = [g.name for g in card.evidence.gaps]
        assert "factor model" in names


class TestOnlyWrittenRulesCreateUrgency:
    """The system never decides something is bad. The user decided in advance."""

    def test_a_broken_rule_is_the_only_route_to_review_now(self, store):
        store.save_claim("CBRS", claim())
        store.emit(dilution_event(0.067))
        card = build_card(store, "CBRS", book(), today=TODAY)
        assert card.action is ActionKind.REVIEW_NOW
        assert card.urgent is True
        assert card.deadline == TODAY + timedelta(days=1)

    def test_the_broken_rule_is_quoted_with_its_number(self, store):
        store.save_claim("CBRS", claim())
        store.emit(dilution_event(0.067))
        card = build_card(store, "CBRS", book(), today=TODAY)
        assert "0.067" in card.because[0]
        assert "above 0.05" in card.because[0]

    def test_a_rule_that_holds_is_not_urgent(self, store):
        store.save_claim("CBRS", claim())
        store.emit(dilution_event(0.02))
        card = build_card(store, "CBRS", book(), today=TODAY)
        assert card.action is not ActionKind.REVIEW_NOW
        assert [c.status for c in card.claims] == ["INTACT"]

    def test_a_material_event_with_no_rule_is_review_not_review_now(self, store):
        """Without a written rule the system has no standing to be urgent."""
        store.emit(stop_event())
        card = build_card(store, "CBRS", book(), today=TODAY)
        assert card.action is ActionKind.REVIEW
        assert card.deadline is None

    def test_nothing_written_and_nothing_material_asks_for_a_thesis(self, store):
        card = build_card(store, "CBRS", book(), today=TODAY)
        assert card.action is ActionKind.WRITE_THESIS

    def test_rules_intact_and_nothing_fired_is_hold(self, store):
        store.save_claim("CBRS", claim())
        card = build_card(store, "CBRS", book(), today=TODAY)
        assert card.action is ActionKind.HOLD

    def test_an_unreachable_rule_is_reported_as_such(self, store):
        """A claim that can never fire must not read as one that held."""
        store.save_claim(
            "CBRS",
            Claim(kind=ClaimKind.RISK, text="management may be optimistic", trigger=Trigger()),
        )
        card = build_card(store, "CBRS", book(), today=TODAY)
        assert [c.status for c in card.claims] == ["UNREACHABLE"]


class TestRecency:
    def test_an_old_event_does_not_drive_todays_card(self, store):
        """Past a week it is history, informative in a story but not a reason
        to look today."""
        old = stop_event(dedup_key="old")
        old.ts = now_et() - timedelta(days=30)
        store.emit(old)
        card = build_card(store, "CBRS", book(), today=TODAY)
        assert card.triggers == []
        assert card.action is ActionKind.WRITE_THESIS

    def test_a_context_event_is_not_a_trigger(self, store):
        """Tier C is logged context and can never drive an action."""
        store.emit(
            Event(
                source=EventSource.YFINANCE,
                kind="NEWS_CONTEXT",
                tier=EventTier.C,
                symbol="CBRS",
                dedup_key="news",
                payload={"title": "A headline"},
            )
        )
        card = build_card(store, "CBRS", book(), today=TODAY)
        assert card.triggers == []


class TestWholeBook:
    def test_cards_are_ordered_by_urgency_then_weight(self, store):
        store.save_claim("CBRS", claim())
        store.emit(dilution_event(0.067))
        b = book(pos("CBRS", qty=2), pos("AAOI", qty=12, entry=129.0, price=95.0))
        cards = build_all(store, b, today=TODAY)
        assert cards[0].symbol == "CBRS"
        assert cards[0].action is ActionKind.REVIEW_NOW

    def test_every_held_symbol_gets_a_card(self, store):
        b = book(pos("CBRS"), pos("AAOI"), pos("SPCX"))
        assert {c.symbol for c in build_all(store, b, today=TODAY)} == {"CBRS", "AAOI", "SPCX"}

    def test_an_empty_book_produces_no_cards(self, store):
        assert build_all(store, BookSnapshot(net_liq=0.0), today=TODAY) == []


class TestItProposesNoTrade:
    """The constraint the module exists under."""

    def test_no_action_kind_names_a_trade(self):
        for kind in ActionKind:
            assert kind.value not in {"BUY", "SELL", "TRIM", "ADD", "EXIT"}

    def test_the_rendering_states_that_it_holds_no_view(self, store):
        from advisor.action.render import render

        store.emit(stop_event())
        card = build_card(store, "CBRS", book(), today=TODAY)
        text = render(card)
        assert "holds no view of its own" in text
        assert "proposes no trade" in text

    def test_every_reason_traces_to_a_stored_number(self, store):
        """A line that cannot carry its number does not belong."""
        store.emit(stop_event())
        card = build_card(store, "CBRS", book(), today=TODAY)
        assert any(ch.isdigit() for ch in card.because[0])
