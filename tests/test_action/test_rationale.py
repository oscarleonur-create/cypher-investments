"""The reasoning chain, and where a proposed action is allowed to come from.

One rule governs this module: there are exactly two sources for an action —
a response the user wrote down in advance, or arithmetic. There is no third,
because a third would be an opinion the system is not entitled to hold.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from advisor.action.assemble import build_card
from advisor.action.models import ActionKind
from advisor.action.rationale import ActionSource, Bearing, build_rationale
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


def pos(symbol="AAOI", *, qty=12, entry=129.31, price=104.90) -> Position:
    return Position(
        account="A",
        symbol=symbol,
        underlying=symbol,
        instrument=EQUITY,
        quantity=qty,
        multiplier=1,
        avg_open_price=entry,
        close_price=price,
    )


def book(*positions, net_liq=7_890.0) -> BookSnapshot:
    return BookSnapshot(as_of=now_et(), positions=list(positions) or [pos()], net_liq=net_liq)


def dilution(pct=0.067) -> Event:
    return Event(
        source=EventSource.EDGAR,
        kind="FILING_DILUTION",
        tier=EventTier.A,
        symbol="AAOI",
        dedup_key="dil",
        payload={"dilution_pct": pct, "offering_usd": 6e8, "form": "424B5"},
    )


def claim(response="") -> Claim:
    return Claim(
        kind=ClaimKind.INVALIDATION,
        text="An equity raise above 5% of market cap breaks this",
        trigger=Trigger(
            event_kinds=["FILING_DILUTION"],
            field="dilution_pct",
            comparator=Comparator.ABOVE,
            threshold=0.05,
        ),
        response=response,
    )


def rationale_for(store, symbol="AAOI", b=None):
    b = b or book()
    card = build_card(store, symbol, b, today=TODAY)
    return build_rationale(store, symbol, b, card), card


class TestTheUsersOwnDecision:
    """The whole point: return the decision to the person who made it."""

    def test_a_written_response_is_quoted_back_verbatim(self, store):
        store.save_claim("AAOI", claim("exit half before the next open"))
        store.emit(dilution())
        rationale, _ = rationale_for(store)
        assert rationale.proposed.source is ActionSource.YOUR_RULE
        assert rationale.proposed.text == "exit half before the next open"

    def test_it_names_which_rule_it_came_from(self, store):
        store.save_claim("AAOI", claim("exit half"))
        store.emit(dilution())
        rationale, _ = rationale_for(store)
        assert "equity raise above 5%" in rationale.proposed.quoted_from

    def test_a_fraction_in_the_response_is_turned_into_shares(self, store):
        store.save_claim("AAOI", claim("exit half before the next open"))
        store.emit(dilution())
        rationale, _ = rationale_for(store)
        assert "6 shares" in rationale.proposed.size

    @pytest.mark.parametrize(
        "phrase,expected", [("salir de la mitad", "6 shares"), ("sell all of it", "12 shares")]
    )
    def test_fractions_are_read_in_either_language(self, store, phrase, expected):
        store.save_claim("AAOI", claim(phrase))
        store.emit(dilution())
        rationale, _ = rationale_for(store)
        assert expected in rationale.proposed.size

    @pytest.mark.parametrize(
        "phrase",
        [
            "call the IR line and reassess",  # contains "all" — the real trap
            "reassess the thesis",
            "wait for the next quarter",
        ],
    )
    def test_a_response_with_no_fraction_gets_no_invented_quantity(self, store, phrase):
        """Putting a number in the user's mouth is the one forbidden thing."""
        store.save_claim("AAOI", claim(phrase))
        store.emit(dilution())
        rationale, _ = rationale_for(store)
        assert rationale.proposed.text == phrase
        assert rationale.proposed.size == ""

    def test_a_broken_rule_with_no_response_says_so_rather_than_inventing_one(self, store):
        store.save_claim("AAOI", claim(""))
        store.emit(dilution())
        rationale, _ = rationale_for(store)
        assert rationale.proposed.source is ActionSource.NONE
        assert "--then" in rationale.proposed.text

    def test_a_rule_that_held_proposes_nothing_from_it(self, store):
        store.save_claim("AAOI", claim("exit half"))
        store.emit(dilution(0.02))
        rationale, _ = rationale_for(store)
        proposed = rationale.proposed
        assert proposed is None or proposed.source is not ActionSource.YOUR_RULE


class TestArithmeticOnly:
    def test_a_position_over_the_limit_proposes_the_trim_that_returns_to_it(self, store):
        """22.1% against a 20% limit: arithmetic, not judgement."""
        big = book(pos("SPCX", qty=11, entry=128.16, price=152.64), net_liq=7_890.0)
        rationale, card = rationale_for(store, "SPCX", big)
        assert rationale.proposed.source is ActionSource.ARITHMETIC
        assert "20%" in rationale.proposed.text
        assert "share" in rationale.proposed.size

    def test_a_position_inside_the_limit_proposes_no_trim(self, store):
        small = book(pos("AAOI", qty=2, entry=129.31, price=104.90), net_liq=7_890.0)
        rationale, _ = rationale_for(store, "AAOI", small)
        assert (
            rationale.proposed is None or rationale.proposed.source is not ActionSource.ARITHMETIC
        )

    def test_a_users_rule_outranks_arithmetic(self, store):
        """What you decided beats what the limit implies."""
        store.save_claim("SPCX", claim("exit half"))
        store.emit(
            Event(
                source=EventSource.EDGAR,
                kind="FILING_DILUTION",
                tier=EventTier.A,
                symbol="SPCX",
                dedup_key="d",
                payload={"dilution_pct": 0.09},
            )
        )
        big = book(pos("SPCX", qty=11, entry=128.16, price=152.64))
        rationale, _ = rationale_for(store, "SPCX", big)
        assert rationale.proposed.source is ActionSource.YOUR_RULE


class TestThereIsNoThirdSource:
    def test_only_three_sources_exist(self):
        assert {s.value for s in ActionSource} == {"YOUR_RULE", "ARITHMETIC", "NONE"}

    def test_a_blocked_card_proposes_nothing_at_all(self, store):
        rationale, card = rationale_for(store, "NVDA")
        assert card.action is ActionKind.CANNOT_SAY
        assert rationale.proposed is None

    def test_material_events_alone_do_not_produce_an_action(self, store):
        """Three stop breaches and an insider cluster, no written rule: the
        system still has nothing of its own to propose."""
        store.emit(
            Event(
                source=EventSource.COMPUTED,
                kind="STOP_BREACHED",
                tier=EventTier.A,
                symbol="AAOI",
                dedup_key="s",
                payload={"entry": 129.31, "price": 104.90, "unrealized_pct": -0.189},
            )
        )
        rationale, card = rationale_for(store)
        assert card.action is ActionKind.REVIEW
        assert rationale.proposed is None or rationale.proposed.source is ActionSource.NONE


class TestTheChain:
    def test_every_step_carries_a_fact(self, store):
        store.emit(dilution())
        rationale, _ = rationale_for(store)
        assert all(step.fact.strip() for step in rationale.steps)

    def test_a_missing_factor_model_is_marked_blind_not_neutral(self, store):
        """ "Cannot see" and "nothing there" must not render alike."""
        rationale, _ = rationale_for(store, "CBRS", book(pos("CBRS")))
        blind = [s for s in rationale.steps if s.bearing is Bearing.BLIND]
        assert any("no factor estimate" in s.fact for s in blind)

    def test_no_rules_written_is_a_blind_spot_not_a_pass(self, store):
        rationale, _ = rationale_for(store)
        assert any(
            s.bearing is Bearing.BLIND and "nothing written" in s.fact for s in rationale.steps
        )

    def test_a_broken_rule_bears_against(self, store):
        store.save_claim("AAOI", claim("exit half"))
        store.emit(dilution())
        rationale, _ = rationale_for(store)
        assert any(
            s.label == "your rule broke" and s.bearing is Bearing.AGAINST for s in rationale.steps
        )

    def test_a_standing_condition_appears_once_not_once_per_observation(self, store):
        """Four concentration warnings in a week are one fact observed four
        times; repeating it buries what only happened once."""
        for i in range(4):
            e = Event(
                source=EventSource.COMPUTED,
                kind="CONCENTRATION_WARNING",
                tier=EventTier.B,
                symbol="AAOI",
                dedup_key=f"c{i}",
                payload={"weight": 0.22, "threshold": 0.20},
            )
            e.ts = now_et() - timedelta(hours=i)
            store.emit(e)
        rationale, _ = rationale_for(store)
        labels = [s.label for s in rationale.steps]
        assert labels.count("concentration warning") == 1
