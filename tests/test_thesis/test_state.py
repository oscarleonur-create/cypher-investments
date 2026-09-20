"""Testing a claim against what is true now, not only against what happened.

The gap this closes, found live: the user wrote "required growth must not
exceed 25% a year for a decade" on SPCX. The stored valuation says 25.8%. No
`IMPLIED_EXPECTATIONS_SHIFT` had fired because the number had not *moved* two
points — the event stream is deliberately edge-triggered — so the card
reported "nothing this week tested it" about a rule being violated the whole
time.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.daemon.store import DaemonStore
from advisor.thesis.models import Claim, ClaimKind, Comparator, Trigger
from advisor.thesis.state import evaluate_against_state
from advisor.valuation.implied import build_snapshot
from advisor.valuation.models import Fundamentals


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def pos(symbol="SPCX", *, qty=11, entry=128.16, price=152.64) -> Position:
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
    return BookSnapshot(positions=list(positions) or [pos()], net_liq=net_liq)


def claim(*, kinds, field, threshold, comparator=Comparator.ABOVE) -> Claim:
    return Claim(
        id="c1",
        kind=ClaimKind.INVALIDATION,
        text="a rule",
        trigger=Trigger(event_kinds=kinds, field=field, comparator=comparator, threshold=threshold),
    )


def spcx_valuation(store, price=152.64):
    """The real SPCX figures from its 10-Q."""
    fundamentals = Fundamentals(
        symbol="SPCX",
        source_accession="0001628280-26-052535",
        period_end=date(2026, 6, 30),
        period_start=date(2026, 4, 1),
        fiscal_period="Q",
        revenue=7_814_000_000.0,
        cash=93_522_000_000.0,
        marketable_securities=6_487_000_000.0,
        total_debt=39_364_000_000.0,
        shares_outstanding=13_181_779_945.0,
    )
    store.save_valuation(build_snapshot(fundamentals, price))


class TestTheRealGap:
    def test_the_spcx_rule_is_reported_as_violated_right_now(self, store):
        """25.8% against a 25% rule, with no event having fired."""
        spcx_valuation(store)
        result = evaluate_against_state(
            store,
            "SPCX",
            claim(kinds=["IMPLIED_EXPECTATIONS_SHIFT"], field="implied_cagr", threshold=0.25),
            book(),
        )
        assert result.tripped is True
        assert result.observed == pytest.approx(0.258, abs=0.005)
        assert "right now" in result.note

    def test_a_rule_the_state_satisfies_is_not_tripped(self, store):
        spcx_valuation(store)
        result = evaluate_against_state(
            store,
            "SPCX",
            claim(kinds=["IMPLIED_EXPECTATIONS_SHIFT"], field="implied_cagr", threshold=0.35),
            book(),
        )
        assert result.tripped is False

    def test_exactly_at_the_threshold_does_not_trip_an_above_rule(self, store):
        spcx_valuation(store)
        base = store.load_latest_valuation("SPCX").base_case().implied_cagr
        result = evaluate_against_state(
            store,
            "SPCX",
            claim(kinds=["IMPLIED_EXPECTATIONS_SHIFT"], field="implied_cagr", threshold=base),
            book(),
        )
        assert result.tripped is False


class TestPositionState:
    def test_concentration_is_read_from_the_current_book(self, store):
        result = evaluate_against_state(
            store,
            "SPCX",
            claim(kinds=["CONCENTRATION_WARNING"], field="weight", threshold=0.20),
            book(),
        )
        assert result.tripped is True
        assert result.observed == pytest.approx(0.213, abs=0.005)

    def test_a_drawdown_rule_reads_the_current_loss(self, store):
        """AAOI's real position: -18.9%, which is not past a -20% rule."""
        result = evaluate_against_state(
            store,
            "AAOI",
            claim(
                kinds=["DEEP_DRAWDOWN"],
                field="unrealized_pct",
                threshold=-0.20,
                comparator=Comparator.BELOW,
            ),
            book(pos("AAOI", qty=12, entry=129.31, price=104.90)),
        )
        assert result.observed == pytest.approx(-0.1888, abs=0.001)
        assert result.tripped is False, "-18.9% is not below -20%"

    def test_the_same_position_trips_a_rule_it_is_actually_past(self, store):
        result = evaluate_against_state(
            store,
            "AAOI",
            claim(
                kinds=["DEEP_DRAWDOWN"],
                field="unrealized_pct",
                threshold=-0.15,
                comparator=Comparator.BELOW,
            ),
            book(pos("AAOI", qty=12, entry=129.31, price=104.90)),
        )
        assert result.tripped is True

    def test_a_symbol_not_held_has_no_standing_reading(self, store):
        assert (
            evaluate_against_state(
                store,
                "NVDA",
                claim(kinds=["CONCENTRATION_WARNING"], field="weight", threshold=0.20),
                book(),
            )
            is None
        )


class TestNotEveryClaimHasAStandingForm:
    """Dilution is inherently an event: there is no "current dilution"."""

    def test_a_dilution_rule_has_no_standing_reading(self, store):
        assert (
            evaluate_against_state(
                store,
                "SPCX",
                claim(kinds=["FILING_DILUTION"], field="dilution_pct", threshold=0.05),
                book(),
            )
            is None
        )

    def test_a_residual_has_no_standing_reading(self, store):
        """A residual is one session's surprise, never a state."""
        assert (
            evaluate_against_state(
                store,
                "SPCX",
                claim(kinds=["RESIDUAL_DIVERGENCE"], field="residual_z", threshold=2.0),
                book(),
            )
            is None
        )

    def test_a_claim_with_no_threshold_has_none(self, store):
        c = Claim(
            id="c",
            kind=ClaimKind.INVALIDATION,
            text="x",
            trigger=Trigger(event_kinds=["FILING_RESTATEMENT"]),
        )
        assert evaluate_against_state(store, "SPCX", c, book()) is None

    def test_a_claim_whose_kinds_disagree_about_the_quantity_says_nothing(self, store):
        """Two kinds reading different fields cannot produce one state."""
        c = claim(
            kinds=["CONCENTRATION_WARNING", "IMPLIED_EXPECTATIONS_SHIFT"],
            field="weight",
            threshold=0.20,
        )
        assert evaluate_against_state(store, "SPCX", c, book()) is None

    def test_a_field_the_kind_does_not_carry_says_nothing(self, store):
        c = claim(kinds=["CONCENTRATION_WARNING"], field="implied_cagr", threshold=0.25)
        assert evaluate_against_state(store, "SPCX", c, book()) is None

    def test_a_missing_valuation_yields_no_reading_rather_than_a_pass(self, store):
        assert (
            evaluate_against_state(
                store,
                "SPCX",
                claim(kinds=["IMPLIED_EXPECTATIONS_SHIFT"], field="implied_cagr", threshold=0.25),
                book(),
            )
            is None
        )


class TestStandingIsNotUrgent:
    """A standing violation would be urgent every day for as long as it lasts,
    which is how an alert channel stops being read."""

    def test_a_standing_violation_reads_as_review_not_review_now(self, store):
        from advisor.action.assemble import build_card
        from advisor.action.models import ActionKind

        spcx_valuation(store)
        store.save_claim(
            "SPCX",
            claim(kinds=["IMPLIED_EXPECTATIONS_SHIFT"], field="implied_cagr", threshold=0.25),
        )
        card = build_card(store, "SPCX", book(), today=date(2026, 9, 20))
        assert card.action is ActionKind.REVIEW
        assert card.deadline is None

    def test_the_claim_is_marked_standing_not_broken(self, store):
        from advisor.action.assemble import build_card

        spcx_valuation(store)
        store.save_claim(
            "SPCX",
            claim(kinds=["IMPLIED_EXPECTATIONS_SHIFT"], field="implied_cagr", threshold=0.25),
        )
        card = build_card(store, "SPCX", book(), today=date(2026, 9, 20))
        assert [c.status for c in card.claims] == ["STANDING"]

    def test_a_standing_violation_still_returns_a_written_response(self, store):
        """It is still the user's rule, still met — it simply is not news."""
        from advisor.action.assemble import build_card
        from advisor.action.rationale import ActionSource, build_rationale

        spcx_valuation(store)
        c = claim(kinds=["IMPLIED_EXPECTATIONS_SHIFT"], field="implied_cagr", threshold=0.25)
        c.response = "trim a quarter and re-read the filing"
        store.save_claim("SPCX", c)
        b = book()
        card = build_card(store, "SPCX", b, today=date(2026, 9, 20))
        rationale = build_rationale(store, "SPCX", b, card)
        assert rationale.proposed.source is ActionSource.YOUR_RULE
        assert "trim a quarter" in rationale.proposed.text
