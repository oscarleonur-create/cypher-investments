"""Shift detection and claim reachability."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from advisor.daemon.models import EventTier
from advisor.daemon.store import DaemonStore
from advisor.macro.sensitivity import FactorLoading, SymbolSensitivity
from advisor.thesis.models import Claim, ClaimKind, Comparator, Trigger
from advisor.thesis.reachability import claim_reachability
from advisor.valuation.implied import build_snapshot
from advisor.valuation.ingest import MATERIAL_CAGR_SHIFT, shift_event
from advisor.valuation.models import Fundamentals

BASE = dict(
    symbol="SPCX",
    source_accession="acc",
    period_end=date(2026, 6, 30),
    period_start=date(2026, 4, 1),
    fiscal_period="Q",
    revenue=7_814_000_000.0,
    cash=93_522_000_000.0,
    marketable_securities=6_487_000_000.0,
    total_debt=39_364_000_000.0,
    shares_outstanding=13_181_779_945.0,
)


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def snap(price: float, asof=date(2026, 9, 13)):
    return build_snapshot(Fundamentals(**BASE), price, asof=asof)


class TestShiftDetection:
    def test_the_first_reading_is_not_news(self):
        """Nothing to compare against; a baseline is not an event."""
        assert shift_event(None, snap(151.21)) is None

    def test_a_drifting_price_does_not_fire(self):
        assert shift_event(snap(151.21), snap(152.50)) is None

    def test_a_material_move_fires_tier_b(self):
        event = shift_event(snap(120.0), snap(200.0))
        assert event is not None
        assert event.tier is EventTier.B  # a valuation is not a deadline
        assert event.symbol == "SPCX"

    def test_the_event_carries_both_readings_and_the_assumptions(self):
        event = shift_event(snap(120.0), snap(200.0))
        payload = event.payload
        assert payload["previous_implied_cagr"] < payload["implied_cagr"]
        assert payload["direction"] == "harder"
        assert payload["terminal_multiple"] and payload["fcf_margin"] and payload["years"]
        assert payload["as_of_filing"] == "acc"

    def test_a_falling_price_reads_as_easier(self):
        event = shift_event(snap(200.0), snap(120.0))
        assert event.payload["direction"] == "easier"
        assert event.payload["change"] < 0

    def test_the_threshold_is_in_cagr_points_not_price(self):
        """A 30% price move on a cheap stock may move required growth little."""
        before, after = snap(120.0), snap(200.0)
        delta = abs(after.base_case().implied_cagr - before.base_case().implied_cagr)
        assert delta >= MATERIAL_CAGR_SHIFT

    def test_a_snapshot_with_no_scenarios_does_not_fire(self):
        empty = snap(151.21)
        empty.scenarios = []
        assert shift_event(snap(120.0), empty) is None

    def test_the_dedup_key_is_symbol_and_day(self, store):
        a = shift_event(snap(120.0), snap(200.0))
        b = shift_event(snap(120.0), snap(200.0))
        assert store.emit(a) is True
        assert store.emit(b) is False


class TestPersistence:
    def test_a_snapshot_round_trips(self, store):
        store.save_valuation(snap(151.21))
        loaded = store.load_latest_valuation("SPCX")
        assert loaded.price == pytest.approx(151.21)
        assert loaded.base_case().implied_cagr == pytest.approx(0.258, abs=0.005)

    def test_before_returns_the_prior_day_not_todays(self, store):
        store.save_valuation(snap(120.0, asof=date(2026, 9, 12)))
        store.save_valuation(snap(200.0, asof=date(2026, 9, 13)))
        prior = store.load_latest_valuation("SPCX", before=date(2026, 9, 13))
        assert prior.price == pytest.approx(120.0)

    def test_saving_twice_on_one_day_replaces(self, store):
        store.save_valuation(snap(120.0))
        store.save_valuation(snap(200.0))
        assert len(store.valuation_history("SPCX")) == 1

    def test_an_unvalued_symbol_returns_none(self, store):
        assert store.load_latest_valuation("NOPE") is None


class TestReachability:
    """A claim monitored in principle and unreachable in practice is a lie."""

    def divergence_claim(self) -> Claim:
        return Claim(
            id="c1",
            kind=ClaimKind.KPI,
            text="a move macro cannot explain",
            trigger=Trigger(
                event_kinds=["RESIDUAL_DIVERGENCE"],
                field="residual_z",
                comparator=Comparator.ABOVE,
                threshold=2.0,
            ),
        )

    def test_the_real_spcx_case_is_blocked(self, store):
        """59 sessions of history, a 120-session floor, so no estimate exists."""
        result = claim_reachability(store, "SPCX", self.divergence_claim())
        assert result.blocked is True
        assert "no factor estimate" in result.reason

    def test_the_same_claim_is_reachable_once_an_estimate_exists(self, store):
        store.save_sensitivity(
            SymbolSensitivity(
                symbol="SPCX",
                asof=date(2026, 9, 4),
                window_days=250,
                n_obs=250,
                r2=0.3,
                resid_vol=0.05,
                loadings=[FactorLoading(factor="MKT", loading=1.0, tstat=3.0)],
            )
        )
        assert claim_reachability(store, "SPCX", self.divergence_claim()).reachable is True

    def test_a_claim_with_no_trigger_is_blocked_with_a_plain_reason(self, store):
        claim = Claim(id="c2", kind=ClaimKind.RISK, text="litigation", trigger=Trigger())
        result = claim_reachability(store, "SPCX", claim)
        assert result.blocked is True
        assert "nothing will ever check this" in result.reason

    def test_a_filing_claim_is_reachable_without_any_stored_state(self, store):
        claim = Claim(
            id="c3",
            kind=ClaimKind.INVALIDATION,
            text="dilution",
            trigger=Trigger(
                event_kinds=["FILING_DILUTION"],
                field="dilution_pct",
                comparator=Comparator.ABOVE,
                threshold=0.05,
            ),
        )
        assert claim_reachability(store, "SPCX", claim).reachable is True

    def test_a_field_the_event_never_carries_is_blocked(self, store):
        """A quieter version of the same failure."""
        claim = Claim(
            id="c4",
            kind=ClaimKind.KPI,
            text="gross margin",
            trigger=Trigger(
                event_kinds=["FILING_DILUTION"],
                field="gross_margin",
                comparator=Comparator.BELOW,
                threshold=0.55,
            ),
        )
        result = claim_reachability(store, "SPCX", claim)
        assert result.blocked is True
        assert "does not carry a 'gross_margin' field" in result.reason

    def test_a_valuation_claim_needs_a_stored_valuation(self, store):
        claim = Claim(
            id="c5",
            kind=ClaimKind.INVALIDATION,
            text="required growth above 25%",
            trigger=Trigger(
                event_kinds=["IMPLIED_EXPECTATIONS_SHIFT"],
                field="implied_cagr",
                comparator=Comparator.ABOVE,
                threshold=0.25,
            ),
        )
        assert claim_reachability(store, "SPCX", claim).blocked is True
        store.save_valuation(snap(151.21))
        assert claim_reachability(store, "SPCX", claim).reachable is True

    def test_blocked_claims_are_excluded_from_the_monitored_count(self, store):
        from advisor.thesis.repo import load_thesis

        store.save_claim("SPCX", self.divergence_claim())
        thesis = load_thesis(store, "SPCX")
        assert len(thesis.claims) == 1
        assert thesis.monitored_claims == []
        assert thesis.blocked
