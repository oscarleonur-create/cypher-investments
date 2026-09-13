"""Evaluating claims against events.

The rule that shapes all of this: a claim untested by an event is not a claim
that passed. "Dilution above 5%" is not satisfied because an earnings filing
arrived — it is simply untouched by it, and reporting otherwise would let a
thesis look checked when nothing checked it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.macro.sensitivity import FactorLoading, SymbolSensitivity
from advisor.thesis.match import evaluate_claim, evaluate_thesis, macro_conflicts
from advisor.thesis.models import (
    Claim,
    ClaimKind,
    Comparator,
    StructuredThesis,
    Trigger,
)


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def dilution(pct=0.067) -> Event:
    return Event(
        source=EventSource.EDGAR,
        kind="FILING_DILUTION",
        tier=EventTier.A,
        symbol="AAOI",
        dedup_key="d1",
        payload={"dilution_pct": pct, "offering_usd": 6e8, "form": "424B5"},
    )


def claim(**kw) -> Claim:
    base = dict(
        id="c1",
        kind=ClaimKind.INVALIDATION,
        text="Any equity raise above 5% of market cap breaks the story",
        trigger=Trigger(
            event_kinds=["FILING_DILUTION"],
            field="dilution_pct",
            comparator=Comparator.ABOVE,
            threshold=0.05,
        ),
    )
    return Claim(**{**base, **kw})


class TestThresholds:
    def test_the_real_aaoi_offering_breaks_the_real_claim(self):
        """6.7% against a 5% invalidation."""
        result = evaluate_claim(claim(), dilution(0.067))
        assert result.tripped is True
        assert result.observed == pytest.approx(0.067)
        assert "above 0.05" in result.note

    def test_a_smaller_raise_does_not_trip_it(self):
        assert evaluate_claim(claim(), dilution(0.02)).tripped is False

    def test_exactly_at_the_threshold_does_not_trip_an_above_claim(self):
        """'Above 5%' means above, not at."""
        assert evaluate_claim(claim(), dilution(0.05)).tripped is False

    def test_a_below_comparator_inverts_the_test(self):
        c = claim(
            trigger=Trigger(
                event_kinds=["FILING_DILUTION"],
                field="dilution_pct",
                comparator=Comparator.BELOW,
                threshold=0.05,
            )
        )
        assert evaluate_claim(c, dilution(0.02)).tripped is True
        assert evaluate_claim(c, dilution(0.067)).tripped is False


class TestApplicability:
    def test_an_unrelated_event_returns_none_not_a_pass(self):
        """The distinction the module exists for."""
        earnings = Event(
            source=EventSource.EDGAR,
            kind="FILING_RESULTS",
            tier=EventTier.A,
            symbol="AAOI",
            dedup_key="r1",
            payload={},
        )
        assert evaluate_claim(claim(), earnings) is None

    def test_a_claim_with_no_trigger_is_never_evaluated(self):
        c = claim(kind=ClaimKind.DRIVER, text="Management is honest", trigger=Trigger())
        assert evaluate_claim(c, dilution()) is None
        assert c.monitored is False

    def test_an_occurrence_claim_trips_on_the_event_alone(self):
        """An auditor resigning needs no threshold."""
        c = claim(
            text="A restatement ends this",
            trigger=Trigger(event_kinds=["FILING_RESTATEMENT"]),
        )
        event = Event(
            source=EventSource.EDGAR,
            kind="FILING_RESTATEMENT",
            tier=EventTier.A,
            symbol="AAOI",
            dedup_key="x",
            payload={},
        )
        assert evaluate_claim(c, event).tripped is True

    def test_a_claim_listing_several_kinds_matches_any_of_them(self):
        c = claim(trigger=Trigger(event_kinds=["FILING_RESTATEMENT", "FILING_AUDITOR_CHANGE"]))
        for kind in ("FILING_RESTATEMENT", "FILING_AUDITOR_CHANGE"):
            event = Event(
                source=EventSource.EDGAR,
                kind=kind,
                tier=EventTier.A,
                symbol="AAOI",
                dedup_key=kind,
                payload={},
            )
            assert evaluate_claim(c, event).tripped is True


class TestMissingData:
    def test_a_matching_event_without_the_field_does_not_trip(self):
        """An unsized offering must not be reported as breaking a 5% rule."""
        unsized = Event(
            source=EventSource.EDGAR,
            kind="FILING_DILUTION",
            tier=EventTier.A,
            symbol="AAOI",
            dedup_key="u",
            payload={"form": "424B5"},
        )
        result = evaluate_claim(claim(), unsized)
        assert result.tripped is False
        assert "no readable dilution_pct" in result.note

    def test_a_non_numeric_field_is_treated_as_absent(self):
        broken = Event(
            source=EventSource.EDGAR,
            kind="FILING_DILUTION",
            tier=EventTier.A,
            symbol="AAOI",
            dedup_key="b",
            payload={"dilution_pct": "a lot"},
        )
        assert evaluate_claim(claim(), broken).tripped is False

    def test_an_empty_payload_does_not_raise(self):
        bare = Event(
            source=EventSource.EDGAR,
            kind="FILING_DILUTION",
            tier=EventTier.A,
            symbol="AAOI",
            dedup_key="e",
            payload={},
        )
        assert evaluate_claim(claim(), bare) is not None


class TestThesisLevel:
    def thesis(self, *claims) -> StructuredThesis:
        return StructuredThesis(symbol="AAOI", title="AAOI", claims=list(claims))

    def test_tripped_invalidations_sort_first(self):
        driver = claim(
            id="d",
            kind=ClaimKind.DRIVER,
            text="demand outruns capacity",
            trigger=Trigger(event_kinds=["FILING_DILUTION"]),
        )
        results = evaluate_thesis(self.thesis(driver, claim()), dilution(0.067))
        assert results[0].kind is ClaimKind.INVALIDATION

    def test_untested_claims_are_absent_from_the_results(self):
        untestable = claim(id="u", kind=ClaimKind.RISK, text="China ban", trigger=Trigger())
        results = evaluate_thesis(self.thesis(untestable, claim()), dilution())
        assert [r.claim_id for r in results] == ["c1"]

    def test_a_thesis_with_no_claims_evaluates_to_nothing(self):
        assert evaluate_thesis(self.thesis(), dilution()) == []

    def test_coverage_reports_the_testable_share(self):
        t = self.thesis(claim(), claim(id="u", text="vibes", trigger=Trigger()))
        assert t.coverage == pytest.approx(0.5)

    def test_coverage_of_an_empty_thesis_does_not_divide_by_zero(self):
        assert self.thesis().coverage == 0.0


class TestMacroDrivers:
    def sens(self, breadth: float) -> SymbolSensitivity:
        from datetime import date

        return SymbolSensitivity(
            symbol="AAOI",
            asof=date(2026, 9, 4),
            window_days=250,
            n_obs=250,
            r2=0.27,
            resid_vol=0.0753,
            loadings=[FactorLoading(factor="BREADTH", loading=breadth, tstat=-2.1)],
        )

    def macro_claim(self, comparator=Comparator.ABOVE) -> Claim:
        return claim(
            id="m",
            kind=ClaimKind.MACRO_DRIVER,
            text="needs broadening markets",
            trigger=Trigger(factor="BREADTH", comparator=comparator),
        )

    def test_a_contradicting_loading_is_reported(self):
        """The thesis wants broadening markets; the book is short breadth."""
        t = StructuredThesis(symbol="AAOI", title="x", claims=[self.macro_claim()])
        result = macro_conflicts(t, self.sens(-3.55))[0]
        assert result.tripped is True
        assert "the measured loading is -3.55" in result.note

    def test_an_agreeing_loading_is_not_a_conflict(self):
        t = StructuredThesis(symbol="AAOI", title="x", claims=[self.macro_claim()])
        assert macro_conflicts(t, self.sens(2.0))[0].tripped is False

    def test_a_loading_near_zero_is_not_called_a_contradiction(self):
        """Noise around zero is not disagreement."""
        t = StructuredThesis(symbol="AAOI", title="x", claims=[self.macro_claim()])
        assert macro_conflicts(t, self.sens(-0.1))[0].tripped is False

    def test_without_a_sensitivity_nothing_is_claimed(self):
        t = StructuredThesis(symbol="AAOI", title="x", claims=[self.macro_claim()])
        assert macro_conflicts(t, None) == []


class TestPersistence:
    def test_a_saved_claim_comes_back_with_its_trigger(self, store):
        claim_id = store.save_claim("aaoi", claim(id=None))
        loaded = store.load_claims("AAOI")
        assert len(loaded) == 1
        assert loaded[0].id == claim_id
        assert loaded[0].trigger.threshold == 0.05

    def test_symbols_are_stored_upper_case(self, store):
        store.save_claim("aaoi", claim(id=None))
        assert store.symbols_with_claims() == ["AAOI"]

    def test_saving_the_same_id_twice_replaces_rather_than_duplicates(self, store):
        store.save_claim("AAOI", claim())
        store.save_claim("AAOI", claim(text="revised"))
        loaded = store.load_claims("AAOI")
        assert len(loaded) == 1
        assert loaded[0].text == "revised"

    def test_a_claim_can_be_deleted(self, store):
        claim_id = store.save_claim("AAOI", claim(id=None))
        assert store.delete_claim(claim_id) is True
        assert store.load_claims("AAOI") == []

    def test_deleting_an_unknown_claim_reports_false(self, store):
        assert store.delete_claim("nope") is False

    def test_another_symbols_claims_are_not_returned(self, store):
        store.save_claim("AAOI", claim(id="a"))
        store.save_claim("CRDO", claim(id="b"))
        assert [c.id for c in store.load_claims("AAOI")] == ["a"]
