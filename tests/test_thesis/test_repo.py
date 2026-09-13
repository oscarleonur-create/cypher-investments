"""Loading a thesis: the three states that must not collapse into one."""

from __future__ import annotations

from pathlib import Path

import pytest
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.thesis.models import Claim, ClaimKind, Comparator, Trigger
from advisor.thesis.repo import ThesisReadError, load_thesis
from advisor.thesis.template import TEMPLATE

REAL = (
    "AAOI fabricates its own lasers, which is the constraint competitors cannot "
    "buy their way around inside this cycle. The AI datacentre build-out pulls "
    "800G demand forward faster than transceiver capacity can be added, and that "
    "gap is where the margin is."
)


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    s._conn.execute(
        "CREATE TABLE theses (symbol TEXT, title TEXT, content TEXT, "
        "conviction TEXT, status TEXT, updated_at TEXT)"
    )
    yield s
    s.close()


def insert(store, symbol, content, title="A thesis"):
    store._conn.execute(
        "INSERT INTO theses VALUES (?, ?, ?, 'MEDIUM', 'DRAFT', '2026-09-01')",
        (symbol, title, content),
    )


def claim() -> Claim:
    return Claim(
        kind=ClaimKind.INVALIDATION,
        text="dilution above 5%",
        trigger=Trigger(
            event_kinds=["FILING_DILUTION"],
            field="dilution_pct",
            comparator=Comparator.ABOVE,
            threshold=0.05,
        ),
    )


class TestThreeStates:
    def test_nothing_written_at_all_returns_none(self, store):
        assert load_thesis(store, "AAOI") is None

    def test_an_untouched_template_returns_a_thesis_marked_empty(self, store):
        """The CRDO case: a row exists, a view does not."""
        insert(store, "CRDO", TEMPLATE)
        thesis = load_thesis(store, "CRDO")
        assert thesis is not None
        assert thesis.substantive is False
        assert "still the blank template" in thesis.prose_note

    def test_a_written_thesis_is_substantive(self, store):
        insert(store, "AAOI", TEMPLATE + "\n\n" + REAL)
        thesis = load_thesis(store, "AAOI")
        assert thesis.substantive is True
        assert "fabricates its own lasers" in thesis.prose_note


class TestClaimsAsContent:
    def test_claims_alone_make_a_thesis_without_any_prose(self, store):
        """Testable claims are worth more than an essay; they count."""
        store.save_claim("AAOI", claim())
        thesis = load_thesis(store, "AAOI")
        assert thesis is not None
        assert thesis.substantive is True
        assert len(thesis.claims) == 1

    def test_claims_rescue_an_otherwise_blank_template(self, store):
        insert(store, "CRDO", TEMPLATE)
        store.save_claim("CRDO", claim())
        assert load_thesis(store, "CRDO").substantive is True

    def test_monitored_count_excludes_untestable_claims(self, store):
        store.save_claim("AAOI", claim())
        store.save_claim("AAOI", Claim(kind=ClaimKind.RISK, text="vibes", trigger=Trigger()))
        thesis = load_thesis(store, "AAOI")
        assert len(thesis.claims) == 2
        assert len(thesis.monitored_claims) == 1

    def test_a_symbol_is_matched_case_insensitively(self, store):
        insert(store, "AAOI", TEMPLATE + REAL)
        assert load_thesis(store, "aaoi") is not None


class TestFailureModes:
    def test_a_missing_theses_table_is_not_an_error(self, tmp_path):
        """A fresh database has no theses table; claims still work."""
        s = DaemonStore(tmp_path / "fresh.db")
        try:
            assert load_thesis(s, "AAOI") is None
            s.save_claim("AAOI", claim())
            assert load_thesis(s, "AAOI").substantive is True
        finally:
            s.close()

    def test_a_malformed_theses_table_raises_rather_than_reading_as_empty(self, tmp_path):
        """'Unreadable' must not be mistaken for 'no opinion held'."""
        s = DaemonStore(tmp_path / "broken.db")
        try:
            s._conn.execute("CREATE TABLE theses (wrong_column TEXT)")
            with pytest.raises(ThesisReadError):
                load_thesis(s, "AAOI")
        finally:
            s.close()


class TestStoryIntegration:
    def test_the_story_slot_reports_a_template_as_not_written(self, store):
        from advisor.story.assemble import _thesis

        insert(store, "CRDO", TEMPLATE)
        event = Event(
            source=EventSource.EDGAR,
            kind="FILING_DILUTION",
            tier=EventTier.A,
            symbol="CRDO",
            dedup_key="x",
            payload={"dilution_pct": 0.067},
        )
        slot = _thesis(store, "CRDO", event)
        assert slot.exists is True
        assert slot.substantive is False

    def test_the_story_slot_carries_the_broken_claim(self, store):
        from advisor.story.assemble import _thesis

        store.save_claim("AAOI", claim())
        event = Event(
            source=EventSource.EDGAR,
            kind="FILING_DILUTION",
            tier=EventTier.A,
            symbol="AAOI",
            dedup_key="y",
            payload={"dilution_pct": 0.067},
        )
        slot = _thesis(store, "AAOI", event)
        assert slot.invalidated is True
        assert slot.tripped[0]["observed"] == pytest.approx(0.067)

    def test_a_read_failure_becomes_an_unavailable_slot(self, tmp_path):
        from advisor.story.assemble import _thesis
        from advisor.story.models import Confidence

        s = DaemonStore(tmp_path / "broken2.db")
        try:
            s._conn.execute("CREATE TABLE theses (wrong_column TEXT)")
            event = Event(
                source=EventSource.EDGAR,
                kind="FILING_DILUTION",
                tier=EventTier.A,
                symbol="AAOI",
                dedup_key="z",
                payload={},
            )
            assert _thesis(s, "AAOI", event).confidence is Confidence.UNAVAILABLE
        finally:
            s.close()
