"""Daemon API surface.

The split these tests protect: store-backed endpoints must answer without a
network round-trip so the frontend can poll them, while anything that reaches
the broker stays user-triggered. A `/status` that quietly called TastyTrade
would turn a 60-second poll into a rate-limit problem.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """An app whose daemon store is an empty temp database.

    A *fresh* store per call, exactly as production does: SQLite connections
    belong to the thread that opened them, and TestClient serves requests on a
    different thread than the test body. Sharing one connection here would
    pass in the test and mean nothing about the real app.
    """
    from advisor.api.app import create_app
    from advisor.api.routers import daemon as router

    db_path = tmp_path / "research.db"
    monkeypatch.setattr(router, "_store", lambda: DaemonStore(db_path))
    with TestClient(create_app()) as c:
        c.store = DaemonStore(db_path)  # type: ignore[attr-defined]
        yield c
        c.store.close()  # type: ignore[attr-defined]


def filing_event(**kw) -> Event:
    base = dict(
        source=EventSource.EDGAR,
        kind="FILING_DILUTION",
        tier=EventTier.A,
        symbol="AAOI",
        dedup_key="0001104659-26-099685",
        payload={
            "form": "424B5",
            "accepted_at": "2026-08-21T20:09:22+00:00",
            "offering_usd": 600_000_000.0,
            "dilution_pct": 0.067,
            "url": "https://www.sec.gov/x.htm",
        },
    )
    return Event(**{**base, **kw})


def source_item(**kw) -> SourceItem:
    base = dict(
        tier=SourceTier.PRIMARY,
        provider="SEC EDGAR",
        url="https://www.sec.gov/x.htm",
        title="424B5: APPLIED OPTOELECTRONICS, INC.",
        published_at=datetime(2026, 8, 21, 20, 9, 22, tzinfo=timezone.utc),
        entity=EntityMatch(symbol="AAOI", cik=1158114, method=MatchMethod.CIK),
        doc_type="424B5",
        accession="0001104659-26-099685",
    )
    return SourceItem(**{**base, **kw})


class TestStatus:
    def test_reports_every_scheduled_job(self, client):
        body = client.get("/api/daemon/status").json()
        names = {j["name"] for j in body["jobs"]}
        assert {
            "brief",
            "watch",
            "review",
            "macro_refresh",
            "reconcile",
            "valuation",
            "insiders",
            "heartbeat",
        } == names

    def test_works_on_a_cold_store(self, client):
        """First run: no heartbeats, no watermarks, no events — not an error."""
        body = client.get("/api/daemon/status").json()
        assert all(j["last_run_at"] is None for j in body["jobs"])
        # A watermark row exists per source from the start, carrying a null
        # timestamp — "never seen anything" rather than "source unknown".
        assert body["watermarks"]
        assert all(w["last_seen_ts"] is None for w in body["watermarks"])
        assert body["event_counts"] == {}

    def test_market_state_is_present_and_boolean(self, client):
        body = client.get("/api/daemon/status").json()
        assert isinstance(body["market_open"], bool)
        assert isinstance(body["trading_day"], bool)

    def test_timestamps_are_timezone_aware(self, client):
        """The UI renders in ET; a naive timestamp would render as browser-local."""
        body = client.get("/api/daemon/status").json()
        assert datetime.fromisoformat(body["now"]).tzinfo is not None


class TestEvents:
    def test_returns_events_newest_first(self, client):
        client.store.emit(filing_event(dedup_key="a"))
        client.store.emit(filing_event(dedup_key="b", kind="FILING_RESULTS"))
        rows = client.get("/api/daemon/events").json()["events"]
        assert len(rows) == 2

    def test_filters_by_tier(self, client):
        client.store.emit(filing_event(dedup_key="a", tier=EventTier.A))
        client.store.emit(filing_event(dedup_key="b", tier=EventTier.C))
        rows = client.get("/api/daemon/events?tier=A").json()["events"]
        assert [r["tier"] for r in rows] == ["A"]

    def test_filters_by_symbol_case_insensitively(self, client):
        client.store.emit(filing_event(dedup_key="a", symbol="AAOI"))
        client.store.emit(filing_event(dedup_key="b", symbol="CRDO"))
        rows = client.get("/api/daemon/events?symbol=crdo").json()["events"]
        assert [r["symbol"] for r in rows] == ["CRDO"]

    def test_an_invalid_tier_is_rejected_not_ignored(self, client):
        assert client.get("/api/daemon/events?tier=Z").status_code == 422

    def test_the_limit_is_bounded(self, client):
        assert client.get("/api/daemon/events?limit=99999").status_code == 422
        assert client.get("/api/daemon/events?limit=0").status_code == 422

    def test_payload_survives_the_round_trip(self, client):
        """The UI reads accepted_at and dilution_pct straight from the payload."""
        client.store.emit(filing_event())
        row = client.get("/api/daemon/events").json()["events"][0]
        assert row["payload"]["accepted_at"] == "2026-08-21T20:09:22+00:00"
        assert row["payload"]["dilution_pct"] == 0.067

    def test_an_empty_stream_returns_an_empty_list(self, client):
        assert client.get("/api/daemon/events").json()["events"] == []


class TestExposure:
    def test_returns_null_before_the_first_refresh(self, client):
        assert client.get("/api/daemon/exposure").json()["exposure"] is None

    def test_serves_the_caveat_alongside_the_numbers(self, client, tmp_path):
        """The UI must never show a loading without the interpretation warning."""
        from datetime import date

        from advisor.macro.exposure import BookExposure, FactorExposure

        client.store.save_exposure(
            BookExposure(
                asof=date(2026, 9, 4),
                net_liq=7761.13,
                covered_weight=0.78,
                uncovered=["CBRS", "SPCX"],
                factors={
                    "BREADTH": FactorExposure(
                        factor="BREADTH", net_loading=-2.17, contributors=[("AAOI", -0.58)]
                    )
                },
            )
        )
        body = client.get("/api/daemon/exposure").json()
        assert body["exposure"]["uncovered"] == ["CBRS", "SPCX"]
        assert "causal betas" in body["caveat"]

    def test_factors_come_back_ranked_by_absolute_size(self, client):
        from datetime import date

        from advisor.macro.exposure import BookExposure, FactorExposure

        client.store.save_exposure(
            BookExposure(
                asof=date(2026, 9, 4),
                net_liq=1000.0,
                covered_weight=1.0,
                factors={
                    "SMALL": FactorExposure(factor="SMALL", net_loading=0.1),
                    "BIG": FactorExposure(factor="BIG", net_loading=-3.0),
                },
            )
        )
        factors = client.get("/api/daemon/exposure").json()["exposure"]["factors"]
        assert [f["factor"] for f in factors] == ["BIG", "SMALL"]


class TestSources:
    def test_returns_provenance_for_every_item(self, client):
        client.store.save_source_item(source_item())
        row = client.get("/api/daemon/sources").json()["items"][0]
        assert row["match"] == "CIK"
        assert row["confidence"] == 1.0
        assert row["tier"] == "PRIMARY"
        assert row["accession"] == "0001104659-26-099685"

    def test_filters_by_symbol(self, client):
        client.store.save_source_item(source_item())
        client.store.save_source_item(
            source_item(
                accession="other",
                entity=EntityMatch(symbol="CRDO", method=MatchMethod.CIK),
            )
        )
        rows = client.get("/api/daemon/sources?symbol=CRDO").json()["items"]
        assert [r["symbol"] for r in rows] == ["CRDO"]

    def test_an_empty_archive_is_not_an_error(self, client):
        assert client.get("/api/daemon/sources").json()["items"] == []


class TestSymbolDetail:
    def test_an_unknown_symbol_returns_empty_sections_not_404(self, client):
        """A symbol the daemon has never seen is a normal, empty answer."""
        body = client.get("/api/daemon/symbol/ZZZZ").json()
        assert body["symbol"] == "ZZZZ"
        assert body["sensitivity"] is None
        assert body["timeline"] == []
        assert body["events"] == []

    def test_symbol_is_upcased(self, client):
        assert client.get("/api/daemon/symbol/aaoi").json()["symbol"] == "AAOI"

    def test_timeline_and_events_are_scoped_to_the_symbol(self, client):
        client.store.save_source_item(source_item())
        client.store.emit(filing_event(symbol="AAOI"))
        client.store.emit(filing_event(dedup_key="z", symbol="CRDO"))
        body = client.get("/api/daemon/symbol/AAOI").json()
        assert len(body["timeline"]) == 1
        assert {e["symbol"] for e in body["events"]} == {"AAOI"}

    def test_the_window_is_bounded(self, client):
        assert client.get("/api/daemon/symbol/AAOI?days=9999").status_code == 422


class TestJobRunner:
    def test_an_unknown_job_is_a_404(self, client):
        assert client.post("/api/daemon/run/not-a-job").status_code == 404
        assert "unknown job" in client.post("/api/daemon/run/nope").json()["detail"]


class TestCoverage:
    def test_reports_zero_on_an_empty_stream_without_dividing_by_zero(self, client):
        body = client.get("/api/daemon/coverage").json()
        assert body["divergences"] == 0
        assert body["rate"] == 0.0


class TestStoryEndpoint:
    def test_a_symbol_with_no_events_returns_an_empty_list(self, client):
        body = client.get("/api/daemon/story/ZZZZ").json()
        assert body["symbol"] == "ZZZZ"
        assert body["stories"] == []

    def test_the_limit_is_bounded(self, client):
        assert client.get("/api/daemon/story/AAOI?limit=0").status_code == 422
        assert client.get("/api/daemon/story/AAOI?limit=99").status_code == 422

    def test_context_events_do_not_become_stories(self, client):
        """Tier C is logged context; it does not merit a narrative."""
        client.store.emit(filing_event(dedup_key="c", tier=EventTier.C))
        assert client.get("/api/daemon/story/AAOI").json()["stories"] == []


class TestThesisEndpoints:
    def test_a_symbol_with_nothing_written_returns_null(self, client):
        assert client.get("/api/daemon/thesis/ZZZZ").json()["thesis"] is None

    def test_a_claim_can_be_added_and_read_back(self, client):
        created = client.post(
            "/api/daemon/thesis/AAOI/claims",
            json={
                "kind": "INVALIDATION",
                "text": "dilution above 5% breaks it",
                "event_kinds": ["FILING_DILUTION"],
                "field": "dilution_pct",
                "comparator": "ABOVE",
                "threshold": 0.05,
            },
        ).json()
        assert created["monitored"] is True
        assert "dilution_pct above 0.05" in created["trigger_description"]

        thesis = client.get("/api/daemon/thesis/AAOI").json()["thesis"]
        assert thesis["substantive"] is True
        assert thesis["claims"][0]["id"] == created["id"]

    def test_a_claim_without_a_trigger_is_recorded_but_not_monitored(self, client):
        created = client.post(
            "/api/daemon/thesis/AAOI/claims",
            json={"kind": "RISK", "text": "management may be optimistic"},
        ).json()
        assert created["monitored"] is False

    def test_coverage_counts_a_claims_only_symbol_as_written(self, client):
        client.post(
            "/api/daemon/thesis/AAOI/claims",
            json={"kind": "RISK", "text": "x"},
        )
        assert client.get("/api/daemon/thesis/AAOI").json()["thesis"]["substantive"] is True

    def test_an_unknown_claim_kind_is_rejected(self, client):
        r = client.post(
            "/api/daemon/thesis/AAOI/claims",
            json={"kind": "NONSENSE", "text": "x"},
        )
        assert r.status_code == 422

    def test_a_claim_can_be_deleted(self, client):
        created = client.post(
            "/api/daemon/thesis/AAOI/claims", json={"kind": "RISK", "text": "x"}
        ).json()
        assert client.delete(f"/api/daemon/thesis/claims/{created['id']}").status_code == 200
        assert client.get("/api/daemon/thesis/AAOI").json()["thesis"] is None

    def test_deleting_an_unknown_claim_is_a_404(self, client):
        assert client.delete("/api/daemon/thesis/claims/nope").status_code == 404

    def test_the_story_slot_reflects_a_claim_added_through_the_api(self, client):
        """End to end: write a claim, then see the event break it."""
        client.post(
            "/api/daemon/thesis/AAOI/claims",
            json={
                "kind": "INVALIDATION",
                "text": "dilution above 5% breaks it",
                "event_kinds": ["FILING_DILUTION"],
                "field": "dilution_pct",
                "comparator": "ABOVE",
                "threshold": 0.05,
            },
        )
        client.store.emit(filing_event())
        stories = client.get("/api/daemon/story/AAOI").json()["stories"]
        assert stories
        tripped = [e for e in stories[0]["thesis"]["evaluations"] if e["tripped"]]
        assert tripped and tripped[0]["observed"] == 0.067


class TestEventSummary:
    """The detail line is served, not recomputed in the browser.

    The frontend had its own copy of this logic and it had drifted: it covered
    offerings and factor shocks but not crossings, concentration, insider
    clusters or news headlines. A stop breach rendered without its numbers and
    eight news items rendered as eight blank rows, in a page that had shipped
    weeks earlier looking correct.
    """

    def test_every_event_carries_a_summary_field(self, client):
        client.store.emit(filing_event())
        row = client.get("/api/daemon/events").json()["events"][0]
        assert "summary" in row

    def test_the_summary_carries_the_numbers(self, client):
        client.store.emit(filing_event())
        row = client.get("/api/daemon/events").json()["events"][0]
        assert "$600M" in row["summary"]
        assert "6.7%" in row["summary"]

    def test_it_matches_the_cli(self, client):
        """One implementation, two readers — the drift this replaces."""
        from advisor.daemon.summarize import summarize

        event = filing_event()
        client.store.emit(event)
        row = client.get("/api/daemon/events").json()["events"][0]
        assert row["summary"] == summarize(event)

    def test_an_empty_payload_yields_an_empty_string_not_null(self, client):
        """The frontend treats it as a string; null would render as 'null'."""
        client.store.emit(filing_event(dedup_key="bare", payload={}))
        rows = client.get("/api/daemon/events").json()["events"]
        assert all(isinstance(r["summary"], str) for r in rows)


class TestCombinedFilters:
    """The Signals page filters by tier and ticker at once.

    The API accepted both from the day it shipped and the page exposed only
    tier, so the one question you actually ask — "what has happened to this
    name?" — could not be asked in the browser.
    """

    def test_tier_and_symbol_narrow_together(self, client):
        client.store.emit(filing_event(dedup_key="a", symbol="CBRS", tier=EventTier.A))
        client.store.emit(filing_event(dedup_key="b", symbol="CBRS", tier=EventTier.C))
        client.store.emit(filing_event(dedup_key="c", symbol="AAOI", tier=EventTier.A))

        rows = client.get("/api/daemon/events?tier=A&symbol=CBRS").json()["events"]
        assert len(rows) == 1
        assert rows[0]["symbol"] == "CBRS"
        assert rows[0]["tier"] == "A"

    def test_a_symbol_with_no_events_at_that_tier_is_empty_not_unfiltered(self, client):
        """The dangerous failure: a filter that silently does nothing."""
        client.store.emit(filing_event(symbol="CBRS", tier=EventTier.C))
        assert client.get("/api/daemon/events?tier=A&symbol=CBRS").json()["events"] == []

    def test_book_level_events_are_excluded_by_a_symbol_filter(self, client):
        """A factor shock carries no symbol; the page says so in the footer."""
        client.store.emit(filing_event(dedup_key="book", symbol=None, tier=EventTier.A))
        client.store.emit(filing_event(dedup_key="named", symbol="CBRS", tier=EventTier.A))
        rows = client.get("/api/daemon/events?symbol=CBRS").json()["events"]
        assert [r["symbol"] for r in rows] == ["CBRS"]

    def test_the_unfiltered_read_still_carries_every_symbol(self, client):
        """The chip row is built from this; filtering it would shrink itself."""
        client.store.emit(filing_event(dedup_key="a", symbol="CBRS"))
        client.store.emit(filing_event(dedup_key="b", symbol="AAOI"))
        rows = client.get("/api/daemon/events").json()["events"]
        assert {r["symbol"] for r in rows} == {"CBRS", "AAOI"}
