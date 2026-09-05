"""Event stream, watermarks and heartbeats."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def make_event(**kw) -> Event:
    base = dict(
        source=EventSource.EDGAR,
        kind="8K_FILED",
        tier=EventTier.B,
        symbol="NVDA",
        dedup_key="0001045810-26-000123",
    )
    return Event(**{**base, **kw})


class TestDedup:
    def test_first_emit_is_new(self, store):
        assert store.emit(make_event()) is True

    def test_same_fact_twice_is_a_noop(self, store):
        store.emit(make_event())
        assert store.emit(make_event()) is False
        assert len(store.recent_events()) == 1

    def test_dedup_survives_a_different_timestamp(self, store):
        """Polling the same filing an hour later must not duplicate it."""
        store.emit(make_event())
        later = make_event(ts=datetime.now() + timedelta(hours=1))
        assert store.emit(later) is False

    def test_distinct_facts_both_land(self, store):
        store.emit(make_event())
        store.emit(make_event(dedup_key="0001045810-26-000999"))
        assert len(store.recent_events()) == 2

    def test_emit_many_counts_only_new(self, store):
        events = [make_event(), make_event(), make_event(dedup_key="other")]
        assert store.emit_many(events) == 2


class TestQuerying:
    def test_filters_by_tier_and_symbol(self, store):
        store.emit(make_event(tier=EventTier.A, symbol="AAPL", dedup_key="a"))
        store.emit(make_event(tier=EventTier.C, symbol="NVDA", dedup_key="b"))
        assert len(store.recent_events(tier=EventTier.A)) == 1
        assert len(store.recent_events(symbol="AAPL")) == 1
        assert len(store.recent_events(symbol="aapl")) == 1  # case-insensitive

    def test_counts_by_tier(self, store):
        store.emit(make_event(tier=EventTier.A, dedup_key="a"))
        store.emit(make_event(tier=EventTier.A, dedup_key="b"))
        store.emit(make_event(tier=EventTier.C, dedup_key="c"))
        assert store.event_counts_by_tier() == {"A": 2, "C": 1}

    def test_book_level_events_have_no_symbol(self, store):
        store.emit(
            make_event(source=EventSource.MACRO, kind="REGIME_BREAK", symbol=None, dedup_key="r")
        )
        assert store.recent_events()[0].symbol is None

    def test_payload_round_trips(self, store):
        store.emit(make_event(payload={"dte": 21, "strike": 145.0, "tags": ["roll"]}))
        assert store.recent_events()[0].payload == {
            "dte": 21,
            "strike": 145.0,
            "tags": ["roll"],
        }


class TestWatermarks:
    def test_missing_watermark_is_empty_not_an_error(self, store):
        wm = store.get_watermark(EventSource.EDGAR)
        assert wm.last_seen_ts is None
        assert wm.last_seen_cursor == ""

    def test_round_trip(self, store):
        ts = datetime(2026, 9, 4, 16, 40)
        store.set_watermark(EventSource.EDGAR, last_seen_ts=ts, last_seen_cursor="acc-123")
        wm = store.get_watermark(EventSource.EDGAR)
        assert wm.last_seen_ts == ts
        assert wm.last_seen_cursor == "acc-123"

    def test_sources_are_independent(self, store):
        store.set_watermark(EventSource.EDGAR, last_seen_ts=datetime(2026, 9, 4, 10, 0))
        assert store.get_watermark(EventSource.BROKER).last_seen_ts is None


class TestHeartbeats:
    def test_unknown_job_starts_at_zero(self, store):
        hb = store.get_heartbeat("brief")
        assert hb.run_count == 0 and hb.last_run_at is None

    def test_success_sets_both_stamps(self, store):
        hb = store.record_run("brief", ok=True)
        assert hb.run_count == 1
        assert hb.last_ok_at is not None
        assert hb.error_count == 0

    def test_failure_stamps_run_but_preserves_last_ok(self, store):
        """`status` must be able to say 'failing since' — so a failure never
        clears the last successful timestamp."""
        ok = store.record_run("brief", ok=True)
        last_ok = ok.last_ok_at
        hb = store.record_run("brief", ok=False, error="boom")
        assert hb.error_count == 1
        assert hb.last_error == "boom"
        assert hb.last_ok_at == last_ok

    def test_success_clears_previous_error(self, store):
        store.record_run("brief", ok=False, error="boom")
        hb = store.record_run("brief", ok=True)
        assert hb.last_error == ""
        assert hb.error_count == 1  # historical count is kept

    def test_long_errors_are_truncated(self, store):
        hb = store.record_run("brief", ok=False, error="x" * 5000)
        assert len(hb.last_error) == 500


class TestTimezoneCorrectness:
    """Regression: the daemon reasons in ET, so it must persist ET-aware stamps.

    With naive local timestamps on a machine that is not on ET, a freshly
    written heartbeat reads back as hours old and every interval job re-fires
    on every tick — the 15-minute sweep would run every 30 seconds.
    """

    def test_heartbeat_stamps_are_timezone_aware(self, store):
        hb = store.record_run("brief", ok=True)
        assert hb.last_run_at.tzinfo is not None
        assert hb.last_ok_at.tzinfo is not None

    def test_heartbeat_reads_back_as_just_now_in_et(self, store):
        from advisor.daemon.market_calendar import now_et

        store.record_run("brief", ok=True)
        age = now_et() - store.get_heartbeat("brief").last_run_at
        assert abs(age.total_seconds()) < 60

    def test_watermark_stamps_survive_the_round_trip_with_tz(self, store):
        from advisor.daemon.market_calendar import now_et

        ts = now_et()
        store.set_watermark(EventSource.BROKER, last_seen_ts=ts)
        assert store.get_watermark(EventSource.BROKER).last_seen_ts == ts

    def test_event_default_timestamp_is_aware(self):
        assert make_event().ts.tzinfo is not None


class TestStoreRobustness:
    def test_creates_a_missing_parent_directory(self, tmp_path: Path):
        """First run on a fresh machine has no data/ directory yet."""
        s = DaemonStore(tmp_path / "nested" / "deeper" / "research.db")
        s.record_run("brief", ok=True)
        assert s.get_heartbeat("brief").run_count == 1
        s.close()

    def test_reopening_preserves_state(self, tmp_path: Path):
        """The daemon restarts; heartbeats and watermarks must survive."""
        path = tmp_path / "research.db"
        first = DaemonStore(path)
        first.emit(make_event())
        first.record_run("brief", ok=True)
        first.close()

        second = DaemonStore(path)
        assert second.get_heartbeat("brief").run_count == 1
        assert len(second.recent_events()) == 1
        # And the dedup index still rejects the same fact after a restart.
        assert second.emit(make_event()) is False
        second.close()

    def test_recent_events_on_an_empty_stream(self, store):
        assert store.recent_events() == []
        assert store.event_counts_by_tier() == {}
