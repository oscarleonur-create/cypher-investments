"""Supervisor behaviour — above all, that a bad job cannot kill the loop."""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import pytest
from advisor.daemon.jobs import (
    AtLeastEvery,
    DailyAt,
    EveryMinutes,
    Job,
    JobContext,
    JobRegistry,
)
from advisor.daemon.market_calendar import MARKET_TZ
from advisor.daemon.models import JobResult
from advisor.daemon.store import DaemonStore
from advisor.daemon.supervisor import Supervisor, build_registry


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


async def ok_handler(ctx: JobContext) -> JobResult:
    return JobResult(job="ok", ok=True, detail="fine")


async def boom_handler(ctx: JobContext) -> JobResult:
    raise RuntimeError("ingest exploded")


class TestRegistry:
    def test_rejects_duplicate_names(self):
        reg = JobRegistry()
        reg.register(Job("a", EveryMinutes(1, False), ok_handler))
        with pytest.raises(ValueError, match="duplicate"):
            reg.register(Job("a", EveryMinutes(1, False), ok_handler))

    def test_default_registry_has_the_designed_schedule(self):
        reg = build_registry()
        assert {j.name for j in reg} == {
            "brief",
            "watch",
            "review",
            "macro_refresh",
            "reconcile",
            "valuation",
            "insiders",
            "scan",
            "scan_outcomes",
            "heartbeat",
        }

    def test_reconcile_runs_before_the_brief_reads_anything(self):
        """Checking the inputs after advising on them would be pointless."""
        by_name = {j.name: j.trigger for j in build_registry()}
        assert by_name["reconcile"].at < by_name["brief"].at

    def test_brief_and_review_are_daily_watch_is_intraday(self):
        by_name = {j.name: j.trigger for j in build_registry()}
        assert isinstance(by_name["brief"], DailyAt)
        assert by_name["brief"].at == time(7, 0)
        assert isinstance(by_name["review"], DailyAt)
        assert by_name["review"].at == time(16, 30)
        assert isinstance(by_name["watch"], EveryMinutes)
        assert by_name["watch"].minutes == 15

    def test_the_insider_scan_is_weekly_not_in_the_brief(self):
        """Each Form 4 is a fetch and an XML parse; a dozen names is several
        hundred round trips, which timed the morning brief out."""
        trigger = {j.name: j.trigger for j in build_registry()}["insiders"]
        assert isinstance(trigger, AtLeastEvery)
        assert trigger.days == 7

    def test_valuation_refreshes_weekly_not_daily(self):
        """Its inputs are a quarterly filing and a price; a daily refit would
        burn an XBRL parse per symbol to produce the same answer."""
        trigger = {j.name: j.trigger for j in build_registry()}["valuation"]
        assert isinstance(trigger, AtLeastEvery)
        assert trigger.days == 7

    def test_sensitivities_refresh_weekly_not_daily(self):
        """Loadings barely move in five sessions; a daily refit is wasted work."""
        trigger = {j.name: j.trigger for j in build_registry()}["macro_refresh"]
        assert isinstance(trigger, AtLeastEvery)
        assert trigger.days == 7


class TestCrashIsolation:
    async def test_a_raising_job_is_recorded_not_propagated(self, store):
        sup = Supervisor(store, JobRegistry())
        job = Job("boom", EveryMinutes(1, False), boom_handler)
        result = await sup.run_job(job)

        assert result.ok is False
        hb = store.get_heartbeat("boom")
        assert hb.error_count == 1
        assert "ingest exploded" in hb.last_error

    async def test_one_failing_job_does_not_stop_the_others(self, store):
        """The contract: a broken EDGAR poller must not stop roll-window checks."""
        reg = JobRegistry()
        reg.register(Job("boom", EveryMinutes(1, False), boom_handler))
        reg.register(Job("fine", EveryMinutes(1, False), ok_handler))
        sup = Supervisor(store, reg)

        results = await sup.tick()

        assert {r.job: r.ok for r in results} == {"boom": False, "fine": True}
        assert store.get_heartbeat("fine").last_ok_at is not None


class TestTick:
    async def test_tick_runs_only_due_jobs(self, store):
        reg = JobRegistry()
        # 07:00 daily won't be due unless the test happens to run in the window,
        # so pin the assertion on a job that is never due: a past-grace slot.
        reg.register(Job("never", DailyAt(time(0, 0), grace_hours=0.0), ok_handler))
        reg.register(Job("always", EveryMinutes(1, during_session_only=False), ok_handler))
        sup = Supervisor(store, reg)

        results = await sup.tick()

        assert [r.job for r in results] == ["always"]

    async def test_interval_job_does_not_refire_within_its_window(self, store):
        reg = JobRegistry()
        reg.register(Job("tick", EveryMinutes(60, during_session_only=False), ok_handler))
        sup = Supervisor(store, reg)

        first = await sup.tick()
        second = await sup.tick()

        assert len(first) == 1
        assert second == []

    async def test_run_job_records_duration(self, store):
        sup = Supervisor(store, JobRegistry())
        result = await sup.run_job(Job("ok", EveryMinutes(1, False), ok_handler))
        assert result.duration_ms >= 0
        assert store.get_heartbeat("ok").run_count == 1


async def soft_fail_handler(ctx: JobContext) -> JobResult:
    """Fails by returning ok=False rather than raising."""
    return JobResult(job="soft", ok=False, detail="upstream returned nothing")


class TestFailureModes:
    async def test_soft_failure_is_recorded_like_a_raise(self, store):
        """A handler that returns ok=False must count as an error, not a run."""
        sup = Supervisor(store, JobRegistry())
        result = await sup.run_job(Job("soft", EveryMinutes(1, False), soft_fail_handler))

        assert result.ok is False
        hb = store.get_heartbeat("soft")
        assert hb.error_count == 1
        assert hb.last_ok_at is None
        assert "upstream returned nothing" in hb.last_error

    async def test_a_recovering_job_clears_its_error(self, store):
        sup = Supervisor(store, JobRegistry())
        await sup.run_job(Job("flaky", EveryMinutes(1, False), boom_handler))
        await sup.run_job(Job("flaky", EveryMinutes(1, False), ok_handler))

        hb = store.get_heartbeat("flaky")
        assert hb.last_error == ""
        assert hb.last_ok_at is not None
        assert hb.error_count == 1  # history preserved

    async def test_empty_registry_ticks_without_error(self, store):
        assert await Supervisor(store, JobRegistry()).tick() == []


class TestCatchUpDetection:
    """A punctual job announced itself as late, every single morning.

    The old test was `isinstance(trigger, DailyAt) and it has run before`,
    which is true of every daily job after its first run. Three days of live
    logs read "fired late for a missed slot" while the daemon was perfectly
    on time — which makes the one line that would matter, an actual catch-up
    after the laptop slept, invisible.
    """

    def _brief(self) -> Job:
        return Job("brief", DailyAt(time(7, 0), grace_hours=6.0), ok_handler)

    def test_firing_on_the_slot_is_not_a_catch_up(self):
        """07:00:16 for a 07:00 slot — the real log line that was wrong."""
        now = datetime(2026, 9, 18, 7, 0, 16, tzinfo=MARKET_TZ)
        assert Supervisor._is_catch_up(self._brief(), now) is False

    def test_a_few_minutes_late_is_still_on_time(self):
        now = datetime(2026, 9, 18, 7, 8, tzinfo=MARKET_TZ)
        assert Supervisor._is_catch_up(self._brief(), now) is False

    def test_hours_late_is_a_catch_up(self):
        """The laptop slept through the slot and woke at noon."""
        now = datetime(2026, 9, 18, 12, 0, tzinfo=MARKET_TZ)
        assert Supervisor._is_catch_up(self._brief(), now) is True

    def test_exactly_at_the_boundary_is_not_yet_a_catch_up(self):
        now = datetime(2026, 9, 18, 7, 15, tzinfo=MARKET_TZ)
        assert Supervisor._is_catch_up(self._brief(), now) is False

    def test_one_minute_past_the_boundary_is(self):
        now = datetime(2026, 9, 18, 7, 16, tzinfo=MARKET_TZ)
        assert Supervisor._is_catch_up(self._brief(), now) is True

    def test_an_interval_job_is_never_a_catch_up(self):
        """Only a slot can be missed; an interval simply runs again."""
        job = Job("watch", EveryMinutes(15, during_session_only=True), ok_handler)
        now = datetime(2026, 9, 18, 12, 0, tzinfo=MARKET_TZ)
        assert Supervisor._is_catch_up(job, now) is False

    def test_the_afternoon_review_uses_its_own_slot(self):
        """16:30, not the brief's 07:00."""
        review = Job("review", DailyAt(time(16, 30), grace_hours=6.0), ok_handler)
        assert (
            Supervisor._is_catch_up(review, datetime(2026, 9, 18, 16, 31, tzinfo=MARKET_TZ))
            is False
        )
        assert (
            Supervisor._is_catch_up(review, datetime(2026, 9, 18, 19, 0, tzinfo=MARKET_TZ)) is True
        )
