"""Supervisor behaviour — above all, that a bad job cannot kill the loop."""

from __future__ import annotations

from datetime import time
from pathlib import Path

import pytest
from advisor.daemon.jobs import DailyAt, EveryMinutes, Job, JobContext, JobRegistry
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
        assert {j.name for j in reg} == {"brief", "watch", "review", "heartbeat"}

    def test_brief_and_review_are_daily_watch_is_intraday(self):
        by_name = {j.name: j.trigger for j in build_registry()}
        assert isinstance(by_name["brief"], DailyAt)
        assert by_name["brief"].at == time(7, 0)
        assert isinstance(by_name["review"], DailyAt)
        assert by_name["review"].at == time(16, 30)
        assert isinstance(by_name["watch"], EveryMinutes)
        assert by_name["watch"].minutes == 15


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
