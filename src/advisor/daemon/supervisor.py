"""The supervised asyncio loop.

Contract: a job that raises must never take the daemon down with it. Failures
are caught, stamped on the job's heartbeat, and the loop continues — a broken
EDGAR poller must not stop the roll-window checks.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from datetime import time

from advisor.daemon import handlers
from advisor.daemon import market_calendar as mc
from advisor.daemon.jobs import DailyAt, EveryMinutes, Job, JobContext, JobRegistry
from advisor.daemon.models import JobResult
from advisor.daemon.store import DaemonStore

logger = logging.getLogger(__name__)

# How often the loop wakes to ask "is anything due?". Independent of any job's
# own cadence — jobs decide against wall clock, this only bounds latency.
TICK_SECONDS = 30


def build_registry() -> JobRegistry:
    """The daemon's schedule: a pre-open brief, an intraday sweep, a close review."""
    reg = JobRegistry()
    reg.register(
        Job(
            name="brief",
            trigger=DailyAt(time(7, 0), grace_hours=6.0),
            handler=handlers.run_brief,
            description="pre-open brief with overnight backfill",
        )
    )
    reg.register(
        Job(
            name="watch",
            trigger=EveryMinutes(15, during_session_only=True),
            handler=handlers.run_watch,
            description="intraday sweep, silent unless actionable",
        )
    )
    reg.register(
        Job(
            name="review",
            trigger=DailyAt(time(16, 30), grace_hours=6.0),
            handler=handlers.run_review,
            description="post-close review",
        )
    )
    reg.register(
        Job(
            name="heartbeat",
            trigger=EveryMinutes(5, during_session_only=False),
            handler=handlers.run_heartbeat,
            description="liveness tick",
        )
    )
    return reg


class Supervisor:
    """Runs the registry until stopped."""

    def __init__(
        self,
        store: DaemonStore,
        registry: JobRegistry | None = None,
        *,
        tick_seconds: int = TICK_SECONDS,
    ) -> None:
        self.store = store
        # `is None`, not `or`: JobRegistry defines __len__, so an empty one is
        # falsy and `or` would silently swap it for the production schedule.
        self.registry = build_registry() if registry is None else registry
        self.tick_seconds = tick_seconds
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def run_job(self, job: Job, *, catch_up: bool = False) -> JobResult:
        """Execute one job, recording success or failure on its heartbeat."""
        now = mc.now_et()
        ctx = JobContext(store=self.store, now=now, catch_up=catch_up)
        started = now
        try:
            result = await job.handler(ctx)
        except Exception as exc:  # noqa: BLE001 — the whole point is not to die
            logger.exception("job %s failed", job.name)
            self.store.record_run(job.name, ok=False, error=f"{type(exc).__name__}: {exc}")
            return JobResult(job=job.name, ok=False, detail=str(exc))

        # Attribute the result to the job that ran, not to whatever the
        # handler named itself — the registry is the source of truth.
        result.job = job.name
        result.duration_ms = int((mc.now_et() - started).total_seconds() * 1000)
        self.store.record_run(job.name, ok=result.ok, error="" if result.ok else result.detail)
        return result

    async def tick(self) -> list[JobResult]:
        """Run everything due right now. Returns their results."""
        now = mc.now_et()
        results = []
        for job in self.registry.due(now, self.store):
            # A DailyAt job firing well after its slot is a catch-up run.
            last_ok = self.store.get_heartbeat(job.name).last_ok_at
            catch_up = isinstance(job.trigger, DailyAt) and last_ok is not None
            results.append(await self.run_job(job, catch_up=catch_up))
        return results

    async def run(self) -> None:
        """Loop until stopped, ticking every ``tick_seconds``."""
        logger.info(
            "daemon up — %d jobs, tick %ds, market %s",
            len(self.registry),
            self.tick_seconds,
            "open" if mc.is_market_open() else "closed",
        )
        while not self._stop.is_set():
            try:
                await self.tick()
            except Exception:  # noqa: BLE001 — a bad tick must not end the loop
                logger.exception("tick failed")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self.tick_seconds)
        logger.info("daemon stopped")


async def serve(store: DaemonStore, *, tick_seconds: int = TICK_SECONDS) -> None:
    """Run a supervisor until SIGINT/SIGTERM."""
    sup = Supervisor(store, tick_seconds=tick_seconds)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, sup.request_stop)
    await sup.run()
