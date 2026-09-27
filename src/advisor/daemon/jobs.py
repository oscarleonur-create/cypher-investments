"""Job definitions and scheduling triggers.

Scheduling is decided against wall-clock ET, never elapsed time, because the
host laptop sleeps. A ``DailyAt`` job that was due at 07:00 while the lid was
shut is still due at 09:12 when it opens — that is what turns "the daemon was
off" into "here is what you missed", and it is why triggers take ``last_run``
rather than counting ticks.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Protocol

from advisor.daemon import market_calendar as mc
from advisor.daemon.models import JobResult
from advisor.daemon.store import DaemonStore


@dataclass
class JobContext:
    """What a job is handed when it runs."""

    store: DaemonStore
    now: datetime  # market-timezone wall clock
    catch_up: bool = False  # True when firing late for a missed slot


class Trigger(Protocol):
    """Decides whether a job is due."""

    def is_due(self, now: datetime, last_run: datetime | None) -> bool: ...

    def last_due(self, now: datetime) -> datetime | None:
        """The latest moment at or before ``now`` by which a run should have succeeded.

        A job whose last success is older is late. None when nothing was due
        yet. Slack for a slow tick is included, so a job running on schedule
        is never reported late.
        """
        ...

    def describe(self) -> str: ...


# A slot that fired this long ago without a success is late, not just running.
LATE_AFTER = timedelta(minutes=15)


def _back_days(now: datetime, days: int = 10):
    """``now``'s date and the ones before it, newest first."""
    d = mc.to_et(now).date()
    for _ in range(days):
        yield d
        d -= timedelta(days=1)


@dataclass(frozen=True)
class DailyAt:
    """Fire once per trading day at a wall-clock time, catching up if missed.

    ``grace_hours`` bounds how late a missed slot may still fire: the 07:00
    brief opening the laptop at 09:00 is useful, at 23:00 it is noise. Past
    days are never replayed — only today's slot can be caught up.
    """

    at: time
    trading_days_only: bool = True
    grace_hours: float = 6.0

    def is_due(self, now: datetime, last_run: datetime | None) -> bool:
        if self.trading_days_only and not mc.is_trading_day(now.date()):
            return False
        slot = datetime.combine(now.date(), self.at, tzinfo=now.tzinfo)
        if now < slot:
            return False
        if now - slot > timedelta(hours=self.grace_hours):
            return False
        return last_run is None or mc.to_et(last_run) < slot

    def last_due(self, now: datetime) -> datetime | None:
        now = mc.to_et(now)
        for d in _back_days(now):
            if self.trading_days_only and not mc.is_trading_day(d):
                continue
            slot = datetime.combine(d, self.at, tzinfo=mc.MARKET_TZ)
            if slot + LATE_AFTER <= now:
                return slot
        return None

    def describe(self) -> str:
        scope = "trading days" if self.trading_days_only else "every day"
        return f"daily at {self.at.strftime('%H:%M')} ET ({scope})"


@dataclass(frozen=True)
class WindowEvery:
    """Fire every ``minutes`` inside a daily wall-clock window, trading days only.

    For work that only makes sense in a slice of the day — the premarket scan
    runs 08:00-09:25 and has nothing to say after the bell. A laptop that
    wakes at 09:10 still gets one run in the window; one that wakes at 10:00
    gets none, because a premarket scan at 10:00 would be a session scan
    with the wrong entry price.
    """

    start: time
    end: time
    minutes: int

    def is_due(self, now: datetime, last_run: datetime | None) -> bool:
        et = mc.to_et(now)
        if not mc.is_trading_day(et.date()):
            return False
        if not (self.start <= et.time() < self.end):
            return False
        if last_run is None:
            return True
        last = mc.to_et(last_run)
        if last.date() != et.date():
            return True
        return et - last >= timedelta(minutes=self.minutes)

    def last_due(self, now: datetime) -> datetime | None:
        """One run inside the latest window that has had time for one."""
        now = mc.to_et(now)
        for d in _back_days(now):
            if not mc.is_trading_day(d):
                continue
            start = datetime.combine(d, self.start, tzinfo=mc.MARKET_TZ)
            if start + LATE_AFTER <= now:
                return start
        return None

    def describe(self) -> str:
        return (
            f"every {self.minutes}m {self.start.strftime('%H:%M')}-"
            f"{self.end.strftime('%H:%M')} ET (trading days)"
        )


@dataclass(frozen=True)
class EveryMinutes:
    """Fire on an interval, optionally only while the market is open."""

    minutes: int
    during_session_only: bool = True
    # Earliest wall-clock time the job may fire each day. The scanner waits
    # for the opening print to settle; firing at 09:30 and then counting the
    # interval from there would push its first real look to 10:00.
    not_before: time | None = None

    def is_due(self, now: datetime, last_run: datetime | None) -> bool:
        if self.during_session_only and not mc.is_market_open(now):
            return False
        if self.not_before is not None and mc.to_et(now).time() < self.not_before:
            return False
        if last_run is None:
            return True
        return now - mc.to_et(last_run) >= timedelta(minutes=self.minutes)

    def last_due(self, now: datetime) -> datetime | None:
        """Within two intervals of now in session (or always), else by the last session's end."""
        now = mc.to_et(now)
        slack = timedelta(minutes=2 * self.minutes) + LATE_AFTER
        if not self.during_session_only:
            return now - slack
        if mc.is_market_open(now):
            first = self.not_before or mc.REGULAR_OPEN
            if now - datetime.combine(now.date(), first, tzinfo=now.tzinfo) >= slack:
                return now - slack
        for d in _back_days(now):
            if not mc.is_trading_day(d):
                continue
            close = datetime.combine(d, mc.session_close(d), tzinfo=mc.MARKET_TZ)
            if close <= now:
                return close - timedelta(minutes=2 * self.minutes)
        return None

    def describe(self) -> str:
        scope = "during session" if self.during_session_only else "always"
        if self.not_before is not None:
            scope += f", from {self.not_before.strftime('%H:%M')} ET"
        return f"every {self.minutes}m ({scope})"


@dataclass(frozen=True)
class AtLeastEvery:
    """Fire when this much time has passed, not on a fixed weekday.

    Used for the weekly sensitivity refresh. A calendar weekday would be
    fragile on a laptop that sleeps — miss Sunday and the estimate goes stale
    for a fortnight. "At least every 7 days, after 06:00" always catches up on
    the next day the machine is awake.
    """

    days: int
    after: time = time(6, 0)

    def is_due(self, now: datetime, last_run: datetime | None) -> bool:
        if now.time() < self.after:
            return False
        if last_run is None:
            return True
        return now - mc.to_et(last_run) >= timedelta(days=self.days)

    def last_due(self, now: datetime) -> datetime | None:
        """A run within ``days`` plus one day of slack for a laptop asleep."""
        return mc.to_et(now) - timedelta(days=self.days + 1)

    def describe(self) -> str:
        return f"every {self.days}d after {self.after.strftime('%H:%M')} ET"


Handler = Callable[[JobContext], Awaitable[JobResult]]


@dataclass
class Job:
    """A named unit of scheduled work."""

    name: str
    trigger: Trigger
    handler: Handler
    description: str = ""


@dataclass
class JobRegistry:
    """The daemon's job table."""

    jobs: list[Job] = field(default_factory=list)

    def register(self, job: Job) -> Job:
        if any(j.name == job.name for j in self.jobs):
            raise ValueError(f"duplicate job name: {job.name}")
        self.jobs.append(job)
        return job

    def due(self, now: datetime, store: DaemonStore) -> list[Job]:
        """Jobs whose trigger fires at ``now``, given their recorded heartbeat."""
        out = []
        for job in self.jobs:
            last_run = store.get_heartbeat(job.name).last_run_at
            if job.trigger.is_due(now, last_run):
                out.append(job)
        return out

    def __iter__(self):
        return iter(self.jobs)

    def __len__(self) -> int:
        return len(self.jobs)
