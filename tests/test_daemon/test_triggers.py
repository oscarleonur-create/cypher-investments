"""Trigger semantics, especially the laptop-sleep catch-up behaviour."""

from __future__ import annotations

from datetime import datetime, time

from advisor.daemon import market_calendar as mc
from advisor.daemon.jobs import AtLeastEvery, DailyAt, EveryMinutes


def et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=mc.MARKET_TZ)


class TestDailyAt:
    trigger = DailyAt(time(7, 0), grace_hours=6.0)

    def test_not_due_before_the_slot(self):
        assert not self.trigger.is_due(et(2026, 9, 4, 6, 59), None)

    def test_due_at_the_slot_when_never_run(self):
        assert self.trigger.is_due(et(2026, 9, 4, 7, 0), None)

    def test_catches_up_when_the_laptop_was_asleep(self):
        """Lid opens at 09:12; the 07:00 brief is still owed."""
        last_run = et(2026, 9, 3, 7, 0)  # ran yesterday
        assert self.trigger.is_due(et(2026, 9, 4, 9, 12), last_run)

    def test_does_not_fire_twice_in_one_day(self):
        already = et(2026, 9, 4, 7, 0)
        assert not self.trigger.is_due(et(2026, 9, 4, 9, 12), already)

    def test_grace_window_expires(self):
        """Opening the laptop at 23:00 should not fire the morning brief."""
        assert not self.trigger.is_due(et(2026, 9, 4, 23, 0), None)

    def test_skips_non_trading_days(self):
        assert not self.trigger.is_due(et(2026, 9, 5, 7, 0), None)  # Saturday
        assert not self.trigger.is_due(et(2026, 9, 7, 7, 0), None)  # Labor Day

    def test_past_days_are_never_replayed(self):
        """A slot missed entirely on Monday does not fire on Tuesday morning."""
        trigger = DailyAt(time(16, 30), grace_hours=6.0)
        # Tuesday 07:00 is before Tuesday's 16:30 slot, so nothing is due
        # even though Monday's slot was never run.
        assert not trigger.is_due(et(2026, 9, 8, 7, 0), et(2026, 9, 4, 16, 30))


class TestEveryMinutes:
    trigger = EveryMinutes(15, during_session_only=True)

    def test_due_when_never_run_during_session(self):
        assert self.trigger.is_due(et(2026, 9, 4, 10, 0), None)

    def test_not_due_outside_session(self):
        assert not self.trigger.is_due(et(2026, 9, 4, 8, 0), None)
        assert not self.trigger.is_due(et(2026, 9, 4, 17, 0), None)
        assert not self.trigger.is_due(et(2026, 9, 5, 10, 0), None)  # Saturday

    def test_respects_the_interval(self):
        last = et(2026, 9, 4, 10, 0)
        assert not self.trigger.is_due(et(2026, 9, 4, 10, 14), last)
        assert self.trigger.is_due(et(2026, 9, 4, 10, 15), last)

    def test_always_on_variant_ignores_session(self):
        trigger = EveryMinutes(5, during_session_only=False)
        assert trigger.is_due(et(2026, 9, 5, 3, 0), None)  # Saturday 3am


class TestDaylightSavingScheduling:
    """A daily slot must not drift by an hour across a DST transition."""

    trigger = DailyAt(time(7, 0), grace_hours=6.0)

    def test_slot_still_fires_the_day_after_spring_forward(self):
        # Ran Friday 2026-03-06 (EST); Monday 2026-03-09 is EDT.
        last_run = et(2026, 3, 6, 7, 0)
        assert self.trigger.is_due(et(2026, 3, 9, 7, 0), last_run)

    def test_slot_does_not_double_fire_after_fall_back(self):
        already = et(2026, 11, 2, 7, 0)
        assert not self.trigger.is_due(et(2026, 11, 2, 9, 0), already)

    def test_interval_measured_across_a_dst_change_is_wall_clock(self):
        """Fall-back repeats 01:00-02:00 locally; a 15m job must not stall."""
        trigger = EveryMinutes(15, during_session_only=False)
        last = et(2026, 11, 2, 9, 0)
        assert trigger.is_due(et(2026, 11, 2, 9, 20), last)


class TestTriggerBoundaries:
    def test_daily_fires_exactly_at_the_slot_not_a_second_before(self):
        trigger = DailyAt(time(7, 0), grace_hours=6.0)
        assert not trigger.is_due(et(2026, 9, 4, 6, 59), None)
        assert trigger.is_due(et(2026, 9, 4, 7, 0), None)

    def test_grace_boundary_is_inclusive_of_the_window(self):
        trigger = DailyAt(time(7, 0), grace_hours=2.0)
        assert trigger.is_due(et(2026, 9, 4, 9, 0), None)  # exactly 2h late
        assert not trigger.is_due(et(2026, 9, 4, 9, 1), None)

    def test_interval_fires_exactly_on_the_boundary(self):
        trigger = EveryMinutes(15, during_session_only=False)
        last = et(2026, 9, 4, 10, 0)
        assert trigger.is_due(et(2026, 9, 4, 10, 15), last)

    def test_session_job_stops_at_the_closing_bell(self):
        trigger = EveryMinutes(15, during_session_only=True)
        assert trigger.is_due(et(2026, 9, 4, 15, 59), None)
        assert not trigger.is_due(et(2026, 9, 4, 16, 0), None)

    def test_session_job_respects_an_early_close(self):
        trigger = EveryMinutes(15, during_session_only=True)
        assert not trigger.is_due(et(2026, 12, 24, 14, 0), None)
        assert trigger.is_due(et(2026, 12, 24, 12, 0), None)


class TestAtLeastEvery:
    """Elapsed-time trigger for the weekly refresh.

    A fixed weekday would be fragile on a laptop that sleeps: miss Sunday and
    the estimate goes stale for a fortnight.
    """

    trigger = AtLeastEvery(days=7, after=time(6, 0))

    def test_due_when_never_run(self):
        assert self.trigger.is_due(et(2026, 9, 4, 7, 0), None)

    def test_not_due_before_the_daily_cutoff(self):
        assert not self.trigger.is_due(et(2026, 9, 4, 5, 0), None)

    def test_not_due_within_the_window(self):
        assert not self.trigger.is_due(et(2026, 9, 4, 7, 0), et(2026, 9, 1, 7, 0))

    def test_due_once_the_window_elapses(self):
        assert self.trigger.is_due(et(2026, 9, 8, 7, 0), et(2026, 9, 1, 7, 0))

    def test_catches_up_after_a_long_gap(self):
        """Machine off for three weeks: the refresh still fires on return."""
        assert self.trigger.is_due(et(2026, 9, 22, 7, 0), et(2026, 9, 1, 7, 0))

    def test_runs_on_non_trading_days(self):
        assert self.trigger.is_due(et(2026, 9, 5, 7, 0), et(2026, 8, 20, 7, 0))
