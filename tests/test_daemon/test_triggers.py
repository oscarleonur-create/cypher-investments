"""Trigger semantics, especially the laptop-sleep catch-up behaviour."""

from __future__ import annotations

from datetime import datetime, time

from advisor.daemon import market_calendar as mc
from advisor.daemon.jobs import DailyAt, EveryMinutes


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
