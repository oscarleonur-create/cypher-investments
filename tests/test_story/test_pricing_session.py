"""Which session first gets to react to an event.

The case that forced this to exist: AAOI's $600m offering was accepted at
16:09 ET on Friday 21 August, nine minutes after the close. Friday's -3.32%
had nothing to do with it; Monday's -13.77% had everything to do with it.
Attributing the move to the wrong session inverts the story — a hand-written
timeline made exactly that mistake before this function existed.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from advisor.daemon.market_calendar import MARKET_TZ
from advisor.story.assemble import pricing_session


def et(y, m, d, hh, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=MARKET_TZ)


class TestTheRealCase:
    def test_a_filing_nine_minutes_after_the_close_is_priced_on_monday(self):
        """The AAOI 424B5, exactly as EDGAR recorded it."""
        assert pricing_session(et(2026, 8, 21, 16, 9)) == date(2026, 8, 24)

    def test_the_same_filing_read_in_utc_lands_on_the_same_session(self):
        """EDGAR stamps UTC; the market runs on ET. 20:09 UTC is 16:09 ET."""
        utc = datetime(2026, 8, 21, 20, 9, 22, tzinfo=timezone.utc)
        assert pricing_session(utc) == date(2026, 8, 24)


class TestWithinASession:
    def test_an_event_during_the_session_is_priced_that_day(self):
        assert pricing_session(et(2026, 8, 21, 11, 30)) == date(2026, 8, 21)

    def test_the_opening_bell_itself_is_priced_that_day(self):
        assert pricing_session(et(2026, 8, 21, 9, 30)) == date(2026, 8, 21)

    def test_before_the_open_is_still_priced_that_day(self):
        """Pre-market news is priced when that day's session opens."""
        assert pricing_session(et(2026, 8, 21, 7, 0)) == date(2026, 8, 21)

    def test_exactly_at_the_close_is_the_next_session(self):
        """16:00:00 is the boundary — the bell has rung."""
        assert pricing_session(et(2026, 8, 21, 16, 0)) == date(2026, 8, 24)

    def test_one_minute_before_the_close_is_that_day(self):
        assert pricing_session(et(2026, 8, 21, 15, 59)) == date(2026, 8, 21)


class TestNonTradingDays:
    def test_a_saturday_event_is_priced_on_monday(self):
        assert pricing_session(et(2026, 8, 22, 10, 0)) == date(2026, 8, 24)

    def test_a_sunday_event_is_priced_on_monday(self):
        assert pricing_session(et(2026, 8, 23, 19, 0)) == date(2026, 8, 24)

    def test_an_event_on_a_holiday_skips_to_the_next_open_session(self):
        """2026-01-19 is a Monday holiday; the market next opens Tuesday."""
        assert pricing_session(et(2026, 1, 19, 10, 0)) == date(2026, 1, 20)

    def test_a_friday_evening_before_a_monday_holiday_skips_two_days(self):
        assert pricing_session(et(2026, 1, 16, 18, 0)) == date(2026, 1, 20)

    def test_new_years_day_rolls_to_the_first_session_of_the_year(self):
        assert pricing_session(et(2026, 1, 1, 12, 0)) == date(2026, 1, 2)


class TestEarlyCloses:
    def test_after_a_1pm_early_close_is_the_next_session(self):
        """2026-11-27 closes at 13:00. 14:00 is after the bell, not during."""
        assert pricing_session(et(2026, 11, 27, 14, 0)) == date(2026, 11, 30)

    def test_before_a_1pm_early_close_is_that_day(self):
        assert pricing_session(et(2026, 11, 27, 12, 30)) == date(2026, 11, 27)

    def test_exactly_at_the_early_close_is_the_next_session(self):
        assert pricing_session(et(2026, 11, 27, 13, 0)) == date(2026, 11, 30)

    def test_a_normal_close_time_on_an_early_close_day_is_still_after(self):
        """15:00 would be mid-session normally; on 27 Nov it is two hours late."""
        assert pricing_session(et(2026, 11, 27, 15, 0)) == date(2026, 11, 30)


class TestTimezoneHandling:
    def test_a_naive_timestamp_is_read_as_et_not_local(self):
        naive = datetime(2026, 8, 21, 16, 9)
        assert pricing_session(naive) == date(2026, 8, 24)

    def test_a_pacific_timestamp_converts_before_comparing(self):
        """13:09 Pacific is 16:09 ET — after the close, not mid-session."""
        pacific = datetime(2026, 8, 21, 13, 9, tzinfo=ZoneInfo("America/Los_Angeles"))
        assert pricing_session(pacific) == date(2026, 8, 24)

    @pytest.mark.parametrize("moment", [et(2026, 3, 7, 18, 0), et(2026, 11, 7, 18, 0)])
    def test_dst_transition_weekends_still_resolve(self, moment):
        """Spring forward and fall back are Sundays; both roll to Monday."""
        assert pricing_session(moment).weekday() == 0
