"""Session logic — the scheduler is only as correct as this."""

from __future__ import annotations

from datetime import date, datetime, time

from advisor.daemon import market_calendar as mc


def test_weekends_are_not_trading_days():
    assert not mc.is_trading_day(date(2026, 9, 5))  # Saturday
    assert not mc.is_trading_day(date(2026, 9, 6))  # Sunday
    assert mc.is_trading_day(date(2026, 9, 4))  # Friday


def test_holidays_are_not_trading_days():
    assert not mc.is_trading_day(date(2026, 9, 7))  # Labor Day
    assert not mc.is_trading_day(date(2026, 11, 26))  # Thanksgiving
    assert not mc.is_trading_day(date(2026, 7, 3))  # Independence Day observed
    assert not mc.is_trading_day(date(2027, 3, 26))  # Good Friday


def test_market_open_only_within_session():
    assert not mc.is_market_open(datetime(2026, 9, 4, 9, 29))
    assert mc.is_market_open(datetime(2026, 9, 4, 9, 30))
    assert mc.is_market_open(datetime(2026, 9, 4, 15, 59))
    assert not mc.is_market_open(datetime(2026, 9, 4, 16, 0))


def test_market_closed_on_holiday_during_normal_hours():
    assert not mc.is_market_open(datetime(2026, 9, 7, 11, 0))


def test_early_close_shortens_the_session():
    assert mc.session_close(date(2026, 12, 24)) == time(13, 0)
    assert mc.session_close(date(2026, 12, 23)) == time(16, 0)
    # 14:00 is inside a normal session but after an early close.
    assert not mc.is_market_open(datetime(2026, 12, 24, 14, 0))
    assert mc.is_market_open(datetime(2026, 12, 23, 14, 0))


def test_previous_trading_day_skips_weekend_and_holiday():
    # Tuesday after Labor Day -> back to the preceding Friday.
    assert mc.previous_trading_day(date(2026, 9, 8)) == date(2026, 9, 4)


class TestDaylightSaving:
    """DST is the highest-risk edge for a wall-clock scheduler.

    The daemon reasons in local market time, so the UTC offset shifts twice a
    year underneath it. A slot must stay pinned to 09:30 *local* through both
    transitions, not drift by an hour.
    """

    def test_offset_flips_between_est_and_edt(self):
        est = datetime(2026, 3, 7, 9, 30, tzinfo=mc.MARKET_TZ)
        edt = datetime(2026, 3, 9, 9, 30, tzinfo=mc.MARKET_TZ)
        assert est.utcoffset() != edt.utcoffset()

    def test_opening_bell_holds_across_spring_forward(self):
        # Friday before, Monday after the 2026-03-08 transition.
        assert mc.is_market_open(datetime(2026, 3, 6, 9, 30))
        assert mc.is_market_open(datetime(2026, 3, 9, 9, 30))
        assert not mc.is_market_open(datetime(2026, 3, 9, 9, 29))

    def test_opening_bell_holds_across_fall_back(self):
        # Friday before, Monday after the 2026-11-01 transition.
        assert mc.is_market_open(datetime(2026, 10, 30, 9, 30))
        assert mc.is_market_open(datetime(2026, 11, 2, 9, 30))
        assert not mc.is_market_open(datetime(2026, 11, 2, 16, 0))


class TestConversion:
    def test_utc_input_is_converted_not_reinterpreted(self):
        """An aware UTC timestamp must shift, not be relabelled as ET."""
        from datetime import timezone

        utc_noon = datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc)
        assert mc.to_et(utc_noon).hour == 12  # 16:00 UTC = 12:00 EDT

    def test_naive_input_is_assumed_to_be_market_time(self):
        assert mc.to_et(datetime(2026, 9, 4, 12, 0)).hour == 12


class TestYearBoundary:
    def test_previous_trading_day_crosses_the_new_year(self):
        # 2027-01-01 is a Friday holiday, 01-02/03 weekend.
        assert mc.previous_trading_day(date(2027, 1, 4)) == date(2026, 12, 31)

    def test_previous_trading_day_crosses_a_long_holiday_stretch(self):
        # Friday after Thanksgiving 2026 is a (short) session, not a closure.
        assert mc.previous_trading_day(date(2026, 11, 30)) == date(2026, 11, 27)
