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
