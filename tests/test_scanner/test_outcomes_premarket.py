"""Outcomes for premarket candidates (entry at the open) and 5/10/20-session horizons."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest
from advisor.daemon import market_calendar as mc
from advisor.scanner.models import Candidate, Phase, Setup
from advisor.scanner.outcomes import (
    PREMARKET_KEYS,
    compute,
    fill_outcomes,
    keys_for,
    nth_trading_day,
)
from advisor.scanner.store import ScannerStore


def et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=mc.MARKET_TZ)


def bars(day: date, start_price=100.0, step=0.1):
    idx = pd.date_range(
        f"{day} 09:30", f"{day} 16:00", freq="5min", inclusive="left", tz="America/New_York"
    )
    px = [start_price + i * step for i in range(len(idx))]
    return pd.DataFrame(
        {"Open": px, "High": [p + 0.5 for p in px], "Low": [p - 0.5 for p in px], "Close": px},
        index=idx,
    )


SESSION = date(2026, 9, 24)
TWO_DAYS = pd.concat([bars(SESSION, 110.0), bars(date(2026, 9, 25), 120.0)])


def pre(price=105.0):
    return Candidate(
        session=SESSION,
        setup=Setup.CATALYST_GAP,
        symbol="AMD",
        detected_at=et(2026, 9, 24, 8, 40),
        phase=Phase.PREMARKET,
        price=price,
        prev_close=100.0,
        change=0.05,
    )


def sess(price=100.0):
    return Candidate(
        session=SESSION,
        setup=Setup.CATALYST_GAP,
        symbol="AMD",
        detected_at=et(2026, 9, 24, 10, 0),
        price=price,
        prev_close=95.0,
        change=0.05,
    )


class TestPremarketEntry:
    def test_entry_is_the_open_and_open_measures_the_gap_given_away(self):
        out = compute(pre(105.0), TWO_DAYS, et(2026, 9, 24, 16, 30))
        assert out["open"] == pytest.approx(110.0 / 105.0 - 1)
        # r30 anchored at 09:30: the 10:00 bar opens at 110 + 6*0.1
        assert out["r30"] == pytest.approx(110.6 / 110.0 - 1)

    def test_nothing_before_the_open(self):
        assert compute(pre(), TWO_DAYS.iloc[0:0], et(2026, 9, 24, 9, 0)) == {}

    def test_r30_waits_until_1005(self):
        out = compute(pre(), bars(SESSION, 110.0), et(2026, 9, 24, 10, 0))
        assert "open" in out and "r30" not in out

    def test_premarket_rows_need_the_open_key(self):
        assert keys_for(pre()) == PREMARKET_KEYS and "open" in keys_for(pre())
        assert "open" not in keys_for(sess())


class TestSwingHorizons:
    def daily(self, closes: dict):
        return closes

    def test_nth_trading_day_skips_weekends_and_holidays(self):
        assert nth_trading_day(date(2026, 9, 24), 5) == date(2026, 10, 1)
        # Thanksgiving 11-26 is skipped.
        assert nth_trading_day(date(2026, 11, 24), 2) == date(2026, 11, 27)

    def test_d5_from_detection_price(self):
        d5 = nth_trading_day(SESSION, 5)
        out = compute(sess(100.0), None, et(2026, 10, 1, 16, 30), {d5: 108.0})
        assert out == {"d5": pytest.approx(0.08)}

    def test_d5_pending_until_that_close_settles(self):
        d5 = nth_trading_day(SESSION, 5)
        assert compute(sess(), None, et(2026, 10, 1, 16, 0), {d5: 108.0}) == {}

    def test_missing_or_nan_day_stays_pending(self):
        d5 = nth_trading_day(SESSION, 5)
        assert compute(sess(), None, et(2026, 10, 2, 9, 0), {}) == {}
        assert compute(sess(), None, et(2026, 10, 2, 9, 0), {d5: float("nan")}) == {}

    def test_premarket_swing_horizon_is_from_the_open(self):
        d5 = nth_trading_day(SESSION, 5)
        out = compute(pre(105.0), TWO_DAYS, et(2026, 10, 1, 16, 30), {d5: 121.0})
        assert out["d5"] == pytest.approx(121.0 / 110.0 - 1)


@pytest.fixture
def store(tmp_path):
    s = ScannerStore(tmp_path / "po.db")
    yield s
    s.close()


def test_fill_writes_daily_keys_and_never_overwrites(store):
    store.add(sess())
    d5 = nth_trading_day(SESSION, 5)
    fill_outcomes(
        store, et(2026, 10, 1, 16, 30), bars_fn=lambda *a: TWO_DAYS, daily_fn=lambda *a: {d5: 108.0}
    )
    assert store.list()[0].outcomes["d5"] == pytest.approx(0.08)
    fill_outcomes(
        store, et(2026, 10, 2, 16, 30), bars_fn=lambda *a: TWO_DAYS, daily_fn=lambda *a: {d5: 999.0}
    )
    assert store.list()[0].outcomes["d5"] == pytest.approx(0.08)


def test_old_session_row_fills_daily_without_intraday_bars(store):
    """Past the 55-day 5m limit, d-horizons still fill from daily closes."""
    old = Candidate(
        session=date(2026, 7, 1),
        setup=Setup.NEWS_DIP,
        symbol="META",
        detected_at=et(2026, 7, 1, 10, 0),
        price=100.0,
        prev_close=106.0,
        change=-0.06,
    )
    old.outcomes = {
        k: 0.0 for k in ("r30", "r60", "r120", "close", "mfe", "mae", "next_open", "next_close")
    }
    store.add(old)
    asked = []
    d20 = nth_trading_day(old.session, 20)
    fill_outcomes(
        store,
        et(2026, 8, 28, 17, 0),
        bars_fn=lambda *a: asked.append(a) or None,
        daily_fn=lambda *a: {d20: 110.0},
    )
    assert asked == [] and store.list()[0].outcomes["d20"] == pytest.approx(0.10)
