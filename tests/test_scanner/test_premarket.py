"""Premarket scan: candle parsing, the three rules, and the scan loop."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.jobs import WindowEvery
from advisor.scanner.models import CatalystItem, Phase, Setup, candidate_id
from advisor.scanner.premarket import (
    Bar,
    PremarketQuote,
    build_quote,
    daily_session,
    detect_premarket,
    is_day_two,
    is_premarket_dip,
    is_premarket_gap,
    run_premarket_scan,
)
from advisor.scanner.store import ScannerStore


def et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=mc.MARKET_TZ)


NOW = et(2026, 9, 24, 8, 40)  # Thursday; previous session Wed 09-23


def daily_bar(day: date, close: float, volume: float = 1e6) -> Bar:
    # DXLink stamps a daily candle at 00:00 UTC of its session.
    ts = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return Bar(ts, close, close, close, close, volume)


def five_min(moment: datetime, close: float, volume: float = 1000) -> Bar:
    return Bar(moment.astimezone(timezone.utc), close, close, close, close, volume)


def daily_history(last_day=date(2026, 9, 23), n=80, start=100.0, step=0.001, vol=1e6):
    """n sessions ending ``last_day``, gently rising, constant volume."""
    days, cursor = [], last_day
    while len(days) < n:
        if mc.is_trading_day(cursor):
            days.append(cursor)
        cursor -= timedelta(days=1)
    days.reverse()
    bars, px = [], start
    for i, d in enumerate(days):
        px *= 1 + (step if i % 2 == 0 else -step / 2)
        bars.append(daily_bar(d, px, vol))
    return bars


# ── Candle parsing ────────────────────────────────────────────────────────


class TestBuildQuote:
    def test_daily_bars_are_dated_in_utc_not_et(self):
        """Thursday's bar, stamped 00:00 UTC, reads as Wednesday 20:00 in ET."""
        bar = daily_bar(date(2026, 9, 24), 10.0)
        assert mc.to_et(bar.ts).date() == date(2026, 9, 23)
        assert daily_session(bar) == date(2026, 9, 24)

    def test_prev_close_is_yesterdays_session_not_todays_partial_bar(self):
        daily = daily_history() + [daily_bar(date(2026, 9, 24), 999.0)]  # today's partial
        q = build_quote("MSFT", [], daily, NOW)
        assert q.prev_close == pytest.approx(daily[-2].close)

    def test_premarket_last_and_volume(self):
        daily = daily_history()
        pre = [
            five_min(et(2026, 9, 24, 4, 0), 101.0, 100),
            five_min(et(2026, 9, 24, 8, 35), 105.0, 400),
        ]
        q = build_quote("MSFT", pre, daily, NOW)
        assert q.last == 105.0 and q.premarket_volume == 500

    def test_overnight_prints_before_0400_are_excluded(self):
        daily = daily_history()
        overnight = [five_min(et(2026, 9, 24, 0, 5), 150.0, 99999)]
        q = build_quote("MSFT", overnight, daily, NOW)
        assert q.last is None and q.premarket_volume == 0

    def test_prints_after_now_are_ignored(self):
        daily = daily_history()
        future = [five_min(et(2026, 9, 24, 9, 0), 150.0)]
        assert build_quote("MSFT", future, daily, NOW).last is None

    def test_no_premarket_trading_is_none_not_zero(self):
        q = build_quote("AOSL", [], daily_history(), NOW)
        assert q.last is None and q.change is None

    def test_prev_close_falls_back_to_last_rth_bar(self):
        rth = [
            five_min(et(2026, 9, 23, 15, 55), 50.0),
            five_min(et(2026, 9, 23, 16, 5), 60.0),  # after hours: not the close
        ]
        q = build_quote("X", rth, [], NOW)
        assert q.prev_close == 50.0

    def test_monday_looks_back_to_friday(self):
        monday = et(2026, 9, 28, 8, 40)
        daily = daily_history(last_day=date(2026, 9, 25))
        q = build_quote("X", [], daily, monday)
        assert q.prev_close == pytest.approx(daily[-1].close)
        assert q.prev_day_change == pytest.approx(daily[-1].close / daily[-2].close - 1)

    def test_after_a_holiday(self):
        """Tuesday after Labor Day: the previous session is Friday 09-04."""
        daily = daily_history(last_day=date(2026, 9, 4))
        q = build_quote("X", [], daily, et(2026, 9, 8, 8, 40))
        assert q.prev_close == pytest.approx(daily[-1].close)

    def test_short_history_gives_no_average_or_sigma(self):
        q = build_quote("NEW", [], daily_history(n=8), NOW)
        assert q.avg_volume is None and q.sigma is None

    def test_zero_or_nan_daily_closes_are_skipped(self):
        daily = daily_history()
        daily[-1] = daily_bar(date(2026, 9, 23), float("nan"))
        q = build_quote("X", [], daily, NOW)
        assert q.prev_close is None  # the lost bar is not replaced by a guess

    def test_avg_volume_excludes_the_previous_day_itself(self):
        daily = daily_history(vol=1e6)
        daily[-1] = daily_bar(date(2026, 9, 23), daily[-1].close, 9e6)
        q = build_quote("X", [], daily, NOW)
        assert q.avg_volume == pytest.approx(1e6) and q.prev_day_rvol == pytest.approx(9.0)


# ── Rules ─────────────────────────────────────────────────────────────────


def quote(**kw) -> PremarketQuote:
    base = dict(
        symbol="META",
        prev_close=100.0,
        last=100.0,
        premarket_volume=0.0,
        prev_day_change=0.0,
        prev_day_volume=1e6,
        avg_volume=1e6,
        sigma=0.02,
        market_cap=1.5e12,
    )
    base.update(kw)
    return PremarketQuote(**base)


class TestPremarketA:
    def test_qualifies(self):
        assert is_premarket_gap(quote(last=106.0, premarket_volume=80_000))

    def test_exactly_at_thresholds(self):
        assert is_premarket_gap(quote(last=104.0, premarket_volume=50_000))

    def test_just_under_gap(self):
        assert not is_premarket_gap(quote(last=103.99, premarket_volume=80_000))

    def test_ordinary_premarket_volume_is_not_a_catalyst(self):
        """3-5% of a day is a normal morning for MRVL/RKLB (measured 2026-09-24)."""
        assert not is_premarket_gap(quote(last=106.0, premarket_volume=40_000))

    def test_thin_dollar_volume_rejected(self):
        q = quote(
            last=6.0, prev_close=5.0, avg_volume=100_000, premarket_volume=60_000, market_cap=1e9
        )
        assert not is_premarket_gap(q)  # $360k traded

    def test_no_premarket_print(self):
        assert not is_premarket_gap(quote(last=None, premarket_volume=0))


class TestPremarketB:
    def test_qualifies(self):
        assert is_day_two(quote(prev_day_change=0.12, prev_day_volume=2e6))

    def test_exactly_nine_percent_and_1_5x(self):
        assert is_day_two(quote(prev_day_change=0.09, prev_day_volume=1.5e6))

    def test_quiet_up_day_is_not_b(self):
        assert not is_day_two(quote(prev_day_change=0.12, prev_day_volume=1.2e6))

    def test_collapsing_premarket_is_not_b(self):
        q = quote(prev_day_change=0.12, prev_day_volume=2e6, last=96.9)
        assert not is_day_two(q)

    def test_giving_back_exactly_three_percent_still_b(self):
        q = quote(prev_day_change=0.12, prev_day_volume=2e6, last=97.0)
        assert is_day_two(q)

    def test_no_premarket_print_still_b(self):
        assert is_day_two(quote(prev_day_change=0.12, prev_day_volume=2e6, last=None))

    def test_missing_history(self):
        assert not is_day_two(quote(prev_day_change=None))
        assert not is_day_two(quote(prev_day_change=0.12, avg_volume=None))


class TestPremarketC:
    def test_qualifies(self):
        ok, z = is_premarket_dip(quote(last=94.0, premarket_volume=20_000))
        assert ok and z == pytest.approx(3.0)

    def test_small_cap_excluded(self):
        q = quote(last=94.0, premarket_volume=200_000, market_cap=2e9)
        assert not is_premarket_dip(q)[0]

    def test_thin_prints_are_not_a_price(self):
        assert not is_premarket_dip(quote(last=94.0, premarket_volume=100))[0]

    def test_under_two_sigma(self):
        assert not is_premarket_dip(quote(last=94.0, premarket_volume=20_000, sigma=0.05))[0]

    def test_no_sigma_falls_back_to_the_floor(self):
        ok, z = is_premarket_dip(quote(last=96.0, premarket_volume=20_000, sigma=None))
        assert ok and z is None


def test_a_name_can_be_b_and_a_at_once():
    q = quote(last=106.0, premarket_volume=80_000, prev_day_change=0.12, prev_day_volume=2e6)
    assert {s for s, _ in detect_premarket(q)} == {Setup.CATALYST_GAP, Setup.DAY_TWO}


# ── The scan ──────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    s = ScannerStore(tmp_path / "pm.db")
    yield s
    s.close()


def news(sym, name, since):
    return [CatalystItem(kind="news", title=f"{sym} news", published_at=NOW)], True


def pm_scan(store, quotes, *, now=NOW, watchlist=None, peer=([], None), **kw):
    syms = watchlist if watchlist is not None else list(quotes)
    return run_premarket_scan(
        store,
        now,
        fetch_watchlist=lambda: (syms, None),
        fetch_quotes=lambda s, n: {k: v for k, v in quotes.items() if k in s},
        fetch_peer_move=lambda s, n: peer,
        fetch_catalysts=news,
        **kw,
    )


GAPPER = quote(symbol="AMD", last=106.0, premarket_volume=80_000)
DIPPER = quote(symbol="INTC", last=94.0, premarket_volume=20_000)
QUIET = quote(symbol="MSFT")


def test_records_premarket_candidates(store):
    r = pm_scan(store, {"AMD": GAPPER, "INTC": DIPPER, "MSFT": QUIET})
    assert {(c.setup, c.symbol) for c in r.new} == {
        (Setup.CATALYST_GAP, "AMD"),
        (Setup.NEWS_DIP, "INTC"),
    }
    assert all(c.phase is Phase.PREMARKET for c in r.new)
    assert all(c.source == "tastytrade:Swing" for c in r.new)


@pytest.mark.parametrize(
    "moment,why",
    [
        (et(2026, 9, 24, 6, 59), "outside"),
        (et(2026, 9, 24, 9, 30), "outside"),  # the bell: the session scan's job
        (et(2026, 9, 26, 8, 40), "not a trading day"),  # Saturday
        (et(2026, 11, 26, 8, 40), "not a trading day"),  # Thanksgiving
    ],
)
def test_window(store, moment, why):
    called = []
    r = run_premarket_scan(
        store,
        moment,
        fetch_watchlist=lambda: called.append(1) or ([], None),
        fetch_quotes=None,
        fetch_peer_move=None,
        fetch_catalysts=None,
    )
    assert not r.ran and why in r.reason and not called


def test_early_close_day_still_scans_premarket(store):
    r = pm_scan(store, {"AMD": GAPPER}, now=et(2026, 11, 27, 8, 40))
    assert r.ran and len(r.new) == 1


def test_watchlist_unavailable_is_an_error_not_a_quiet_morning(store):
    r = run_premarket_scan(
        store,
        NOW,
        fetch_watchlist=lambda: ([], "watchlist 'Swing' unavailable: 401"),
        fetch_quotes=None,
        fetch_peer_move=None,
        fetch_catalysts=None,
    )
    assert r.ran and r.error and "401" in r.summary()


def test_empty_watchlist(store):
    r = pm_scan(store, {}, watchlist=[])
    assert r.ran and r.new == [] and r.error is None


def test_symbol_without_data_is_named(store):
    r = pm_scan(store, {"AMD": GAPPER}, watchlist=["AMD", "ZZZZ"])
    assert r.no_data == ["ZZZZ"] and len(r.new) == 1


def test_rescan_does_not_duplicate_and_keeps_first_price(store):
    pm_scan(store, {"AMD": GAPPER})
    later = quote(symbol="AMD", last=110.0, premarket_volume=90_000)
    r = pm_scan(store, {"AMD": later}, now=et(2026, 9, 24, 9, 20))
    assert r.new == [] and r.already_recorded == 1
    assert store.list()[0].price == 106.0


def test_premarket_and_session_candidates_are_separate(store):
    pm_scan(store, {"AMD": GAPPER})
    assert store.exists(date(2026, 9, 24), Setup.CATALYST_GAP, "AMD", Phase.PREMARKET)
    assert not store.exists(date(2026, 9, 24), Setup.CATALYST_GAP, "AMD", Phase.SESSION)


def test_session_ids_keep_their_original_form():
    """Rows written before premarket existed must still be found."""
    assert candidate_id(date(2026, 9, 23), Setup.CATALYST_GAP, "nnbr") == "2026-09-23:A:NNBR"
    assert candidate_id(date(2026, 9, 23), Setup.CATALYST_GAP, "NNBR", Phase.PREMARKET) == (
        "2026-09-23:A@pre:NNBR"
    )


def test_premarket_sector_move_is_not_c(store):
    r = pm_scan(store, {"INTC": DIPPER}, peer=(["AMD", "NVDA"], -0.05))
    assert r.new == [] and r.sector_moves == ["INTC"]


def test_b_without_premarket_prints_records_change_as_none(store):
    q = quote(symbol="AMZN", prev_day_change=0.12, prev_day_volume=2e6, last=None)
    r = pm_scan(store, {"AMZN": q})
    (c,) = r.new
    assert c.setup is Setup.DAY_TWO and c.change is None and c.price == 100.0


# ── Trigger ───────────────────────────────────────────────────────────────


class TestWindowEvery:
    trigger = WindowEvery(start=time(8, 0), end=time(9, 25), minutes=20)

    def test_first_run_at_0800(self):
        assert not self.trigger.is_due(et(2026, 9, 24, 7, 59), None)
        assert self.trigger.is_due(et(2026, 9, 24, 8, 0), None)

    def test_every_twenty_minutes(self):
        last = et(2026, 9, 24, 8, 0)
        assert not self.trigger.is_due(et(2026, 9, 24, 8, 19), last)
        assert self.trigger.is_due(et(2026, 9, 24, 8, 20), last)

    def test_closed_at_0925(self):
        assert not self.trigger.is_due(et(2026, 9, 24, 9, 25), et(2026, 9, 24, 9, 0))

    def test_yesterdays_run_does_not_block_today(self):
        assert self.trigger.is_due(et(2026, 9, 24, 8, 5), et(2026, 9, 23, 9, 20))

    def test_laptop_waking_late_in_the_window_runs_once(self):
        assert self.trigger.is_due(et(2026, 9, 24, 9, 10), et(2026, 9, 23, 9, 20))

    def test_laptop_waking_after_the_window_does_not(self):
        assert not self.trigger.is_due(et(2026, 9, 24, 10, 0), et(2026, 9, 23, 9, 20))

    def test_weekend_and_holiday(self):
        assert not self.trigger.is_due(et(2026, 9, 26, 8, 30), None)
        assert not self.trigger.is_due(et(2026, 11, 26, 8, 30), None)

    def test_dst_change_uses_wall_clock(self):
        # 13:00 UTC on 2026-11-02 is 08:00 EST (after DST ends).
        moment = datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc)
        assert self.trigger.is_due(moment, None)
