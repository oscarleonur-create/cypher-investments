"""E0 eligibility: floors at the boundary, point in time, and the day's snapshot."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from advisor.breadth import universe as U
from advisor.breadth.listings import build_directory
from advisor.breadth.store import BreadthStore

DAY = date(2026, 9, 25)


def rows(n: int, close: float, volume: float | None, end: date = DAY):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append((d, close, volume))
        d -= timedelta(days=1)
    return sorted(out)


def liquid(n=U.MIN_SESSIONS):
    return rows(n, 50.0, U.MIN_DOLLAR_VOLUME / 50.0)


def test_eligible():
    e = U.eligibility(liquid(), DAY)
    assert e.eligible and e.reason is None and e.sessions == U.MIN_SESSIONS


def test_exactly_at_every_floor_is_eligible():
    r = rows(U.MIN_SESSIONS, U.MIN_PRICE, U.MIN_DOLLAR_VOLUME / U.MIN_PRICE)
    assert U.eligibility(r, DAY).eligible


def test_one_cent_under_the_price_floor():
    r = rows(U.MIN_SESSIONS, U.MIN_PRICE - 0.01, 1e9)
    assert U.eligibility(r, DAY).reason == "price below floor"


def test_one_session_short():
    assert U.eligibility(liquid(U.MIN_SESSIONS - 1), DAY).reason == "short history"


def test_dollar_volume_is_a_median_one_block_trade_does_not_qualify():
    r = rows(U.MIN_SESSIONS, 50.0, 1000.0)
    r[-1] = (r[-1][0], 50.0, 1e9)  # one huge session
    assert U.eligibility(r, DAY).reason == "dollar volume below floor"


def test_zero_and_missing_volume_count_as_nothing_traded():
    assert U.eligibility(rows(U.MIN_SESSIONS, 50.0, 0.0), DAY).reason == (
        "dollar volume below floor"
    )
    assert U.eligibility(rows(U.MIN_SESSIONS, 50.0, None), DAY).reason == (
        "dollar volume below floor"
    )


def test_no_bars():
    assert U.eligibility([], DAY).reason == "no bars"


def test_future_bars_are_ignored():
    # Illiquid up to DAY, liquid afterwards: on DAY it is not eligible.
    past = rows(U.MIN_SESSIONS, 50.0, 10.0)
    future = [(DAY + timedelta(days=i), 50.0, 1e9) for i in range(1, 30)]
    assert U.eligibility(past + future, DAY).reason == "dollar volume below floor"
    assert U.eligibility(future + past, DAY).reason == "dollar volume below floor"


def test_stale_bars_are_not_judged():
    old = rows(U.MIN_SESSIONS, 50.0, 1e9, end=DAY - timedelta(days=U.STALE_BAR_DAYS + 1))
    assert U.eligibility(old, DAY).reason == "stale bars"
    ok = rows(U.MIN_SESSIONS, 50.0, 1e9, end=DAY - timedelta(days=U.STALE_BAR_DAYS))
    assert U.eligibility(ok, DAY).eligible


NASDAQ = """\
Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
LIQ|Liquid Co - Common Stock|G|N|N|100|N|N
THIN|Thin Co - Common Stock|G|N|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
"""
OTHER = "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
SEC = {
    "0": {"cik_str": 10, "ticker": "LIQ", "title": ""},
    "1": {"cik_str": 11, "ticker": "THIN", "title": ""},
}


@pytest.fixture
def store(tmp_path):
    s = BreadthStore(tmp_path / "b.db")
    yield s
    s.close()


def _put(store, symbol, data):
    with store.conn:
        store.conn.executemany(
            "INSERT INTO breadth_bars (symbol, day, close, volume) VALUES (?, ?, ?, ?)",
            [(symbol, d.isoformat(), c, v) for d, c, v in data],
        )


def test_snapshot_keeps_every_listing_with_its_reason_and_rewrites_the_day(store):
    _put(store, "LIQ", liquid())
    _put(store, "THIN", rows(U.MIN_SESSIONS, 50.0, 10.0))
    d = build_directory(NASDAQ, OTHER, SEC)
    out = U.snapshot(store, d, DAY, "v1")
    by = {r["symbol"]: r for r in out}
    assert by["LIQ"]["eligible"] and by["LIQ"]["cik"] == 10
    assert by["THIN"]["reason"] == "dollar volume below floor"
    assert by["QQQ"]["reason"] == "etf"
    U.snapshot(store, d, DAY, "v1")  # a re-run of the same day
    assert len(store.universe(DAY)) == 3
    assert [r["symbol"] for r in store.universe(DAY, eligible_only=True)] == ["LIQ"]
    f = U.funnel(out)
    assert f == {
        "listings": 3,
        "common_stock_sec_filers": 2,
        "eligible": 1,
        "excluded": {"dollar volume below floor": 1, "etf": 1},
    }
