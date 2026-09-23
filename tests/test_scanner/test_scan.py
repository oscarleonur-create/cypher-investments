"""The scan loop: idempotency, source failure, news budget, catalyst window."""

from __future__ import annotations

from datetime import datetime

import pytest
from advisor.daemon import market_calendar as mc
from advisor.scanner.models import CatalystItem, Mover, Setup
from advisor.scanner.scan import catalyst_window_start, run_scan
from advisor.scanner.store import ScannerStore


def et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=mc.MARKET_TZ)


NOW = et(2026, 9, 23, 10, 5)

GAP = Mover(
    symbol="AMPG",
    name="AmpliTech",
    price=10.8,
    prev_close=10.0,
    open=10.6,
    volume=2_000_000,
    avg_volume=1_000_000,
    market_cap=500e6,
)
DIP = Mover(
    symbol="AAPL",
    name="Apple",
    price=94.0,
    prev_close=100.0,
    open=97.0,
    volume=30_000_000,
    avg_volume=50_000_000,
    market_cap=3e12,
)
NOISE = Mover(
    symbol="XYZ",
    price=10.1,
    prev_close=10.0,
    open=10.0,
    volume=10_000,
    avg_volume=1_000_000,
    market_cap=1e9,
)


@pytest.fixture
def store(tmp_path):
    s = ScannerStore(tmp_path / "t.db")
    yield s
    s.close()


def news(symbol, name, since):
    return [CatalystItem(kind="news", title=f"{symbol} headline", published_at=NOW)], True


def no_news(symbol, name, since):
    return [], True


def scan(store, movers, *, now=NOW, catalysts=news, sigma=None, **kw):
    return run_scan(
        store,
        now,
        fetch_movers=lambda: (movers, None),
        fetch_sigma=lambda syms: sigma if sigma is not None else {s: 0.02 for s in syms},
        fetch_catalysts=catalysts,
        **kw,
    )


def test_records_both_setups_and_ignores_noise(store):
    r = scan(store, [GAP, DIP, NOISE])
    assert r.ran and {(c.setup, c.symbol) for c in r.new} == {
        (Setup.CATALYST_GAP, "AMPG"),
        (Setup.NEWS_DIP, "AAPL"),
    }
    assert store.count() == 2


def test_same_candidate_twice_is_one_row_priced_at_first_sight(store):
    scan(store, [GAP])
    later = GAP.model_copy(update={"price": 11.5})
    r = scan(store, [later], now=et(2026, 9, 23, 11, 0))
    assert r.new == [] and r.already_recorded == 1
    assert store.list()[0].price == pytest.approx(10.8)


def test_next_session_is_a_new_candidate(store):
    scan(store, [GAP])
    r = scan(store, [GAP], now=et(2026, 9, 24, 10, 5))
    assert len(r.new) == 1 and store.count() == 2


def test_two_stores_racing_on_one_db(tmp_path):
    a, b = ScannerStore(tmp_path / "r.db"), ScannerStore(tmp_path / "r.db")
    try:
        scan(a, [GAP])
        # b's exists() check passed before a wrote: add() must still refuse.
        cand = a.list()[0]
        assert b.add(cand) is False
        assert b.count() == 1
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize(
    "moment,reason",
    [
        (et(2026, 9, 23, 9, 32), "before 09:35"),
        (et(2026, 9, 26, 11, 0), "market closed"),  # Saturday
        (et(2026, 11, 26, 11, 0), "market closed"),  # Thanksgiving
        (et(2026, 11, 27, 13, 30), "market closed"),  # after an early close
    ],
)
def test_does_not_run_outside_the_window(store, moment, reason):
    called = []
    r = run_scan(store, moment, fetch_movers=lambda: called.append(1) or ([], None))
    assert not r.ran and reason in r.reason and not called


def test_screener_failure_is_reported_not_hidden(store):
    r = run_scan(store, NOW, fetch_movers=lambda: ([], "gainers: HTTP 429"))
    assert r.ran and r.movers_seen == 0 and "429" in r.summary()


def test_empty_market(store):
    r = scan(store, [])
    assert r.ran and r.new == [] and store.count() == 0


def test_no_news_found_is_recorded_as_checked(store):
    scan(store, [GAP], catalysts=no_news)
    c = store.list()[0]
    assert c.news_checked and c.has_catalyst is False


def test_news_budget_leaves_the_rest_unchecked(store):
    movers = [GAP.model_copy(update={"symbol": f"G{i}"}) for i in range(4)]
    r = scan(store, movers, news_budget=2)
    assert r.news_lookups == 2 and r.over_budget == 2
    flags = sorted(str(c.has_catalyst) for c in store.list())
    assert flags == ["None", "None", "True", "True"]


def test_budget_carries_across_scans_in_a_session(store):
    scan(store, [GAP.model_copy(update={"symbol": "G1"})], news_budget=1)
    r = scan(
        store, [GAP.model_copy(update={"symbol": "G2"})], now=et(2026, 9, 23, 11, 0), news_budget=1
    )
    assert r.news_lookups == 0 and r.over_budget == 1


def test_no_news_mode_spends_nothing(store):
    calls = []
    scan(store, [GAP], catalysts=lambda *a: calls.append(a) or ([], True), check_news=False)
    assert calls == [] and store.list()[0].has_catalyst is None


def test_missing_volatility_still_records_the_dip_without_sigma(store):
    r = scan(store, [DIP], sigma={})
    assert len(r.new) == 1 and r.new[0].sigma is None


def test_catalyst_window_starts_at_previous_close():
    assert catalyst_window_start(et(2026, 9, 23, 10, 0)) == et(2026, 9, 22, 16, 0)
    # Monday looks back to Friday's close, not Sunday's.
    assert catalyst_window_start(et(2026, 9, 21, 10, 0)) == et(2026, 9, 18, 16, 0)
    # After a holiday Monday (Labor Day), Tuesday looks back to Friday.
    assert catalyst_window_start(et(2026, 9, 8, 10, 0)) == et(2026, 9, 4, 16, 0)
    # The day after an early close looks back to 13:00, not 16:00.
    assert catalyst_window_start(et(2026, 11, 30, 10, 0)) == et(2026, 11, 27, 13, 0)


def test_news_is_unlimited_by_default(store):
    movers = [GAP.model_copy(update={"symbol": f"G{i}"}) for i in range(60)]
    r = scan(store, movers)
    assert r.news_lookups == 60 and r.over_budget == 0
    assert all(c.news_checked for c in store.list())
