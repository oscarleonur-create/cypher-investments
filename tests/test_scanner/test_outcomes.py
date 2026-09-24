"""Outcome filling: horizons past the close, pending vs final, re-runs."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest
from advisor.daemon import market_calendar as mc
from advisor.scanner.models import Candidate, Setup
from advisor.scanner.outcomes import KEYS, compute, fill_outcomes, next_trading_day
from advisor.scanner.report import summarize
from advisor.scanner.store import ScannerStore


def et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=mc.MARKET_TZ)


def bars(day: date, close_at="16:00", start_price=100.0, step=0.1):
    idx = pd.date_range(
        f"{day} 09:30", f"{day} {close_at}", freq="5min", inclusive="left", tz="America/New_York"
    )
    px = [start_price + i * step for i in range(len(idx))]
    return pd.DataFrame(
        {"Open": px, "High": [p + 0.5 for p in px], "Low": [p - 0.5 for p in px], "Close": px},
        index=idx,
    )


def cand(detected=et(2026, 9, 23, 10, 0), price=100.0, session=date(2026, 9, 23)):
    return Candidate(
        session=session,
        setup=Setup.CATALYST_GAP,
        symbol="AMPG",
        detected_at=detected,
        price=price,
        prev_close=95.0,
        change=0.05,
    )


TWO_DAYS = pd.concat([bars(date(2026, 9, 23)), bars(date(2026, 9, 24), start_price=110.0)])


def test_all_horizons_after_next_close():
    out = compute(cand(), TWO_DAYS, et(2026, 9, 24, 16, 30))
    assert set(out) == set(KEYS)
    # detection 10:00 at 100; bar at 10:30 opens at 100 + 12*0.1
    assert out["r30"] == pytest.approx(101.2 / 100 - 1)
    assert out["next_open"] == pytest.approx(0.10)


def test_mid_session_only_fills_what_is_knowable():
    out = compute(cand(), TWO_DAYS, et(2026, 9, 23, 10, 40))
    assert set(out) == {"r30"}


def test_horizon_past_the_close_is_final_none():
    out = compute(cand(detected=et(2026, 9, 23, 15, 0)), TWO_DAYS, et(2026, 9, 23, 16, 30))
    assert out["r120"] is None and out["r60"] is None and out["r30"] is not None


def test_early_close_day():
    day = date(2026, 11, 27)
    b = bars(day, close_at="13:00")
    c = cand(detected=et(2026, 11, 27, 12, 0), session=day)
    out = compute(c, b, et(2026, 11, 27, 13, 20))
    assert out["r60"] is None  # 13:00 close comes first
    assert "close" in out


def test_next_session_skips_weekend_and_holiday():
    assert next_trading_day(date(2026, 9, 25)) == date(2026, 9, 28)
    assert next_trading_day(date(2026, 9, 4)) == date(2026, 9, 8)  # Labor Day


def test_zero_price_candidate_yields_nothing():
    assert compute(cand(price=0.0), TWO_DAYS, et(2026, 9, 24, 16, 30)) == {}


def test_no_bars_yields_nothing():
    assert compute(cand(), None, et(2026, 9, 24, 16, 30)) == {}
    assert compute(cand(), TWO_DAYS.iloc[0:0], et(2026, 9, 24, 16, 30)) == {}


def test_mfe_counts_only_bars_after_detection():
    b = bars(date(2026, 9, 23))
    b.loc[b.index[0], "High"] = 999.0  # 09:30 spike, before a 10:02 detection
    out = compute(cand(detected=et(2026, 9, 23, 10, 2)), b, et(2026, 9, 23, 16, 30))
    assert out["mfe"] < 1.0


@pytest.fixture
def store(tmp_path):
    s = ScannerStore(tmp_path / "o.db")
    yield s
    s.close()


def test_fill_is_idempotent_and_never_overwrites(store):
    store.add(cand())
    calls = []

    def fetch(symbol, start, end):
        calls.append(symbol)
        return TWO_DAYS

    first = fill_outcomes(store, et(2026, 9, 23, 16, 30), bars_fn=fetch)
    assert first.updated == 1
    before = dict(store.list()[0].outcomes)

    # A later run with different bars must not rewrite what was written.
    shifted = TWO_DAYS.assign(Open=TWO_DAYS.Open * 2, Close=TWO_DAYS.Close * 2)
    fill_outcomes(store, et(2026, 9, 24, 16, 30), bars_fn=lambda *a: shifted)
    after = store.list()[0].outcomes
    for k, v in before.items():
        assert after[k] == v
    assert {"next_open", "next_close"} <= set(after)

    # Complete rows are not fetched again.
    calls.clear()
    fill_outcomes(store, et(2026, 9, 25, 16, 30), bars_fn=fetch)
    assert calls == []


def test_missing_bars_reported(store):
    store.add(cand())
    r = fill_outcomes(store, et(2026, 9, 23, 16, 30), bars_fn=lambda *a: None)
    assert r.no_bars == ["AMPG"] and r.updated == 0


def test_report_keeps_unchecked_apart_from_no_news():
    checked = cand()
    checked.news_checked = True
    checked.outcomes = {"close": 0.02}
    unchecked = cand()
    unchecked.outcomes = {"close": -0.01}
    rows = summarize([checked, unchecked])
    labels = {r["catalyst"] for r in rows}
    assert labels == {"no catalyst", "unchecked"}


def test_report_empty():
    assert summarize([]) == []
