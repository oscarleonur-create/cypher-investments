"""Picks: who qualifies, the order, the rationale, storage and the endpoint."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from advisor.breadth import measure as M
from advisor.breadth import picks as PK
from advisor.breadth import signals as S
from advisor.breadth.store import BreadthStore
from advisor.daemon.market_calendar import MARKET_TZ

NOW = datetime(2026, 9, 25, 21, 0, tzinfo=MARKET_TZ)


def pick(families, accel=0.0, since_row=100, symbol="X"):
    f = {"growth": 0.3 + accel, "growth_before": 0.3} if "F" in families else None
    return {"symbol": symbol, "families": families, "f": f, "since_row": since_row}


class TestOrder:
    def test_more_families_first_then_f_then_acceleration_then_freshness(self):
        ps = [
            pick(["I", "P"], symbol="ip"),
            pick(["F", "P"], accel=0.05, symbol="fp_small"),
            pick(["F", "I", "P"], symbol="fip"),
            pick(["F", "P"], accel=0.30, symbol="fp_big"),
            pick(["F", "P"], accel=0.30, since_row=200, symbol="fp_big_newer"),
        ]
        ps.sort(key=PK.order_key)
        assert [p["symbol"] for p in ps] == ["fip", "fp_big_newer", "fp_big", "fp_small", "ip"]


class TestRationale:
    def full(self):
        return {
            "since": "2026-09-14",
            "sessions_since": 9,
            "move_since": 0.034,
            "f": {"quarter": "2026Q3", "quarter_end": "2026-07-31", "revenue": 23.5e6,
                  "growth": 0.698, "growth_before": -0.23, "known": "2026-09-14"},
            "i": {"buyers": 3, "value": 548_278.0, "filed": "2026-09-12"},
            "p": {"momentum_on": True, "momentum_12_1": 1.072, "high_on": True,
                  "from_52w_high": -0.015, "breakout_on": True, "breakout_day": "2026-09-11",
                  "breakout_move": 0.424, "breakout_rvol": 6.1},
        }  # fmt: skip

    def test_every_reason_has_a_source_and_uses_the_picks_numbers(self):
        r = PK.rationale(self.full(), 2302)
        assert all(x["source"] for x in r["reasons"])
        text = " ".join(x["text"] for x in r["reasons"])
        for expected in ("2026-07-31", "$23.5M", "+69.8%", "-23.0%", "+107.2%", "2,302",
                         "-1.5%", "+42.4%", "6.1x", "3 officers", "$548.3k", "+3.4%"):  # fmt: skip
            assert expected in text, expected
        assert len(r["invalidates"]) == 3

    def test_it_never_says_buy_or_names_a_value(self):
        r = PK.rationale(self.full(), 2302)
        words = " ".join(x["text"] for x in r["reasons"]).lower() + " ".join(r["invalidates"])
        for banned in ("buy now", "target", "fair value", "worth", "undervalued", "should"):
            assert banned not in words

    def test_only_what_is_active(self):
        p = self.full()
        p["i"] = None
        p["p"] = None
        r = PK.rationale(p, 100)
        assert {x["family"] for x in r["reasons"]} == {"F", ""}
        assert r["invalidates"] == ["the next quarter's year-on-year growth slows"]

    def test_missing_numbers_do_not_crash(self):
        p = self.full()
        p["move_since"] = None
        p["p"] = {
            "momentum_on": True,
            "momentum_12_1": None,
            "breakout_on": True,
            "breakout_day": "2026-09-11",
            "breakout_move": None,
            "breakout_rvol": None,
        }
        r = PK.rationale(p, 10)
        assert "n/a" in " ".join(x["text"] for x in r["reasons"])


class TestActive:
    N = 120

    def arrays(self):
        elig = np.ones((self.N, 1), bool)
        p_recent = np.zeros((self.N, 1), bool)
        return elig, p_recent

    def test_two_families_needed(self):
        elig, p_recent = self.arrays()
        f = {"A": [S.FEvent("A", 100, "Q", 1e8, 0.3, 0.1)]}
        assert PK.active_at("A", 110, 0, f, {}, p_recent, elig) is None
        p_recent[95:, 0] = True
        a = PK.active_at("A", 110, 0, f, {}, p_recent, elig)
        assert a.families == ["F", "P"] and a.since_row == 100

    def test_f_expires_after_the_window(self):
        elig, p_recent = self.arrays()
        p_recent[:, 0] = True
        f = {"A": [S.FEvent("A", 50, "Q", 1e8, 0.3, 0.1)]}
        assert PK.active_at("A", 50 + S.CONVERGE_SESSIONS, 0, f, {}, p_recent, elig)
        assert PK.active_at("A", 51 + S.CONVERGE_SESSIONS, 0, f, {}, p_recent, elig) is None

    def test_not_eligible_today_is_no_pick(self):
        elig, p_recent = self.arrays()
        p_recent[:, 0] = True
        elig[110, 0] = False
        f = {"A": [S.FEvent("A", 100, "Q", 1e8, 0.3, 0.1)]}
        assert PK.active_at("A", 110, 0, f, {}, p_recent, elig) is None

    def test_since_stops_at_a_break_in_eligibility(self):
        elig, p_recent = self.arrays()
        p_recent[:, 0] = True
        elig[104, 0] = False
        f = {"A": [S.FEvent("A", 100, "Q", 1e8, 0.3, 0.1)]}
        assert PK.active_at("A", 110, 0, f, {}, p_recent, elig).since_row == 105


# ── building, storing, serving ────────────────────────────────────────────

SYMBOLS = ["AAA", "BBB", "CCC"]


def _days(n=300, end=date(2026, 9, 25)):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


@pytest.fixture
def store(tmp_path):
    s = BreadthStore(tmp_path / "research-breadth.db")
    days = _days()
    with s.conn:
        for sym in SYMBOLS:
            s.conn.executemany(
                "INSERT INTO breadth_bars (symbol, day, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(sym, d.isoformat(), 50, 50, 50, 50.0, 1e6) for d in days],
            )
        s.conn.executemany(
            "INSERT INTO breadth_universe (day, symbol, cik, exchange, name, eligible, reason, "
            "price, dollar_volume, sessions, rules) VALUES (?, ?, ?, 'X', ?, 1, NULL, 50, 5e7, "
            "300, 'v')",
            [("2026-09-25", s_, i + 1, f"{s_} Inc") for i, s_ in enumerate(SYMBOLS)],
        )
    yield s
    s.close()


@pytest.fixture
def inputs(monkeypatch):
    """AAA: F and I and (flat at its high, so) P. BBB: P only. CCC: nothing (below its high)."""

    def fake(store, panel, tally=None, t=S.DEFAULT):
        elig = pd.DataFrame(True, index=panel.close.index, columns=panel.close.columns)
        last = len(panel.sessions) - 1
        panel.high.loc[:, "CCC"] = 60.0  # CCC sits 17% under its high: no P
        events = [S.FEvent("AAA", last - 3, "2026Q3", 1e8, 0.4, 0.2, date(2026, 7, 31))]
        ievents = [S.IEvent("AAA", last - 2, 3, 5e5, date(2026, 9, 21))]
        return {"AAA": 1, "BBB": 2, "CCC": 3}, elig, events, ievents

    monkeypatch.setattr(M, "_inputs", fake)


def test_build_store_and_read_back(store, inputs):
    r = PK.build_picks(store, date(2026, 9, 25), NOW)
    assert r["ok"] and r["candidates"] == 1
    (p,) = r["picks"]
    assert p["symbol"] == "AAA" and p["families"] == ["F", "I", "P"]
    assert p["f"]["quarter_end"] == "2026-07-31" and p["reasons"]
    back = PK.latest_picks(store)
    assert back["day"] == "2026-09-25" and back["picks"][0]["rank"] == 1
    assert back["picks"][0]["reasons"] == p["reasons"]


def test_rebuilding_a_day_rewrites_it(store, inputs):
    PK.build_picks(store, date(2026, 9, 25), NOW)
    with store.conn:
        store.conn.execute(
            "INSERT INTO breadth_picks VALUES ('2026-09-25', 9, 'GONE', 'v', '{}', 'x')"
        )
    PK.build_picks(store, date(2026, 9, 25), NOW)
    assert [r[0] for r in store.conn.execute("SELECT symbol FROM breadth_picks")] == ["AAA"]


def test_a_day_without_bars_is_reported(store, inputs):
    r = PK.build_picks(store, date(2026, 9, 26), NOW)  # a Saturday
    assert not r["ok"] and "no bars" in r["error"]


def test_no_picks_yet(tmp_path):
    with BreadthStore(tmp_path / "b.db") as s:
        out = PK.latest_picks(s)
    assert out["day"] is None and out["picks"] == [] and out["track_record"] == []


def test_track_record_shows_every_window_of_the_current_rules(store):
    from advisor.breadth.ruleset import signal_rules

    cur = signal_rules().version
    cell = {"group": "2+", "horizon": "d20", "n": 5, "verdict": "UNDETERMINED"}
    runs = [
        ("r3", "2026-09-28T10:00", cur, 3),
        ("r2old", "2026-09-28T09:00", cur, 2),
        ("r2", "2026-09-28T11:00", cur, 2),
        ("rold", "2026-09-28T12:00", "old-rules", 3),
    ]
    with store.conn:
        for rid, at, rules, years in runs:
            store.conn.execute(
                "INSERT INTO breadth_replay_runs VALUES (?, ?, ?, 'c', ?, ?, ?)",
                (rid, at, at, rules, json.dumps({"years": years}),
                 json.dumps({"from": "a", "to": "b", "cells": [cell], "cells_tested": 15})),
            )  # fmt: skip
    got = PK._replay_record(store)
    assert [(w["years"], w["run_id"]) for w in got] == [(3, "r3"), (2, "r2")]


def test_the_endpoint_serves_the_stored_picks(store, inputs, monkeypatch, tmp_path):
    from advisor.api.routers import breadth as router

    PK.build_picks(store, date(2026, 9, 25), NOW)
    monkeypatch.setattr(router, "_db", lambda: tmp_path / "research.db")
    out = router._latest()
    assert [p["symbol"] for p in out["picks"]] == ["AAA"]


def test_the_endpoint_without_a_breadth_store(monkeypatch, tmp_path):
    from advisor.api.routers import breadth as router

    monkeypatch.setattr(router, "_db", lambda: tmp_path / "nothing" / "research.db")
    assert router._latest()["picks"] == []
