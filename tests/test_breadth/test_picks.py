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

    def test_a_high_left_since_is_said_as_such(self):
        p = self.full()
        p["p"]["from_52w_high"] = -0.081
        text = " ".join(x["text"] for x in PK.rationale(p, 10)["reasons"])
        assert "Came within 2% of its 52-week high in the last month; now -8.1%" in text
        assert "Within 2% of its 52-week high (now -8.1%)" not in text

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


# ── the entry plan ────────────────────────────────────────────────────────


def track(group="F+P", n=1561, mean=0.023, tail=-0.067, verdict="EDGE", years=3):
    cell = {"group": group, "horizon": "d20", "n": n, "mean": mean, "control": 0.011,
            "excess": 0.012, "ci": [0.003, 0.022], "beat": 0.52, "tail": tail,
            "verdict": verdict}  # fmt: skip
    return {"years": years, "from": "2023-09-27", "to": "2026-09-25", "cells": [cell]}


def plan_pick(families=("F", "P"), sessions_since=0, price=100.0, **kw):
    p = TestRationale().full()
    p.update(symbol="X", families=list(families), sessions_since=sessions_since, price=price,
             since="2026-09-28", held=False)  # fmt: skip
    if "F" not in families:
        p["f"] = None
    if "I" not in families:
        p["i"] = None
    p.update(kw)
    return p


def plan(p, trk=None, sigma=0.02, measured=100.0, net_liq=10_000.0):
    return PK.entry_plan(
        p, [track()] if trk is None else trk, sigma=sigma, measured_price=measured,
        measured_end="2026-10-26", review_on="2026-10-26", net_liq=net_liq, day="2026-09-28",
    )  # fmt: skip


class TestEntryPlan:
    """User, 2026-09-29: "no me dice entra a x precio dado que está sucediendo esto y
    esperamos esto"."""

    def test_fresh_pick_enter_because_expecting_with_the_engines_stop_and_size(self):
        from advisor.entry.proposal import position_stop_pct

        pl = plan(plan_pick())
        assert pl["ok"] and pl["stage"] == "fresh" and pl["group"] == "F+P"
        assert pl["summary"].startswith("Enter near $100.00 (close 2026-09-28) because revenue")
        assert "+69.8% a year" in pl["because"] and "the price is strong" in pl["because"]
        stop_pct = position_stop_pct(0.02)  # 2·0.02·√10 = 12.6%
        assert pl["stop"] == pytest.approx(100 * (1 - stop_pct))
        assert pl["size"]["shares"] == int(10_000 * 0.02 // (100 - pl["stop"]))
        (e,) = pl["expect"]
        assert e["price_mean"] == pytest.approx(102.3) and e["price_tail"] == pytest.approx(93.3)
        assert "rose +2.3% on average over 20 sessions against +1.1%" in e["text"]
        assert "EDGE" in e["text"] and "52%" in e["text"]
        assert pl["expects"][0].startswith("the next quarter's revenue grows at least +69.8%")

    def test_a_late_pick_is_told_the_record_does_not_reach_it(self):
        pl = plan(plan_pick(sessions_since=17, move_since=-0.134), measured=410.4)
        assert pl["stage"] == "late"
        assert pl["summary"].startswith("If entered now: near $100.00")
        assert "17 of the 20 measured sessions" in pl["timing"]
        assert "entering now was not measured" in pl["timing"]
        # The group's average is priced from the measured entry, not today's.
        assert pl["expect"][0]["price_mean"] == pytest.approx(410.4 * 1.023)

    @pytest.mark.parametrize("n,stage", [(0, "fresh"), (1, "late"), (19, "late"), (20, "past"),
                                         (45, "past")])  # fmt: skip
    def test_stage_at_the_window_boundaries(self, n, stage):
        pl = plan(plan_pick(sessions_since=n))
        assert pl["stage"] == stage
        if stage == "past":
            assert "the record says nothing about entering now" in pl["timing"]

    @pytest.mark.parametrize(
        "fams,group",
        [(("F", "P"), "F+P"), (("I", "P"), "2+"), (("F", "I", "P"), "2+"), (("F", "I"), "2+")],
    )
    def test_group_is_the_one_the_replay_judged(self, fams, group):
        assert PK.plan_group(list(fams)) == group

    def test_every_window_is_quoted_and_empty_cells_skipped(self):
        trk = [track(), track(mean=0.016, verdict="UNDETERMINED", years=2), track(group="2+")]
        trk.append({"years": 1, "cells": [{"group": "F+P", "horizon": "d20", "n": 0}]})
        pl = plan(plan_pick(), trk=trk)
        assert [e["years"] for e in pl["expect"]] == [3, 2]
        assert plan(plan_pick(), trk=[])["expect"] == []

    def test_no_volatility_no_stop_no_size(self):
        pl = plan(plan_pick(), sigma=None)
        assert pl["stop"] is None and pl["size"]["note"].startswith("no volatility")
        assert "Stop" not in pl["summary"]

    def test_no_book_or_held_is_no_size(self):
        assert plan(plan_pick(), net_liq=None)["size"]["note"] == "no book on file: no size"
        assert plan(plan_pick(), net_liq=0.0)["size"]["note"] == "no book on file: no size"
        assert "already held" in plan(plan_pick(held=True))["size"]["note"]

    def test_size_is_capped_at_the_book_limit(self):
        pl = plan(plan_pick(), sigma=0.001)  # stop at the 8% floor: 2% risk = 25% of the book
        assert pl["stop_pct"] == pytest.approx(0.08)
        assert pl["size"]["shares"] == 20 and "20% book limit" in pl["size"]["note"]

    def test_one_share_above_the_budget_is_said(self):
        pl = plan(plan_pick(price=5000.0), measured=5000.0, net_liq=8000.0)
        assert pl["size"]["shares"] == 0 and "one share ($5,000.00)" in pl["size"]["note"]

    @pytest.mark.parametrize("price", [None, 0.0, float("nan"), -3.0])
    def test_no_price_no_plan(self, price):
        pl = plan(plan_pick(price=price))
        assert not pl["ok"] and "no price" in pl["gap"]

    def test_no_measured_price_quotes_percentages_only(self):
        (e,) = plan(plan_pick(), measured=None)["expect"]
        assert "price_mean" not in e and "rose +2.3%" in e["text"]

    def test_provisional_says_live(self):
        assert "(live 2026-09-28)" in plan(plan_pick(provisional=True))["summary"]

    def test_it_names_no_value_and_no_target(self):
        pl = plan(plan_pick())
        words = (pl["summary"] + pl["because"] + " ".join(pl["expects"])).lower()
        for banned in ("target", "fair value", "worth", "undervalued"):
            assert banned not in words


def plan_run(years=3, group="F+P", offsets=(0, 5, 10, 15)):
    def side(mean, ex, verdict):
        return {"mean": mean, "excess": ex, "ci": [ex - 0.01, ex + 0.01], "verdict": verdict}

    cells = [
        {"group": group, "offset": k, "n": 1000 - k, "peers": 0.01, "stopped": 0.15,
         "stop_effect": -0.004, "with_stop": side(0.018, 0.008, "UNDETERMINED"),
         "plain": side(0.022, 0.012, "EDGE")}
        for k in offsets
    ]  # fmt: skip
    return {"years": years, "from": "2023-09-29", "to": "2026-09-28", "cells": cells}


class TestPlanReplayed:
    """The plan as shown — its stop, the delay it is entered at — replayed."""

    @pytest.mark.parametrize("n,offset", [(0, 0), (2, 0), (3, 5), (7, 5), (8, 10), (12, 10),
                                          (13, 15), (19, 15)])  # fmt: skip
    def test_the_nearest_measured_delay(self, n, offset):
        (r,) = PK.plan_record("F+P", n, [plan_run()])
        assert r["offset"] == offset

    def test_past_the_window_nothing_was_measured(self):
        assert PK.plan_record("F+P", 20, [plan_run()]) == []

    def test_other_group_or_missing_cell_is_left_out(self):
        assert PK.plan_record("2+", 0, [plan_run()]) == []
        assert PK.plan_record("F+P", 7, [plan_run(offsets=(0,))]) == []
        assert PK.plan_record("F+P", 0, []) == []

    def test_text_says_both_with_and_without_the_stop(self):
        (r,) = PK.plan_record("F+P", 6, [plan_run()])
        assert "entering 5 sessions after the first day (995 trades)" in r["text"]
        assert "with the stop +1.8%" in r["text"] and "stopped out 15%" in r["text"]
        assert "Without the stop +2.2%" in r["text"] and "EDGE" in r["text"]

    def test_a_late_plan_points_to_the_measured_delay(self):
        p = plan_pick(sessions_since=6, move_since=0.02)
        pl = PK.entry_plan(p, [track()], sigma=0.02, measured_price=100.0,
                           measured_end="2026-10-26", review_on="2026-10-26", net_liq=1e4,
                           day="2026-09-28", plan_runs=[plan_run(), plan_run(years=2)])  # fmt: skip
        assert [r["years"] for r in pl["replayed"]] == [3, 2]
        assert "the plan replay measured entering 5 sessions in" in pl["timing"]
        assert "not measured" not in pl["timing"]

    def test_without_a_plan_run_the_plan_reads_as_before(self):
        pl = plan(plan_pick(sessions_since=6))
        assert pl["replayed"] == [] and "entering now was not measured" in pl["timing"]


class TestFaded:
    """P counts a state on at any session of the window; the text says which still holds."""

    def test_momentum_left_the_top_decile(self):
        p = plan_pick(families=("I", "P"))
        p["p"] = {"momentum_on": True, "momentum_now": False, "momentum_12_1": -0.026}
        text = " ".join(x["text"] for x in PK.rationale(p, 2298)["reasons"])
        assert "now -2.6%, out of it" in text and "-2.6%, in the top decile" not in text
        because = PK._because(p)
        assert "the price was strong (was top decile of 12-month returns, now -2.6%)" in because

    def test_an_old_payload_without_the_flag_reads_as_before(self):
        p = plan_pick()
        p["p"] = {"momentum_on": True, "momentum_12_1": 1.14}
        assert "in the top decile" in PK.rationale(p, 10)["reasons"][1]["text"]

    def test_high_near_or_left(self):
        p = plan_pick(families=("I", "P"))
        p["p"] = {"high_on": True, "from_52w_high": -0.02}
        assert "within 2% of its 52-week high (-2.0%)" in PK._because(p)
        p["p"]["from_52w_high"] = -0.109
        assert "came within 2% of its 52-week high this month, now -10.9%" in PK._because(p)


class TestHelpers:
    def test_sigma_leaves_the_last_session_out_and_needs_twenty_returns(self):
        closes = np.array([100.0, 101.0] * 15 + [200.0])  # the last session's jump is excluded
        s = PK.sigma_before(closes)
        assert s == pytest.approx(np.std(np.diff(closes[:-1]) / closes[:-2], ddof=1))
        assert PK.sigma_before(np.array([100.0] * 10)) is None
        assert PK.sigma_before(np.array([100.0] * 40)) is None  # flat: no volatility
        with_gaps = np.array([100.0, np.nan, 101.0] * 15 + [100.0])
        assert PK.sigma_before(with_gaps) is not None

    def test_sessions_skip_weekends_and_holidays(self):
        assert PK.after_sessions(date(2026, 11, 25), 1) == date(2026, 11, 27)  # Thanksgiving
        assert PK.after_sessions(date(2026, 9, 25), 1) == date(2026, 9, 28)  # a Friday
        assert PK.after_sessions(date(2026, 9, 28), 20) == date(2026, 10, 26)
        assert PK.after_sessions(date(2026, 9, 28), 0) == date(2026, 9, 28)


def test_built_picks_carry_a_plan(store, inputs, tmp_path):
    from advisor.daemon.book import BookSnapshot
    from advisor.daemon.store import DaemonStore

    db = tmp_path / "research.db"
    DaemonStore(db).save_book(BookSnapshot(net_liq=7700.0))
    r = PK.build_picks(store, date(2026, 9, 25), NOW, db)
    pl = r["picks"][0]["plan"]
    assert pl["ok"] and pl["group"] == "2+" and pl["review_on"] == "2026-10-23"
    # Flat test bars: no volatility, so no stop and no size — said, not guessed.
    assert pl["stop"] is None and pl["size"]["note"].startswith("no volatility")
    assert PK.latest_picks(store)["picks"][0]["plan"] == pl


def test_every_pick_gets_its_own_verdict_with_the_measured_record(store, monkeypatch):
    """Two picks, a stored value test: each verdict quotes it (a shadowed name once
    handed the second pick a price where the record should be)."""
    from advisor.breadth import verdict as V

    def fake(store, panel, tally=None, t=S.DEFAULT):
        elig = pd.DataFrame(True, index=panel.close.index, columns=panel.close.columns)
        last = len(panel.sessions) - 1
        events = [S.FEvent(s, last - 3, "2026Q3", 1e8, 0.4, 0.2, date(2026, 7, 31))
                  for s in ("AAA", "BBB")]  # fmt: skip
        ievents = [S.IEvent(s, last - 2, 3, 5e5, date(2026, 9, 21)) for s in ("AAA", "BBB")]
        return {"AAA": 1, "BBB": 2, "CCC": 3}, elig, events, ievents

    monkeypatch.setattr(M, "_inputs", fake)
    card = {"verdict": "above_bear", "own_margins": [{"value": 0.2}], "rationale": [],
            "cases": [{"name": "bear", "value_per_share": 40.0},
                      {"name": "base", "value_per_share": 55.0},
                      {"name": "bull", "value_per_share": 70.0},
                      {"name": "market", "value_per_share": 50.0, "margin": 0.15}]}  # fmt: skip
    monkeypatch.setattr(V, "value_card", lambda *a, **k: (card, ""))
    record = {"below_base": {"n": 402, "excess": 0.0231, "ci": None, "verdict": "EDGE"}}
    monkeypatch.setattr(V, "evidence", lambda store: record)
    r = PK.build_picks(store, date(2026, 9, 25), NOW)
    assert [p["symbol"] for p in r["picks"]] == ["AAA", "BBB"]
    for p in r["picks"]:
        v = p["verdict"]
        assert v["action"] == "ENTER" and v["entry"] == 55.0 and v["target"] == 70.0
        assert v["evidence"].startswith("Measured: 402 past picks like this beat peers by +2.3%")
