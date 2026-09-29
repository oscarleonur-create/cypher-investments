"""The entry plan replayed: its stop and a later entry, which the signal replay never tested.

User, 2026-09-29: "si esto no estaba, ¿cómo hacíamos backtesting?"
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest
from advisor.breadth import plan_replay as PR
from advisor.breadth.store import BreadthStore
from advisor.daemon.market_calendar import MARKET_TZ

NAN = np.nan


def series(closes, lows=None, opens=None):
    c = np.array(closes, float)
    lo = np.array(lows if lows is not None else closes, float)
    op = np.array(opens if opens is not None else closes, float)
    return c, lo, op


class TestTrade:
    def test_no_stop_touched_both_are_the_close_at_the_horizon(self):
        c, lo, op = series([100, 101, 102, 103, 104, 110])
        t = PR.trade(c, lo, op, 0, 90.0, 5)
        assert t["plain"] == pytest.approx(0.10) and t["with_stop"] == pytest.approx(0.10)
        assert not t["stopped"] and t["held"] == 5

    def test_a_low_through_the_stop_exits_at_the_stop(self):
        c, lo, op = series([100, 99, 98, 97, 120, 130], lows=[100, 99, 89, 97, 120, 130])
        t = PR.trade(c, lo, op, 0, 90.0, 5)
        assert t["stopped"] and t["held"] == 2
        assert t["with_stop"] == pytest.approx(-0.10) and t["plain"] == pytest.approx(0.30)

    def test_a_gap_below_the_stop_fills_at_the_open(self):
        c, lo, op = series([100, 80, 81, 82], opens=[100, 82, 81, 82], lows=[100, 79, 80, 81])
        t = PR.trade(c, lo, op, 0, 90.0, 3)
        assert t["stopped"] and t["with_stop"] == pytest.approx(-0.18)

    def test_exactly_at_the_stop_is_stopped(self):
        c, lo, op = series([100, 95, 96], lows=[100, 90, 96])
        assert PR.trade(c, lo, op, 0, 90.0, 2)["stopped"]

    def test_incomplete_window_or_no_entry_price_is_no_trade(self):
        c, lo, op = series([100, 101, 102])
        assert PR.trade(c, lo, op, 0, 90.0, 5) is None
        c, lo, op = series([NAN, 101, 102, 103])
        assert PR.trade(c, lo, op, 0, 90.0, 2) is None
        c, lo, op = series([0.0, 101, 102, 103])
        assert PR.trade(c, lo, op, 0, 90.0, 2) is None

    def test_missing_bars_are_skipped_not_read_as_zero(self):
        c = np.array([100, NAN, NAN, 105], float)
        t = PR.trade(c, c.copy(), c.copy(), 0, 90.0, 3)
        assert not t["stopped"] and t["plain"] == pytest.approx(0.05)
        c = np.array([100, NAN, NAN, NAN], float)
        assert PR.trade(c, c.copy(), c.copy(), 0, 90.0, 3) is None

    def test_missing_low_falls_back_to_the_close(self):
        c = np.array([100, 85, 100], float)
        lo = np.array([100, NAN, 100], float)
        op = np.array([100, NAN, 100], float)
        t = PR.trade(c, lo, op, 0, 90.0, 2)
        assert t["stopped"] and t["with_stop"] == pytest.approx(-0.15)  # the close gapped below

    def test_no_stop_given_is_the_plain_trade(self):
        c, lo, op = series([100, 50, 110])
        t = PR.trade(c, lo, op, 0, None, 2)
        assert not t["stopped"] and t["with_stop"] == t["plain"]


def row(grp, k, day, plain, with_stop, peers, stopped=False):
    return {"grp": grp, "offset": k, "day": day, "plain": plain, "with_stop": with_stop,
            "peers": peers, "stopped": stopped, "stop_pct": 0.15, "held": 20}  # fmt: skip


class TestSummarize:
    def test_the_stops_effect_is_the_difference_and_every_cell_is_listed(self):
        rows = [row("F+P", 0, date(2024, 1, 1), 0.05, -0.15, 0.01, True),
                row("F+P", 0, date(2024, 3, 1), 0.10, 0.10, 0.02)]  # fmt: skip
        cells = {(c["group"], c["offset"]): c for c in PR.summarize(rows)}
        assert set(cells) == {(g, k) for g in PR.GROUPS for k in PR.OFFSETS}
        c = cells[("F+P", 0)]
        assert c["n"] == 2 and c["stopped"] == 0.5
        assert c["stop_effect"] == pytest.approx(-0.10)
        assert c["plain"]["excess"] == pytest.approx(0.06)
        assert c["with_stop"]["verdict"] == "UNDETERMINED"  # two records prove nothing
        assert cells[("2+", 5)]["n"] == 0 and cells[("2+", 5)]["reason"] == "no trades"


def test_runs_are_kept_per_window_under_the_current_rules(tmp_path):
    import json

    from advisor.breadth.ruleset import signal_rules

    cur = signal_rules().version
    with BreadthStore(tmp_path / "b.db") as s:
        s.conn.executescript(PR._SCHEMA)
        runs = [
            ("a", "2026-09-29T10", cur, 3),
            ("b", "2026-09-29T11", cur, 3),
            ("c", "2026-09-29T12", "old", 3),
            ("d", "2026-09-29T09", cur, 2),
        ]
        for rid, at, rules, years in runs:
            s.conn.execute(
                "INSERT INTO breadth_plan_runs VALUES (?, ?, ?, '{}', ?)",
                (rid, at, rules, json.dumps({"run_id": rid, "years": years})),
            )
        got = PR.latest_plan_runs(s)
    assert [r["run_id"] for r in got] == ["b", "d"]


def test_no_bars_is_said(tmp_path):
    with BreadthStore(tmp_path / "b.db") as s:
        out = PR.replay_plan(s, datetime(2026, 9, 29, 10, tzinfo=MARKET_TZ))
    assert not out["ok"] and "no bars" in out["error"]
