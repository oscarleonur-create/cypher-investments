"""The pick's bracket replayed: +5% target, -2.5% stop from the entry, else out on time.

User, 2026-09-30: "los picks son diarios, el backtest tiene que tener una salida
clara, +5% -2.5% del entry price".
"""

from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
import pytest
from advisor.breadth import plan_replay as PR
from advisor.breadth.store import BreadthStore
from advisor.daemon.market_calendar import MARKET_TZ

NAN = np.nan


def bars(closes, highs=None, lows=None, opens=None):
    c = np.array(closes, float)
    pick = lambda x: np.array(x if x is not None else closes, float)  # noqa: E731
    return c, pick(highs), pick(lows), pick(opens)


def run(closes, hold=3, **kw):
    c, h, lo, o = bars(closes, **kw)
    return PR.trade(c, h, lo, o, 0, hold)


class TestTrade:
    def test_levels_are_fixed_from_the_entry(self):
        assert PR.levels(100.0) == pytest.approx((105.0, 97.5))

    def test_target_reached_fills_at_the_target(self):
        t = run([100, 101, 103, 102], highs=[100, 102, 106, 102])
        assert t["exit"] == "target" and t["held"] == 2 and t["ret"] == pytest.approx(0.05)
        assert t["plain"] == pytest.approx(0.02)  # the 20-session-style hold, beside it

    def test_stop_reached_fills_at_the_stop(self):
        t = run([100, 99, 98, 110], lows=[100, 97.5, 98, 110])  # exactly at the stop
        assert t["exit"] == "stop" and t["held"] == 1 and t["ret"] == pytest.approx(-0.025)

    def test_gaps_fill_at_the_open(self):
        down = run([100, 90, 91, 92], opens=[100, 91, 91, 92], lows=[100, 89, 90, 91])
        assert down["exit"] == "stop" and down["ret"] == pytest.approx(-0.09)
        up = run([100, 108, 107, 106], opens=[100, 107, 107, 106], highs=[100, 109, 108, 107])
        assert up["exit"] == "target" and up["ret"] == pytest.approx(0.07)

    def test_both_in_one_session_counts_the_stop(self):
        t = run([100, 100, 101, 102], highs=[100, 106, 101, 102], lows=[100, 97, 101, 102])
        assert t["exit"] == "stop" and t["both"] and t["ret"] == pytest.approx(-0.025)

    def test_neither_reached_is_out_at_the_last_close(self):
        t = run([100, 101, 102, 103])
        assert t["exit"] == "time" and t["held"] == 3 and t["ret"] == pytest.approx(0.03)

    def test_incomplete_window_or_no_entry_price_is_no_trade(self):
        assert run([100, 101, 102]) is None
        assert run([NAN, 101, 102, 103]) is None
        assert run([0.0, 101, 102, 103]) is None

    def test_missing_sessions_are_skipped_not_read_as_zero(self):
        t = run([100, NAN, NAN, 103])
        assert t["exit"] == "time" and t["ret"] == pytest.approx(0.03)
        assert run([100, NAN, NAN, NAN]) is None

    def test_missing_high_low_open_fall_back_to_the_close(self):
        t = run([100, 96, 100, 100], highs=[100, NAN, 100, 100], lows=[100, NAN, 100, 100],
                opens=[100, NAN, 100, 100])  # fmt: skip
        assert t["exit"] == "stop" and t["ret"] == pytest.approx(-0.04)  # the close is below


def row(grp, k, day, ret, exit_, peers=0.0, plain=0.02, both=False, held=2):
    return {"grp": grp, "offset": k, "day": day, "ret": ret, "exit": exit_, "peers": peers,
            "peers_hold": 0.01, "plain": plain, "both": both, "held": held}  # fmt: skip


class TestSummarize:
    def test_exit_shares_expectancy_and_every_cell_listed(self):
        rows = [row("F+P", 0, date(2024, 1, 1), 0.05, "target", peers=0.01),
                row("F+P", 0, date(2024, 3, 1), -0.025, "stop", both=True),
                row("F+P", 0, date(2024, 5, 1), 0.01, "time", held=20)]  # fmt: skip
        cells = {(c["group"], c["offset"]): c for c in PR.summarize(rows)}
        assert set(cells) == {(g, k) for g in PR.GROUPS for k in PR.OFFSETS}
        c = cells[("F+P", 0)]
        assert c["exits"] == pytest.approx({"target": 1 / 3, "stop": 1 / 3, "time": 1 / 3})
        assert c["both"] == pytest.approx(1 / 3) and c["held_median"] == 2
        assert c["ret"] == pytest.approx(0.035 / 3)
        assert c["excess"]["mean"] == pytest.approx((0.035 - 0.01) / 3)
        assert c["time_ret"] == pytest.approx(0.01)
        assert c["raw"]["verdict"] == "UNDETERMINED"  # three trades prove nothing
        assert cells[("2+", 5)]["n"] == 0 and cells[("2+", 5)]["reason"] == "no trades"

    def test_no_time_exits_has_no_time_return(self):
        (c, *_) = PR.summarize([row("F+P", 0, date(2024, 1, 1), 0.05, "target")])
        assert c["time_ret"] is None


def test_runs_are_kept_per_window_under_the_current_plan_rules(tmp_path):
    from advisor.breadth.ruleset import plan_rules

    cur = plan_rules().version
    with BreadthStore(tmp_path / "b.db") as s:
        s.conn.executescript(PR._SCHEMA)
        runs = [
            ("a", "2026-09-30T10", cur, 3),
            ("b", "2026-09-30T11", cur, 3),
            ("c", "2026-09-30T12", "sigma-stop", 3),  # the 2026-09-29 plan: not shown
            ("d", "2026-09-30T09", cur, 2),
        ]
        for rid, at, rules, years in runs:
            s.conn.execute(
                "INSERT INTO breadth_plan_runs VALUES (?, ?, ?, '{}', ?)",
                (rid, at, rules, json.dumps({"run_id": rid, "years": years})),
            )
        got = PR.latest_plan_runs(s)
    assert [r["run_id"] for r in got] == ["b", "d"]


def test_the_exits_version_the_plan_not_the_signals(monkeypatch):
    from advisor.breadth.ruleset import plan_rules, signal_rules

    before_plan, before_signals = plan_rules().version, signal_rules().version
    monkeypatch.setattr(PR, "STOP_LOSS", 0.03)
    assert plan_rules().version != before_plan
    assert signal_rules().version == before_signals


def test_no_bars_is_said(tmp_path):
    with BreadthStore(tmp_path / "b.db") as s:
        out = PR.replay_plan(s, datetime(2026, 9, 30, 10, tzinfo=MARKET_TZ))
    assert not out["ok"] and "no bars" in out["error"]
