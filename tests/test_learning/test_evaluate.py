"""The judge: clustered intervals, own-drift baseline, UNDETERMINED by default, calibration."""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest
from advisor.learning.evaluate import (
    MIN_SESSIONS,
    Baselines,
    Record,
    Verdict,
    chance_edges,
    cluster_ci,
    drift,
    evaluate,
    stance_calibration,
    stop_calibration,
    touch_probability,
    verdict,
    wilson,
)

START = date(2025, 1, 2)


def days(n):
    out, d = [], START
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def flat_closes(symbol):
    return {d: 100.0 for d in days(300)}


def rising_closes(symbol):
    # +0.1% a session: a 20-session drift of about +2.02%.
    return {d: 100.0 * 1.001**i for i, d in enumerate(days(300))}


def rec(session, ret, horizon="d20", symbol="AAA", group="action ENTER", **extra):
    return Record(
        ruleset="entry",
        version="v1",
        group=group,
        session=session,
        symbol=symbol,
        outcomes={horizon: ret},
        extra=extra,
    )


class TestDrift:
    def test_flat_prices_have_no_drift(self):
        assert drift(flat_closes("X"), 20) == pytest.approx(0)

    def test_steady_rise(self):
        assert drift(rising_closes("X"), 20) == pytest.approx(1.001**20 - 1)

    def test_same_session_horizon_is_zero(self):
        assert drift(rising_closes("X"), 0) == 0.0

    def test_too_little_history_is_none(self):
        closes = {d: 100.0 for d in days(30)}
        assert drift(closes, 20) is None

    def test_zero_and_missing_prices_are_skipped(self):
        closes = rising_closes("X")
        closes[days(300)[10]] = 0.0
        assert drift(closes, 5) is not None

    def test_closes_fetched_once_per_symbol(self):
        calls = []

        def fn(s):
            calls.append(s)
            return flat_closes(s)

        b = Baselines(fn)
        b.get("A", 5), b.get("A", 20), b.get("B", 5)
        assert calls == ["A", "B"]

    def test_no_price_data_is_none_not_zero(self):
        assert Baselines(lambda s: None).get("A", 20) is None


class TestInterval:
    def test_deterministic(self):
        vals = [(d, 0.01 * (i % 7 - 3)) for i, d in enumerate(days(40))]
        assert cluster_ci(vals) == cluster_ci(vals)

    def test_one_session_has_no_interval(self):
        assert cluster_ci([(START, 0.1), (START, 0.2)]) is None

    def test_many_records_on_one_day_count_as_one_day(self):
        # 100 records on one session plus 11 on others: the interval must be
        # as wide as twelve sessions allow, not as narrow as 111 records would.
        rng = random.Random(1)
        big = [(START, 0.05)] * 100
        rest = [(d, rng.gauss(0, 0.05)) for d in days(12)[1:]]
        lo, hi = cluster_ci(big + rest)
        assert hi - lo > 0.02


class TestVerdict:
    def test_too_few_records(self):
        assert verdict(5, 5, (0.01, 0.02))[0] is Verdict.UNDETERMINED

    def test_enough_records_on_too_few_sessions(self):
        v, why = verdict(100, MIN_SESSIONS - 1, (0.01, 0.02))
        assert v is Verdict.UNDETERMINED and "sessions" in why

    def test_edge_negative_straddle(self):
        assert verdict(50, 20, (0.002, 0.02))[0] is Verdict.EDGE
        assert verdict(50, 20, (-0.02, -0.002))[0] is Verdict.NEGATIVE
        assert verdict(50, 20, (-0.01, 0.01))[0] is Verdict.UNDETERMINED

    def test_exactly_zero_lower_bound_is_not_an_edge(self):
        assert verdict(50, 20, (0.0, 0.02))[0] is Verdict.UNDETERMINED


class TestEvaluate:
    def test_a_rule_riding_the_drift_has_no_edge(self):
        # Every record returns exactly its names' own 20-session drift.
        d20 = 1.001**20 - 1
        records = [rec(d, d20) for d in days(30)]
        (cell,) = evaluate(records, Baselines(rising_closes))
        assert cell.mean == pytest.approx(d20) and cell.excess == pytest.approx(0)
        assert cell.verdict is Verdict.UNDETERMINED

    def test_a_real_edge_over_enough_sessions(self):
        rng = random.Random(3)
        records = [rec(d, 0.03 + rng.gauss(0, 0.01)) for d in days(30)]
        (cell,) = evaluate(records, Baselines(flat_closes))
        assert cell.verdict is Verdict.EDGE and cell.sessions == 30

    def test_few_sessions_show_no_interval(self):
        records = [rec(d, 0.05) for d in days(3) for _ in range(10)]
        (cell,) = evaluate(records, Baselines(flat_closes))
        assert cell.n == 30 and cell.sessions == 3
        assert cell.ci is None and cell.verdict is Verdict.UNDETERMINED

    def test_versions_and_groups_are_never_pooled(self):
        a = [rec(d, 0.05) for d in days(25)]
        b = [rec(d, -0.05, group="action WAIT") for d in days(25)]
        c = [Record("entry", "v2", "action ENTER", d, "AAA", {"d20": 0.0}) for d in days(25)]
        cells = evaluate(a + b + c, Baselines(flat_closes))
        assert {(x.version, x.group) for x in cells} == {
            ("v1", "action ENTER"),
            ("v1", "action WAIT"),
            ("v2", "action ENTER"),
        }

    def test_live_and_replay_are_separate_cells(self):
        live = [rec(d, 0.01) for d in days(25)]
        replay = [rec(d, 0.01) for d in days(25)]
        for r in replay:
            r.origin = "replay"
        assert {c.origin for c in evaluate(live + replay, Baselines(flat_closes))} == {
            "live",
            "replay",
        }

    def test_records_without_price_history_are_counted_not_guessed(self):
        records = [rec(d, 0.01, symbol="NEW") for d in days(25)]
        (cell,) = evaluate(records, Baselines(lambda s: None))
        assert cell.unbased == 25 and cell.excess is None
        assert cell.verdict is Verdict.UNDETERMINED

    def test_tail_reported_with_the_mean(self):
        records = [
            Record("entry", "v1", "g", d, "AAA", {"d20": 0.02, "mae20": -0.10}) for d in days(25)
        ]
        (cell,) = evaluate(records, Baselines(flat_closes))
        assert cell.tail == pytest.approx(-0.10)

    def test_empty_is_empty(self):
        assert evaluate([], Baselines(flat_closes)) == []

    def test_chance_edges_counts_only_judged_cells(self):
        few = [rec(d, 0.05) for d in days(3)]
        many = [rec(d, 0.05, group="g2") for d in days(40)]
        cells = evaluate(few + many, Baselines(flat_closes))
        assert chance_edges(cells) == pytest.approx(0.025)


class TestCalibration:
    def test_touch_probability(self):
        assert touch_probability(0, 1) == pytest.approx(1.0)
        assert touch_probability(1.5, 1) == pytest.approx(0.1336, abs=1e-3)
        assert touch_probability(2, 0) == 0.0

    def test_wilson(self):
        lo, hi = wilson(5, 10)
        assert lo < 0.5 < hi
        assert wilson(0, 0) is None

    def _stop_records(self, hits: int, n: int):
        out = []
        for i, d in enumerate(days(n)):
            r = Record("entry", "v1", "action ENTER", d, "AAA", {"trade_stop": float(i < hits)})
            r.extra = {"sigma": 0.02, "trade_stop_pct": 0.03}
            out.append(r)
        return out

    def test_stops_hit_far_more_than_sigma_implies(self):
        (s,) = stop_calibration(self._stop_records(hits=35, n=40))
        assert "more often" in s["finding"]
        assert s["expected"] == pytest.approx(touch_probability(1.5, 1.5))

    def test_consistent_with_sigma(self):
        (s,) = stop_calibration(self._stop_records(hits=9, n=40))
        assert s["finding"] == "consistent with σ"

    def test_too_few_stops(self):
        (s,) = stop_calibration(self._stop_records(hits=1, n=5))
        assert s["finding"] == "too little data"

    def test_records_without_sigma_are_skipped(self):
        r = Record("entry", "v1", "g", START, "A", {"trade_stop": 1.0})
        assert stop_calibration([r]) == []

    def test_stances_ordered(self):
        rng = random.Random(5)
        recs = []
        for d in days(30):
            recs.append(
                rec(d, 0.04 + rng.gauss(0, 0.01), stance="CONSTRUCTIVE", reading_prompt="p1")
            )
            recs.append(rec(d, -0.04 + rng.gauss(0, 0.01), stance="AT_RISK", reading_prompt="p1"))
        out = stance_calibration(recs, Baselines(flat_closes))
        assert out["p1"]["finding"].startswith("ORDERED")

    def test_stances_inverted(self):
        recs = []
        for d in days(30):
            recs.append(rec(d, -0.04, stance="CONSTRUCTIVE", reading_prompt="p1"))
            recs.append(rec(d, 0.04, stance="AT_RISK", reading_prompt="p1"))
        assert stance_calibration(recs, Baselines(flat_closes))["p1"]["finding"].startswith(
            "INVERTED"
        )

    def test_prompts_are_judged_separately(self):
        recs = [rec(d, 0.04, stance="CONSTRUCTIVE", reading_prompt="p1") for d in days(30)]
        recs += [rec(d, 0.04, stance="AT_RISK", reading_prompt="p2") for d in days(30)]
        out = stance_calibration(recs, Baselines(flat_closes))
        assert set(out) == {"p1", "p2"}
        assert all(v["finding"].startswith("UNDETERMINED") for v in out.values())
