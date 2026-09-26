"""The generator: a value is proposed only if it keeps winning out of sample."""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta

import pytest
from advisor.learning.actuator import ChangeStore, Status
from advisor.learning.replay import DayBar, SymbolData
from advisor.learning.sweep import (
    FOLDS,
    MIN_RECORDS,
    Target,
    Verdict,
    _grid,
    _position_result,
    _trade_result,
    file_proposals,
    sweep,
    walk_forward,
)
from advisor.learning.sweep import SweepResult as Result

T = Target("dip trigger", "entry", "proposal.DIP_SIGMAS", "dip_sigmas",
           (1.0, 1.5, 2.0, 2.5, 3.0), 1, "test")  # fmt: skip
DAYS = [date(2025, 1, 1) + timedelta(days=i) for i in range(400)]


def records(effects: dict, n_per_day=1, noise=0.002, seed=1, by_fold=None):
    """{value: [(day, excess)]} with each value's mean effect, optionally per fold."""
    rng = random.Random(seed)
    out = {}
    fold_len = len(DAYS) // FOLDS
    for v, eff in effects.items():
        recs = []
        for i, d in enumerate(DAYS):
            e = by_fold[v][min(i // fold_len, FOLDS - 1)] if by_fold and v in by_fold else eff
            recs += [(d, e + rng.gauss(0, noise)) for _ in range(n_per_day)]
        out[v] = recs
    return out


def sessions_of(by_value):
    return [d for recs in by_value.values() for d, _ in recs]


class TestGrid:
    def test_float_grid_brackets_the_current_value(self):
        assert _grid(2.0) == (1.0, 1.5, 2.0, 2.5, 3.0)

    def test_whole_grid_never_below_one(self):
        assert _grid(2, whole=True) == (1, 2, 3, 4)


class TestResults:
    def bars(self, lows, closes):
        return [DayBar(date(2025, 1, 1) + timedelta(i), 1, 1, lo, c, 1) for i, (lo, c) in
                enumerate(zip(lows, closes))]  # fmt: skip

    def test_trade_stopped_next_day(self):
        b = self.bars([100, 90], [100, 105])
        assert _trade_result(b, 0, 95.0, 100.0) == pytest.approx(-0.05)

    def test_trade_out_at_next_close(self):
        b = self.bars([100, 99], [100, 103])
        assert _trade_result(b, 0, 95.0, 100.0) == pytest.approx(0.03)

    def test_trade_without_a_next_day(self):
        assert _trade_result(self.bars([100], [100]), 0, 95.0, 100.0) is None

    def test_position_stop_within_twenty(self):
        lows = [100] * 10 + [80] + [100] * 15
        closes = [100] * 26
        assert _position_result(self.bars(lows, closes), 0, 90.0, 100.0) == pytest.approx(-0.1)


class TestWalkForward:
    def test_a_value_that_keeps_winning_is_proposed(self):
        by = records({1.0: 0.0, 1.5: 0.004, 2.0: 0.0, 2.5: 0.006, 3.0: 0.002})
        v = walk_forward(T, 2.0, by, sessions_of(by))
        assert v.proposed == 2.5 and v.reason == "survived walk-forward"
        assert v.evidence["wins"] >= 2 and v.evidence["variants_tested"] == 5

    def test_the_current_value_winning_proposes_nothing(self):
        by = records({1.0: 0.0, 1.5: 0.001, 2.0: 0.006, 2.5: 0.001, 3.0: 0.0})
        v = walk_forward(T, 2.0, by, sessions_of(by))
        assert v.proposed is None and "current value" in v.reason

    def test_a_value_that_won_once_is_not_proposed(self):
        # 3.0 is best in the first stretch only; afterwards it is worst.
        by = records(
            {1.0: 0.0, 1.5: 0.0, 2.0: 0.001, 2.5: 0.0, 3.0: 0.0},
            by_fold={3.0: [0.02, -0.01, -0.01, -0.01]},
        )
        v = walk_forward(T, 2.0, by, sessions_of(by))
        assert v.proposed is None

    def test_a_lone_peak_is_not_proposed(self):
        by = records({1.0: 0.0, 1.5: -0.004, 2.0: 0.0, 2.5: -0.004, 3.0: 0.006})
        t = Target("x", "entry", "proposal.DIP_SIGMAS", "dip_sigmas",
                   (1.0, 1.5, 2.0, 2.5, 3.0), 1, "t")  # fmt: skip
        v = walk_forward(t, 1.0, by, sessions_of(by))
        assert v.proposed is None and "lone peak" in v.reason

    def test_a_better_mean_bought_with_the_tail_is_not_proposed(self):
        rng = random.Random(9)
        by = records({1.0: 0.0, 1.5: 0.0, 2.0: 0.0, 3.0: 0.0})
        by[2.5] = [(d, 0.02 if rng.random() > 0.2 else -0.06) for d in DAYS]
        v = walk_forward(T, 2.0, by, sessions_of(by))
        assert v.proposed is None

    def test_too_few_records_proposes_nothing(self):
        by = {v: [(DAYS[i * 50], 0.01)] for i, v in enumerate(T.grid)}
        v = walk_forward(T, 2.0, by, sessions_of(by))
        assert v.proposed is None

    def test_no_sessions(self):
        assert walk_forward(T, 2.0, {}, []).reason == "no sessions"

    def test_choosing_is_purged_of_outcomes_that_reach_the_judged_stretch(self):
        # With a 60-session horizon, records in the last 60 sessions before a
        # judged stretch cannot be used to choose: make them the only reason
        # 2.5 would look best, and it must not be chosen for that stretch.
        long = Target("x", "entry", "proposal.DIP_SIGMAS", "dip_sigmas", T.grid, 60, "t")
        by = records({1.0: 0.0, 1.5: 0.0, 2.0: 0.001, 2.5: 0.0, 3.0: 0.0}, noise=0.0)
        fold_start = sorted(set(DAYS))[len(DAYS) // FOLDS]
        by[2.5] = [(d, 0.5 if fold_start - timedelta(days=50) <= d < fold_start else 0.0)
                   for d, _ in by[2.5]]  # fmt: skip
        v = walk_forward(long, 2.0, by, sessions_of(by))
        first = v.evidence["folds"][0]
        assert first.get("chosen") != 2.5


class TestFiling:
    @pytest.fixture
    def store(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "r.db"))
        yield ChangeStore(conn)
        conn.close()

    def survived(self, value=2.5):
        return Result(
            verdicts=[
                Verdict("dip trigger", "proposal.DIP_SIGMAS", 2.0, value, "survived", {"x": 1})
            ]
        )

    def test_files_pending_with_its_evidence(self, store):
        r = file_proposals(self.survived(), store)
        (c,) = store.list()
        assert r.proposed == [c.id] and c.status is Status.PENDING and c.source == "sweep"
        assert c.evidence == {"x": 1}

    def test_not_filed_twice(self, store):
        file_proposals(self.survived(), store)
        r = file_proposals(self.survived(), store)
        assert r.proposed == [] and "already on file" in r.skipped[0]

    def test_a_recent_rejection_is_remembered(self, store):
        file_proposals(self.survived(), store)
        (c,) = store.list()
        store.reject(c.id, "not convinced")
        r = file_proposals(self.survived(), store)
        assert r.proposed == [] and "rejected" in r.skipped[0]

    def test_a_different_value_after_a_rejection_is_filed(self, store):
        file_proposals(self.survived(2.5), store)
        store.reject(store.list()[0].id)
        assert file_proposals(self.survived(3.0), store).proposed

    def test_nothing_survived_nothing_filed(self, store):
        r = Result(verdicts=[Verdict("t", "proposal.DIP_SIGMAS", 2.0, None, "no")])
        assert file_proposals(r, store).proposed == [] and store.list() == []


def test_a_sweep_over_synthetic_symbols_runs_every_target():
    import math

    days, d = [], date(2024, 1, 2)
    from advisor.daemon import market_calendar as mc

    while len(days) < 700:
        if mc.is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)

    def loader(symbol, start, end):
        rng = random.Random(hash(symbol) % 1000)
        bars, px = [], 100.0
        for day in days:
            px *= 1 + rng.gauss(0.0003, 0.02) + (0.06 if rng.random() < 0.02 else 0.0)
            o = px * (1 + rng.gauss(0, 0.01))
            bars.append(DayBar(day, o, max(o, px) * 1.01, min(o, px) * 0.99, px, 1e6))
        return SymbolData(symbol, bars, series=None)

    r = sweep(["AAA", "BBB"], days[400], days[650], loader=loader)
    assert r.used == 2 and len(r.verdicts) == 7
    assert all(v.reason for v in r.verdicts)
    assert math.isfinite(len(r.verdicts))
    _ = MIN_RECORDS


def test_the_grid_and_the_baseline_are_the_rules_in_force():
    from dataclasses import replace

    from advisor.entry.proposal import current_params
    from advisor.learning.sweep import _current, targets

    active = replace(current_params(), trade_stop_sigmas=2.0)
    t = next(x for x in targets(active) if x.field == "trade_stop_sigmas")
    assert t.grid == (1.0, 1.5, 2.0, 2.5, 3.0)
    assert _current(t, active) == 2.0
    assert _current(t) == current_params().trade_stop_sigmas
