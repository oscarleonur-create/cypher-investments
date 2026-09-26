"""Replay: same code as live, nothing from the future, and its records kept apart."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import replace
from datetime import date, timedelta

import pytest
from advisor.daemon import market_calendar as mc
from advisor.entry.ruleset import entry_rules
from advisor.learning.evaluate import Baselines, Cell, Verdict, agreement, load_replay
from advisor.learning.replay import (
    DAILY_A,
    DAILY_C,
    DayBar,
    ReplayStore,
    SymbolData,
    compact,
    daily_setups,
    replay_symbol,
    run,
    setup_outcomes,
    sigma_before,
)
from advisor.learning.rules import Origin
from advisor.scanner import detect
from advisor.valuation.history import Point, Series


def sessions(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if mc.is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


DAYS = sessions(date(2024, 1, 2), 700)


def wave(i: int) -> float:
    return 100 * (1 + 0.25 * math.sin(i / 40)) * (1 + 0.002 * math.sin(i * 1.7))


def make_bars(px=wave, days=DAYS) -> list[DayBar]:
    out = []
    for i, d in enumerate(days):
        c = px(i)
        out.append(DayBar(d, c, c * 1.01, c * 0.99, c, 1e6))
    return out


def series() -> Series:
    # Revenue flat at 100/quarter-TTM, shares flat: P/S moves only with price.
    pts = [Point(end=d, value=400.0, known=d + timedelta(days=45)) for d in DAYS[::63]]
    pts.insert(0, Point(end=date(2023, 9, 30), value=400.0, known=date(2023, 11, 14)))
    shares = [Point(end=date(2023, 9, 30), value=10.0, known=date(2023, 11, 14))]
    return Series(symbol="ZZZ", revenue_ttm=pts, shares=shares)


def data(bars=None) -> SymbolData:
    return SymbolData(symbol="ZZZ", bars=bars or make_bars(), series=series())


START, END = DAYS[450], DAYS[650]


class TestDailySetups:
    def bars(self, prev_close, open_, close):
        return [DayBar(DAYS[0], 1, 1, 1, prev_close, 1), DayBar(DAYS[1], open_, 1, 1, close, 1)]

    def test_gap_exactly_at_threshold_qualifies(self):
        assert daily_setups(self.bars(100, 104, 104), 1, None) == [(DAILY_A, 104)]

    def test_just_under_gap(self):
        assert daily_setups(self.bars(100, 103.99, 104), 1, None) == []

    def test_a_price_floor(self):
        assert daily_setups(self.bars(1.0, 1.9, 1.9), 1, None) == []

    def test_c_needs_drop_and_sigma(self):
        assert daily_setups(self.bars(100, 99, 95), 1, 0.02) == [(DAILY_C, 95)]
        assert daily_setups(self.bars(100, 99, 95), 1, 0.03) == []  # 5% is 1.7σ
        assert daily_setups(self.bars(100, 99, 95), 1, None) == []  # no σ, no C

    def test_thresholds_are_the_scanners(self):
        looser = replace(detect.DEFAULT, gap_min=0.02)
        assert daily_setups(self.bars(100, 102.5, 103), 1, None, looser) == [(DAILY_A, 102.5)]

    def test_first_bar_has_no_setup(self):
        assert daily_setups(self.bars(100, 110, 110), 0, 0.01) == []

    def test_outcomes_from_the_entry(self):
        bars = make_bars()
        out = setup_outcomes(bars, 10, DAILY_A, bars[10].open)
        assert out["close"] == pytest.approx(0) and "d20" in out
        assert "d20" not in setup_outcomes(bars, len(bars) - 3, DAILY_C, 100.0)

    def test_sigma_excludes_today(self):
        bars = make_bars()
        crashed = bars[:100] + [replace(bars[100], close=1.0)]
        assert sigma_before(crashed, 100) == sigma_before(bars, 100)


class TestReplaySymbol:
    def test_one_proposal_per_session_in_window(self):
        props, _ = replay_symbol(data(), START, END)
        assert [p.session for p in props] == DAYS[450:651]

    def test_replayed_records_say_what_they_are(self):
        props, _ = replay_symbol(data(), START, END)
        p = props[0]
        assert p.origin is Origin.REPLAY
        assert p.rules.version == entry_rules().version
        assert any("no event stream" in g for g in p.gaps)

    def test_nothing_after_the_day_changes_the_decision(self):
        """The look-ahead test: rewrite every price after day k; decisions to k stay."""
        k = 520
        base = make_bars()
        future_changed = base[: k + 1] + [
            replace(b, open=b.open * 3, high=b.high * 3, low=b.low * 0.2, close=b.close * 0.3)
            for b in base[k + 1 :]
        ]
        a, sa = replay_symbol(data(base), START, DAYS[k])
        b, sb = replay_symbol(data(future_changed), START, DAYS[k])
        assert [(p.action, p.features, [g.stop for g in p.legs]) for p in a] == [
            (p.action, p.features, [g.stop for g in p.legs]) for p in b
        ]
        assert [s["id"] for s in sa] == [s["id"] for s in sb]
        # ...while the outcomes, which are the future, do differ.
        assert [p.outcomes for p in a] != [p.outcomes for p in b]

    def test_a_revenue_figure_is_not_used_before_it_was_known(self):
        s = series()
        late = Point(end=DAYS[555], value=4000.0, known=DAYS[560])
        s.revenue_ttm = sorted(s.revenue_ttm + [late], key=lambda p: p.end)
        props, _ = replay_symbol(SymbolData("ZZZ", make_bars(), s), START, END)
        before = next(p for p in props if p.session == DAYS[559])
        after = next(p for p in props if p.session == DAYS[560])
        assert before.features["ps"] > after.features["ps"] * 5

    def test_results_guard_applies_in_replay(self):
        d = data()
        props, _ = replay_symbol(d, START, END)
        acting = next((p for p in props if p.legs), None)
        assert acting is not None
        d.earnings = [acting.session + timedelta(days=1)]
        again, _ = replay_symbol(d, acting.session, acting.session)
        assert again[0].action.value == "WAIT"

    def test_no_series_still_replays_setups_without_a_zone(self):
        d = SymbolData("ZZZ", make_bars(), series=None)
        props, _ = replay_symbol(d, START, END)
        assert all(p.features["in_zone"] is None for p in props)


class FakeLoader:
    def __init__(self, missing=()):
        self.missing = set(missing)

    def __call__(self, symbol, start, end):
        if symbol in self.missing:
            return None
        return SymbolData(symbol, make_bars(), series())


@pytest.fixture
def store(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "research.db"))
    yield ReplayStore(conn)
    conn.close()


class TestRun:
    def test_before_the_calendar_is_refused(self, store):
        with pytest.raises(ValueError, match="market calendar"):
            run(store, ["ZZZ"], date(2023, 6, 1), END, loader=FakeLoader())

    def test_end_before_start_is_refused(self, store):
        with pytest.raises(ValueError):
            run(store, ["ZZZ"], END, START, loader=FakeLoader())

    def test_missing_data_is_listed_not_skipped_silently(self, store):
        r = run(store, ["ZZZ", "GONE"], START, END, run_id="r1", loader=FakeLoader({"GONE"}))
        assert r.no_data == ["GONE"] and r.replayed == 1

    def test_cannot_say_is_not_stored(self, store):
        def no_zone(symbol, start, end):
            return SymbolData(symbol, make_bars(px=lambda i: 100.0), series=None)

        r = run(store, ["FLAT"], START, END, run_id="r2", loader=no_zone)
        assert r.proposals == 0 and store.proposals("r2") == []

    def test_rerunning_a_run_id_replaces_its_records(self, store):
        run(store, ["ZZZ"], START, END, run_id="r3", loader=FakeLoader())
        first = len(store.proposals("r3"))
        run(store, ["ZZZ"], START, END, run_id="r3", loader=FakeLoader())
        assert len(store.proposals("r3")) == first

    def test_only_the_latest_runs_keep_their_records(self, store):
        for rid in ("old", "mid", "new"):
            run(store, ["ZZZ"], START, END, run_id=rid, loader=FakeLoader())
        assert store.proposals("old") == []
        assert store.proposals("mid") and store.proposals("new")
        assert {r["id"] for r in store.runs()} >= {"old", "mid", "new"}  # summaries kept

    def test_run_records_the_rule_versions_it_replayed(self, store):
        run(store, ["ZZZ"], START, END, run_id="r4", loader=FakeLoader())
        (r,) = [x for x in store.runs() if x["id"] == "r4"]
        assert r["params"]["entry_version"] == entry_rules().version
        assert r["finished_at"] is not None

    def test_compact_keeps_what_the_judge_needs(self):
        props, _ = replay_symbol(data(), START, END)
        c = compact(props[0])
        assert set(c) >= {"action", "session", "version", "features", "legs", "outcomes"}
        assert "reasons" not in c


class TestLoadAndAgreement:
    def test_load_replay_latest(self, tmp_path):
        path = tmp_path / "research.db"
        conn = sqlite3.connect(str(path))
        run(ReplayStore(conn), ["ZZZ"], START, END, run_id="r5", loader=FakeLoader())
        conn.close()
        run_id, records = load_replay(path)
        assert run_id == "r5" and records
        assert all(r.origin == "replay" for r in records)

    def test_no_run_on_file(self, tmp_path):
        assert load_replay(tmp_path / "empty.db") == (None, [])

    def cell(self, origin, excess, ci):
        return Cell(
            "entry", "v", "action ENTER", "d20", origin, 30, 30, excess, excess, 0.5, 0.0,
            excess, ci, None, 0, Verdict.UNDETERMINED, "",
        )  # fmt: skip

    def test_live_inside_replay_interval_agrees(self):
        rows = agreement([self.cell("replay", 0.01, (0.0, 0.02)), self.cell("live", 0.015, None)])
        assert rows[0]["agrees"] is True

    def test_live_outside_replay_interval_is_flagged(self):
        rows = agreement([self.cell("replay", 0.01, (0.0, 0.02)), self.cell("live", -0.03, None)])
        assert rows[0]["agrees"] is False

    def test_no_replay_interval_no_comparison(self):
        assert agreement([self.cell("replay", 0.01, None), self.cell("live", 0.0, None)]) == []


class TestCalendar:
    @pytest.mark.parametrize(
        "day",
        [date(2024, 12, 25), date(2025, 1, 1), date(2025, 1, 9), date(2024, 11, 28)],
    )
    def test_past_holidays_are_closed(self, day):
        assert not mc.is_trading_day(day)

    def test_past_early_closes(self):
        assert mc.session_close(date(2024, 12, 24)) == mc.EARLY_CLOSE

    def test_covered_from(self):
        assert mc.COVERED_FROM == date(2024, 1, 1)


_ = Baselines  # imported for symmetry with the evaluator tests
