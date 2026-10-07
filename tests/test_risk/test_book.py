"""The book's risk as it moves: one volatility, each name's share, the bets it is made of.

The live reading of 2026-10-07 that motivated it: AAOI, COHR, CRDO, DRAM and
NBIS correlated 0.57 on average — 24.4% of net liq carrying 49.9% of the
book's risk, of which SMH explained 65%. SPCX, 22% of net liq, carried 42%
on its own and nothing hedges it.
"""

from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.macro import calendar as cal
from advisor.risk import book as R

ET = mc.MARKET_TZ
NOW = datetime(2026, 10, 7, 16, 45, tzinfo=ET)
NL = 10_000.0


def frame(n=120, seed=7, *, etf_noise=0.004, loose=0.03):
    """Three names on one factor, one loose name, and an ETF that tracks the factor."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.02, n)
    idx = pd.bdate_range("2026-03-02", periods=n)
    names = pd.DataFrame(
        {
            "AAOI": 1.5 * f + rng.normal(0, 0.01, n),
            "COHR": 1.2 * f + rng.normal(0, 0.01, n),
            "CRDO": 1.3 * f + rng.normal(0, 0.01, n),
            "SPCX": rng.normal(0, loose, n),
        },
        index=idx,
    )
    etfs = pd.DataFrame({"SMH": f + rng.normal(0, etf_noise, n)}, index=idx)
    return names, etfs


W = {"AAOI": 0.06, "COHR": 0.04, "CRDO": 0.08, "SPCX": 0.22}
PRICES = {"SMH": 625.0}


def measure(weights=W, names=None, etfs=None, prices=PRICES, net_liq=NL):
    if names is None:
        names, etfs = frame()
    return R.measure(weights, names, etfs, prices, net_liq, NOW)


class TestBook:
    def test_the_volatility_is_the_covariance_of_the_holdings(self):
        names, etfs = frame()
        got = measure(names=names, etfs=etfs)
        w = pd.Series(W)
        expect = math.sqrt(w @ names[list(W)].cov() @ w)
        assert got.daily_vol == pytest.approx(expect)
        assert got.two_week == pytest.approx(2 * expect * math.sqrt(10) * NL)
        assert got.two_week_pct == pytest.approx(got.two_week / NL)
        assert got.invested == pytest.approx(0.40) and got.sessions == 120

    def test_risk_shares_add_to_the_whole(self):
        got = measure()
        assert sum(n.risk_share for n in got.names) == pytest.approx(1.0)
        assert got.names == sorted(got.names, key=lambda n: -n.risk_share)

    def test_a_diversifier_carries_a_negative_share(self):
        names, etfs = frame()
        names["HEDGE"] = -names["AAOI"] + np.random.default_rng(1).normal(0, 0.002, 120)
        got = measure({**W, "HEDGE": 0.03}, names, etfs)
        assert {n.symbol: n.risk_share for n in got.names}["HEDGE"] < 0

    def test_only_the_last_window_is_read(self):
        names, etfs = frame(n=200)
        crash = names.copy()
        crash.iloc[:80] = 0.5  # a regime the window no longer covers
        assert measure(names=crash, etfs=etfs).daily_vol == pytest.approx(
            measure(names=names, etfs=etfs).daily_vol
        )


class TestHistory:
    @pytest.mark.parametrize("sessions,kept", [(39, False), (40, True)])
    def test_a_name_needs_forty_sessions(self, sessions, kept):
        names, etfs = frame()
        names.iloc[: 120 - sessions, names.columns.get_loc("SPCX")] = np.nan
        got = measure(names=names, etfs=etfs)
        assert ("SPCX" in {n.symbol for n in got.names}) is kept
        assert bool(got.excluded) is not kept
        if not kept:
            assert got.excluded == ["SPCX: 39 sessions of prices, under 40"]

    def test_a_name_with_no_prices_at_all_is_named(self):
        got = measure({**W, "CCXI": 0.05})
        assert got.excluded == ["CCXI: 0 sessions of prices, under 40"]

    @pytest.mark.parametrize("net_liq", [0.0, -5.0, None])
    def test_no_net_liq_is_nothing(self, net_liq):
        assert measure(net_liq=net_liq) is None

    def test_no_holdings_or_none_measurable_is_nothing(self):
        assert measure(weights={}) is None
        assert measure(weights={"CCXI": 0.05}) is None

    def test_a_flat_book_is_nothing(self):
        names, etfs = frame()
        names[:] = 0.0
        assert measure(names=names, etfs=etfs) is None


class TestGroups:
    def test_names_that_move_together_are_one_group_and_the_loose_one_is_not(self):
        [g] = measure().groups
        assert g.members == ["AAOI", "COHR", "CRDO"] and g.label == "AAOI, COHR, CRDO"
        assert g.weight == pytest.approx(0.18)
        assert g.correlation > 0.8
        shares = {n.symbol: n.risk_share for n in measure().names}
        assert g.risk_share == pytest.approx(shares["AAOI"] + shares["COHR"] + shares["CRDO"])

    def test_the_cut_is_an_average_correlation_of_one_half(self):
        corr = pd.DataFrame(
            [[1, 0.51, 0.1], [0.51, 1, 0.1], [0.1, 0.1, 1]],
            index=list("ABC"),
            columns=list("ABC"),
        )
        assert R._groups(corr) == [["A", "B"]]
        assert R._groups(corr.replace(0.51, 0.49)) == []

    def test_one_name_is_no_group(self):
        assert R._groups(pd.DataFrame([[1.0]], index=["A"], columns=["A"])) == []

    def test_the_group_is_read_on_days_every_member_traded(self):
        names, etfs = frame()
        full = measure(names=names, etfs=etfs).groups[0]
        names.iloc[:30, names.columns.get_loc("COHR")] = np.nan  # listed later
        late = measure(names=names, etfs=etfs).groups[0]
        assert late.members == full.members
        # A missing day read as "no move" would shrink the group's volatility.
        assert late.two_week == pytest.approx(full.two_week, rel=0.15)


class TestHedge:
    def test_the_best_fitting_etf_is_sized_by_its_dollar_beta(self):
        names, etfs = frame()
        [g] = measure(names=names, etfs=etfs).groups
        h = g.hedge
        gr = names[g.members] @ pd.Series({m: W[m] for m in g.members})
        beta = np.cov(gr, etfs["SMH"])[0, 1] / etfs["SMH"].var()
        assert h.etf == "SMH" and h.beta == pytest.approx(beta) and h.r2 > 0.8
        # Beta is ~0.06·1.5 + 0.04·1.2 + 0.08·1.3 = 0.242 of net liq per unit of SMH.
        assert h.beta == pytest.approx(0.242, abs=0.02)
        assert h.shares == round(h.beta * NL / 625.0) and h.notional == pytest.approx(h.beta * NL)

    def test_a_hedge_that_explains_under_half_is_not_named(self):
        names, etfs = frame(etf_noise=0.05)
        [g] = measure(names=names, etfs=etfs).groups
        assert g.hedge is None
        assert g.hedge_note.startswith("the best fit, SMH, explains only")
        assert g.hedge_note.endswith("no hedge fits")

    def test_without_a_price_or_history_there_is_no_hedge(self):
        [g] = measure(prices={}).groups
        assert g.hedge is None and "enough shared history" in g.hedge_note

    def test_a_beta_that_rounds_to_no_shares_is_said(self):
        [g] = measure(prices={"SMH": 1e6}).groups
        assert g.hedge is None and g.hedge_note == "SMH's beta rounds to no shares"


class TestLive:
    def test_returns_are_on_the_union_of_sessions(self):
        d = [date(2026, 10, i) for i in (1, 2, 5, 6)]
        got = R.returns_from({"A": [(d[0], 10.0), (d[1], 11.0), (d[3], 11.0)],
                              "B": [(x, 5.0) for x in d], "C": []})  # fmt: skip
        assert list(got.columns) == ["A", "B"]
        assert len(got) == 3 and got["A"].iloc[0] == pytest.approx(0.1)
        assert math.isnan(got["A"].iloc[1])  # no close on 10-05: not a flat day

    def test_the_book_is_priced_from_closes(self):
        names, etfs = frame()

        def closes(sym):
            col = names[sym] if sym in names else etfs[sym] if sym in etfs else None
            if col is None:
                return []
            px = 100 * (1 + col).cumprod()
            return [(ts.date(), float(v)) for ts, v in px.items()]

        held = [
            Position(account="A", symbol=s, underlying=s, instrument=EQUITY, quantity=1,
                     avg_open_price=1.0, mark_price=W[s] * NL, close_price=W[s] * NL)
            for s in W
        ]  # fmt: skip
        book = BookSnapshot(as_of=NOW, net_liq=NL, positions=held)
        got = R.measure_book(book, NOW, closes=closes)
        assert {n.symbol for n in got.names} == set(W)
        assert got.groups[0].hedge.etf == "SMH"
        assert got.book_asof == NOW

    def test_no_book_or_no_prices_is_nothing(self):
        assert R.measure_book(None, NOW) is None
        book = BookSnapshot(as_of=NOW, net_liq=NL, positions=[])
        assert R.measure_book(book, NOW, closes=lambda s: []) is None


class TestStore:
    def test_one_reading_a_day_the_newest_wins(self, tmp_path):
        db = tmp_path / "research.db"
        assert R.latest_risk(db) is None  # no table yet
        first = measure()
        R.save_risk(db, first)
        R.save_risk(db, first.model_copy(update={"daily_vol": 0.5}))  # a re-run the same day
        later = first.model_copy(update={"asof": NOW + timedelta(days=1), "daily_vol": 0.7})
        R.save_risk(db, later)
        assert R.latest_risk(db).daily_vol == 0.7
        import sqlite3

        assert sqlite3.connect(db).execute("SELECT count(*) FROM book_risk").fetchone()[0] == 2

    def test_the_job_stores_a_reading_and_says_so(self, tmp_path, monkeypatch):
        from advisor.daemon.handlers import run_book_risk
        from advisor.daemon.jobs import JobContext
        from advisor.daemon.store import DaemonStore

        store = DaemonStore(tmp_path / "research.db")
        try:
            r = asyncio.run(run_book_risk(JobContext(store=store, now=NOW)))
            assert r.ok and r.detail == "nothing to measure (no book or prices)"
            monkeypatch.setattr(R, "measure_book", lambda book, now: measure())
            r = asyncio.run(run_book_risk(JobContext(store=store, now=NOW)))
            assert r.ok and "AAOI, COHR, CRDO" in r.detail
            assert R.latest_risk(store.db_path) is not None
        finally:
            store.close()


class TestCalendar:
    @pytest.mark.parametrize("today,expect", [
        (date(2026, 10, 27), [(date(2026, 11, 3), "the US midterm elections", 5)]),
        (date(2026, 10, 26), []),          # the sixth session
        (date(2026, 11, 3), [(date(2026, 11, 3), "the US midterm elections", 0)]),
        (date(2026, 11, 4), []),           # past
        (date(2026, 10, 31), [(date(2026, 11, 3), "the US midterm elections", 2)]),  # a Saturday
    ])  # fmt: skip
    def test_upcoming_within_the_sessions(self, today, expect):
        assert cal.upcoming(today, 5) == expect

    def test_the_midterms_are_the_first_tuesday_after_the_first_monday(self):
        day = next(d for d, label in cal.MACRO_EVENTS if "midterm" in label)
        assert day.weekday() == 1 and 2 <= day.day <= 8 and day.month == 11

    @pytest.mark.parametrize("days,read", [(4, True), (5, False)])
    def test_an_old_reading_is_not_read(self, tmp_path, days, read):
        db = tmp_path / "research.db"
        R.save_risk(db, measure())
        assert (R.latest_risk(db, (NOW + timedelta(days=days)).date()) is not None) is read
        assert R.latest_risk(db) is not None  # no today: the CLI shows whatever is stored
