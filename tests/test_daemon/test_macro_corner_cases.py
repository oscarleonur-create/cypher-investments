"""Adversarial corner cases for the macro layer.

The 53 tests shipped with Phase 2b cover the model's happy path and its
obvious degeneracies. These probe the seams *between* components — stale
state, misaligned calendars, degenerate books — which is where this repo's
real bugs have all lived so far (a naive `datetime.now()`, a level-triggered
threshold, a partial snapshot diffed against a full one).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.daemon.macro_ingest import (
    FACTOR_SHOCK_Z,
    MATERIAL_BOOK_LOADING,
    MAX_SENSITIVITY_AGE_DAYS,
    MIN_EXPECTED_BOOK_MOVE,
    factor_shock_events,
    fresh_sensitivities,
    residual_divergence_events,
)
from advisor.daemon.market_calendar import MARKET_TZ
from advisor.daemon.models import EventTier
from advisor.daemon.store import DaemonStore
from advisor.macro.exposure import BookExposure, FactorExposure, build_book_exposure
from advisor.macro.factors import Factor, log_returns
from advisor.macro.sensitivity import (
    FactorLoading,
    SymbolSensitivity,
    estimate_sensitivity,
    residual_z,
)


@pytest.fixture
def rng():
    return np.random.default_rng(11)


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def factor_frame(rng, *, last_move=None, n=300, end="2026-09-04"):
    idx = pd.bdate_range(end=end, periods=n)
    df = pd.DataFrame({f.value: rng.normal(0, 0.01, n) for f in Factor}, index=idx)
    for factor, move in (last_move or {}).items():
        df.iloc[-1, df.columns.get_loc(factor)] = move
    return df


def pos(symbol="AMD", *, qty=10, price=100.0, account="A", mark=0.0) -> Position:
    return Position(
        account=account,
        symbol=symbol,
        underlying=symbol,
        instrument=EQUITY,
        quantity=qty,
        multiplier=1,
        avg_open_price=price,
        close_price=price,
        mark_price=mark,
    )


def sens(symbol="AMD", *, loadings=None, resid_vol=0.02, r2=0.5) -> SymbolSensitivity:
    loadings = loadings or {Factor.MKT.value: 1.5}
    return SymbolSensitivity(
        symbol=symbol,
        asof=date(2026, 9, 4),
        window_days=250,
        n_obs=250,
        r2=r2,
        resid_vol=resid_vol,
        loadings=[FactorLoading(factor=f, loading=v, tstat=4.0) for f, v in loadings.items()],
    )


def exposure(loadings, *, net_liq=10_000.0, contributor="AMD", asof=date(2026, 9, 4)):
    return BookExposure(
        asof=asof,
        net_liq=net_liq,
        covered_weight=1.0,
        factors={
            f: FactorExposure(factor=f, net_loading=v, contributors=[(contributor, v)])
            for f, v in loadings.items()
        },
    )


# ── Stale state ──────────────────────────────────────────────────────────


class TestStaleState:
    """Exposure describes a book that may no longer exist."""

    def test_the_daily_path_reweights_against_todays_book(self, store):
        """Sold AMD, bought NVDA. Last week's exposure row still says AMD.

        The stored row is a cached artifact for the CLI. The daily path must
        rebuild from the book actually held, or a Tier A interrupt will name
        a position closed days ago.
        """
        store.save_exposure(exposure({Factor.MKT.value: 1.5}, contributor="AMD"))
        store.save_sensitivity(sens("AMD"))
        store.save_sensitivity(sens("NVDA"))
        book = BookSnapshot(positions=[pos("NVDA")], net_liq=10_000.0)

        rebuilt = build_book_exposure(book, fresh_sensitivities(store, book, asof=date(2026, 9, 8)))
        cited = {s for entry in rebuilt.factors.values() for s, _ in entry.contributors}
        assert cited == {"NVDA"}

    def test_a_position_opened_since_the_refresh_is_covered_immediately(self, store):
        """The stored row predates the position; the rebuild does not."""
        store.save_exposure(exposure({Factor.MKT.value: 1.5}, contributor="AMD"))
        store.save_sensitivity(sens("AMD"))
        store.save_sensitivity(sens("NVDA", loadings={Factor.MKT.value: 2.0}))
        book = BookSnapshot(
            positions=[pos("AMD", price=100.0), pos("NVDA", price=100.0)], net_liq=10_000.0
        )
        rebuilt = build_book_exposure(book, fresh_sensitivities(store, book, asof=date(2026, 9, 8)))
        assert rebuilt.uncovered == []
        assert rebuilt.loading(Factor.MKT.value) == pytest.approx(0.35)  # .1*1.5 + .1*2.0

    def test_a_stale_sensitivity_is_dropped_not_reused(self, store):
        """A laptop off for a month must not fire interrupts off July's fit."""
        ancient = sens("AMD")
        ancient.asof = date(2026, 7, 1)
        store.save_sensitivity(ancient)
        book = BookSnapshot(positions=[pos("AMD")], net_liq=10_000.0)
        assert fresh_sensitivities(store, book, asof=date(2026, 9, 4)) == {}

    def test_a_dropped_estimate_makes_the_symbol_uncovered_not_neutral(self, store):
        ancient = sens("AMD")
        ancient.asof = date(2026, 7, 1)
        store.save_sensitivity(ancient)
        book = BookSnapshot(positions=[pos("AMD")], net_liq=10_000.0)
        rebuilt = build_book_exposure(book, fresh_sensitivities(store, book, asof=date(2026, 9, 4)))
        assert rebuilt.uncovered == ["AMD"]
        assert rebuilt.factors == {}

    def test_exactly_at_the_age_limit_is_still_used(self, store):
        """The boundary itself is inside — a Friday fit read the next Friday."""
        estimate = sens("AMD")
        estimate.asof = date(2026, 9, 4)
        store.save_sensitivity(estimate)
        book = BookSnapshot(positions=[pos("AMD")], net_liq=10_000.0)
        asof = date(2026, 9, 4) + timedelta(days=MAX_SENSITIVITY_AGE_DAYS)
        assert "AMD" in fresh_sensitivities(store, book, asof=asof)
        assert fresh_sensitivities(store, book, asof=asof + timedelta(days=1)) == {}


# ── Calendar and alignment ───────────────────────────────────────────────


class TestAlignment:
    def test_stock_bars_lagging_the_factor_panel_do_not_produce_a_residual(self, rng):
        """A halted name's last bar is stale; today's factor moves are not.

        Comparing Friday's stock return against Monday's factor moves
        manufactures a residual out of a calendar mismatch.
        """
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.04}, end="2026-09-04")
        # Symbol prices end two sessions earlier — a halt.
        idx = pd.bdate_range(end="2026-09-02", periods=300)
        prices = pd.DataFrame({"AMD": np.linspace(100, 140, 300)}, index=idx)
        book = BookSnapshot(positions=[pos("AMD")], net_liq=10_000.0)

        events = residual_divergence_events(
            book, {"AMD": sens("AMD")}, log_returns(prices), factors
        )
        assert events == [], "residual computed across a two-session calendar gap"

    def test_halted_symbol_with_a_nan_bar_is_skipped(self, rng):
        """The common shape: yfinance keeps the row, leaves the value NaN."""
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.04})
        prices = pd.DataFrame(
            {"AMD": np.linspace(100, 140, 300), "NVDA": np.linspace(50, 90, 300)},
            index=factors.index,
        )
        prices.iloc[-1, prices.columns.get_loc("AMD")] = np.nan
        book = BookSnapshot(positions=[pos("AMD"), pos("NVDA")], net_liq=10_000.0)
        events = residual_divergence_events(
            book,
            {"AMD": sens("AMD"), "NVDA": sens("NVDA")},
            log_returns(prices),
            factors,
        )
        assert all(e.symbol != "AMD" for e in events)

    def test_weekend_run_dedups_against_the_same_friday_session(self, rng, store):
        """Saturday and Sunday runs both see Friday's last row — one event."""
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.05}, end="2026-09-04")
        exp = exposure({Factor.MKT.value: 1.5})
        first = factor_shock_events(factors, exp)
        second = factor_shock_events(factors, exp)
        assert len(first) == 1
        assert store.emit(first[0]) is True
        assert store.emit(second[0]) is False, "a second weekend run re-fired the same shock"


# ── Degenerate books ─────────────────────────────────────────────────────


class TestDegenerateBook:
    def test_negative_net_liq_yields_no_exposure_rather_than_inverted_signs(self):
        """A margin call must not silently flip every loading's sign."""
        book = BookSnapshot(positions=[pos("AMD")], net_liq=-500.0)
        exp = build_book_exposure(book, {"AMD": sens("AMD")})
        assert exp.factors == {}
        assert exp.uncovered == ["AMD"]

    def test_near_zero_net_liq_stays_finite(self):
        book = BookSnapshot(positions=[pos("AMD", qty=10, price=100.0)], net_liq=0.01)
        exp = build_book_exposure(book, {"AMD": sens("AMD")})
        assert all(np.isfinite(f.net_loading) for f in exp.factors.values())

    def test_positions_marked_at_zero_do_not_divide_by_zero(self):
        """Pre-market with no mark and a zero close: gross notional is 0."""
        book = BookSnapshot(positions=[pos("AMD", price=0.0)], net_liq=10_000.0)
        exp = build_book_exposure(book, {"AMD": sens("AMD")})
        assert exp.covered_weight == 0.0
        assert np.isfinite(exp.loading(Factor.MKT.value))

    def test_same_underlying_across_two_accounts_is_merged_once(self):
        book = BookSnapshot(
            positions=[
                pos("AMD", qty=10, price=100.0, account="5WI30382"),
                pos("AMD", qty=5, price=100.0, account="5WI47366"),
            ],
            net_liq=10_000.0,
        )
        exp = build_book_exposure(book, {"AMD": sens("AMD", loadings={Factor.MKT.value: 1.0})})
        mkt = exp.factors[Factor.MKT.value]
        assert len(mkt.contributors) == 1, "one underlying should appear once"
        assert mkt.net_loading == pytest.approx(0.15)  # (1000 + 500) / 10000 * 1.0

    def test_a_short_subtracts_from_the_same_names_long(self):
        book = BookSnapshot(
            positions=[
                pos("AMD", qty=10, price=100.0, account="A"),
                pos("AMD", qty=-10, price=100.0, account="B"),
            ],
            net_liq=10_000.0,
        )
        exp = build_book_exposure(book, {"AMD": sens("AMD", loadings={Factor.MKT.value: 1.0})})
        assert exp.loading(Factor.MKT.value) == pytest.approx(0.0)
        assert exp.covered_weight == pytest.approx(1.0)  # both legs still covered

    def test_holding_a_factor_leg_itself_does_not_blow_up(self, rng):
        """Own SPY and MKT explains it perfectly: resid_vol -> 0."""
        factors = factor_frame(rng, n=300)
        y = factors[Factor.MKT.value]  # SPY *is* the market factor
        estimate = estimate_sensitivity("SPY", y, factors)
        assert estimate is not None
        assert np.isfinite(estimate.resid_vol)
        z = residual_z(estimate, 0.02, {Factor.MKT.value: 0.02})
        assert np.isfinite(z)

    def test_constant_price_series_gives_no_divergence(self, rng):
        factors = factor_frame(rng, n=300)
        flat = pd.Series(np.zeros(len(factors)), index=factors.index)
        estimate = estimate_sensitivity("FLAT", flat, factors)
        assert estimate is not None
        assert residual_z(estimate, 0.0, {Factor.MKT.value: 0.03}) == 0.0


# ── Threshold boundaries ─────────────────────────────────────────────────


class TestBoundaries:
    def test_exactly_at_every_threshold_fires(self, rng):
        """Strict `<` comparisons mean the boundary itself is inside."""
        factors = factor_frame(rng, n=300)
        sigma = float(factors[Factor.MKT.value].iloc[-251:-1].std())
        move = FACTOR_SHOCK_Z * sigma
        factors.iloc[-1, factors.columns.get_loc(Factor.MKT.value)] = move
        loading = MIN_EXPECTED_BOOK_MOVE / move
        assert abs(loading) >= MATERIAL_BOOK_LOADING
        events = factor_shock_events(factors, exposure({Factor.MKT.value: loading}))
        assert len(events) == 1

    def test_one_basis_point_of_book_impact_below_threshold_is_silent(self, rng):
        factors = factor_frame(rng, n=300)
        sigma = float(factors[Factor.MKT.value].iloc[-251:-1].std())
        move = 3.0 * sigma  # comfortably past the z gate
        factors.iloc[-1, factors.columns.get_loc(Factor.MKT.value)] = move
        loading = (MIN_EXPECTED_BOOK_MOVE - 0.0001) / move
        assert abs(loading) >= MATERIAL_BOOK_LOADING  # so only book impact can gate it
        events = factor_shock_events(factors, exposure({Factor.MKT.value: loading}))
        assert events == []

    def test_a_shock_does_not_damp_its_own_z_score(self, rng):
        """Including today in its own baseline scored a true 8-sigma day at 7.2."""
        factors = factor_frame(rng, n=300)
        sigma = float(factors[Factor.MKT.value].iloc[-251:-1].std())
        factors.iloc[-1, factors.columns.get_loc(Factor.MKT.value)] = 8.0 * sigma
        events = factor_shock_events(factors, exposure({Factor.MKT.value: 1.5}))
        assert events[0].payload["z"] == pytest.approx(8.0, abs=0.01)

    def test_short_factor_history_is_skipped_not_extrapolated(self, rng):
        """Fewer than 60 observations cannot support a sigma."""
        factors = factor_frame(rng, n=50, last_move={Factor.MKT.value: 0.20})
        assert factor_shock_events(factors, exposure({Factor.MKT.value: 2.0})) == []


# ── Idempotency ──────────────────────────────────────────────────────────


class TestIdempotency:
    def test_two_racing_pollers_emit_one_row(self, rng, store):
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.05})
        exp = exposure({Factor.MKT.value: 1.5})
        a = factor_shock_events(factors, exp)[0]
        b = factor_shock_events(factors, exp)[0]
        assert store.emit(a) is True
        assert store.emit(b) is False

    def test_residual_events_dedup_within_a_session(self, rng, store):
        factors = factor_frame(rng, n=300)
        prices = pd.DataFrame({"AMD": np.linspace(100, 200, 300)}, index=factors.index)
        prices.iloc[-1, 0] = prices.iloc[-2, 0] * 1.15  # a 15% day
        book = BookSnapshot(positions=[pos("AMD")], net_liq=10_000.0)
        rets = log_returns(prices)
        first = residual_divergence_events(book, {"AMD": sens("AMD")}, rets, factors)
        assert first and first[0].tier == EventTier.B
        assert store.emit(first[0]) is True
        again = residual_divergence_events(book, {"AMD": sens("AMD")}, rets, factors)
        assert store.emit(again[0]) is False


# ── Timezone ─────────────────────────────────────────────────────────────


class TestTimezone:
    def test_macro_events_carry_et_aware_timestamps(self, rng):
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.05})
        event = factor_shock_events(factors, exposure({Factor.MKT.value: 1.5}))[0]
        assert event.ts.tzinfo is not None
        assert event.ts.utcoffset() == datetime.now(MARKET_TZ).utcoffset()

    def test_exposure_asof_survives_a_store_round_trip(self, store):
        exp = exposure({Factor.MKT.value: 1.0}, asof=date(2026, 3, 8))  # DST spring forward
        store.save_exposure(exp)
        assert store.load_latest_exposure().asof == date(2026, 3, 8)
