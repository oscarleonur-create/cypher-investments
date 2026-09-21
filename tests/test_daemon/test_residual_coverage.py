"""A residual is only a residual if the model could have explained the move.

Found reviewing the daemon's first full trading session with the NaN fixes in
place. The 07:00 brief of 21 September ran with **8 of 9 factors unobserved** —
yfinance has no daily bar pre-market — and the fix from before correctly
dropped them from the factor-move vector. But `expected_return` sums over the
factors it is *given*, so a dropped factor contributes zero silently, and
everything it would have accounted for lands in the residual instead.

Measured on the real book, with every holding moving *exactly* what the full
nine-factor panel predicts — a day that is idiosyncratic by 0.00 standard
deviations, by construction:

    sym    z (9 factors)   z (VOL only)
    OUST            0.00          -1.69
    CRDO            0.00          -1.14
    AAOI            0.00          -1.09

Scale that risk-off day by 1.2 and OUST reads -2.03, crossing the firing
threshold and announcing that it "moved for a reason macro cannot explain".
Macro explained all of it. The model simply could not see.

Nothing bad had been published: `RESIDUAL_DIVERGENCE` has never fired in this
book's history. It was one wider morning away from doing so.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.daemon.macro_ingest import (
    MIN_LOADING_COVERAGE,
    RESIDUAL_DIVERGENCE_Z,
    _loading_coverage,
    residual_divergence_events,
)
from advisor.macro.factors import Factor, log_returns
from advisor.macro.sensitivity import FactorLoading, SymbolSensitivity

ALL_FACTORS = [f.value for f in Factor]


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def sens(symbol="OUST", *, loadings, resid_vol=0.0467) -> SymbolSensitivity:
    return SymbolSensitivity(
        symbol=symbol,
        asof=date(2026, 9, 15),
        window_days=250,
        n_obs=250,
        r2=0.36,
        resid_vol=resid_vol,
        loadings=[FactorLoading(factor=f, loading=v, tstat=3.0) for f, v in loadings.items()],
    )


# OUST's real stored profile, the worst case in the book.
OUST_LOADINGS = {
    "MKT": 2.6383,
    "DURATION": -1.1187,
    "DOLLAR": 1.3549,
    "ENERGY": 0.2041,
    "CREDIT": -2.0197,
    "GROWTH_VALUE": 0.9382,
    "SIZE": 1.4285,
    "VOL": -0.0157,
    "BREADTH": -1.3161,
}


# ── the measure itself ────────────────────────────────────────────────────
class TestCoverage:
    def test_the_whole_panel_is_full_coverage(self):
        estimate = sens(loadings=OUST_LOADINGS)
        assert _loading_coverage(estimate, dict.fromkeys(ALL_FACTORS, 0.0)) == 1.0

    def test_the_morning_case_is_near_zero(self):
        """8 of 9 unobserved, and the survivor is the one OUST barely loads on."""
        estimate = sens(loadings=OUST_LOADINGS)
        assert _loading_coverage(estimate, {"VOL": 0.0}) < 0.01

    def test_coverage_is_weighted_by_loading_not_by_count(self):
        """Missing one of nine factors is 89% by count and can be far less by
        weight — which is the number that decides whether a residual means
        anything."""
        estimate = sens(loadings=OUST_LOADINGS)
        without_mkt = {f: 0.0 for f in ALL_FACTORS if f != "MKT"}
        assert _loading_coverage(estimate, without_mkt) < 8 / 9

    def test_a_symbol_that_does_not_load_on_the_missing_factor_keeps_its_coverage(self):
        """The reason this is per symbol: a name with no dollar exposure does
        not care that DOLLAR is missing."""
        estimate = sens(loadings={"MKT": 1.5, "DOLLAR": 0.0})
        without_dollar = {f: 0.0 for f in ALL_FACTORS if f != "DOLLAR"}
        assert _loading_coverage(estimate, without_dollar) == 1.0

    def test_a_symbol_with_no_sensitivity_at_all_cannot_be_misexplained(self):
        """Nothing to miss, so nothing is blocked on account of missing it."""
        estimate = sens(loadings={"MKT": 0.0})
        assert _loading_coverage(estimate, {}) == 1.0

    def test_coverage_ignores_the_sign_of_a_loading(self):
        """A -4.5 credit loading does as much work as a +4.5 one."""
        negative = sens(loadings={"MKT": 1.0, "CREDIT": -3.0})
        positive = sens(loadings={"MKT": 1.0, "CREDIT": 3.0})
        moves = {"MKT": 0.0}
        assert _loading_coverage(negative, moves) == _loading_coverage(positive, moves)


# ── what it blocks, and what it must not ──────────────────────────────────
def frame(rng, *, observed: list[str], symbol_return: float, n=300):
    """A factor panel where only `observed` carry a reading for the last bar."""
    idx = pd.bdate_range(end="2026-09-21", periods=n)
    factors = pd.DataFrame({f: rng.normal(0, 0.01, n) for f in ALL_FACTORS}, index=idx)
    for f in ALL_FACTORS:
        if f not in observed:
            factors.iloc[-1, factors.columns.get_loc(f)] = np.nan
    # A price path whose final log return is exactly `symbol_return`.
    prices = np.full(n, 100.0)
    prices[-1] = 100.0 * np.exp(symbol_return)
    return factors, pd.DataFrame({"OUST": prices}, index=idx)


def book() -> BookSnapshot:
    return BookSnapshot(
        positions=[
            Position(
                account="A",
                symbol="OUST",
                underlying="OUST",
                instrument=EQUITY,
                quantity=10,
                multiplier=1,
                avg_open_price=100.0,
                close_price=100.0,
            )
        ],
        net_liq=10_000.0,
    )


class TestBlocking:
    def test_a_move_the_blind_model_would_call_idiosyncratic_is_not_reported(self, rng):
        """The live failure, reduced: a huge move, one factor observed."""
        factors, prices = frame(rng, observed=["VOL"], symbol_return=-0.20)
        events = residual_divergence_events(
            book(), {"OUST": sens(loadings=OUST_LOADINGS)}, log_returns(prices), factors
        )
        assert events == []

    def test_the_same_move_is_reported_when_the_panel_is_whole(self, rng):
        """The detector must still work. Blocking everything would be a fix
        that removed the feature."""
        factors, prices = frame(rng, observed=ALL_FACTORS, symbol_return=-0.20)
        events = residual_divergence_events(
            book(), {"OUST": sens(loadings=OUST_LOADINGS)}, log_returns(prices), factors
        )
        assert len(events) == 1
        assert abs(events[0].payload["residual_z"]) >= RESIDUAL_DIVERGENCE_Z

    def test_one_light_factor_missing_still_reports(self, rng):
        """OUST's VOL loading is -0.016. Losing it costs 0.4% of coverage and
        must not silence the detector — that would trade one failure for the
        opposite one."""
        observed = [f for f in ALL_FACTORS if f != "VOL"]
        factors, prices = frame(rng, observed=observed, symbol_return=-0.20)
        events = residual_divergence_events(
            book(), {"OUST": sens(loadings=OUST_LOADINGS)}, log_returns(prices), factors
        )
        assert len(events) == 1

    def test_the_heaviest_factor_missing_blocks(self, rng):
        """MKT carries 25% of OUST's loading mass. Without it the residual is
        a quarter guesswork."""
        observed = [f for f in ALL_FACTORS if f != "MKT"]
        factors, prices = frame(rng, observed=observed, symbol_return=-0.20)
        events = residual_divergence_events(
            book(), {"OUST": sens(loadings=OUST_LOADINGS)}, log_returns(prices), factors
        )
        assert events == []

    def test_blocking_is_per_symbol_not_per_run(self, rng):
        """One blind symbol must not silence a name the panel can still see."""
        factors, prices = frame(rng, observed=["VOL"], symbol_return=-0.20)
        prices["FLAT"] = prices["OUST"]
        positions = list(book().positions) + [
            Position(
                account="A",
                symbol="FLAT",
                underlying="FLAT",
                instrument=EQUITY,
                quantity=10,
                multiplier=1,
                avg_open_price=100.0,
                close_price=100.0,
            )
        ]
        two = BookSnapshot(positions=positions, net_liq=10_000.0)
        events = residual_divergence_events(
            two,
            {
                "OUST": sens(loadings=OUST_LOADINGS),
                # Loads on nothing but VOL, so VOL alone is the whole model.
                "FLAT": sens("FLAT", loadings={"VOL": 1.0}, resid_vol=0.02),
            },
            log_returns(prices),
            factors,
        )
        assert [e.symbol for e in events] == ["FLAT"]

    def test_exactly_at_the_coverage_threshold_is_allowed(self, rng):
        """A boundary the constant makes meaningful: 80% observed passes."""
        estimate = sens(loadings={"MKT": 8.0, "CREDIT": 2.0}, resid_vol=0.02)
        factors, prices = frame(rng, observed=["MKT"], symbol_return=-0.20)
        assert _loading_coverage(estimate, {"MKT": 0.0}) == pytest.approx(MIN_LOADING_COVERAGE)
        events = residual_divergence_events(
            book(), {"OUST": estimate}, log_returns(prices), factors
        )
        assert len(events) == 1

    def test_just_under_the_coverage_threshold_blocks(self, rng):
        estimate = sens(loadings={"MKT": 7.9, "CREDIT": 2.1}, resid_vol=0.02)
        factors, prices = frame(rng, observed=["MKT"], symbol_return=-0.20)
        assert _loading_coverage(estimate, {"MKT": 0.0}) < MIN_LOADING_COVERAGE
        events = residual_divergence_events(
            book(), {"OUST": estimate}, log_returns(prices), factors
        )
        assert events == []

    def test_no_factor_observed_at_all_still_returns_nothing(self, rng):
        """The pre-existing guard, kept: this must not regress into an
        exception now that a second check runs after it."""
        factors, prices = frame(rng, observed=[], symbol_return=-0.20)
        events = residual_divergence_events(
            book(), {"OUST": sens(loadings=OUST_LOADINGS)}, log_returns(prices), factors
        )
        assert events == []
