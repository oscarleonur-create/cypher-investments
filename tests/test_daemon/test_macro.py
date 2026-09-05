"""Factor panel, sensitivity regression, and book exposure.

Deterministic throughout: factor returns are synthesised so a known answer
exists, rather than depending on the network or on what the market did today.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.macro.exposure import build_book_exposure
from advisor.macro.factors import (
    FACTOR_SPECS,
    Factor,
    all_factor_tickers,
    build_factor_returns,
    log_returns,
)
from advisor.macro.sector_map import get_sector_etf, sector_etf_for_name
from advisor.macro.sensitivity import (
    MIN_OBSERVATIONS,
    estimate_sensitivity,
    expected_return,
    residual_z,
    ridge_fit,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20260904)


@pytest.fixture
def factors(rng):
    """300 sessions of independent, well-scaled factor returns."""
    idx = pd.bdate_range("2025-01-01", periods=300)
    return pd.DataFrame({f.value: rng.normal(0, 0.01, len(idx)) for f in Factor}, index=idx)


def pos(symbol="AMD", *, qty=10, price=100.0, account="A") -> Position:
    return Position(
        account=account,
        symbol=symbol,
        underlying=symbol,
        instrument=EQUITY,
        quantity=qty,
        multiplier=1,
        avg_open_price=price,
        close_price=price,
    )


class TestFactorPanel:
    def test_every_factor_has_a_spec(self):
        assert set(FACTOR_SPECS) == set(Factor)

    def test_spread_factors_declare_two_legs(self):
        assert FACTOR_SPECS[Factor.CREDIT].tickers == ("HYG", "LQD")
        assert FACTOR_SPECS[Factor.MKT].tickers == ("SPY",)

    def test_ticker_list_is_deduped(self):
        tickers = all_factor_tickers()
        assert len(tickers) == len(set(tickers))
        assert "SPY" in tickers  # used by MKT, SIZE and BREADTH

    def test_spread_factor_is_the_difference_of_its_legs(self):
        idx = pd.bdate_range("2025-01-01", periods=10)
        prices = pd.DataFrame(
            {
                "HYG": np.linspace(100, 110, 10),
                "LQD": np.linspace(100, 105, 10),
                "SPY": np.linspace(100, 120, 10),
            },
            index=idx,
        )
        built = build_factor_returns(prices)
        rets = log_returns(prices)
        expected = rets["HYG"] - rets["LQD"]
        pd.testing.assert_series_equal(built[Factor.CREDIT.value], expected, check_names=False)

    def test_missing_leg_skips_the_factor_rather_than_zeroing_it(self):
        """A silently zeroed factor reads as 'no exposure' instead of 'unknown'."""
        idx = pd.bdate_range("2025-01-01", periods=10)
        prices = pd.DataFrame({"SPY": np.linspace(100, 110, 10)}, index=idx)
        built = build_factor_returns(prices)
        assert Factor.MKT.value in built.columns
        assert Factor.CREDIT.value not in built.columns  # LQD absent

    def test_empty_prices_produce_an_empty_panel(self):
        assert build_factor_returns(pd.DataFrame()).empty


class TestRidge:
    def test_recovers_a_known_relationship(self, factors, rng):
        """y built as 2*MKT - 1*DURATION should come back close to that."""
        y = (
            2.0 * factors[Factor.MKT.value]
            - 1.0 * factors[Factor.DURATION.value]
            + rng.normal(0, 0.001, len(factors))
        )
        sens = estimate_sensitivity("TEST", y, factors)
        assert sens is not None
        assert sens.loading(Factor.MKT.value) == pytest.approx(2.0, abs=0.15)
        assert sens.loading(Factor.DURATION.value) == pytest.approx(-1.0, abs=0.15)
        assert sens.r2 > 0.95

    def test_pure_noise_gives_near_zero_loadings_and_r2(self, factors, rng):
        y = pd.Series(rng.normal(0, 0.02, len(factors)), index=factors.index)
        sens = estimate_sensitivity("NOISE", y, factors)
        assert sens is not None
        assert sens.r2 < 0.15
        assert all(abs(entry.loading) < 0.5 for entry in sens.loadings)

    def test_shrinks_rather_than_exploding_on_collinear_factors(self, factors, rng):
        """The reason for ridge: OLS on duplicated regressors blows up."""
        collinear = factors.copy()
        collinear["DUP"] = collinear[Factor.MKT.value] + rng.normal(0, 1e-6, len(collinear))
        y = 2.0 * collinear[Factor.MKT.value]
        sens = estimate_sensitivity("DUP", y, collinear)
        assert sens is not None
        assert all(abs(entry.loading) < 10 for entry in sens.loadings)

    def test_constant_regressor_does_not_divide_by_zero(self, factors):
        flat = factors.copy()
        flat[Factor.VOL.value] = 0.0
        y = flat[Factor.MKT.value] * 1.5
        sens = estimate_sensitivity("FLAT", y, flat)
        assert sens is not None
        assert np.isfinite(sens.loading(Factor.VOL.value))

    def test_r2_is_bounded(self, factors, rng):
        y = pd.Series(rng.normal(0, 0.01, len(factors)), index=factors.index)
        sens = estimate_sensitivity("X", y, factors)
        assert 0.0 <= sens.r2 <= 1.0

    def test_ridge_fit_returns_finite_values_on_degenerate_input(self):
        x = np.zeros((200, 3))
        y = np.zeros(200)
        betas, tstats, r2, resid = ridge_fit(y, x)
        assert np.all(np.isfinite(betas))
        assert np.isfinite(r2)
        assert resid == 0.0


class TestInsufficientData:
    def test_short_history_returns_none_not_a_bogus_estimate(self, factors, rng):
        """Real case: two live holdings are recent listings with <120 sessions."""
        short = factors.iloc[-60:]
        y = pd.Series(rng.normal(0, 0.01, len(short)), index=short.index)
        assert estimate_sensitivity("NEW", y, short) is None

    def test_exactly_at_the_minimum_is_accepted(self, factors, rng):
        window = factors.iloc[-MIN_OBSERVATIONS:]
        y = pd.Series(rng.normal(0, 0.01, len(window)), index=window.index)
        assert estimate_sensitivity("EDGE", y, window) is not None

    def test_non_overlapping_dates_yield_none(self, factors, rng):
        other = pd.bdate_range("2020-01-01", periods=200)
        y = pd.Series(rng.normal(0, 0.01, len(other)), index=other)
        assert estimate_sensitivity("GAP", y, factors) is None


class TestPrediction:
    def test_expected_return_recombines_loadings(self, factors, rng):
        y = 2.0 * factors[Factor.MKT.value] + rng.normal(0, 0.001, len(factors))
        sens = estimate_sensitivity("T", y, factors)
        got = expected_return(sens, {Factor.MKT.value: 0.01})
        assert got == pytest.approx(0.02, abs=0.002)

    def test_residual_z_is_zero_when_the_move_matches_the_model(self, factors, rng):
        y = 2.0 * factors[Factor.MKT.value] + rng.normal(0, 0.001, len(factors))
        sens = estimate_sensitivity("T", y, factors)
        moves = {Factor.MKT.value: 0.01}
        assert residual_z(sens, expected_return(sens, moves), moves) == pytest.approx(0, abs=0.1)

    def test_residual_z_flags_an_unexplained_move(self, factors, rng):
        y = 2.0 * factors[Factor.MKT.value] + rng.normal(0, 0.001, len(factors))
        sens = estimate_sensitivity("T", y, factors)
        moves = {Factor.MKT.value: 0.0}
        assert abs(residual_z(sens, 0.15, moves)) > 3

    def test_zero_residual_vol_does_not_divide_by_zero(self, factors, rng):
        y = 2.0 * factors[Factor.MKT.value]
        sens = estimate_sensitivity("T", y, factors)
        sens.resid_vol = 0.0
        assert residual_z(sens, 0.05, {}) == 0.0


class TestBookExposure:
    def _sens(self, symbol, factors, rng, beta):
        y = beta * factors[Factor.MKT.value] + rng.normal(0, 0.001, len(factors))
        return estimate_sensitivity(symbol, y, factors)

    def test_weights_loadings_by_signed_notional(self, factors, rng):
        book = BookSnapshot(positions=[pos("AMD", qty=10, price=100.0)], net_liq=10_000.0)
        sens = {"AMD": self._sens("AMD", factors, rng, 2.0)}
        exp = build_book_exposure(book, sens)
        # 1000 notional / 10000 net liq = 0.1 weight, x beta 2.0
        assert exp.loading(Factor.MKT.value) == pytest.approx(0.2, abs=0.02)

    def test_a_short_position_subtracts_exposure(self, factors, rng):
        long_book = BookSnapshot(positions=[pos(qty=10, price=100.0)], net_liq=10_000.0)
        short_book = BookSnapshot(positions=[pos(qty=-10, price=100.0)], net_liq=10_000.0)
        sens = {"AMD": self._sens("AMD", factors, rng, 2.0)}
        assert build_book_exposure(long_book, sens).loading(Factor.MKT.value) == pytest.approx(
            -build_book_exposure(short_book, sens).loading(Factor.MKT.value)
        )

    def test_symbols_without_an_estimate_are_reported_not_zeroed(self, factors, rng):
        """'Unknown' and 'neutral' are different answers."""
        book = BookSnapshot(
            positions=[pos("AMD", price=100.0), pos("NEW", price=100.0)], net_liq=10_000.0
        )
        exp = build_book_exposure(book, {"AMD": self._sens("AMD", factors, rng, 2.0)})
        assert exp.uncovered == ["NEW"]
        assert exp.covered_weight == pytest.approx(0.5, abs=0.01)

    def test_zero_net_liq_is_handled(self, factors, rng):
        book = BookSnapshot(positions=[pos()], net_liq=0.0)
        exp = build_book_exposure(book, {"AMD": self._sens("AMD", factors, rng, 2.0)})
        assert exp.factors == {}
        assert exp.uncovered == ["AMD"]

    def test_empty_book_produces_empty_exposure(self):
        exp = build_book_exposure(BookSnapshot(net_liq=1000.0), {})
        assert exp.factors == {}
        assert exp.covered_weight == 0.0

    def test_contributors_are_ranked_by_absolute_share(self, factors, rng):
        book = BookSnapshot(
            positions=[pos("BIG", qty=50, price=100.0), pos("SML", qty=1, price=100.0)],
            net_liq=10_000.0,
        )
        sens = {
            "BIG": self._sens("BIG", factors, rng, 2.0),
            "SML": self._sens("SML", factors, rng, 2.0),
        }
        exp = build_book_exposure(book, sens)
        assert exp.factors[Factor.MKT.value].contributors[0][0] == "BIG"

    def test_ranked_orders_by_absolute_loading(self, factors, rng):
        book = BookSnapshot(positions=[pos(qty=10, price=100.0)], net_liq=10_000.0)
        exp = build_book_exposure(book, {"AMD": self._sens("AMD", factors, rng, 2.0)})
        magnitudes = [abs(f.net_loading) for f in exp.ranked()]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_expected_move_recombines_book_loadings(self, factors, rng):
        book = BookSnapshot(positions=[pos(qty=10, price=100.0)], net_liq=10_000.0)
        exp = build_book_exposure(book, {"AMD": self._sens("AMD", factors, rng, 2.0)})
        assert exp.expected_move({Factor.MKT.value: 0.01}) == pytest.approx(0.002, abs=0.001)


class TestSectorMap:
    def test_membership_lookup(self):
        assert get_sector_etf("AAPL") == "XLK"
        assert get_sector_etf("aapl") == "XLK"

    def test_unknown_symbol_returns_none(self):
        assert get_sector_etf("ZZZZ") is None

    def test_sector_name_lookup_is_case_insensitive(self):
        assert sector_etf_for_name("Technology") == "XLK"
        assert sector_etf_for_name("  technology  ") == "XLK"

    def test_unknown_or_empty_sector_name(self):
        assert sector_etf_for_name("Widgets") is None
        assert sector_etf_for_name(None) is None
