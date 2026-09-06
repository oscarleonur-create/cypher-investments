"""Macro event generation — above all, that it stays quiet.

Measured on 503 real sessions, a naive 2-sigma trigger across nine factors
fires on ~34% of days (1 - 0.954^9). The book-impact filter is what turns that
into roughly one interrupt every three weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.daemon.macro_ingest import (
    FACTOR_SHOCK_Z,
    MIN_EXPECTED_BOOK_MOVE,
    factor_shock_events,
    residual_divergence_events,
)
from advisor.daemon.models import EventTier
from advisor.macro.exposure import BookExposure, FactorExposure
from advisor.macro.factors import Factor
from advisor.macro.sensitivity import FactorLoading, SymbolSensitivity


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def factor_frame(rng, *, last_move: dict[str, float] | None = None, n: int = 300):
    idx = pd.bdate_range("2025-01-01", periods=n)
    df = pd.DataFrame({f.value: rng.normal(0, 0.01, n) for f in Factor}, index=idx)
    for factor, move in (last_move or {}).items():
        df.iloc[-1, df.columns.get_loc(factor)] = move
    return df


def exposure(loadings: dict[str, float], *, net_liq=10_000.0) -> BookExposure:
    return BookExposure(
        asof=pd.Timestamp("2026-01-05").date(),
        net_liq=net_liq,
        covered_weight=1.0,
        factors={
            f: FactorExposure(factor=f, net_loading=v, contributors=[("AMD", v)])
            for f, v in loadings.items()
        },
    )


class TestFactorShock:
    def test_big_move_on_an_exposed_factor_fires_tier_a(self, rng):
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.05})  # ~5 sigma
        events = factor_shock_events(factors, exposure({Factor.MKT.value: 1.5}))
        assert len(events) == 1
        assert events[0].tier == EventTier.A
        assert events[0].symbol is None  # book-level
        assert events[0].payload["expected_book_move"] == pytest.approx(0.075, abs=0.001)

    def test_big_move_on_an_unexposed_factor_is_silent(self, rng):
        """A 5-sigma move in something the book has no exposure to is not news."""
        factors = factor_frame(rng, last_move={Factor.ENERGY.value: 0.05})
        assert factor_shock_events(factors, exposure({Factor.ENERGY.value: 0.01})) == []

    def test_large_exposure_on_a_quiet_day_is_silent(self, rng):
        factors = factor_frame(rng)  # nothing unusual
        assert factor_shock_events(factors, exposure({Factor.MKT.value: 2.0})) == []

    def test_material_move_but_immaterial_book_impact_is_silent(self, rng):
        """The filter that fixed the alert rate: a shock the book barely feels."""
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.03})
        # loading clears MATERIAL_BOOK_LOADING but 0.31 * 0.03 < 1% of net liq
        assert factor_shock_events(factors, exposure({Factor.MKT.value: 0.31})) == []

    def test_threshold_is_stricter_than_two_sigma(self):
        assert FACTOR_SHOCK_Z > 2.0
        assert MIN_EXPECTED_BOOK_MOVE >= 0.01

    def test_zero_net_liq_produces_nothing(self, rng):
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.05})
        assert factor_shock_events(factors, exposure({Factor.MKT.value: 1.5}, net_liq=0)) == []

    def test_empty_panel_produces_nothing(self):
        assert factor_shock_events(pd.DataFrame(), exposure({Factor.MKT.value: 1.5})) == []

    def test_short_history_factor_is_skipped(self, rng):
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.05}, n=40)
        assert factor_shock_events(factors, exposure({Factor.MKT.value: 1.5})) == []

    def test_dedups_within_a_session(self, rng):
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.05})
        exp = exposure({Factor.MKT.value: 1.5})
        first = factor_shock_events(factors, exp)[0]
        second = factor_shock_events(factors, exp)[0]
        assert first.dedup_hash() == second.dedup_hash()


class TestResidualDivergence:
    def _sens(self, symbol="AMD", *, beta=1.0, resid_vol=0.02):
        return SymbolSensitivity(
            symbol=symbol,
            asof=pd.Timestamp("2026-01-05").date(),
            window_days=250,
            n_obs=250,
            r2=0.5,
            resid_vol=resid_vol,
            loadings=[FactorLoading(factor=Factor.MKT.value, loading=beta, tstat=5.0)],
        )

    def _book(self, symbol="AMD"):
        return BookSnapshot(
            positions=[
                Position(
                    account="A",
                    symbol=symbol,
                    underlying=symbol,
                    instrument=EQUITY,
                    quantity=10,
                    avg_open_price=100.0,
                    close_price=100.0,
                )
            ],
            net_liq=10_000.0,
        )

    def _returns(self, factors, symbol="AMD", *, last=0.0):
        s = pd.Series(0.0, index=factors.index)
        s.iloc[-1] = last
        return pd.DataFrame({symbol: s})

    def test_a_move_the_model_explains_is_silent(self, rng):
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.02})
        rets = self._returns(factors, last=0.02)  # beta 1 -> exactly expected
        assert residual_divergence_events(self._book(), {"AMD": self._sens()}, rets, factors) == []

    def test_an_unexplained_move_fires_tier_b(self, rng):
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.0})
        rets = self._returns(factors, last=0.10)  # 5 residual sigma
        events = residual_divergence_events(self._book(), {"AMD": self._sens()}, rets, factors)
        assert len(events) == 1
        assert events[0].tier == EventTier.B
        assert events[0].payload["direction"] == "outperformed"

    def test_direction_is_labelled(self, rng):
        factors = factor_frame(rng, last_move={Factor.MKT.value: 0.0})
        rets = self._returns(factors, last=-0.10)
        events = residual_divergence_events(self._book(), {"AMD": self._sens()}, rets, factors)
        assert events[0].payload["direction"] == "underperformed"

    def test_symbol_without_an_estimate_is_skipped(self, rng):
        factors = factor_frame(rng)
        rets = self._returns(factors, last=0.10)
        assert residual_divergence_events(self._book(), {}, rets, factors) == []

    def test_nan_return_is_skipped(self, rng):
        factors = factor_frame(rng)
        rets = self._returns(factors, last=np.nan)
        assert residual_divergence_events(self._book(), {"AMD": self._sens()}, rets, factors) == []

    def test_empty_inputs_produce_nothing(self, rng):
        assert residual_divergence_events(self._book(), {}, pd.DataFrame(), pd.DataFrame()) == []
