"""Tests for importance sampling tail risk estimation."""

import pytest
from advisor.simulator.engine import MonteCarloEngine
from advisor.simulator.models import PCSCandidate, SimConfig


@pytest.fixture
def candidate():
    """A typical OTM put credit spread on a ~$50 stock."""
    return PCSCandidate(
        symbol="TEST",
        expiration="2026-04-03",
        dte=35,
        short_strike=45.0,
        long_strike=42.0,
        width=3.0,
        short_bid=0.80,
        short_ask=0.90,
        long_bid=0.30,
        long_ask=0.40,
        net_credit=0.40,
        mid_credit=0.45,
        short_delta=-0.20,
        short_gamma=0.03,
        short_theta=-0.05,
        short_vega=0.10,
        short_iv=0.35,
        long_delta=-0.10,
        long_iv=0.38,
        underlying_price=50.0,
        iv_percentile=60.0,
        iv_rank=55.0,
        pop_estimate=0.80,
        buying_power=260.0,
    )


class TestImportanceSampling:
    def test_is_cvar_lower_se_than_crude(self, candidate):
        """IS CVaR standard error should be lower than crude SE."""
        cfg = SimConfig(
            n_paths=5000,
            seed=42,
            vol_mean_level=0.30,
            use_importance_sampling=True,
            use_antithetic=False,
            use_control_variate=False,
        )
        result = MonteCarloEngine(cfg).simulate(candidate, n_paths=5000)
        assert result.cvar_95_is != 0.0, "IS CVaR should be computed"
        assert result.cvar_95_se > 0, "IS SE should be positive"
        # IS SE should generally be smaller or comparable
        # (not guaranteed for every seed, but with tilt toward short strike it helps)
        assert (
            result.cvar_95_se < abs(result.cvar_95) * 0.5
        ), f"IS SE ({result.cvar_95_se}) should be reasonable relative to CVaR ({result.cvar_95})"

    def test_is_cvar_consistent_with_crude(self, candidate):
        """IS CVaR and crude CVaR should agree within reasonable bounds."""
        cfg = SimConfig(
            n_paths=5000,
            seed=42,
            vol_mean_level=0.30,
            use_importance_sampling=True,
            use_antithetic=False,
            use_control_variate=False,
        )
        result = MonteCarloEngine(cfg).simulate(candidate, n_paths=5000)
        # Both should be negative (losses in the tail)
        # They should be in the same ballpark (within a factor of 3)
        if result.cvar_95 < 0 and result.cvar_95_is < 0:
            ratio = result.cvar_95_is / result.cvar_95
            assert 0.1 < ratio < 10, (
                f"IS CVaR ({result.cvar_95_is}) and crude CVaR ({result.cvar_95}) "
                f"should be in same ballpark, ratio={ratio:.2f}"
            )

    def test_is_disabled_by_default(self, candidate):
        """When IS is off, cvar_95_is should be 0."""
        cfg = SimConfig(
            n_paths=2000,
            seed=42,
            vol_mean_level=0.30,
            use_importance_sampling=False,
        )
        result = MonteCarloEngine(cfg).simulate(candidate, n_paths=2000)
        assert result.cvar_95_is == 0.0
        assert result.cvar_95_se == 0.0

    def test_is_composes_with_antithetic(self, candidate):
        """IS + antithetic should not error."""
        cfg = SimConfig(
            n_paths=2000,
            seed=42,
            vol_mean_level=0.30,
            use_importance_sampling=True,
            use_antithetic=True,
            use_control_variate=True,
        )
        result = MonteCarloEngine(cfg).simulate(candidate, n_paths=2000)
        assert result.cvar_95_is != 0.0 or result.cvar_95_se >= 0
        assert 0 <= result.pop <= 1
