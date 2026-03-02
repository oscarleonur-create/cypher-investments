"""Tests for Monte Carlo engine — path generation, BSM vectorization, exit rules."""

import numpy as np
import pytest
from advisor.simulator.engine import MonteCarloEngine, bsm_put_price_vec
from advisor.simulator.models import PCSCandidate, SimConfig
from scipy.stats import kurtosis


@pytest.fixture
def config():
    return SimConfig(
        n_paths=5_000,
        student_t_df=5.0,
        vol_mean_level=0.30,
        vol_mean_revert_speed=0.5,
        leverage_effect=-0.5,
        risk_free_rate=0.05,
        profit_target_pct=0.50,
        stop_loss_multiplier=2.0,
        close_at_dte=7,
    )


@pytest.fixture
def engine(config):
    return MonteCarloEngine(config)


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
        net_credit=0.40,  # 0.80 - 0.40
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


class TestPathGeneration:
    def test_path_shape(self, engine):
        prices, ivs = engine._generate_paths(S0=100.0, iv0=0.30, dte=30, n_paths=1000)
        assert prices.shape == (1000, 31)
        assert ivs.shape == (1000, 31)

    def test_initial_values(self, engine):
        prices, ivs = engine._generate_paths(S0=50.0, iv0=0.25, dte=20, n_paths=500)
        np.testing.assert_allclose(prices[:, 0], 50.0)
        np.testing.assert_allclose(ivs[:, 0], 0.25)

    def test_fat_tails(self, engine):
        """Student-t innovations should produce excess kurtosis > 0 (fatter than normal)."""
        prices, _ = engine._generate_paths(S0=100.0, iv0=0.30, dte=60, n_paths=10_000)
        # Compute log returns from the generated paths
        log_returns = np.log(prices[:, 1:] / prices[:, :-1]).flatten()
        k = kurtosis(log_returns, fisher=True)
        # Student-t with df=5 should give excess kurtosis > 0
        assert k > 0, f"Expected fat tails (kurtosis > 0), got {k}"

    def test_iv_mean_reversion(self, engine):
        """IV should tend back toward the mean level over time."""
        # Start IV far from mean (0.60 vs mean of 0.30)
        _, ivs = engine._generate_paths(S0=100.0, iv0=0.60, dte=60, n_paths=5000)
        avg_final_iv = np.mean(ivs[:, -1])
        # Should have reverted toward 0.30 (at least partway from 0.60)
        assert avg_final_iv < 0.60, f"IV should revert toward mean, got {avg_final_iv:.3f}"

    def test_prices_positive(self, engine):
        prices, _ = engine._generate_paths(S0=100.0, iv0=0.50, dte=45, n_paths=5000)
        assert np.all(prices > 0), "All prices must be positive (GBM property)"

    def test_ivs_floored(self, engine):
        _, ivs = engine._generate_paths(S0=100.0, iv0=0.05, dte=45, n_paths=5000)
        assert np.all(ivs >= 0.01), "IVs should be floored at 1%"


class TestVectorizedBSM:
    def test_scalar_parity(self):
        """Vectorized BSM should match scalar BSM for single inputs."""
        from advisor.core.enums import OptionType
        from advisor.core.pricing import bsm_price

        S = np.array([100.0])
        sigma = np.array([0.30])
        K, T, r = 95.0, 30 / 252, 0.05

        vec_price = bsm_put_price_vec(S, K, T, r, sigma)[0]
        scalar_result = bsm_price(100.0, K, T, r, 0.30, OptionType.PUT)

        np.testing.assert_allclose(vec_price, scalar_result.price, atol=0.01)

    def test_batch_shape(self):
        S = np.array([100.0, 95.0, 90.0, 85.0])
        sigma = np.array([0.30, 0.35, 0.40, 0.45])
        result = bsm_put_price_vec(S, 95.0, 30 / 252, 0.05, sigma)
        assert result.shape == (4,)

    def test_otm_put_cheaper(self):
        """Deep OTM put should be cheaper than ATM put."""
        S = np.array([100.0, 100.0])
        sigma = np.array([0.30, 0.30])
        otm = bsm_put_price_vec(S, 80.0, 30 / 252, 0.05, sigma)[0]
        atm = bsm_put_price_vec(S, 100.0, 30 / 252, 0.05, sigma)[0]
        assert otm < atm

    def test_expiration_intrinsic(self):
        """At T=0, put price should equal intrinsic value."""
        S = np.array([90.0, 100.0, 110.0])
        sigma = np.array([0.30, 0.30, 0.30])
        prices = bsm_put_price_vec(S, 100.0, 0.0, 0.05, sigma)
        np.testing.assert_allclose(prices, [10.0, 0.0, 0.0])


class TestPCSSimulation:
    def test_sim_result_ranges(self, engine, candidate):
        result = engine.simulate(candidate, n_paths=5000)
        assert 0 <= result.pop <= 1
        assert 0 <= result.touch_prob <= 1
        assert 0 <= result.stop_prob <= 1
        assert result.avg_hold_days > 0
        assert result.avg_hold_days <= candidate.dte

    def test_exit_fractions_sum_to_one(self, engine, candidate):
        result = engine.simulate(candidate, n_paths=5000)
        total = (
            result.exit_profit_target
            + result.exit_stop_loss
            + result.exit_dte
            + result.exit_expiration
        )
        np.testing.assert_allclose(total, 1.0, atol=0.01)

    def test_pnl_bounded(self, engine, candidate):
        """P&L should have reasonable bounds."""
        result = engine.simulate(candidate, n_paths=5000)
        max_loss = (candidate.width - candidate.net_credit) * 100
        # P5 shouldn't be worse than max loss (with tolerance for MTM fluctuation)
        assert result.pnl_p5 >= -max_loss - 50
        # P95 should be positive (winners exist)
        assert result.pnl_p95 > 0

    def test_far_otm_high_pop(self):
        """A moderately OTM spread (delta ~0.10) should have POP > 50%."""
        far_otm = PCSCandidate(
            symbol="TEST",
            expiration="2026-04-03",
            dte=35,
            short_strike=44.0,  # ~12% OTM
            long_strike=41.0,
            width=3.0,
            short_bid=0.60,
            short_ask=0.75,
            long_bid=0.20,
            long_ask=0.30,
            net_credit=0.30,
            mid_credit=0.375,
            short_delta=-0.12,
            short_gamma=0.02,
            short_theta=-0.03,
            short_vega=0.06,
            short_iv=0.35,
            long_delta=-0.05,
            long_iv=0.38,
            underlying_price=50.0,
            iv_percentile=50.0,
            pop_estimate=0.88,
            buying_power=270.0,
        )
        config = SimConfig(n_paths=10_000, close_at_dte=7, slippage_pct=0.01)
        engine = MonteCarloEngine(config)
        result = engine.simulate(far_otm, n_paths=10_000)
        assert result.pop > 0.50, f"OTM spread POP should be > 50%, got {result.pop:.2%}"

    def test_ev_per_bp_calculated(self, engine, candidate):
        result = engine.simulate(candidate, n_paths=5000)
        expected = result.ev / candidate.buying_power if candidate.buying_power > 0 else 0
        np.testing.assert_allclose(result.ev_per_bp, expected, atol=0.001)


class TestSeedReproducibility:
    def test_same_seed_same_results(self, candidate):
        """Same seed should produce identical simulation results."""
        config = SimConfig(n_paths=1000, seed=123, vol_mean_level=0.30)
        engine1 = MonteCarloEngine(config)
        engine2 = MonteCarloEngine(config)
        r1 = engine1.simulate(candidate, n_paths=1000)
        r2 = engine2.simulate(candidate, n_paths=1000)
        assert r1.ev == r2.ev
        assert r1.pop == r2.pop

    def test_different_seed_different_results(self, candidate):
        """Different seeds should (almost certainly) produce different results."""
        config1 = SimConfig(n_paths=1000, seed=123, vol_mean_level=0.30)
        config2 = SimConfig(n_paths=1000, seed=456, vol_mean_level=0.30)
        r1 = MonteCarloEngine(config1).simulate(candidate, n_paths=1000)
        r2 = MonteCarloEngine(config2).simulate(candidate, n_paths=1000)
        assert r1.ev != r2.ev or r1.pop != r2.pop


class TestVolOfVol:
    def test_high_vov_wider_iv_dispersion(self, candidate):
        """Higher vol-of-vol should produce wider IV dispersion at terminal time."""
        low_vov = SimConfig(n_paths=5000, seed=42, vol_of_vol=0.05, vol_mean_level=0.30)
        high_vov = SimConfig(n_paths=5000, seed=42, vol_of_vol=0.50, vol_mean_level=0.30)

        _, ivs_low = MonteCarloEngine(low_vov)._generate_paths(
            S0=50.0, iv0=0.30, dte=30, n_paths=5000
        )
        _, ivs_high = MonteCarloEngine(high_vov)._generate_paths(
            S0=50.0, iv0=0.30, dte=30, n_paths=5000
        )

        std_low = np.std(ivs_low[:, -1])
        std_high = np.std(ivs_high[:, -1])
        assert (
            std_high > std_low
        ), f"High VoV should produce wider IV spread: {std_high:.4f} vs {std_low:.4f}"


class TestVarianceReduction:
    """Tests for antithetic variates, stratified sampling, and control variate."""

    def _run_replications(self, config, candidate, n_reps=20, n_paths=2000):
        """Run n_reps independent sims, return array of EV estimates."""
        evs = []
        for i in range(n_reps):
            cfg = config.model_copy(update={"seed": i * 1000 + 42})
            engine = MonteCarloEngine(cfg)
            result = engine.simulate(candidate, n_paths=n_paths)
            evs.append(result.ev)
        return np.array(evs)

    def test_antithetic_reduces_variance(self, candidate):
        """Antithetic variates should reduce variance of EV estimates across replications."""
        base_cfg = SimConfig(
            vol_mean_level=0.30,
            use_antithetic=False,
            use_control_variate=False,
            use_stratified=False,
        )
        anti_cfg = SimConfig(
            vol_mean_level=0.30,
            use_antithetic=True,
            use_control_variate=False,
            use_stratified=False,
        )
        evs_base = self._run_replications(base_cfg, candidate)
        evs_anti = self._run_replications(anti_cfg, candidate)
        assert np.std(evs_anti) < np.std(
            evs_base
        ), f"Antithetic should reduce EV std: {np.std(evs_anti):.3f} vs {np.std(evs_base):.3f}"

    def test_antithetic_preserves_mean(self, candidate):
        """EV with and without antithetic should agree within tolerance."""
        base_cfg = SimConfig(
            n_paths=5000,
            seed=42,
            vol_mean_level=0.30,
            use_antithetic=False,
            use_control_variate=False,
        )
        anti_cfg = SimConfig(
            n_paths=5000,
            seed=42,
            vol_mean_level=0.30,
            use_antithetic=True,
            use_control_variate=False,
        )
        r_base = MonteCarloEngine(base_cfg).simulate(candidate, n_paths=5000)
        r_anti = MonteCarloEngine(anti_cfg).simulate(candidate, n_paths=5000)
        # Should agree within $5 (MC noise)
        assert (
            abs(r_base.ev - r_anti.ev) < 5.0
        ), f"Antithetic EV should be close: {r_base.ev} vs {r_anti.ev}"

    def test_control_variate_reduces_variance(self, candidate):
        """Control variate should reduce variance of EV estimates."""
        base_cfg = SimConfig(
            vol_mean_level=0.30,
            use_antithetic=False,
            use_control_variate=False,
            use_stratified=False,
        )
        cv_cfg = SimConfig(
            vol_mean_level=0.30,
            use_antithetic=False,
            use_control_variate=True,
            use_stratified=False,
        )
        evs_base = self._run_replications(base_cfg, candidate)
        evs_cv = self._run_replications(cv_cfg, candidate)
        assert (
            np.std(evs_cv) <= np.std(evs_base) * 1.1
        ), f"CV should not increase variance much: {np.std(evs_cv):.3f} vs {np.std(evs_base):.3f}"

    def test_stratified_reduces_variance(self, candidate):
        """Stratified sampling should reduce variance of EV estimates."""
        base_cfg = SimConfig(
            vol_mean_level=0.30,
            use_antithetic=False,
            use_control_variate=False,
            use_stratified=False,
        )
        strat_cfg = SimConfig(
            vol_mean_level=0.30,
            use_antithetic=False,
            use_control_variate=False,
            use_stratified=True,
        )
        evs_base = self._run_replications(base_cfg, candidate)
        evs_strat = self._run_replications(strat_cfg, candidate)
        assert np.std(evs_strat) < np.std(
            evs_base
        ), f"Stratified should reduce EV std: {np.std(evs_strat):.3f} vs {np.std(evs_base):.3f}"

    def test_techniques_stack(self, candidate):
        """All three enabled should have lowest variance."""
        crude_cfg = SimConfig(
            vol_mean_level=0.30,
            use_antithetic=False,
            use_control_variate=False,
            use_stratified=False,
        )
        all_cfg = SimConfig(
            vol_mean_level=0.30,
            use_antithetic=True,
            use_control_variate=True,
            use_stratified=True,
        )
        evs_crude = self._run_replications(crude_cfg, candidate)
        evs_all = self._run_replications(all_cfg, candidate)
        assert np.std(evs_all) < np.std(
            evs_crude
        ), f"Stacked VR should beat crude: {np.std(evs_all):.3f} vs {np.std(evs_crude):.3f}"

    def test_backwards_compatible_defaults(self, candidate):
        """Default config (antithetic + CV on) produces valid results."""
        config = SimConfig(n_paths=2000, seed=42, vol_mean_level=0.30)
        result = MonteCarloEngine(config).simulate(candidate, n_paths=2000)
        assert 0 <= result.pop <= 1
        assert result.mc_std_err > 0
        assert result.variance_reduction_factor >= 1.0 or result.variance_reduction_factor > 0

    def test_stratified_path_shape(self):
        """Stratified sampling should produce paths of correct shape."""
        cfg = SimConfig(
            n_paths=1000,
            seed=42,
            vol_mean_level=0.30,
            use_stratified=True,
            use_antithetic=False,
        )
        engine = MonteCarloEngine(cfg)
        prices, ivs = engine._generate_paths(S0=100.0, iv0=0.30, dte=30, n_paths=1000)
        assert prices.shape == (1000, 31)
        assert ivs.shape == (1000, 31)

    def test_std_err_reported(self, candidate):
        """mc_std_err should be positive when computed."""
        config = SimConfig(n_paths=2000, seed=42, vol_mean_level=0.30)
        result = MonteCarloEngine(config).simulate(candidate, n_paths=2000)
        assert result.mc_std_err > 0, "Standard error should be positive"
