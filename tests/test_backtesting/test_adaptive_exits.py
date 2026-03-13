"""Tests for adaptive exit rules (regime-based parameter adjustment)."""

from __future__ import annotations

import pytest
from advisor.backtesting.adaptive_exits import REGIME_PARAMS, AdaptiveExitPolicy
from advisor.backtesting.options_backtester import BacktestConfig


class TestRegimeParams:
    def test_all_regimes_defined(self):
        assert "low" in REGIME_PARAMS
        assert "mid" in REGIME_PARAMS
        assert "high" in REGIME_PARAMS

    def test_low_vol_tighter(self):
        """Low vol should have tighter profit target and closer DTE exit."""
        assert REGIME_PARAMS["low"]["profit_target_pct"] < REGIME_PARAMS["mid"]["profit_target_pct"]
        assert REGIME_PARAMS["low"]["close_at_dte"] < REGIME_PARAMS["mid"]["close_at_dte"]
        assert (
            REGIME_PARAMS["low"]["stop_loss_multiplier"]
            < REGIME_PARAMS["mid"]["stop_loss_multiplier"]
        )

    def test_high_vol_wider(self):
        """High vol should have wider profit target and further DTE exit."""
        assert (
            REGIME_PARAMS["high"]["profit_target_pct"] > REGIME_PARAMS["mid"]["profit_target_pct"]
        )
        assert REGIME_PARAMS["high"]["close_at_dte"] > REGIME_PARAMS["mid"]["close_at_dte"]
        assert (
            REGIME_PARAMS["high"]["stop_loss_multiplier"]
            > REGIME_PARAMS["mid"]["stop_loss_multiplier"]
        )


class TestAdaptiveExitPolicy:
    def test_low_vol_regime(self):
        config = BacktestConfig()
        policy = AdaptiveExitPolicy(config, vol_regime="low")
        adapted = policy.adapt()

        assert adapted.profit_target_pct == 0.40
        assert adapted.stop_loss_multiplier == 2.0
        assert adapted.close_at_dte == 14

    def test_mid_vol_regime(self):
        config = BacktestConfig()
        policy = AdaptiveExitPolicy(config, vol_regime="mid")
        adapted = policy.adapt()

        assert adapted.profit_target_pct == 0.50
        assert adapted.stop_loss_multiplier == 3.0
        assert adapted.close_at_dte == 21

    def test_high_vol_regime(self):
        config = BacktestConfig()
        policy = AdaptiveExitPolicy(config, vol_regime="high")
        adapted = policy.adapt()

        assert adapted.profit_target_pct == 0.65
        assert adapted.stop_loss_multiplier == 4.0
        assert adapted.close_at_dte == 28

    def test_unknown_regime_defaults_to_mid(self):
        config = BacktestConfig()
        policy = AdaptiveExitPolicy(config, vol_regime="unknown")
        adapted = policy.adapt()

        assert adapted.profit_target_pct == 0.50
        assert adapted.stop_loss_multiplier == 3.0
        assert adapted.close_at_dte == 21

    def test_high_conviction_widens_profit_target(self):
        config = BacktestConfig()
        policy = AdaptiveExitPolicy(config, vol_regime="mid", conviction_score=80)
        adapted = policy.adapt()

        # 0.50 * 1.10 = 0.55
        assert adapted.profit_target_pct == pytest.approx(0.55, abs=0.01)
        # Stop loss unchanged
        assert adapted.stop_loss_multiplier == 3.0

    def test_low_conviction_tightens_stop_loss(self):
        config = BacktestConfig()
        policy = AdaptiveExitPolicy(config, vol_regime="mid", conviction_score=20)
        adapted = policy.adapt()

        # 3.0 * 0.80 = 2.4
        assert adapted.stop_loss_multiplier == pytest.approx(2.4, abs=0.01)
        # Profit target unchanged
        assert adapted.profit_target_pct == 0.50

    def test_medium_conviction_no_adjustment(self):
        config = BacktestConfig()
        policy = AdaptiveExitPolicy(config, vol_regime="mid", conviction_score=50)
        adapted = policy.adapt()

        assert adapted.profit_target_pct == 0.50
        assert adapted.stop_loss_multiplier == 3.0

    def test_no_conviction_no_adjustment(self):
        config = BacktestConfig()
        policy = AdaptiveExitPolicy(config, vol_regime="mid", conviction_score=None)
        adapted = policy.adapt()

        assert adapted.profit_target_pct == 0.50
        assert adapted.stop_loss_multiplier == 3.0

    def test_does_not_mutate_original_config(self):
        config = BacktestConfig()
        original_pt = config.profit_target_pct
        original_sl = config.stop_loss_multiplier

        policy = AdaptiveExitPolicy(config, vol_regime="high", conviction_score=80)
        adapted = policy.adapt()

        # Original should be unchanged
        assert config.profit_target_pct == original_pt
        assert config.stop_loss_multiplier == original_sl
        # Adapted should be different
        assert adapted.profit_target_pct != original_pt

    def test_preserves_non_exit_config(self):
        """Non-exit fields from base_config should be preserved."""
        config = BacktestConfig(
            use_adaptive_delta=False,
            use_regime_filter=False,
            slippage_pct=0.10,
        )
        policy = AdaptiveExitPolicy(config, vol_regime="high")
        adapted = policy.adapt()

        assert adapted.use_adaptive_delta is False
        assert adapted.use_regime_filter is False
        assert adapted.slippage_pct == 0.10

    def test_combined_regime_and_conviction(self):
        """High vol + high conviction should produce widest settings."""
        config = BacktestConfig()
        policy = AdaptiveExitPolicy(config, vol_regime="high", conviction_score=90)
        adapted = policy.adapt()

        # 0.65 * 1.10 = 0.715
        assert adapted.profit_target_pct > 0.65
        assert adapted.stop_loss_multiplier == 4.0
        assert adapted.close_at_dte == 28
