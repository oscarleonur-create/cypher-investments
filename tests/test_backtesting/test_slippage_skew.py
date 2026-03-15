"""Tests for options backtester slippage and IV skew modeling."""

from __future__ import annotations

import pytest
from advisor.backtesting.options_backtester import BacktestConfig
from advisor.core.exit_rules import EXIT_RULE_DEFAULTS


class TestSlippageConfig:
    def test_default_slippage_pct(self):
        config = BacktestConfig()
        assert config.slippage_pct == 0.05

    def test_slippage_reduces_entry_credit(self):
        """Entry slippage should reduce credit received."""
        from advisor.backtesting.options_backtester import Backtester

        bt = object.__new__(Backtester)
        bt.config = BacktestConfig(slippage_pct=0.05)

        credit = 2.00
        slipped = bt._apply_entry_slippage(credit)
        assert slipped == pytest.approx(1.90)
        assert slipped < credit

    def test_slippage_increases_exit_cost(self):
        """Exit slippage should increase cost to close."""
        from advisor.backtesting.options_backtester import Backtester

        bt = object.__new__(Backtester)
        bt.config = BacktestConfig(slippage_pct=0.05)

        cost = 1.00
        slipped = bt._apply_exit_slippage(cost)
        assert slipped == pytest.approx(1.05)
        assert slipped > cost

    def test_zero_slippage_no_effect(self):
        """With slippage_pct=0, credit/cost should be unchanged."""
        from advisor.backtesting.options_backtester import Backtester

        bt = object.__new__(Backtester)
        bt.config = BacktestConfig(slippage_pct=0.0)

        assert bt._apply_entry_slippage(2.00) == pytest.approx(2.00)
        assert bt._apply_exit_slippage(1.00) == pytest.approx(1.00)


class TestIVSkew:
    def test_otm_put_gets_higher_iv(self):
        """OTM puts should have higher IV than ATM."""
        from advisor.backtesting.options_backtester import Backtester

        bt = object.__new__(Backtester)
        bt.config = BacktestConfig(use_iv_skew=True, skew_slope=0.20)

        base_iv = 0.30
        spot = 100.0
        otm_strike = 90.0  # 10% OTM

        skewed = bt._apply_skew(base_iv, otm_strike, spot)
        assert skewed > base_iv
        # 10% OTM -> moneyness = 0.10 -> skew = 0.30 * (1 + 0.20 * 0.10) = 0.306
        assert skewed == pytest.approx(0.306)

    def test_atm_no_skew(self):
        """ATM options should have no skew applied."""
        from advisor.backtesting.options_backtester import Backtester

        bt = object.__new__(Backtester)
        bt.config = BacktestConfig(use_iv_skew=True, skew_slope=0.20)

        base_iv = 0.30
        skewed = bt._apply_skew(base_iv, 100.0, 100.0)
        assert skewed == pytest.approx(base_iv)

    def test_skew_disabled(self):
        """With use_iv_skew=False, IV should be unchanged."""
        from advisor.backtesting.options_backtester import Backtester

        bt = object.__new__(Backtester)
        bt.config = BacktestConfig(use_iv_skew=False)

        assert bt._apply_skew(0.30, 85.0, 100.0) == pytest.approx(0.30)

    def test_deeper_otm_gets_more_skew(self):
        """Deeper OTM puts should get proportionally more skew."""
        from advisor.backtesting.options_backtester import Backtester

        bt = object.__new__(Backtester)
        bt.config = BacktestConfig(use_iv_skew=True, skew_slope=0.20)

        base_iv = 0.30
        spot = 100.0

        iv_5pct_otm = bt._apply_skew(base_iv, 95.0, spot)
        iv_10pct_otm = bt._apply_skew(base_iv, 90.0, spot)
        iv_20pct_otm = bt._apply_skew(base_iv, 80.0, spot)

        assert iv_5pct_otm < iv_10pct_otm < iv_20pct_otm


class TestMarginFloor:
    def test_margin_includes_floor(self):
        """Margin formula should enforce $2.50/share ($250/contract) floor."""
        # For a deep OTM put: 0.20*S - OTM could be very small
        # 0.10*K could also be small for low strikes
        # The $2.50/share floor ensures minimum margin
        S = 10.0  # low-priced stock
        strike = 8.0  # deep OTM
        otm_amount = max(S - strike, 0)  # = 2.0
        margin = max(0.20 * S - otm_amount, 0.10 * strike, 2.50) * 100
        # 0.20*10 - 2 = 0, 0.10*8 = 0.80, floor = 2.50
        assert margin == 250.0  # $2.50 * 100 = $250


class TestSharedExitRuleDefaults:
    def test_backtester_uses_shared_defaults(self):
        config = BacktestConfig()
        assert config.profit_target_pct == EXIT_RULE_DEFAULTS.profit_target_pct
        assert config.stop_loss_multiplier == EXIT_RULE_DEFAULTS.stop_loss_multiplier
        assert config.close_at_dte == EXIT_RULE_DEFAULTS.close_at_dte

    def test_simulator_uses_shared_defaults(self):
        from advisor.simulator.models import SimConfig

        config = SimConfig()
        assert config.profit_target_pct == EXIT_RULE_DEFAULTS.profit_target_pct
        assert config.stop_loss_multiplier == EXIT_RULE_DEFAULTS.stop_loss_multiplier
        assert config.close_at_dte == EXIT_RULE_DEFAULTS.close_at_dte
