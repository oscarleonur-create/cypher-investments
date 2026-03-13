"""Tests for enhanced exit rules: trailing stop, theta decay, delta breach."""

from __future__ import annotations

import pandas as pd
from advisor.backtesting.options_backtester import BacktestConfig, Backtester, Trade
from advisor.core.enums import OptionType
from advisor.core.pricing import bsm_price


def _make_backtester_stub(config: BacktestConfig | None = None) -> Backtester:
    """Create a Backtester without calling _load_data (no network)."""
    bt = object.__new__(Backtester)
    bt.config = config or BacktestConfig()
    bt.r = 0.045
    bt.trades = []
    bt.equity_curve = []
    bt.initial_cash = 10000
    bt.cash = 10000
    bt.symbol = "TEST"
    bt.start = "2024-01-01"
    bt.end = "2024-12-31"
    return bt


# ── Trailing Stop ────────────────────────────────────────────────────────────


class TestTrailingStop:
    def test_trailing_stop_triggers(self):
        """Trade that hit profit peak then fell should trigger trailing stop."""
        config = BacktestConfig(
            use_trailing_stop=True,
            trailing_activation_pct=0.30,
            trailing_floor_pct=0.10,
            close_at_dte=0,
            use_gamma_exit=False,
            use_iv_crush_exit=False,
            # Set negative to effectively disable profit target (premium >= 0 always)
            profit_target_pct=-1.0,
            stop_loss_multiplier=100.0,
        )
        bt = _make_backtester_stub(config)
        # Use ATM put to keep premium high and prevent profit target
        S = 97.0  # slightly below strike so put stays ITM-ish
        K = 100.0
        T_entry = 45 / 365.0
        entry_iv = 0.30
        entry_premium = bsm_price(100.0, K, T_entry, 0.045, entry_iv, OptionType.PUT).price

        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-15",
            strike=K,
            option_type="put",
            premium=round(entry_premium, 2),
            entry_iv=entry_iv,
            max_unrealized_pnl=0.0,
        )

        # Simulate: trade was very profitable at some point
        # (price was well above strike, premium near zero)
        trade.max_unrealized_pnl = entry_premium * 0.60  # 60% of credit captured at peak

        # Now price dropped back to just below strike — premium is high again
        # _manage_position will compute current_premium via BSM
        # With S=97, K=100, ~30 DTE, premium should be significant (>= entry_premium)
        # So unrealized = entry_premium - current_premium will be small or negative
        date = pd.Timestamp("2024-01-15")
        closed, result = bt._manage_position(trade, S, entry_iv, date, "2024-01-15")

        # activation = 0.30 * entry_premium. max_unrealized (0.60*ep) >= activation. CHECK.
        # current unrealized = entry_premium - current_premium (could be negative)
        # floor = 0.10 * 0.60 * entry_premium = 0.06 * entry_premium
        # If current_unrealized < floor -> trailing stop
        assert closed is True
        assert result.reason == "trailing_stop"

    def test_trailing_stop_not_activated(self):
        """Trade that never reached activation threshold should not trigger."""
        config = BacktestConfig(
            use_trailing_stop=True,
            trailing_activation_pct=0.50,
            trailing_floor_pct=0.25,
            close_at_dte=0,
            use_gamma_exit=False,
            use_iv_crush_exit=False,
            profit_target_pct=0.99,
            stop_loss_multiplier=10.0,
        )
        bt = _make_backtester_stub(config)
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=95.0,
            option_type="put",
            premium=2.00,
            entry_iv=0.30,
            max_unrealized_pnl=0.50,  # only 25% of credit captured at peak
        )
        # max_unrealized_pnl (0.50) < activation (0.50 * 2.00 = 1.00), not activated
        date = pd.Timestamp("2024-01-15")
        closed, result = bt._manage_position(trade, 100.0, 0.30, date, "2024-01-15")
        if closed:
            assert result.reason != "trailing_stop"

    def test_trailing_stop_disabled(self):
        """When use_trailing_stop=False, trailing stop should never trigger."""
        config = BacktestConfig(
            use_trailing_stop=False,
            close_at_dte=0,
            use_gamma_exit=False,
            use_iv_crush_exit=False,
            profit_target_pct=0.99,
            stop_loss_multiplier=10.0,
        )
        bt = _make_backtester_stub(config)
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=95.0,
            option_type="put",
            premium=2.00,
            entry_iv=0.30,
            max_unrealized_pnl=1.80,  # way above activation
        )
        date = pd.Timestamp("2024-01-15")
        closed, result = bt._manage_position(trade, 100.0, 0.30, date, "2024-01-15")
        if closed:
            assert result.reason != "trailing_stop"


# ── Theta Decay Exit ─────────────────────────────────────────────────────────


class TestThetaDecayExit:
    def test_theta_decay_exit_triggers(self):
        """Trade with high cumulative theta should trigger theta decay exit."""
        config = BacktestConfig(
            use_theta_decay_exit=True,
            theta_decay_target_pct=0.70,
            close_at_dte=0,
            use_gamma_exit=False,
            use_iv_crush_exit=False,
            use_trailing_stop=False,
            profit_target_pct=0.99,
            stop_loss_multiplier=10.0,
        )
        bt = _make_backtester_stub(config)
        S = 100.0
        K = 90.0
        T_entry = 35 / 365.0
        entry_iv = 0.30
        entry_premium = bsm_price(S, K, T_entry, 0.045, entry_iv, OptionType.PUT).price

        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=K,
            option_type="put",
            premium=round(entry_premium, 2),
            entry_iv=entry_iv,
            cumulative_theta=entry_premium * 0.65,  # already accumulated 65% of theta
        )

        # After one more day with some theta, it should cross 70% threshold
        # and with OTM put the position should be profitable
        date = pd.Timestamp("2024-01-20")
        closed, result = bt._manage_position(trade, S, entry_iv, date, "2024-01-20")

        # The cumulative theta gets updated in _manage_position, adding daily theta
        # If total crosses 70% and position is profitable -> theta_decay_exit
        if closed:
            assert result.reason in ("theta_decay_exit", "profit_target")

    def test_theta_decay_disabled(self):
        """When use_theta_decay_exit=False, theta exit should never trigger."""
        config = BacktestConfig(
            use_theta_decay_exit=False,
            close_at_dte=0,
            use_gamma_exit=False,
            use_iv_crush_exit=False,
            profit_target_pct=0.99,
            stop_loss_multiplier=10.0,
        )
        bt = _make_backtester_stub(config)
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=90.0,
            option_type="put",
            premium=2.00,
            entry_iv=0.30,
            cumulative_theta=100.0,  # artificially high
        )
        date = pd.Timestamp("2024-01-15")
        closed, result = bt._manage_position(trade, 100.0, 0.30, date, "2024-01-15")
        if closed:
            assert result.reason != "theta_decay_exit"


# ── Delta Breach Exit ────────────────────────────────────────────────────────


class TestDeltaBreachExit:
    def test_delta_breach_triggers_itm(self):
        """Short put going deep ITM should trigger delta breach."""
        config = BacktestConfig(
            use_delta_breach_exit=True,
            delta_breach_threshold=0.50,
            close_at_dte=0,
            use_gamma_exit=False,
            use_iv_crush_exit=False,
            use_trailing_stop=False,
            use_theta_decay_exit=False,
            profit_target_pct=0.99,
            stop_loss_multiplier=10.0,
        )
        bt = _make_backtester_stub(config)

        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=100.0,
            option_type="put",
            premium=3.00,
            entry_iv=0.30,
        )

        # Price drops well below strike -> delta > 0.50
        date = pd.Timestamp("2024-01-15")
        S = 95.0  # ITM by $5
        closed, result = bt._manage_position(trade, S, 0.30, date, "2024-01-15")

        # Check that delta is indeed > 0.50 for this ITM put
        T = (pd.Timestamp("2024-02-05") - date).days / 365.0
        delta = abs(bsm_price(S, 100.0, T, 0.045, 0.30, OptionType.PUT).delta)
        if delta > 0.50:
            assert closed is True
            assert result.reason == "delta_breach_exit"

    def test_delta_breach_otm(self):
        """OTM put with low delta should not trigger delta breach."""
        config = BacktestConfig(
            use_delta_breach_exit=True,
            delta_breach_threshold=0.50,
            close_at_dte=0,
            use_gamma_exit=False,
            use_iv_crush_exit=False,
            use_trailing_stop=False,
            use_theta_decay_exit=False,
            profit_target_pct=0.99,
            stop_loss_multiplier=10.0,
        )
        bt = _make_backtester_stub(config)
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=85.0,
            option_type="put",
            premium=1.00,
            entry_iv=0.30,
        )

        # Price well above strike -> very low delta
        date = pd.Timestamp("2024-01-15")
        closed, result = bt._manage_position(trade, 105.0, 0.25, date, "2024-01-15")
        if closed:
            assert result.reason != "delta_breach_exit"

    def test_delta_breach_disabled(self):
        """When use_delta_breach_exit=False, delta breach should never trigger."""
        config = BacktestConfig(
            use_delta_breach_exit=False,
            close_at_dte=0,
            use_gamma_exit=False,
            use_iv_crush_exit=False,
            profit_target_pct=0.99,
            stop_loss_multiplier=10.0,
        )
        bt = _make_backtester_stub(config)
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=100.0,
            option_type="put",
            premium=3.00,
            entry_iv=0.30,
        )
        date = pd.Timestamp("2024-01-15")
        closed, result = bt._manage_position(trade, 90.0, 0.30, date, "2024-01-15")
        if closed:
            assert result.reason != "delta_breach_exit"


# ── Trade dataclass new fields ───────────────────────────────────────────────


class TestTradeNewFields:
    def test_max_unrealized_pnl_default(self):
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=95.0,
            option_type="put",
            premium=2.00,
        )
        assert trade.max_unrealized_pnl == 0.0

    def test_cumulative_theta_default(self):
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=95.0,
            option_type="put",
            premium=2.00,
        )
        assert trade.cumulative_theta == 0.0

    def test_new_fields_in_to_dict(self):
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=95.0,
            option_type="put",
            premium=2.00,
            max_unrealized_pnl=1.50,
            cumulative_theta=0.80,
        )
        d = trade.to_dict()
        assert d["max_unrealized_pnl"] == 1.50
        assert d["cumulative_theta"] == 0.80


# ── BacktestConfig new fields ───────────────────────────────────────────────


class TestBacktestConfigNewFields:
    def test_trailing_stop_defaults(self):
        config = BacktestConfig()
        assert config.use_trailing_stop is False
        assert config.trailing_activation_pct == 0.50
        assert config.trailing_floor_pct == 0.25

    def test_theta_decay_defaults(self):
        config = BacktestConfig()
        assert config.use_theta_decay_exit is False
        assert config.theta_decay_target_pct == 0.70

    def test_delta_breach_defaults(self):
        config = BacktestConfig()
        assert config.use_delta_breach_exit is False
        assert config.delta_breach_threshold == 0.50

    def test_adaptive_exits_default(self):
        config = BacktestConfig()
        assert config.adaptive_exits is False
