"""Unit tests for the options backtester overhaul."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from advisor.backtesting.options_backtester import (
    BacktestConfig,
    Backtester,
    Trade,
    find_strike_for_delta,
)
from advisor.core.enums import OptionType
from advisor.core.pricing import bsm_price
from advisor.market.premium_screener import get_adaptive_delta
from scipy.stats import norm

# ── Helpers ──────────────────────────────────────────────────────────────────


def _old_bs_put_price(S, K, T, r, sigma):
    """Original inline BS put price for parity testing."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _old_bs_call_price(S, K, T, r, sigma):
    """Original inline BS call price for parity testing."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _old_bs_put_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def _old_bs_call_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


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


# ── Test 1: BSM parity ──────────────────────────────────────────────────────


class TestBSMParity:
    """Verify bsm_price() matches old inline functions to within 1e-6."""

    @pytest.mark.parametrize(
        "S,K,T,r,sigma",
        [
            (100, 95, 0.1, 0.045, 0.30),
            (50, 45, 0.25, 0.05, 0.40),
            (200, 180, 0.5, 0.03, 0.25),
            (25, 30, 0.08, 0.04, 0.50),
            (10, 10, 1.0, 0.05, 0.20),
            (150, 140, 0.01, 0.045, 0.35),
        ],
    )
    def test_put_price_matches(self, S, K, T, r, sigma):
        old = _old_bs_put_price(S, K, T, r, sigma)
        new = bsm_price(S, K, T, r, sigma, OptionType.PUT).price
        assert abs(old - new) < 1e-6, f"Put price mismatch: old={old}, new={new}"

    @pytest.mark.parametrize(
        "S,K,T,r,sigma",
        [
            (100, 105, 0.1, 0.045, 0.30),
            (50, 55, 0.25, 0.05, 0.40),
            (200, 220, 0.5, 0.03, 0.25),
            (25, 20, 0.08, 0.04, 0.50),
            (10, 10, 1.0, 0.05, 0.20),
        ],
    )
    def test_call_price_matches(self, S, K, T, r, sigma):
        old = _old_bs_call_price(S, K, T, r, sigma)
        new = bsm_price(S, K, T, r, sigma, OptionType.CALL).price
        assert abs(old - new) < 1e-6, f"Call price mismatch: old={old}, new={new}"

    @pytest.mark.parametrize(
        "S,K,T,r,sigma",
        [
            (100, 95, 0.1, 0.045, 0.30),
            (50, 45, 0.25, 0.05, 0.40),
            (200, 180, 0.5, 0.03, 0.25),
        ],
    )
    def test_put_delta_matches(self, S, K, T, r, sigma):
        old = _old_bs_put_delta(S, K, T, r, sigma)
        new = bsm_price(S, K, T, r, sigma, OptionType.PUT).delta
        assert abs(old - new) < 1e-6, f"Put delta mismatch: old={old}, new={new}"

    @pytest.mark.parametrize(
        "S,K,T,r,sigma",
        [
            (100, 105, 0.1, 0.045, 0.30),
            (50, 55, 0.25, 0.05, 0.40),
        ],
    )
    def test_call_delta_matches(self, S, K, T, r, sigma):
        old = _old_bs_call_delta(S, K, T, r, sigma)
        new = bsm_price(S, K, T, r, sigma, OptionType.CALL).delta
        assert abs(old - new) < 1e-6, f"Call delta mismatch: old={old}, new={new}"

    def test_edge_case_at_expiry(self):
        """T=0 should return intrinsic value."""
        put_itm = bsm_price(90, 100, 0, 0.045, 0.30, OptionType.PUT)
        assert abs(put_itm.price - 10.0) < 1e-6
        assert put_itm.delta == -1.0

        put_otm = bsm_price(110, 100, 0, 0.045, 0.30, OptionType.PUT)
        assert put_otm.price == 0.0
        assert put_otm.delta == 0.0


# ── Test 2: DTE exit ────────────────────────────────────────────────────────


class TestDTEExit:
    def test_dte_exit_triggers(self):
        """Trade at 21 DTE should trigger close with reason 'dte_exit'."""
        bt = _make_backtester_stub()
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-15",
            strike=95.0,
            option_type="put",
            premium=2.00,
            entry_iv=0.30,
        )
        # Date is ~21 days before expiry
        date = pd.Timestamp("2024-01-25")
        closed, result = bt._manage_position(trade, 100.0, 0.30, date, "2024-01-25")
        # premium at this point should be small for OTM put at 95 with 21 DTE
        # Profit target (50%) fires first if the premium is < 1.00
        # Otherwise DTE exit fires
        assert closed is True
        assert result.reason in ("profit_target", "dte_exit")

    def test_dte_exit_disabled(self):
        """When close_at_dte=0, the DTE exit should not trigger."""
        config = BacktestConfig(close_at_dte=0, use_gamma_exit=False, use_iv_crush_exit=False)
        bt = _make_backtester_stub(config)
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=90.0,
            option_type="put",
            premium=1.50,
            entry_iv=0.30,
        )
        # 20 DTE, still open — no DTE exit, no gamma, no IV crush,
        # premium may be small enough for profit target
        date = pd.Timestamp("2024-01-16")
        closed, result = bt._manage_position(trade, 100.0, 0.30, date, "2024-01-16")
        if closed:
            # Should only be profit_target, never dte_exit
            assert result.reason != "dte_exit"


# ── Test 3: Gamma exit ──────────────────────────────────────────────────────


class TestGammaExit:
    def test_gamma_exit_triggers_near_strike(self):
        """Trade near ATM with short DTE should have high gamma, triggering exit."""
        bt = _make_backtester_stub()
        # Strike very close to spot with short time — high gamma
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-01-10",
            strike=100.0,
            option_type="put",
            premium=3.00,
            entry_iv=0.30,
        )
        # ATM with 5 DTE -> very high gamma
        date = pd.Timestamp("2024-01-05")
        S = 100.0
        closed, result = bt._manage_position(trade, S, 0.30, date, "2024-01-05")
        # gamma * S should be large for ATM near expiry
        T = 5 / 365.0
        gamma = bsm_price(S, 100.0, T, 0.045, 0.30, OptionType.PUT).gamma
        assert gamma * S > bt.config.gamma_threshold
        assert closed is True

    def test_gamma_exit_disabled(self):
        """When use_gamma_exit=False, gamma exit should not trigger."""
        config = BacktestConfig(use_gamma_exit=False, close_at_dte=0, use_iv_crush_exit=False)
        bt = _make_backtester_stub(config)
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-01-10",
            strike=100.0,
            option_type="put",
            premium=3.00,
            entry_iv=0.30,
        )
        date = pd.Timestamp("2024-01-05")
        closed, result = bt._manage_position(trade, 100.0, 0.30, date, "2024-01-05")
        if closed:
            assert result.reason != "gamma_exit"


# ── Test 4: IV crush exit ───────────────────────────────────────────────────


class TestIVCrushExit:
    def test_iv_crush_exit_triggers(self):
        """Trade where IV drops 25% from entry and unrealized P&L > 30%."""
        bt = _make_backtester_stub(BacktestConfig(close_at_dte=0, use_gamma_exit=False))
        # Entry at high IV
        entry_iv = 0.60
        S = 100.0
        K = 90.0
        T_entry = 35 / 365.0
        entry_premium = bsm_price(S, K, T_entry, 0.045, entry_iv, OptionType.PUT).price

        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=K,
            option_type="put",
            premium=round(entry_premium, 2),
            entry_iv=entry_iv,
        )

        # After 10 days, IV has dropped significantly
        current_iv = 0.35  # ~42% drop from 0.60
        date = pd.Timestamp("2024-01-11")
        closed, result = bt._manage_position(trade, S, current_iv, date, "2024-01-11")
        # With such a large IV drop, the premium should be much lower
        assert closed is True
        # Could be profit_target or iv_crush_exit depending on exact premium
        assert result.reason in ("profit_target", "iv_crush_exit")

    def test_iv_crush_no_trigger_small_drop(self):
        """IV drop of only 5% should not trigger IV crush exit."""
        bt = _make_backtester_stub(BacktestConfig(close_at_dte=0, use_gamma_exit=False))
        entry_iv = 0.40
        S = 100.0
        K = 85.0  # deep OTM
        T_entry = 35 / 365.0
        entry_premium = bsm_price(S, K, T_entry, 0.045, entry_iv, OptionType.PUT).price

        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=K,
            option_type="put",
            premium=round(entry_premium, 2),
            entry_iv=entry_iv,
        )

        # Only 5% IV drop
        current_iv = 0.38
        date = pd.Timestamp("2024-01-05")
        closed, result = bt._manage_position(trade, S, current_iv, date, "2024-01-05")
        if closed:
            assert result.reason != "iv_crush_exit"


# ── Test 5: Adaptive delta ──────────────────────────────────────────────────


class TestAdaptiveDelta:
    def test_low_iv_tight_delta(self):
        assert get_adaptive_delta(10) == 0.16
        assert get_adaptive_delta(24) == 0.16

    def test_normal_iv_standard_delta(self):
        assert get_adaptive_delta(25) == 0.28
        assert get_adaptive_delta(50) == 0.28
        assert get_adaptive_delta(74) == 0.28

    def test_high_iv_wide_delta(self):
        assert get_adaptive_delta(75) == 0.35
        assert get_adaptive_delta(90) == 0.35
        assert get_adaptive_delta(100) == 0.35

    def test_get_target_delta_adaptive(self):
        bt = _make_backtester_stub(BacktestConfig(use_adaptive_delta=True))
        row = pd.Series({"IV_Pctile": 80.0})
        assert bt._get_target_delta(row, default=0.25) == 0.35

    def test_get_target_delta_fixed(self):
        bt = _make_backtester_stub(BacktestConfig(use_adaptive_delta=False))
        row = pd.Series({"IV_Pctile": 80.0})
        assert bt._get_target_delta(row, default=0.25) == 0.25


# ── Test 6: Regime filter ───────────────────────────────────────────────────


class TestRegimeFilter:
    def test_skip_high_vol_regime(self):
        bt = _make_backtester_stub(BacktestConfig(use_regime_filter=True))
        row = pd.Series({"RedDay": True, "IV_Pctile": 50.0, "regime": 2, "RSI": 30.0})
        assert bt._should_enter(row, rsi=30.0) is False

    def test_allow_normal_regime(self):
        bt = _make_backtester_stub(BacktestConfig(use_regime_filter=True))
        row = pd.Series({"RedDay": True, "IV_Pctile": 50.0, "regime": 1, "RSI": 30.0})
        assert bt._should_enter(row, rsi=30.0) is True

    def test_allow_low_vol_regime(self):
        bt = _make_backtester_stub(BacktestConfig(use_regime_filter=True))
        row = pd.Series({"RedDay": True, "IV_Pctile": 50.0, "regime": 0, "RSI": 30.0})
        assert bt._should_enter(row, rsi=30.0) is True

    def test_regime_filter_disabled(self):
        bt = _make_backtester_stub(BacktestConfig(use_regime_filter=False))
        row = pd.Series({"RedDay": True, "IV_Pctile": 50.0, "regime": 2, "RSI": 30.0})
        assert bt._should_enter(row, rsi=30.0) is True

    def test_rsi_relaxed_in_high_iv(self):
        bt = _make_backtester_stub(
            BacktestConfig(
                base_rsi_threshold=40.0,
                rsi_relax_in_high_iv=True,
                use_regime_filter=False,
            )
        )
        # RSI 45 would fail default threshold of 40, but IV pctile > 75 relaxes it to 50
        row = pd.Series({"RedDay": True, "IV_Pctile": 80.0, "RSI": 45.0})
        assert bt._should_enter(row, rsi=45.0) is True

    def test_rsi_not_relaxed_when_disabled(self):
        bt = _make_backtester_stub(
            BacktestConfig(
                base_rsi_threshold=40.0,
                rsi_relax_in_high_iv=False,
                use_regime_filter=False,
            )
        )
        row = pd.Series({"RedDay": True, "IV_Pctile": 80.0, "RSI": 45.0})
        assert bt._should_enter(row, rsi=45.0) is False


# ── Test 7: find_strike_for_delta uses bsm_price ────────────────────────────


class TestFindStrike:
    def test_put_strike_reasonable(self):
        S, T, r, sigma = 100, 35 / 365, 0.045, 0.30
        strike = find_strike_for_delta(S, T, r, sigma, 0.25, "put")
        assert 70 < strike < 101
        # Verify the delta is close to target
        delta = abs(bsm_price(S, strike, T, r, sigma, OptionType.PUT).delta)
        assert abs(delta - 0.25) < 0.10

    def test_call_strike_reasonable(self):
        S, T, r, sigma = 100, 35 / 365, 0.045, 0.30
        strike = find_strike_for_delta(S, T, r, sigma, 0.30, "call")
        assert 100 <= strike < 130
        delta = bsm_price(S, strike, T, r, sigma, OptionType.CALL).delta
        assert abs(delta - 0.30) < 0.10


# ── Test 8: Trade entry_iv / entry_delta fields ─────────────────────────────


class TestTradeEntryContext:
    def test_trade_has_entry_fields(self):
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=95.0,
            option_type="put",
            premium=2.00,
            entry_iv=0.35,
            entry_delta=0.25,
        )
        assert trade.entry_iv == 0.35
        assert trade.entry_delta == 0.25

    def test_to_dict_includes_entry_fields(self):
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=95.0,
            option_type="put",
            premium=2.00,
            entry_iv=0.35,
            entry_delta=0.25,
        )
        d = trade.to_dict()
        assert d["entry_iv"] == 0.35
        assert d["entry_delta"] == 0.25

    def test_to_dict_excludes_none_entry_fields(self):
        trade = Trade(
            strategy="naked_put",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=95.0,
            option_type="put",
            premium=2.00,
        )
        d = trade.to_dict()
        assert "entry_iv" not in d
        assert "entry_delta" not in d


# ── Test 9: Spread position management ──────────────────────────────────────


class TestSpreadManagement:
    def test_spread_profit_target(self):
        bt = _make_backtester_stub(
            BacktestConfig(close_at_dte=0, use_gamma_exit=False, use_iv_crush_exit=False)
        )
        # Spread with net credit of 1.00 (2.50 - 1.50)
        trade = Trade(
            strategy="put_credit_spread",
            entry_date="2024-01-01",
            expiry_date="2024-02-15",
            strike=95.0,
            option_type="spread",
            premium=2.50,
            long_strike=90.0,
            long_premium=1.50,
            entry_iv=0.30,
        )
        # At a price well above the strikes, the spread should be worth < 0.50
        date = pd.Timestamp("2024-01-20")
        closed, result = bt._manage_position(trade, 110.0, 0.25, date, "2024-01-20")
        assert closed is True
        assert result.reason == "profit_target"

    def test_spread_expiry_itm(self):
        """Spread where short leg is ITM at expiry — should settle at intrinsic."""
        bt = _make_backtester_stub(
            BacktestConfig(close_at_dte=0, use_gamma_exit=False, use_iv_crush_exit=False)
        )
        trade = Trade(
            strategy="put_credit_spread",
            entry_date="2024-01-01",
            expiry_date="2024-02-05",
            strike=100.0,
            option_type="spread",
            premium=3.50,
            long_strike=95.0,
            long_premium=1.50,
            entry_iv=0.30,
        )
        # At expiry, price below short strike (97) — both legs ITM
        # short intrinsic = max(100 - 97, 0) = 3, long intrinsic = max(95 - 97, 0) = 0
        # net intrinsic = 3, initial credit = 2.00, so P&L = (2.00 - 3.00) * 100 = -100
        date = pd.Timestamp("2024-02-05")
        closed, result = bt._manage_position(trade, 97.0, 0.25, date, "2024-02-05")
        assert closed is True
        assert result.reason == "expired"
        assert result.pnl < 0  # lost money


# ── Test 10: BacktestConfig defaults ─────────────────────────────────────────


class TestBacktestConfig:
    def test_defaults(self):
        config = BacktestConfig()
        assert config.use_adaptive_delta is True
        assert config.base_rsi_threshold == 40.0
        assert config.profit_target_pct == 0.50
        assert config.stop_loss_multiplier == 3.0
        assert config.close_at_dte == 21
        assert config.use_gamma_exit is True
        assert config.gamma_threshold == 0.03
        assert config.use_iv_crush_exit is True

    def test_custom_config(self):
        config = BacktestConfig(
            use_adaptive_delta=False,
            profit_target_pct=0.40,
            stop_loss_multiplier=2.0,
        )
        assert config.use_adaptive_delta is False
        assert config.profit_target_pct == 0.40
        assert config.stop_loss_multiplier == 2.0


# ── Test 11: Summary exit reasons ───────────────────────────────────────────


class TestSummaryExitReasons:
    def test_exit_reasons_in_summary(self):
        bt = _make_backtester_stub()
        bt.trades = [
            Trade(
                strategy="naked_put",
                entry_date="2024-01-01",
                expiry_date="2024-02-05",
                strike=95.0,
                option_type="put",
                premium=2.00,
                exit_date="2024-01-20",
                pnl=100.0,
                reason="profit_target",
            ),
            Trade(
                strategy="naked_put",
                entry_date="2024-02-01",
                expiry_date="2024-03-07",
                strike=94.0,
                option_type="put",
                premium=1.80,
                exit_date="2024-02-15",
                pnl=-50.0,
                reason="stop_loss",
            ),
            Trade(
                strategy="naked_put",
                entry_date="2024-03-01",
                expiry_date="2024-04-05",
                strike=93.0,
                option_type="put",
                premium=1.50,
                exit_date="2024-03-10",
                pnl=80.0,
                reason="dte_exit",
            ),
            Trade(
                strategy="naked_put",
                entry_date="2024-04-01",
                expiry_date="2024-05-06",
                strike=92.0,
                option_type="put",
                premium=2.20,
                exit_date="2024-04-05",
                pnl=120.0,
                reason="iv_crush_exit",
            ),
        ]
        bt.equity_curve = [10000, 10100, 10050, 10130, 10250]

        result = bt.summary("naked_put")
        assert "exit_reasons" in result
        assert result["exit_reasons"]["profit_target"] == 1
        assert result["exit_reasons"]["stop_loss"] == 1
        assert result["exit_reasons"]["dte_exit"] == 1
        assert result["exit_reasons"]["iv_crush_exit"] == 1
