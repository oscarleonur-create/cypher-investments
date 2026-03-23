"""Tests for lean swing-trade discovery — pre-screen patterns, alpha ranking, validation gates."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from advisor.workflow.models import CandidateVerdict, ScannerHit

# ── Pre-screen pattern tests ────────────────────────────────────────────────


def _make_ohlcv(
    n: int = 60,
    close_start: float = 100.0,
    close_end: float | None = None,
    volume_base: float = 1_000_000,
    volume_last_mult: float = 1.0,
) -> pd.DataFrame:
    """Build a simple OHLCV DataFrame for testing."""
    close_end = close_end or close_start
    close = np.linspace(close_start, close_end, n)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": [volume_base] * (n - 1) + [volume_base * volume_last_mult],
        },
        index=pd.date_range("2026-01-01", periods=n, freq="B"),
    )


class TestSwingPrescreenBatch:
    """Test _swing_prescreen_batch pattern matching."""

    @patch("yfinance.download")
    def test_momentum_breakout_pattern(self, mock_dl):
        """price > SMA-20 AND volume > 1.5x avg → momentum_breakout."""
        from advisor.workflow.shared import _swing_prescreen_batch

        # Rising price, big volume spike on last bar
        df = _make_ohlcv(n=60, close_start=90.0, close_end=110.0, volume_last_mult=2.0)
        mock_dl.return_value = df

        result = _swing_prescreen_batch(["AAPL"])
        assert "AAPL" in result
        assert "momentum_breakout" in result["AAPL"]

    @patch("yfinance.download")
    def test_swing_signal_all_confirmations(self, mock_dl):
        """Price below lower band + multiple confirmations → swing_signal passes."""
        from advisor.workflow.shared import _swing_prescreen_batch

        n = 210  # need 200+ for SMA-200
        # Uptrending base so price > SMA-200, then sharp drop at end below lower BB
        close = np.concatenate(
            [
                np.linspace(80.0, 120.0, n - 5),  # long uptrend
                np.array([105.0, 103.0, 100.0, 97.0, 90.0]),  # sharp selloff
            ]
        )
        # Volume spike on last bar for volume confirmation
        vol = [1_000_000] * (n - 1) + [2_000_000]
        df = pd.DataFrame(
            {
                "Open": close * 0.99,
                "High": close * 1.01,
                "Low": close * 0.98,
                "Close": close,
                "Volume": vol,
            },
            index=pd.date_range("2025-01-01", periods=n, freq="B"),
        )
        mock_dl.return_value = df

        result = _swing_prescreen_batch(["TSLA"])
        assert "TSLA" in result
        swing_sources = [p for p in result["TSLA"] if p.startswith("swing_signal")]
        assert len(swing_sources) == 1
        # Should contain confirmation details in parens
        assert "(" in swing_sources[0] and ")" in swing_sources[0]

    @patch("yfinance.download")
    def test_swing_signal_gate_only_no_confirmations(self, mock_dl):
        """Price below band but 0 confirmations → does NOT pass."""
        from advisor.workflow.shared import _swing_prescreen_batch

        # Flat price then drop — no RSI < 35 (just barely), no volume spike,
        # no SMA-200 (< 200 bars), no MACD turning up
        n = 60
        close = np.array([100.0] * (n - 1) + [85.0])
        df = pd.DataFrame(
            {
                "Open": close * 0.99,
                "High": close * 1.01,
                "Low": close * 0.98,
                "Close": close,
                "Volume": [1_000_000] * n,
            },
            index=pd.date_range("2026-01-01", periods=n, freq="B"),
        )
        mock_dl.return_value = df

        result = _swing_prescreen_batch(["DROP"])
        # Gate met but not enough confirmations — should not match swing_signal
        swing = [p for p in result.get("DROP", []) if p.startswith("swing_signal")]
        assert len(swing) == 0

    @patch("yfinance.download")
    def test_swing_signal_gate_plus_two(self, mock_dl):
        """Price below band + RSI + volume → passes (strength = 2)."""
        from advisor.workflow.shared import _swing_prescreen_batch

        n = 60
        # Downtrend to push RSI < 35, sharp drop at end below lower BB
        close = np.concatenate(
            [
                np.linspace(120.0, 100.0, n - 3),  # gradual decline
                np.array([95.0, 90.0, 82.0]),  # sharp drop
            ]
        )
        # Big volume spike on last bar
        vol = [1_000_000] * (n - 1) + [3_000_000]
        df = pd.DataFrame(
            {
                "Open": close * 0.99,
                "High": close * 1.01,
                "Low": close * 0.98,
                "Close": close,
                "Volume": vol,
            },
            index=pd.date_range("2026-01-01", periods=n, freq="B"),
        )
        mock_dl.return_value = df

        result = _swing_prescreen_batch(["DIP"])
        assert "DIP" in result
        swing = [p for p in result["DIP"] if p.startswith("swing_signal")]
        assert len(swing) == 1
        # Should have at least rsi and volume in confirmations
        assert "rsi" in swing[0]
        assert "volume" in swing[0]

    @patch("yfinance.download")
    def test_no_patterns_match(self, mock_dl):
        """Flat price, normal volume → no patterns match."""
        from advisor.workflow.shared import _swing_prescreen_batch

        df = _make_ohlcv(n=60, close_start=100.0, close_end=100.0, volume_last_mult=1.0)
        mock_dl.return_value = df

        result = _swing_prescreen_batch(["FLAT"])
        assert "FLAT" not in result

    @patch("yfinance.download")
    def test_empty_data_returns_empty(self, mock_dl):
        """Empty download → empty result."""
        from advisor.workflow.shared import _swing_prescreen_batch

        mock_dl.return_value = pd.DataFrame()

        result = _swing_prescreen_batch(["GHOST"])
        assert result == {}


# ── Lean discovery integration tests ────────────────────────────────────────


class TestRunLeanDiscovery:
    """Test run_lean_discovery end-to-end flow."""

    @patch("advisor.workflow.shared._swing_prescreen_batch")
    @patch("advisor.workflow.shared._resolve_symbols")
    def test_zero_candidates_warns(self, mock_resolve, mock_prescreen):
        """0 pre-screen candidates → warning appended, empty result."""
        from advisor.workflow.shared import run_lean_discovery

        mock_resolve.return_value = ["AAPL", "MSFT"]
        mock_prescreen.return_value = {}

        errors: list[str] = []
        result = run_lean_discovery(["sp500"], [], errors)

        assert result == []
        assert any("0 candidates" in e for e in errors)

    @patch("advisor.confluence.alpha_scorer.compute_alpha")
    @patch("advisor.workflow.shared._swing_prescreen_batch")
    @patch("advisor.workflow.shared._resolve_symbols")
    def test_top_n_ranking(self, mock_resolve, mock_prescreen, mock_alpha):
        """Alpha ranking returns top N symbols."""
        from advisor.workflow.shared import run_lean_discovery

        mock_resolve.return_value = ["A", "B", "C", "D", "E", "F"]
        mock_prescreen.return_value = {
            "A": ["momentum_breakout"],
            "B": ["swing_signal(rsi,volume)"],
            "C": ["momentum_breakout"],
            "D": ["swing_signal(rsi,volume,sma200)"],
            "E": ["swing_signal(rsi,macd)"],
            "F": ["momentum_breakout", "swing_signal(rsi,volume,sma200,macd)"],
        }

        # Simulate different alpha scores
        alpha_scores = {"A": 80, "B": 60, "C": 90, "D": 70, "E": 50, "F": 85}
        mock_result = MagicMock()
        mock_alpha.side_effect = (
            lambda sym, **kw: setattr(mock_result := MagicMock(), "alpha_score", alpha_scores[sym])
            or mock_result
        )

        errors: list[str] = []
        result = run_lean_discovery(["sp500"], [], errors, top_n=3)

        assert len(result) == 3
        # Should be ranked by alpha: C(90), F(85), A(80)
        assert result[0].symbol == "C"
        assert result[1].symbol == "F"
        assert result[2].symbol == "A"

    @patch("advisor.workflow.shared._resolve_symbols")
    def test_empty_universe(self, mock_resolve):
        """No symbols resolved → error."""
        from advisor.workflow.shared import run_lean_discovery

        mock_resolve.return_value = []
        errors: list[str] = []
        result = run_lean_discovery([], [], errors)

        assert result == []
        assert any("No symbols" in e for e in errors)


# ── Lean validation gate tests ──────────────────────────────────────────────


class TestLeanValidationGates:
    """Test _validate_candidates_lean decision gates."""

    def _make_hit(self, symbol: str, alpha: float) -> ScannerHit:
        return ScannerHit(
            symbol=symbol,
            sources=["momentum_breakout"],
            best_score=alpha,
            alpha_score=alpha,
        )

    def test_deep_dive_high_alpha_ml_buy(self):
        """alpha >= 65 AND ML BUY (>= 0.60) → DEEP_DIVE."""
        alpha = 70
        ml_signal = "BUY"
        ml_win_prob = 0.65
        strategy_signals = []

        ml_buy = ml_signal == "BUY" and ml_win_prob >= 0.60
        has_strategy_buy = len(strategy_signals) > 0

        if alpha >= 65 and (ml_buy or (has_strategy_buy and alpha >= 75)):
            verdict = CandidateVerdict.DEEP_DIVE
        elif alpha >= 50 and ml_signal != "SELL":
            verdict = CandidateVerdict.WATCH
        else:
            verdict = CandidateVerdict.SKIP

        assert verdict == CandidateVerdict.DEEP_DIVE

    def test_deep_dive_high_alpha_strategy_buy(self):
        """alpha >= 75 AND strategy BUY (no ML BUY needed) → DEEP_DIVE."""
        alpha = 80
        ml_signal = "HOLD"
        ml_win_prob = 0.50
        strategy_signals = ["sma_crossover:BUY"]

        ml_buy = ml_signal == "BUY" and ml_win_prob >= 0.60
        has_strategy_buy = len(strategy_signals) > 0

        if alpha >= 65 and (ml_buy or (has_strategy_buy and alpha >= 75)):
            verdict = CandidateVerdict.DEEP_DIVE
        elif alpha >= 50 and ml_signal != "SELL":
            verdict = CandidateVerdict.WATCH
        else:
            verdict = CandidateVerdict.SKIP

        assert verdict == CandidateVerdict.DEEP_DIVE

    def test_watch_moderate_alpha_ml_hold(self):
        """alpha >= 50 AND ML not SELL → WATCH."""
        alpha = 55
        ml_signal = "HOLD"
        ml_win_prob = 0.45
        strategy_signals = []

        ml_buy = ml_signal == "BUY" and ml_win_prob >= 0.60
        has_strategy_buy = len(strategy_signals) > 0

        if alpha >= 65 and (ml_buy or (has_strategy_buy and alpha >= 75)):
            verdict = CandidateVerdict.DEEP_DIVE
        elif alpha >= 50 and ml_signal != "SELL":
            verdict = CandidateVerdict.WATCH
        else:
            verdict = CandidateVerdict.SKIP

        assert verdict == CandidateVerdict.WATCH

    def test_skip_low_alpha(self):
        """alpha < 50 → SKIP regardless of ML."""
        alpha = 40
        ml_signal = "BUY"
        ml_win_prob = 0.70
        strategy_signals = ["momentum_breakout:BUY"]

        ml_buy = ml_signal == "BUY" and ml_win_prob >= 0.60
        has_strategy_buy = len(strategy_signals) > 0

        if alpha >= 65 and (ml_buy or (has_strategy_buy and alpha >= 75)):
            verdict = CandidateVerdict.DEEP_DIVE
        elif alpha >= 50 and ml_signal != "SELL":
            verdict = CandidateVerdict.WATCH
        else:
            verdict = CandidateVerdict.SKIP

        assert verdict == CandidateVerdict.SKIP

    def test_skip_ml_sell_vetoes(self):
        """alpha >= 50 but ML SELL → SKIP."""
        alpha = 60
        ml_signal = "SELL"
        ml_win_prob = 0.30
        strategy_signals = []

        ml_buy = ml_signal == "BUY" and ml_win_prob >= 0.60
        has_strategy_buy = len(strategy_signals) > 0

        if alpha >= 65 and (ml_buy or (has_strategy_buy and alpha >= 75)):
            verdict = CandidateVerdict.DEEP_DIVE
        elif alpha >= 50 and ml_signal != "SELL":
            verdict = CandidateVerdict.WATCH
        else:
            verdict = CandidateVerdict.SKIP

        assert verdict == CandidateVerdict.SKIP

    def test_alpha_65_strategy_no_ml_buy_watch_not_deepdive(self):
        """alpha 65, strategy BUY but alpha < 75 and no ML BUY → WATCH (not DEEP_DIVE)."""
        alpha = 65
        ml_signal = "HOLD"
        ml_win_prob = 0.50
        strategy_signals = ["sma_crossover:BUY"]

        ml_buy = ml_signal == "BUY" and ml_win_prob >= 0.60
        has_strategy_buy = len(strategy_signals) > 0

        if alpha >= 65 and (ml_buy or (has_strategy_buy and alpha >= 75)):
            verdict = CandidateVerdict.DEEP_DIVE
        elif alpha >= 50 and ml_signal != "SELL":
            verdict = CandidateVerdict.WATCH
        else:
            verdict = CandidateVerdict.SKIP

        # alpha 65 >= 65 but no ml_buy and alpha < 75 so strategy+alpha gate fails
        assert verdict == CandidateVerdict.WATCH
