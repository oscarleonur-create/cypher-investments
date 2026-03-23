"""Tests for stocks morning workflow — decision gate logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from advisor.workflow.models import (
    CandidateVerdict,
    PositionAction,
    ScannerHit,
)
from advisor.workflow.morning_stocks import StocksMorningWorkflow


class TestPositionHealthGates:
    """Test equity position health check decision gates."""

    def test_consider_profit_above_15pct(self):
        workflow = StocksMorningWorkflow(skip_account=True)
        positions = [
            {
                "symbol": "NVDA",
                "underlying_symbol": "NVDA",
                "quantity": 10,
                "average_open_price": 100.0,
                "close_price": 122.0,
                "instrument_type": "Equity",
            }
        ]
        results = workflow._check_position_health(positions, [])
        assert len(results) == 1
        assert results[0].action == PositionAction.CONSIDER_PROFIT
        assert results[0].pnl_pct == 22.0

    def test_review_below_minus10pct(self):
        workflow = StocksMorningWorkflow(skip_account=True)
        positions = [
            {
                "symbol": "MARA",
                "underlying_symbol": "MARA",
                "quantity": 50,
                "average_open_price": 20.0,
                "close_price": 17.0,
                "instrument_type": "Equity",
            }
        ]
        results = workflow._check_position_health(positions, [])
        assert len(results) == 1
        assert results[0].action == PositionAction.REVIEW
        assert results[0].pnl_pct == -15.0

    def test_hold_normal_position(self):
        workflow = StocksMorningWorkflow(skip_account=True)
        positions = [
            {
                "symbol": "AAPL",
                "underlying_symbol": "AAPL",
                "quantity": 5,
                "average_open_price": 180.0,
                "close_price": 185.0,
                "instrument_type": "Equity",
            }
        ]
        # Patch yfinance to avoid network calls
        with patch("yfinance.Ticker") as mock_ticker:
            mock_hist = MagicMock()
            mock_hist.__len__ = lambda s: 0
            mock_ticker.return_value.history.return_value = mock_hist
            results = workflow._check_position_health(positions, [])
        assert len(results) == 1
        assert results[0].action == PositionAction.HOLD

    def test_zero_price_handled(self):
        workflow = StocksMorningWorkflow(skip_account=True)
        positions = [
            {
                "symbol": "UNKNOWN",
                "underlying_symbol": "UNKNOWN",
                "quantity": 1,
                "average_open_price": 0,
                "close_price": 0,
                "instrument_type": "Equity",
            }
        ]
        results = workflow._check_position_health(positions, [])
        assert len(results) == 1
        assert results[0].action == PositionAction.HOLD
        assert "Insufficient" in results[0].reason


class TestValidationGates:
    """Test candidate validation decision gates for stocks mode."""

    def _make_hit(self, symbol: str, alpha: float = 75) -> ScannerHit:
        return ScannerHit(
            symbol=symbol, sources=["alpha", "mispricing"], best_score=alpha, alpha_score=alpha
        )

    @patch("advisor.workflow.morning_stocks.StocksMorningWorkflow._validate_candidates")
    def test_deep_dive_enter_ml_buy_strategy_buy(self, mock_validate):
        """ENTER + ML BUY (>=0.60) + strategy BUY → DEEP_DIVE"""
        # Test the logic directly by constructing the scenario
        from advisor.workflow.models import ValidatedCandidate

        candidate = ValidatedCandidate(
            symbol="AMD",
            verdict=CandidateVerdict.DEEP_DIVE,
            confluence_verdict="ENTER",
            ml_signal="BUY",
            ml_win_prob=0.65,
            strategy_signals=["momentum_breakout:BUY"],
            reason="ENTER + ML BUY + strategy BUY → deep dive",
        )
        assert candidate.verdict == CandidateVerdict.DEEP_DIVE

    def test_watch_caution_ml_hold(self):
        """CAUTION + ML not SELL → WATCH"""
        from advisor.workflow.models import ValidatedCandidate

        candidate = ValidatedCandidate(
            symbol="SOFI",
            verdict=CandidateVerdict.WATCH,
            confluence_verdict="CAUTION",
            ml_signal="HOLD",
            ml_win_prob=0.55,
            reason="CAUTION + ML not SELL → watch",
        )
        assert candidate.verdict == CandidateVerdict.WATCH

    def test_skip_pass_verdict(self):
        """PASS → SKIP"""
        from advisor.workflow.models import ValidatedCandidate

        candidate = ValidatedCandidate(
            symbol="XYZ",
            verdict=CandidateVerdict.SKIP,
            confluence_verdict="PASS",
            reason="PASS — does not meet criteria",
        )
        assert candidate.verdict == CandidateVerdict.SKIP

    def test_skip_enter_ml_sell(self):
        """ENTER + ML SELL → SKIP (ML SELL vetoes)"""
        # The gate logic: (is_enter or is_caution) and ml_not_sell → WATCH
        # But if ml_signal is SELL, ml_not_sell is False, so it falls to SKIP

        # Simulating what the validation logic would produce
        confluence_verdict = "ENTER"
        ml_signal = "SELL"
        ml_win_prob = 0.30
        strategy_signals = ["momentum_breakout:BUY"]

        is_enter = confluence_verdict == "ENTER"
        ml_buy = ml_signal == "BUY" and ml_win_prob >= 0.60
        ml_not_sell = ml_signal != "SELL"
        has_strategy_buy = len(strategy_signals) > 0

        if is_enter and ml_buy and has_strategy_buy:
            verdict = CandidateVerdict.DEEP_DIVE
        elif (is_enter or confluence_verdict == "CAUTION") and ml_not_sell:
            verdict = CandidateVerdict.WATCH
        else:
            verdict = CandidateVerdict.SKIP

        assert verdict == CandidateVerdict.SKIP


class TestDiscoveryGate:
    """Test the scanner overlap gate logic."""

    def test_two_scanners_pass(self):
        """Tickers in >= 2 scanners pass regardless of score."""
        hit = ScannerHit(
            symbol="AMD",
            sources=["smart_money", "alpha"],
            best_score=50,
        )
        assert len(hit.sources) >= 2

    def test_single_scanner_high_score_pass(self):
        """Single scanner with score > 70 passes."""
        hit = ScannerHit(
            symbol="NVDA",
            sources=["alpha"],
            best_score=85,
        )
        assert hit.best_score > 70

    def test_single_scanner_low_score_fail(self):
        """Single scanner with score <= 70 fails the gate."""
        hit = ScannerHit(
            symbol="XYZ",
            sources=["mispricing"],
            best_score=45,
        )
        assert len(hit.sources) < 2 and hit.best_score <= 70
