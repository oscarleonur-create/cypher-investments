"""Tests for the StrategyBase.scan() classmethod."""

from __future__ import annotations

from unittest.mock import patch

from advisor.confluence.models import (
    ConfluenceResult,
    ConfluenceVerdict,
    FundamentalResult,
    SentimentResult,
    TechnicalResult,
)


def _make_result(strategy_name: str) -> ConfluenceResult:
    return ConfluenceResult(
        symbol="AAPL",
        strategy_name=strategy_name,
        verdict=ConfluenceVerdict.PASS,
        technical=TechnicalResult(
            signal="NEUTRAL",
            price=0.0,
            sma_20=0.0,
            volume_ratio=0.0,
            is_bullish=False,
        ),
        sentiment=SentimentResult(
            score=0.0,
            positive_pct=0.0,
            key_headlines=[],
            sources=[],
            is_bullish=False,
        ),
        fundamental=FundamentalResult(
            earnings_within_7_days=False,
            earnings_date=None,
            insider_buying_detected=False,
            is_clear=False,
        ),
        reasoning="test",
    )


class TestBaseScan:
    @patch("advisor.confluence.orchestrator.run_confluence")
    def test_scan_delegates_with_strategy_name(self, mock_run):
        """SMACrossover.scan() should call run_confluence with strategy_name='sma_crossover'."""
        from advisor.strategies.equity.sma_crossover import SMACrossover

        mock_run.return_value = _make_result("sma_crossover")

        result = SMACrossover.scan("AAPL")

        mock_run.assert_called_once_with("AAPL", strategy_name="sma_crossover")
        assert result.strategy_name == "sma_crossover"

    @patch("advisor.confluence.orchestrator.run_confluence")
    def test_buy_hold_scan_delegates(self, mock_run):
        """BuyAndHold.scan() should call run_confluence with strategy_name='buy_hold'."""
        from advisor.strategies.equity.buy_hold import BuyAndHold

        mock_run.return_value = _make_result("buy_hold")

        result = BuyAndHold.scan("MSFT")

        mock_run.assert_called_once_with("MSFT", strategy_name="buy_hold")
        assert result.strategy_name == "buy_hold"

    @patch("advisor.confluence.orchestrator.run_confluence")
    def test_momentum_breakout_scan_classmethod(self, mock_run):
        """MomentumBreakout.scan() calls run_confluence with momentum_breakout."""
        from advisor.strategies.equity.momentum_breakout import MomentumBreakout

        mock_run.return_value = _make_result("momentum_breakout")

        result = MomentumBreakout.scan("TSLA")

        mock_run.assert_called_once_with("TSLA", strategy_name="momentum_breakout")
        assert result.strategy_name == "momentum_breakout"

    @patch("advisor.confluence.orchestrator.run_confluence")
    def test_momentum_breakout_module_scan_backward_compat(self, mock_run):
        """Module-level scan() should still work as a backward-compat wrapper."""
        from advisor.strategies.equity.momentum_breakout import scan

        mock_run.return_value = _make_result("momentum_breakout")

        result = scan("AAPL")

        mock_run.assert_called_once_with("AAPL", strategy_name="momentum_breakout")
        assert result.strategy_name == "momentum_breakout"
