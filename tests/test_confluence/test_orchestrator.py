"""Tests for the confluence orchestrator.

The orchestrator runs the full pipeline (technical → sentiment → fundamental)
for any strategy. We mock the three agents to test gating and verdict logic.
"""

from __future__ import annotations

from unittest.mock import patch

from advisor.confluence.models import (
    ConfluenceVerdict,
    FundamentalResult,
    SentimentResult,
    TechnicalResult,
)
from advisor.confluence.orchestrator import run_confluence


def _tech(is_bullish: bool) -> TechnicalResult:
    return TechnicalResult(
        signal="BUY" if is_bullish else "NEUTRAL",
        price=150.0,
        sma_20=145.0,
        volume_ratio=1.8 if is_bullish else 0.9,
        is_bullish=is_bullish,
    )


def _sent(is_bullish: bool) -> SentimentResult:
    return SentimentResult(
        score=85.0 if is_bullish else 40.0,
        positive_pct=80.0 if is_bullish else 30.0,
        key_headlines=["Test headline"],
        sources=[],
        is_bullish=is_bullish,
    )


def _fund(is_clear: bool, insider: bool = False) -> FundamentalResult:
    return FundamentalResult(
        earnings_within_7_days=not is_clear,
        earnings_date=None,
        insider_buying_detected=insider,
        is_clear=is_clear,
    )


class TestRunConfluence:
    # ── Gate: no breakout → PASS immediately, skip sentiment/fundamental ──

    @patch("advisor.confluence.orchestrator.check_technical")
    def test_no_breakout_returns_pass_immediately(self, mock_tech):
        """When technical is not bullish, verdict is PASS and no API calls."""
        mock_tech.return_value = _tech(False)

        result = run_confluence("AAPL")

        assert result.verdict == ConfluenceVerdict.PASS
        assert result.suggested_hold_days == 0
        assert "No breakout" in result.reasoning
        assert "skipped" in result.reasoning.lower()

    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_no_breakout_skips_sentiment_and_fundamental(self, mock_tech, mock_sent, mock_fund):
        """Sentiment and fundamental should NOT be called when technical fails."""
        mock_tech.return_value = _tech(False)

        run_confluence("AAPL")

        mock_sent.assert_not_called()
        mock_fund.assert_not_called()

    # ── Breakout fires → run all three checks ────────────────────────────

    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_all_bullish_returns_enter(self, mock_tech, mock_sent, mock_fund):
        mock_tech.return_value = _tech(True)
        mock_sent.return_value = _sent(True)
        mock_fund.return_value = _fund(True)

        result = run_confluence("AAPL")

        assert result.verdict == ConfluenceVerdict.ENTER
        assert result.symbol == "AAPL"
        assert result.suggested_hold_days == 5

    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_breakout_plus_sentiment_returns_caution(self, mock_tech, mock_sent, mock_fund):
        """Breakout + sentiment pass, fundamental fails."""
        mock_tech.return_value = _tech(True)
        mock_sent.return_value = _sent(True)
        mock_fund.return_value = _fund(False)

        result = run_confluence("AAPL")

        assert result.verdict == ConfluenceVerdict.CAUTION
        assert result.suggested_hold_days == 3
        assert "earnings risk" in result.reasoning

    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_breakout_plus_fundamental_returns_caution(self, mock_tech, mock_sent, mock_fund):
        """Breakout + fundamental pass, sentiment fails."""
        mock_tech.return_value = _tech(True)
        mock_sent.return_value = _sent(False)
        mock_fund.return_value = _fund(True)

        result = run_confluence("AAPL")

        assert result.verdict == ConfluenceVerdict.CAUTION
        assert "sentiment below 70%" in result.reasoning

    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_breakout_only_returns_pass(self, mock_tech, mock_sent, mock_fund):
        """Breakout fires but both confirmations fail."""
        mock_tech.return_value = _tech(True)
        mock_sent.return_value = _sent(False)
        mock_fund.return_value = _fund(False)

        result = run_confluence("AAPL")

        assert result.verdict == ConfluenceVerdict.PASS
        assert result.suggested_hold_days == 0
        assert "both sentiment and fundamental" in result.reasoning.lower()

    # ── Insider buying bonus ─────────────────────────────────────────────

    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_insider_buying_in_reasoning_for_enter(self, mock_tech, mock_sent, mock_fund):
        mock_tech.return_value = _tech(True)
        mock_sent.return_value = _sent(True)
        mock_fund.return_value = _fund(True, insider=True)

        result = run_confluence("AAPL")

        assert result.verdict == ConfluenceVerdict.ENTER
        assert "Insider buying" in result.reasoning

    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_insider_buying_in_reasoning_for_caution(self, mock_tech, mock_sent, mock_fund):
        mock_tech.return_value = _tech(True)
        mock_sent.return_value = _sent(False)
        mock_fund.return_value = _fund(True, insider=True)

        result = run_confluence("AAPL")

        assert result.verdict == ConfluenceVerdict.CAUTION
        assert "Insider buying" in result.reasoning

    # ── Symbol normalization ─────────────────────────────────────────────

    @patch("advisor.confluence.orchestrator.check_technical")
    def test_symbol_uppercased(self, mock_tech):
        mock_tech.return_value = _tech(False)

        result = run_confluence("aapl")

        assert result.symbol == "AAPL"

    # ── Strategy name parameterization ───────────────────────────────────

    @patch("advisor.confluence.orchestrator.check_technical")
    def test_default_strategy_is_momentum_breakout(self, mock_tech):
        mock_tech.return_value = _tech(False)

        result = run_confluence("AAPL")

        assert result.strategy_name == "momentum_breakout"
        mock_tech.assert_called_once_with("AAPL", strategy_name="momentum_breakout")

    @patch("advisor.confluence.orchestrator.check_technical")
    def test_scan_with_custom_strategy_name(self, mock_tech):
        mock_tech.return_value = _tech(False)

        result = run_confluence("AAPL", strategy_name="sma_crossover")

        assert result.strategy_name == "sma_crossover"
        mock_tech.assert_called_once_with("AAPL", strategy_name="sma_crossover")

    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_strategy_name_in_result(self, mock_tech, mock_sent, mock_fund):
        mock_tech.return_value = _tech(True)
        mock_sent.return_value = _sent(True)
        mock_fund.return_value = _fund(True)

        result = run_confluence("AAPL", strategy_name="buy_hold")

        assert result.strategy_name == "buy_hold"

    @patch("advisor.confluence.orchestrator.check_technical")
    def test_strategy_name_in_pass_reasoning(self, mock_tech):
        mock_tech.return_value = _tech(False)

        result = run_confluence("AAPL", strategy_name="sma_crossover")

        assert "sma_crossover" in result.reasoning
