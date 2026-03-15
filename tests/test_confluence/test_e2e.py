"""End-to-end confluence pipeline tests with mocked external APIs."""

from __future__ import annotations

from unittest.mock import patch

from advisor.confluence.models import (
    ConfluenceVerdict,
    FundamentalResult,
    SentimentResult,
    TechnicalResult,
    VolumeConfirmationResult,
)
from advisor.confluence.orchestrator import run_confluence


def _mock_technical(is_bullish: bool) -> TechnicalResult:
    return TechnicalResult(
        signal="breakout" if is_bullish else "no_signal",
        price=150.0,
        sma_20=145.0,
        volume_ratio=1.8 if is_bullish else 0.7,
        is_bullish=is_bullish,
    )


def _mock_sentiment(is_bullish: bool, confidence: float = 1.0) -> SentimentResult:
    return SentimentResult(
        score=80.0 if is_bullish else 30.0,
        positive_pct=80.0 if is_bullish else 30.0,
        key_headlines=["Test headline"],
        sources=[],
        is_bullish=is_bullish,
        confidence=confidence,
    )


def _mock_fundamental(is_clear: bool) -> FundamentalResult:
    return FundamentalResult(
        earnings_within_7_days=not is_clear,
        earnings_date=None,
        insider_buying_detected=False,
        is_clear=is_clear,
    )


def _mock_volume(score: float = 0.0) -> VolumeConfirmationResult:
    return VolumeConfirmationResult(
        volume_ratio=1.5,
        capitulation_detected=score >= 30,
        capitulation_ratio=2.5 if score >= 30 else 0.0,
        volume_dryup=score >= 50,
        obv_divergence=score >= 70,
        score=score,
    )


class TestConfluenceE2E:
    """Test full ENTER/CAUTION/PASS paths through the pipeline."""

    @patch("advisor.confluence.orchestrator.check_volume_confirmation")
    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_enter_all_aligned(self, mock_tech, mock_sent, mock_fund, mock_vol):
        """All three checks aligned -> ENTER."""
        mock_tech.return_value = _mock_technical(is_bullish=True)
        mock_sent.return_value = _mock_sentiment(is_bullish=True, confidence=0.9)
        mock_fund.return_value = _mock_fundamental(is_clear=True)
        mock_vol.return_value = _mock_volume(score=0.0)

        result = run_confluence("AAPL", include_ml=False)
        assert result.verdict == ConfluenceVerdict.ENTER
        assert result.suggested_hold_days == 5

    @patch("advisor.confluence.orchestrator.check_volume_confirmation")
    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_caution_two_of_three(self, mock_tech, mock_sent, mock_fund, mock_vol):
        """Breakout + clear fundamentals but bearish sentiment -> CAUTION."""
        mock_tech.return_value = _mock_technical(is_bullish=True)
        mock_sent.return_value = _mock_sentiment(is_bullish=False)
        mock_fund.return_value = _mock_fundamental(is_clear=True)
        mock_vol.return_value = _mock_volume(score=0.0)

        result = run_confluence("AAPL", include_ml=False)
        assert result.verdict == ConfluenceVerdict.CAUTION

    @patch("advisor.confluence.orchestrator.check_volume_confirmation")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_pass_no_breakout(self, mock_tech, mock_vol):
        """No breakout -> PASS (skips sentiment/fundamental)."""
        mock_tech.return_value = _mock_technical(is_bullish=False)
        mock_vol.return_value = _mock_volume(score=0.0)

        result = run_confluence("AAPL", include_ml=False)
        assert result.verdict == ConfluenceVerdict.PASS

    @patch("advisor.confluence.orchestrator.check_volume_confirmation")
    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_pass_all_fail(self, mock_tech, mock_sent, mock_fund, mock_vol):
        """Breakout but both sentiment and fundamental fail -> PASS."""
        mock_tech.return_value = _mock_technical(is_bullish=True)
        mock_sent.return_value = _mock_sentiment(is_bullish=False)
        mock_fund.return_value = _mock_fundamental(is_clear=False)
        mock_vol.return_value = _mock_volume(score=0.0)

        result = run_confluence("AAPL", include_ml=False)
        assert result.verdict == ConfluenceVerdict.PASS


class TestGroundingConfidenceGating:
    """Test that low-confidence sentiment is treated as unconfirmed."""

    @patch("advisor.confluence.orchestrator.check_volume_confirmation")
    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_low_confidence_demotes_sentiment(self, mock_tech, mock_sent, mock_fund, mock_vol):
        """Bullish sentiment with 30% confidence should not count as bullish."""
        mock_tech.return_value = _mock_technical(is_bullish=True)
        mock_sent.return_value = _mock_sentiment(is_bullish=True, confidence=0.3)
        mock_fund.return_value = _mock_fundamental(is_clear=True)
        mock_vol.return_value = _mock_volume(score=0.0)

        result = run_confluence("AAPL", include_ml=False)
        # With sentiment demoted, only 1 confirmation (fundamental)
        assert result.verdict == ConfluenceVerdict.CAUTION
        assert "grounding confidence is low" in result.reasoning

    @patch("advisor.confluence.orchestrator.check_volume_confirmation")
    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_high_confidence_counts(self, mock_tech, mock_sent, mock_fund, mock_vol):
        """Bullish sentiment with 90% confidence should count as bullish."""
        mock_tech.return_value = _mock_technical(is_bullish=True)
        mock_sent.return_value = _mock_sentiment(is_bullish=True, confidence=0.9)
        mock_fund.return_value = _mock_fundamental(is_clear=True)
        mock_vol.return_value = _mock_volume(score=0.0)

        result = run_confluence("AAPL", include_ml=False)
        assert result.verdict == ConfluenceVerdict.ENTER
        assert "grounding confidence" not in result.reasoning


class TestVolumeConfirmationIntegration:
    """Test that volume confirmation is included in confluence results."""

    @patch("advisor.confluence.orchestrator.check_volume_confirmation")
    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_volume_confirmation_in_result(self, mock_tech, mock_sent, mock_fund, mock_vol):
        """Volume confirmation should be included in the result."""
        mock_tech.return_value = _mock_technical(is_bullish=True)
        mock_sent.return_value = _mock_sentiment(is_bullish=True, confidence=0.9)
        mock_fund.return_value = _mock_fundamental(is_clear=True)
        mock_vol.return_value = _mock_volume(score=60.0)

        result = run_confluence("AAPL", include_ml=False)
        assert result.volume_confirmation is not None
        assert result.volume_confirmation.score == 60.0

    @patch("advisor.confluence.orchestrator.check_volume_confirmation")
    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_high_volume_score_in_reasoning(self, mock_tech, mock_sent, mock_fund, mock_vol):
        """High volume score (>=40) should appear in reasoning."""
        mock_tech.return_value = _mock_technical(is_bullish=True)
        mock_sent.return_value = _mock_sentiment(is_bullish=True, confidence=0.9)
        mock_fund.return_value = _mock_fundamental(is_clear=True)
        mock_vol.return_value = _mock_volume(score=65.0)

        result = run_confluence("AAPL", include_ml=False)
        assert "Volume confirmation score" in result.reasoning

    @patch("advisor.confluence.orchestrator.check_volume_confirmation")
    @patch("advisor.confluence.orchestrator.check_fundamental")
    @patch("advisor.confluence.orchestrator.check_sentiment")
    @patch("advisor.confluence.orchestrator.check_technical")
    def test_low_volume_score_not_in_reasoning(self, mock_tech, mock_sent, mock_fund, mock_vol):
        """Low volume score (<40) should NOT appear in reasoning."""
        mock_tech.return_value = _mock_technical(is_bullish=True)
        mock_sent.return_value = _mock_sentiment(is_bullish=True, confidence=0.9)
        mock_fund.return_value = _mock_fundamental(is_clear=True)
        mock_vol.return_value = _mock_volume(score=10.0)

        result = run_confluence("AAPL", include_ml=False)
        assert "Volume confirmation score" not in result.reasoning
