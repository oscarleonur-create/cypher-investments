"""Tests for the confluence CLI commands."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from typer.testing import CliRunner

from advisor.cli.app import app
from advisor.confluence.models import (
    ConfluenceResult,
    ConfluenceVerdict,
    FundamentalResult,
    SentimentResult,
    SourceInfo,
    TechnicalResult,
)

runner = CliRunner()

_MOCK_TARGET = "advisor.confluence.orchestrator.run_confluence"


def _make_result(
    verdict: ConfluenceVerdict = ConfluenceVerdict.ENTER,
    strategy_name: str = "momentum_breakout",
) -> ConfluenceResult:
    return ConfluenceResult(
        symbol="AAPL",
        strategy_name=strategy_name,
        verdict=verdict,
        technical=TechnicalResult(
            signal="BUY",
            price=150.0,
            sma_20=145.0,
            volume_ratio=1.8,
            is_bullish=True,
        ),
        sentiment=SentimentResult(
            score=85.0,
            positive_pct=80.0,
            key_headlines=["Stock surges on earnings beat"],
            sources=[
                SourceInfo(source_id="s1", url="https://reuters.com/article1", title="Reuters: AAPL surges", tier=2),
            ],
            is_bullish=True,
        ),
        fundamental=FundamentalResult(
            earnings_within_7_days=False,
            earnings_date=None,
            insider_buying_detected=True,
            is_clear=True,
        ),
        reasoning="All three checks aligned for AAPL.",
        suggested_hold_days=5,
        scanned_at=datetime(2025, 6, 15, 10, 30),
    )


class TestConfluenceCLI:
    @patch(_MOCK_TARGET)
    def test_scan_rich_output(self, mock_run):
        mock_run.return_value = _make_result()

        result = runner.invoke(app, ["confluence", "scan", "AAPL"])

        assert result.exit_code == 0
        assert "ENTER" in result.output
        assert "AAPL" in result.output
        assert "BULLISH" in result.output

    @patch(_MOCK_TARGET)
    @patch("advisor.cli.confluence_cmds.output_json")
    def test_scan_json_output(self, mock_json, mock_run):
        mock_run.return_value = _make_result()

        result = runner.invoke(app, ["confluence", "scan", "AAPL", "--output", "json"])

        assert result.exit_code == 0
        mock_json.assert_called_once()
        data = mock_json.call_args[0][0]
        assert data.symbol == "AAPL"
        assert data.verdict == ConfluenceVerdict.ENTER

    @patch(_MOCK_TARGET)
    def test_scan_verbose_shows_headlines(self, mock_run):
        mock_run.return_value = _make_result()

        result = runner.invoke(app, ["confluence", "scan", "AAPL", "-v"])

        assert result.exit_code == 0
        assert "Stock surges on earnings beat" in result.output

    @patch(_MOCK_TARGET)
    def test_scan_pass_verdict(self, mock_run):
        mock_run.return_value = _make_result(ConfluenceVerdict.PASS)

        result = runner.invoke(app, ["confluence", "scan", "AAPL"])

        assert result.exit_code == 0
        assert "PASS" in result.output

    @patch(_MOCK_TARGET)
    def test_scan_caution_verdict(self, mock_run):
        mock_run.return_value = _make_result(ConfluenceVerdict.CAUTION)

        result = runner.invoke(app, ["confluence", "scan", "AAPL"])

        assert result.exit_code == 0
        assert "CAUTION" in result.output

    @patch(_MOCK_TARGET)
    def test_scan_verbose_shows_sources(self, mock_run):
        mock_run.return_value = _make_result()

        result = runner.invoke(app, ["confluence", "scan", "AAPL", "-v"])

        assert result.exit_code == 0
        assert "reuters.com" in result.output
        assert "Sources" in result.output

    @patch(_MOCK_TARGET)
    def test_scan_error_handling(self, mock_run):
        mock_run.side_effect = RuntimeError("Connection failed")

        result = runner.invoke(app, ["confluence", "scan", "AAPL"])

        assert result.exit_code != 0

    # ── --strategy flag tests ────────────────────────────────────────────

    @patch(_MOCK_TARGET)
    def test_scan_default_strategy(self, mock_run):
        """Default strategy is momentum_breakout when --strategy not provided."""
        mock_run.return_value = _make_result()

        runner.invoke(app, ["confluence", "scan", "AAPL"])

        mock_run.assert_called_once_with("AAPL", strategy_name="momentum_breakout")

    @patch(_MOCK_TARGET)
    def test_scan_with_strategy_flag(self, mock_run):
        """--strategy flag forwards the strategy name to run_confluence."""
        mock_run.return_value = _make_result(strategy_name="sma_crossover")

        result = runner.invoke(app, ["confluence", "scan", "AAPL", "--strategy", "sma_crossover"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with("AAPL", strategy_name="sma_crossover")

    @patch(_MOCK_TARGET)
    def test_scan_with_strategy_short_flag(self, mock_run):
        """-s flag works the same as --strategy."""
        mock_run.return_value = _make_result(strategy_name="buy_hold")

        result = runner.invoke(app, ["confluence", "scan", "AAPL", "-s", "buy_hold"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with("AAPL", strategy_name="buy_hold")

    @patch(_MOCK_TARGET)
    def test_scan_shows_strategy_in_panel_title(self, mock_run):
        """Panel title should include the strategy name."""
        mock_run.return_value = _make_result(strategy_name="sma_crossover")

        result = runner.invoke(app, ["confluence", "scan", "AAPL", "-s", "sma_crossover"])

        assert result.exit_code == 0
        assert "sma_crossover" in result.output
