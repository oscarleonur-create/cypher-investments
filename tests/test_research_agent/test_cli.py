"""Tests for research_agent.cli (CLI commands via CliRunner)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.models import InputMode

runner = CliRunner()


class TestHistoryCommand:
    @patch("research_agent.cli.Store")
    def test_history_empty(self, MockStore):
        mock_store = MagicMock()
        mock_store.list_runs.return_value = []
        MockStore.return_value = mock_store

        result = runner.invoke(app, ["history"])
        assert result.exit_code == 0
        assert "No runs found" in result.output

    @patch("research_agent.cli.Store")
    def test_history_with_runs(self, MockStore):
        mock_store = MagicMock()
        mock_store.list_runs.return_value = [
            {
                "id": "abc123",
                "mode": "ticker",
                "input_value": "AAPL",
                "verdict": "BUY_THE_DIP",
                "dip_type": "TEMPORARY",
                "created_at": "2026-01-15 10:00:00",
            }
        ]
        MockStore.return_value = mock_store

        result = runner.invoke(app, ["history"])
        assert result.exit_code == 0
        assert "abc123" in result.output

    @patch("research_agent.cli.Store")
    def test_history_mode_filter(self, MockStore):
        mock_store = MagicMock()
        mock_store.list_runs.return_value = []
        MockStore.return_value = mock_store

        result = runner.invoke(app, ["history", "--mode", "sector"])
        assert result.exit_code == 0
        mock_store.list_runs.assert_called_once_with(ticker=None, mode="sector", limit=20)


class TestShowCommand:
    @patch("research_agent.cli.Store")
    def test_show_not_found(self, MockStore):
        mock_store = MagicMock()
        mock_store.load_run.return_value = None
        MockStore.return_value = mock_store

        result = runner.invoke(app, ["show", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output


class TestRunCommand:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # Typer returns exit code 0 for help display
        assert "research_agent" in result.output or "Usage" in result.output

    def test_no_mode_option_errors(self):
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 1
        assert "exactly one" in result.output.lower()

    def test_multiple_modes_errors(self):
        result = runner.invoke(app, ["run", "--ticker", "AAPL", "--sector", "Technology"])
        assert result.exit_code == 1
        assert "exactly one" in result.output.lower()

    @patch("research_agent.cli.render_markdown", return_value="# Card")
    @patch("research_agent.result.write_outputs", return_value=("out.json", "out.md"))
    @patch("research_agent.pipeline.run")
    def test_sector_creates_sector_input(self, mock_run, mock_write, mock_render):
        from research_agent.models import OpportunityCard, ResearchInput

        inp = ResearchInput(mode=InputMode.SECTOR, value="Technology")
        mock_card = OpportunityCard(id="s1", input=inp)
        mock_run.return_value = mock_card

        result = runner.invoke(app, ["run", "--sector", "Technology"])
        assert result.exit_code == 0
        call_input = mock_run.call_args[0][0]
        assert call_input.mode == InputMode.SECTOR
        assert call_input.value == "Technology"

    @patch("research_agent.cli.render_markdown", return_value="# Card")
    @patch("research_agent.result.write_outputs", return_value=("out.json", "out.md"))
    @patch("research_agent.pipeline.run")
    def test_thesis_creates_thesis_input(self, mock_run, mock_write, mock_render):
        from research_agent.models import OpportunityCard, ResearchInput

        inp = ResearchInput(mode=InputMode.THESIS, value="AI infrastructure spending")
        mock_card = OpportunityCard(id="t1", input=inp)
        mock_run.return_value = mock_card

        result = runner.invoke(app, ["run", "--thesis", "AI infrastructure spending"])
        assert result.exit_code == 0
        call_input = mock_run.call_args[0][0]
        assert call_input.mode == InputMode.THESIS
        assert call_input.value == "AI infrastructure spending"
