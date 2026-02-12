"""Tests for research_agent.cli (CLI commands via CliRunner)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from research_agent.cli import app

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
