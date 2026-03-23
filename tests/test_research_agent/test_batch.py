"""Tests for batch execution mode in the research agent."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

from research_agent.batch import RateLimiter, _generate_batch_id, run_batch
from research_agent.card import render_batch_summary
from research_agent.config import ResearchConfig
from research_agent.models import (
    BatchResult,
    Catalyst,
    DipType,
    InputMode,
    KeyMetrics,
    OpportunityCard,
    ResearchInput,
    Verdict,
)
from research_agent.store import Store


def _make_config(**overrides) -> ResearchConfig:
    defaults = dict(
        _env_file=None,
        perplexity_api_key="test",
        anthropic_api_key="test",
    )
    defaults.update(overrides)
    return ResearchConfig(**defaults)


def _make_card(ticker: str) -> OpportunityCard:
    return OpportunityCard(
        id=f"test-{ticker.lower()}",
        input=ResearchInput(mode=InputMode.TICKER, value=ticker.upper()),
        verdict=Verdict.BUY_THE_DIP,
        catalyst=Catalyst(summary=f"{ticker} dipped on earnings miss"),
        dip_type=DipType.TEMPORARY,
        bull_case=["Strong fundamentals"],
        bear_case=["Macro risk"],
        key_metrics=KeyMetrics(revenue_growth="10% YoY"),
    )


class TestRateLimiter:
    def test_enforces_interval(self):
        limiter = RateLimiter(min_interval=0.1)
        start = time.time()
        limiter.wait()
        limiter.wait()
        elapsed = time.time() - start
        # Second wait should have slept ~0.1s
        assert elapsed >= 0.09

    def test_thread_safety(self):
        limiter = RateLimiter(min_interval=0.05)
        timestamps: list[float] = []
        lock = threading.Lock()

        def worker():
            limiter.wait()
            with lock:
                timestamps.append(time.time())

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 5 calls should complete
        assert len(timestamps) == 5
        # Sort timestamps and check intervals
        timestamps.sort()
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i - 1]
            # Each should be at least ~min_interval apart (with small tolerance)
            assert interval >= 0.04


class TestBatchId:
    def test_deterministic(self):
        id1 = _generate_batch_id(["AAPL", "MSFT", "GOOG"])
        id2 = _generate_batch_id(["GOOG", "AAPL", "MSFT"])  # Different order
        assert id1 == id2

    def test_different_for_different_symbols(self):
        id1 = _generate_batch_id(["AAPL", "MSFT"])
        id2 = _generate_batch_id(["AAPL", "GOOG"])
        assert id1 != id2


class TestRunBatch:
    @patch("research_agent.batch._run_single")
    def test_all_succeed(self, mock_run_single):
        cards = [_make_card("AAPL"), _make_card("MSFT"), _make_card("GOOG")]
        mock_run_single.side_effect = cards

        config = _make_config()
        result = run_batch(["AAPL", "MSFT", "GOOG"], config, concurrency=3)

        assert result.total_requested == 3
        assert result.total_completed == 3
        assert result.total_failed == 0
        assert len(result.cards) == 3
        assert result.elapsed_seconds >= 0

    @patch("research_agent.batch._run_single")
    def test_partial_failure(self, mock_run_single):
        def side_effect(symbol, config, rate_limiter):
            if symbol == "MSFT":
                raise RuntimeError("API error for MSFT")
            return _make_card(symbol)

        mock_run_single.side_effect = side_effect

        config = _make_config()
        result = run_batch(["AAPL", "MSFT", "GOOG"], config, concurrency=3)

        assert result.total_requested == 3
        assert result.total_completed == 2
        assert result.total_failed == 1
        assert "MSFT" in result.failures

    @patch("research_agent.batch._run_single")
    def test_deduplication(self, mock_run_single):
        mock_run_single.return_value = _make_card("AAPL")

        config = _make_config()
        result = run_batch(["AAPL", "aapl", "AAPL", " aapl "], config, concurrency=3)

        assert result.total_requested == 1
        assert mock_run_single.call_count == 1


class TestBatchResultModel:
    def test_serialization_roundtrip(self):
        cards = [_make_card("AAPL"), _make_card("MSFT")]
        batch = BatchResult(
            batch_id="test123",
            cards=cards,
            failures={"GOOG": "API error"},
            total_requested=3,
            total_completed=2,
            total_failed=1,
            elapsed_seconds=12.5,
        )
        json_str = batch.model_dump_json()
        restored = BatchResult.model_validate_json(json_str)
        assert restored.batch_id == "test123"
        assert len(restored.cards) == 2
        assert restored.failures["GOOG"] == "API error"
        assert restored.total_requested == 3
        assert restored.elapsed_seconds == 12.5


class TestRenderBatchSummary:
    def test_renders_comparison_table(self):
        cards = [_make_card("AAPL"), _make_card("MSFT")]
        batch = BatchResult(
            batch_id="test123",
            cards=cards,
            total_requested=2,
            total_completed=2,
            total_failed=0,
            elapsed_seconds=5.0,
        )
        md = render_batch_summary(batch)
        assert "Batch Research Summary" in md
        assert "AAPL" in md
        assert "MSFT" in md
        assert "BUY_THE_DIP" in md
        assert "10% YoY" in md
        assert "test123" in md

    def test_renders_failures(self):
        batch = BatchResult(
            batch_id="test456",
            cards=[_make_card("AAPL")],
            failures={"MSFT": "API timeout"},
            total_requested=2,
            total_completed=1,
            total_failed=1,
            elapsed_seconds=3.0,
        )
        md = render_batch_summary(batch)
        assert "Failures" in md
        assert "MSFT" in md
        assert "API timeout" in md


class TestBatchCli:
    def test_parse_tickers(self):
        """Test ticker parsing logic (unit test of the parsing, not full CLI)."""
        tickers_str = "AAPL,MSFT, GOOG , "
        symbols = [t.strip() for t in tickers_str.split(",") if t.strip()]
        assert symbols == ["AAPL", "MSFT", "GOOG"]

    def test_parse_file(self, tmp_path):
        """Test file parsing logic."""
        watchlist = tmp_path / "watchlist.txt"
        watchlist.write_text("AAPL\nMSFT\n# This is a comment\nGOOG\n\n")
        symbols = []
        for line in watchlist.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line.split()[0])
        assert symbols == ["AAPL", "MSFT", "GOOG"]


class TestStoreBatch:
    def test_save_and_load_batch(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = Store(db_path)
        try:
            cards = [_make_card("AAPL"), _make_card("MSFT")]
            batch = BatchResult(
                batch_id="batch123",
                cards=cards,
                failures={},
                total_requested=2,
                total_completed=2,
                total_failed=0,
                elapsed_seconds=10.0,
            )
            store.save_batch(batch)

            # List batches
            batches = store.list_batches()
            assert len(batches) == 1
            assert batches[0]["batch_id"] == "batch123"
            assert batches[0]["total_completed"] == 2

            # Load batch cards
            loaded_cards = store.load_batch_cards("batch123")
            assert len(loaded_cards) == 2
            tickers = {c.input.value.upper() for c in loaded_cards}
            assert tickers == {"AAPL", "MSFT"}
        finally:
            store.close()

    def test_wal_mode_enabled(self, tmp_path):
        db_path = tmp_path / "test_wal.db"
        store = Store(db_path)
        try:
            result = store._conn.execute("PRAGMA journal_mode").fetchone()
            assert result[0] == "wal"
        finally:
            store.close()
