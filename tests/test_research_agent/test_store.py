"""Tests for research_agent.store (uses tmp_path for ephemeral DB)."""

from __future__ import annotations

from research_agent.models import (
    DipType,
    InputMode,
    OpportunityCard,
    ResearchInput,
    Source,
    Verdict,
)
from research_agent.store import Store


def test_save_and_load_run(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    try:
        inp = ResearchInput(mode=InputMode.TICKER, value="AAPL")
        card = OpportunityCard(
            id="test123",
            input=inp,
            verdict=Verdict.BUY_THE_DIP,
            dip_type=DipType.TEMPORARY,
            sources=[
                Source(url="https://example.com/1", title="Article 1", tier=1),
                Source(url="https://example.com/2", title="Article 2", tier=2),
            ],
        )
        store.save_run(card)
        loaded = store.load_run("test123")
        assert loaded is not None
        assert loaded.id == "test123"
        assert loaded.verdict == Verdict.BUY_THE_DIP
        assert len(loaded.sources) == 2
    finally:
        store.close()


def test_load_nonexistent_run(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    try:
        assert store.load_run("nonexistent") is None
    finally:
        store.close()


def test_list_runs(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    try:
        for i, ticker in enumerate(["AAPL", "MSFT", "AAPL"]):
            inp = ResearchInput(mode=InputMode.TICKER, value=ticker)
            card = OpportunityCard(id=f"run{i}", input=inp, verdict=Verdict.WATCH)
            store.save_run(card)

        all_runs = store.list_runs()
        assert len(all_runs) == 3

        aapl_runs = store.list_runs(ticker="AAPL")
        assert len(aapl_runs) == 2

        limited = store.list_runs(limit=1)
        assert len(limited) == 1
    finally:
        store.close()


def test_search_cache(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    try:
        store.cache_search("AAPL stock price", {"results": [{"url": "https://example.com"}]})
        cached = store.get_cached_search("AAPL stock price")
        assert cached is not None
        assert cached["results"][0]["url"] == "https://example.com"

        # Same query (case-insensitive, whitespace-trimmed) hits cache
        cached2 = store.get_cached_search("  aapl stock price  ")
        assert cached2 is not None
    finally:
        store.close()


def test_cache_miss(tmp_path):
    db = tmp_path / "test.db"
    store = Store(db)
    try:
        assert store.get_cached_search("unknown query") is None
    finally:
        store.close()


def test_save_run_upsert(tmp_path):
    """Saving the same run_id twice updates the record."""
    db = tmp_path / "test.db"
    store = Store(db)
    try:
        inp = ResearchInput(mode=InputMode.TICKER, value="AAPL")
        card1 = OpportunityCard(id="same_id", input=inp, verdict=Verdict.WATCH)
        store.save_run(card1)

        card2 = OpportunityCard(id="same_id", input=inp, verdict=Verdict.BUY_THE_DIP)
        store.save_run(card2)

        loaded = store.load_run("same_id")
        assert loaded.verdict == Verdict.BUY_THE_DIP
    finally:
        store.close()
