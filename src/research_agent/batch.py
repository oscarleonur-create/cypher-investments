"""Batch execution engine with shared rate limiting for multi-symbol research."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from research_agent.config import ResearchConfig
from research_agent.models import (
    BatchResult,
    InputMode,
    OpportunityCard,
    ResearchInput,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter that enforces a minimum interval between requests."""

    def __init__(self, min_interval: float = 0.5) -> None:
        self._min_interval = min_interval
        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request_time = time.time()


def _generate_batch_id(symbols: list[str]) -> str:
    """Deterministic batch ID from sorted symbols + today's date."""
    raw = f"{','.join(sorted(s.upper() for s in symbols))}:{date.today().isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _run_single(
    symbol: str,
    config: ResearchConfig,
    rate_limiter: RateLimiter,
) -> OpportunityCard:
    """Run the research pipeline for a single symbol with a shared rate limiter.

    Each thread gets its own Store instance (own SQLite connection) for thread safety.
    """
    from research_agent.agent import run_loop
    from research_agent.evidence import SourceRegistry
    from research_agent.llm import OpenRouterLLM
    from research_agent.search import TavilyClient
    from research_agent.store import Store

    store = Store(config.db_path)
    try:
        search = TavilyClient(config, store, rate_limiter=rate_limiter)
        llm = OpenRouterLLM(config)
        registry = SourceRegistry()
        state_input = ResearchInput(mode=InputMode.TICKER, value=symbol.upper())
        from research_agent.models import AgentState

        state = AgentState(input=state_input)
        card = run_loop(state, search, llm, registry, config)
        return card
    finally:
        store.close()


def run_batch(
    symbols: list[str],
    config: ResearchConfig,
    concurrency: int = 3,
) -> BatchResult:
    """Run research pipelines for multiple symbols concurrently.

    Uses a shared RateLimiter across all threads and separate Store instances
    per thread for SQLite thread safety.
    """
    # Deduplicate and normalize
    seen: set[str] = set()
    unique_symbols: list[str] = []
    for s in symbols:
        upper = s.strip().upper()
        if upper and upper not in seen:
            seen.add(upper)
            unique_symbols.append(upper)

    batch_id = _generate_batch_id(unique_symbols)
    logger.info(
        "Starting batch %s: %d symbols, concurrency=%d", batch_id, len(unique_symbols), concurrency
    )

    rate_limiter = RateLimiter(min_interval=0.5)
    cards: list[OpportunityCard] = []
    failures: dict[str, str] = {}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_symbol = {
            executor.submit(_run_single, symbol, config, rate_limiter): symbol
            for symbol in unique_symbols
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                card = future.result()
                cards.append(card)
                logger.info("Completed research for %s", symbol)
            except Exception as e:
                logger.error("Research failed for %s: %s", symbol, e)
                failures[symbol] = str(e)

    elapsed = time.time() - start_time

    return BatchResult(
        batch_id=batch_id,
        cards=cards,
        failures=failures,
        total_requested=len(unique_symbols),
        total_completed=len(cards),
        total_failed=len(failures),
        elapsed_seconds=round(elapsed, 2),
    )
