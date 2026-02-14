"""Market scanner orchestrator — filters universe then runs confluence."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from advisor.confluence.models import ConfluenceResult, ConfluenceVerdict
from advisor.data.cache import DiskCache
from advisor.data.universe import fetch_sp500
from advisor.market.filters import FilterConfig, apply_filters
from advisor.market.models import FilterStatsModel, MarketScanResult
from advisor.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)

_VERDICT_ORDER = {
    ConfluenceVerdict.ENTER: 0,
    ConfluenceVerdict.CAUTION: 1,
    ConfluenceVerdict.PASS: 2,
}


class MarketScanner:
    """Orchestrates market-wide scanning: universe → filters → confluence."""

    def __init__(self, cache: DiskCache | None = None):
        self.cache = cache or DiskCache()

    def scan(
        self,
        strategy_name: str = "momentum_breakout",
        filter_config: FilterConfig | None = None,
        max_workers: int = 4,
        dry_run: bool = False,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> MarketScanResult:
        """Run a full market scan.

        Args:
            strategy_name: Registered strategy for confluence pipeline.
            filter_config: Filter thresholds (uses defaults if None).
            max_workers: Parallel confluence workers.
            dry_run: If True, skip confluence (filters only, zero API cost).
            on_progress: Callback(phase, advance) for progress updates.

        Returns:
            MarketScanResult with filter stats, qualifiers, and results.
        """
        start = time.time()
        config = filter_config or FilterConfig()

        # ── Load universe ─────────────────────────────────────────────────
        if on_progress:
            on_progress("universe", 0)
        universe = fetch_sp500(cache=self.cache)
        symbols = [s.symbol for s in universe]
        sector_map = {s.symbol: s.sector for s in universe}
        if on_progress:
            on_progress("universe_done", len(symbols))

        # ── Run filters ───────────────────────────────────────────────────
        qualifiers, filter_stats = apply_filters(
            symbols=symbols,
            strategy_name=strategy_name,
            config=config,
            sector_map=sector_map,
            on_progress=on_progress,
        )

        stats_model = FilterStatsModel(
            universe_total=filter_stats.universe_total,
            after_volume_cap=filter_stats.after_volume_cap,
            after_sector=filter_stats.after_sector,
            after_technical=filter_stats.after_technical,
            volume_cap_rejected_count=filter_stats.volume_cap_rejected_count,
            sector_rejected_count=filter_stats.sector_rejected_count,
            technical_rejected_count=filter_stats.technical_rejected_count,
            fetch_error_count=filter_stats.fetch_error_count,
        )

        if dry_run or not qualifiers:
            return MarketScanResult(
                strategy_name=strategy_name,
                universe_size=len(symbols),
                filter_stats=stats_model,
                qualifiers=qualifiers,
                results=[],
                errors=[],
                elapsed_seconds=time.time() - start,
            )

        # ── Run confluence on qualifiers ──────────────────────────────────
        registry = StrategyRegistry()
        registry.discover()
        strategy_cls = registry.get_strategy(strategy_name)

        results: list[ConfluenceResult] = []
        errors: list[dict] = []

        if on_progress:
            on_progress("confluence_start", len(qualifiers))

        def _run_one(sym: str) -> ConfluenceResult:
            return strategy_cls.scan(sym)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_one, sym): sym for sym in qualifiers}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.warning("Confluence failed for %s: %s", sym, e)
                    errors.append({"symbol": sym, "error": str(e)})
                if on_progress:
                    on_progress("confluence_tick", 1)

        # Sort: ENTER first, then CAUTION, then PASS
        results.sort(key=lambda r: _VERDICT_ORDER.get(r.verdict, 99))

        return MarketScanResult(
            strategy_name=strategy_name,
            universe_size=len(symbols),
            filter_stats=stats_model,
            qualifiers=qualifiers,
            results=results,
            errors=errors,
            elapsed_seconds=time.time() - start,
        )
