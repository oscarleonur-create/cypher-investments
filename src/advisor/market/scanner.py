"""Market scanner orchestrator — filters universe then runs confluence."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Callable

import yfinance as yf

from advisor.confluence.models import ConfluenceResult, ConfluenceVerdict
from advisor.data.cache import DiskCache
from advisor.data.universe import fetch_universe
from advisor.market.filters import FilterConfig, apply_filters
from advisor.market.models import FilterStatsModel, MarketScanResult
from advisor.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)

_PEAD_EARNINGS_WINDOW = 7  # calendar days

_VERDICT_ORDER = {
    ConfluenceVerdict.ENTER: 0,
    ConfluenceVerdict.CAUTION: 1,
    ConfluenceVerdict.PASS: 2,
}


def _pead_has_recent_earnings(symbol: str) -> bool:
    """Quick check: did this stock report earnings within the PEAD window?

    This is a cheap pre-screen to avoid sending stocks without recent
    earnings into the full (expensive) confluence pipeline.
    """
    try:
        ticker = yf.Ticker(symbol)
        earnings_dates = ticker.earnings_dates
        if earnings_dates is None or earnings_dates.empty:
            return False

        if "Reported EPS" not in earnings_dates.columns:
            return False

        reported = earnings_dates[earnings_dates["Reported EPS"].notna()]
        if reported.empty:
            return False

        report_idx = reported.index[0]
        if hasattr(report_idx, "date"):
            report_date = report_idx.date()
        elif isinstance(report_idx, date):
            report_date = report_idx
        else:
            return False

        days_since = (date.today() - report_date).days
        return 0 <= days_since <= _PEAD_EARNINGS_WINDOW
    except Exception:
        logger.debug("PEAD earnings pre-screen failed for %s", symbol)
        return False


def _pead_prescreen(
    qualifiers: list[str],
    max_workers: int,
    on_progress: Callable[[str, int], None] | None = None,
) -> list[str]:
    """Filter qualifiers to only those with recent earnings reports.

    Runs in parallel to keep the pre-screen fast. This avoids wasting
    sentiment API calls on stocks that the PEAD screener would immediately
    reject for "no recent earnings."
    """
    passed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_pead_has_recent_earnings, sym): sym for sym in qualifiers}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                if future.result():
                    passed.append(sym)
            except Exception:
                pass
            if on_progress:
                on_progress("pead_prescreen_tick", 1)

    logger.info(
        "PEAD earnings pre-screen: %d/%d have recent earnings",
        len(passed),
        len(qualifiers),
    )
    return passed


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
        universe: str = "sp500",
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
        stocks = fetch_universe(universe, cache=self.cache)
        symbols = [s.symbol for s in stocks]
        sector_map = {s.symbol: s.sector for s in stocks}
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

        # ── PEAD earnings pre-screen (before expensive confluence) ────────
        if strategy_name == "pead" and qualifiers:
            if on_progress:
                on_progress("pead_prescreen_start", len(qualifiers))
            qualifiers = _pead_prescreen(
                qualifiers,
                max_workers=max_workers,
                on_progress=on_progress,
            )
            if on_progress:
                on_progress("pead_prescreen_done", len(qualifiers))
            if not qualifiers:
                return MarketScanResult(
                    strategy_name=strategy_name,
                    universe_size=len(symbols),
                    filter_stats=stats_model,
                    qualifiers=[],
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
