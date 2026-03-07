"""Pipeline orchestrator — SCAN → PRE-SCORE → QUICK SIM → DEEP SIM → OUTPUT."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Callable

import numpy as np

from advisor.simulator.calibration import calibrate
from advisor.simulator.candidates import scan_and_generate
from advisor.simulator.db import SimulatorStore
from advisor.simulator.engine import MonteCarloEngine
from advisor.simulator.models import (
    CalibrationRecord,
    PCSCandidate,
    PipelineResult,
    SimConfig,
    SimResult,
)

logger = logging.getLogger(__name__)


class SimulatorPipeline:
    """Orchestrates the full MC simulation pipeline."""

    def __init__(
        self,
        config: SimConfig | None = None,
        store: SimulatorStore | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.config = config or SimConfig()
        self.store = store or SimulatorStore()
        self.progress = progress_callback or (lambda msg: None)

    def _calibrate_per_symbol(
        self,
        symbols: list[str],
    ) -> tuple[dict[str, MonteCarloEngine], dict[str, dict]]:
        """Calibrate a separate engine per symbol for accurate fat-tail + vol dynamics."""
        engines: dict[str, MonteCarloEngine] = {}
        cal_params: dict[str, dict] = {}
        for sym in symbols:
            self.progress(f"Calibrating {sym}...")
            try:
                cfg = calibrate(sym, self.config)
            except Exception as e:
                logger.warning("Calibration failed for %s, using defaults: %s", sym, e)
                cfg = self.config
            engines[sym] = MonteCarloEngine(cfg)
            cal_params[sym] = {
                "student_t_df": cfg.student_t_df,
                "vol_mean_level": cfg.vol_mean_level,
                "vol_mean_revert_speed": cfg.vol_mean_revert_speed,
                "leverage_effect": cfg.leverage_effect,
            }
        return engines, cal_params

    def _update_engines_with_live_iv(
        self,
        candidates: list[PCSCandidate],
        engines: dict[str, MonteCarloEngine],
        cal_params: dict[str, dict],
    ) -> None:
        """Update engine vol_mean_level using live IV from scanned candidates.

        The calibration step fits vol_mean_level from historical volatility (HV),
        but the market-implied vol (IVx) from the live chain often differs
        significantly. Using HV alone understates vol during high-IV regimes,
        inflating POP estimates.

        Strategy: for each symbol, compute median short_iv across candidates.
        If live IV > calibrated HV, update vol_mean_level to the live IV so the
        OU mean-reversion target reflects current market conditions.
        """
        from collections import defaultdict

        iv_by_symbol: dict[str, list[float]] = defaultdict(list)
        for c in candidates:
            if c.short_iv > 0:
                iv_by_symbol[c.symbol].append(c.short_iv)

        for sym, iv_values in iv_by_symbol.items():
            if sym not in engines:
                continue
            live_iv = float(np.median(iv_values))
            engine = engines[sym]
            hv_vol = engine.config.vol_mean_level

            if live_iv > hv_vol:
                logger.info(
                    "Updating %s vol_mean_level: HV=%.1f%% → live IV=%.1f%%",
                    sym,
                    hv_vol * 100,
                    live_iv * 100,
                )
                updated_config = engine.config.model_copy(update={"vol_mean_level": live_iv})
                engines[sym] = MonteCarloEngine(updated_config)
                cal_params[sym]["vol_mean_level"] = live_iv
                cal_params[sym]["vol_source"] = "live_iv"
            else:
                cal_params[sym]["vol_source"] = "historical"

    def _simulate_batch_per_symbol(
        self,
        candidates: list[PCSCandidate],
        engines: dict[str, MonteCarloEngine],
        n_paths: int,
        label: str,
    ) -> list[tuple[PCSCandidate, SimResult]]:
        """Simulate a batch using per-symbol calibrated engines."""
        results = []
        fallback_engine = MonteCarloEngine(self.config)
        for i, cand in enumerate(candidates):
            self.progress(
                f"{label}: {i + 1}/{len(candidates)} — {cand.symbol} "
                f"${cand.short_strike}/{cand.long_strike}"
            )
            engine = engines.get(cand.symbol, fallback_engine)
            try:
                result = engine.simulate(cand, n_paths=n_paths)
                results.append((cand, result))
            except Exception as e:
                logger.warning(
                    "Sim failed for %s %s/%s: %s",
                    cand.symbol,
                    cand.short_strike,
                    cand.long_strike,
                    e,
                )

        results.sort(key=lambda x: x[1].ev_per_bp, reverse=True)
        return results

    def run(
        self,
        symbols: list[str],
        top_n: int = 5,
        quick_paths: int = 10_000,
        deep_paths: int = 100_000,
        pre_score_limit: int = 200,
        quick_limit: int = 20,
    ) -> PipelineResult:
        """Run full pipeline: SCAN → PRE-SCORE → QUICK SIM → DEEP SIM → OUTPUT."""

        # Step 1: Calibrate per-symbol
        self.progress("Calibrating model parameters...")
        engines, cal_params = self._calibrate_per_symbol(symbols)

        # Step 1.5: Drawdown analysis
        dd_data = {}
        for sym in symbols:
            self.progress(f"Drawdown analysis for {sym}...")
            try:
                from advisor.market.drawdown_analysis import analyze_max_move

                dd_data[sym] = analyze_max_move(sym, dtes=[21, 30, 45, 60])
            except Exception:
                logger.debug("Drawdown analysis failed for %s", sym, exc_info=True)

        # Step 2: SCAN — fetch chains + generate candidates
        # Pass store=None to scan_and_generate to avoid double-saving candidates.
        # We save only the final top results to DB below.
        self.progress(f"Scanning {len(symbols)} symbols...")
        try:
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop and loop.is_running():
                # Already inside an event loop — run in a thread to avoid nesting
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    all_candidates = pool.submit(
                        asyncio.run,
                        scan_and_generate(
                            symbols, self.config, self.store, self.progress, dd_data=dd_data
                        ),
                    ).result()
            else:
                all_candidates = asyncio.run(
                    scan_and_generate(
                        symbols, self.config, self.store, self.progress, dd_data=dd_data
                    )
                )
        except Exception as e:
            logger.error("Scan failed: %s", e)
            all_candidates = []

        if not all_candidates:
            self.progress("No candidates found.")
            return PipelineResult(
                symbols_scanned=len(symbols),
                candidates_generated=0,
                candidates_simulated=0,
                top_results=[],
                calibration_params=cal_params,
                config=self.config,
            )

        n_generated = len(all_candidates)
        self.progress(f"Generated {n_generated} candidates")

        # Step 2.5: Update engine vol using live IV from chain data
        self._update_engines_with_live_iv(all_candidates, engines, cal_params)

        # Step 3: PRE-SCORE — take top N by sell_score
        pre_scored = all_candidates[:pre_score_limit]

        # Step 4: QUICK SIM — 10K paths each (per-symbol engines)
        self.progress(f"Quick sim ({quick_paths:,} paths) on top {len(pre_scored)} candidates...")
        quick_results = self._simulate_batch_per_symbol(
            pre_scored, engines, quick_paths, "Quick sim"
        )

        # Take top N from quick sim
        quick_top = quick_results[:quick_limit]

        # Step 5: DEEP SIM — 100K paths on top candidates
        deep_candidates = [c for c, _ in quick_top]
        self.progress(
            f"Deep sim ({deep_paths:,} paths) on top {len(deep_candidates)} candidates..."
        )
        deep_results = self._simulate_batch_per_symbol(
            deep_candidates, engines, deep_paths, "Deep sim"
        )

        # Step 6: Take final top N
        final_results = deep_results[:top_n]

        # Save only final results to DB (scan_and_generate already saved snapshots)
        for cand, result in final_results:
            cand_ids = self.store.save_candidates_batch([cand])
            if cand_ids:
                result.candidate_id = cand_ids[0]
                self.store.save_sim_result(result, candidate_id=cand_ids[0])
                # Save calibration record for Brier score tracking
                cal_record = CalibrationRecord(
                    candidate_id=cand_ids[0],
                    symbol=result.symbol,
                    predicted_pop=result.pop,
                    predicted_touch=result.touch_prob,
                    predicted_stop=result.stop_prob,
                    predicted_ev=result.ev,
                )
                self.store.save_calibration_record(cal_record)

        top_sim_results = [r for _, r in final_results]

        self.progress(f"Done — {len(top_sim_results)} top results")

        return PipelineResult(
            symbols_scanned=len(symbols),
            candidates_generated=n_generated,
            candidates_simulated=len(quick_results) + len(deep_results),
            top_results=top_sim_results,
            calibration_params=cal_params,
            config=self.config,
        )

    def run_single(
        self,
        candidates: list[PCSCandidate],
        n_paths: int | None = None,
    ) -> list[SimResult]:
        """Simulate a list of pre-built candidates (skip scan phase)."""
        engine = MonteCarloEngine(self.config)
        paths = n_paths or self.config.n_paths
        # Use single engine for all candidates (no per-symbol calibration)
        engines = {c.symbol: engine for c in candidates}
        results = self._simulate_batch_per_symbol(candidates, engines, paths, "Simulating")
        return [r for _, r in results]
