"""Pipeline orchestrator — DISCOVER → VALIDATE → IV TIMING → SIMULATE → SCORE."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from typing import Callable

from advisor.pipeline.models import (
    ConvictionTier,
    IVTimingResult,
    MCEdgeResult,
    PipelineConfig,
    PipelineRunResult,
    SignalBreakdown,
    SignalDiscoveryResult,
    TradeRecommendation,
    ValidationResult,
)
from advisor.pipeline.sizing import compute_position_size

logger = logging.getLogger(__name__)

# Score string → numeric for dip/pead normalization
_DIP_SCORE_MAP = {
    "FAIL": 0,
    "WEAK": 20,
    "WATCH": 40,
    "LEAN_BUY": 60,
    "BUY": 80,
    "STRONG_BUY": 100,
}

_PEAD_SCORE_MAP = {
    "FAIL": 0,
    "WATCH": 30,
    "LEAN_BUY": 55,
    "BUY": 80,
    "STRONG_BUY": 100,
}

# Confluence verdict → numeric for scoring
_VERDICT_SCORE_MAP = {
    "ENTER": 100,
    "CAUTION": 50,
    "PASS": 0,
}


class PipelineOrchestrator:
    """Integrated daily workflow: discover, validate, time IV, simulate, score."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.config = config or PipelineConfig()
        self.progress = progress_callback or (lambda msg: None)

    def run(
        self,
        symbols: list[str],
        top_n: int = 5,
    ) -> PipelineRunResult:
        """Run the full pipeline and return ranked trade recommendations."""
        start = time.time()
        errors: list[str] = []
        symbols = [s.upper() for s in symbols]

        # Layer 1: Signal Discovery
        self.progress("Layer 1: Discovering signals...")
        discovered = self._discover_signals(symbols, errors)
        self.progress(f"  Discovered {len(discovered)} symbols with signals")

        if not discovered:
            return PipelineRunResult(
                config=self.config,
                symbols_scanned=len(symbols),
                errors=errors,
                elapsed_seconds=round(time.time() - start, 2),
            )

        # Layer 2: Validation
        self.progress("Layer 2: Validating candidates...")
        disc_symbols = [d.symbol for d in discovered]
        validated = self._validate_candidates(disc_symbols, errors)
        self.progress(f"  Validated {len(validated)} symbols")

        # Layer 3: IV Timing
        self.progress("Layer 3: Checking IV timing...")
        val_symbols = list(validated.keys())
        iv_results = self._check_iv_timing(val_symbols, errors)
        # No hard gate — conviction scoring handles filtering via min_conviction.
        # PASS verdict → 0/10 pts, low IV → low iv_timing_score → fewer pts.
        iv_passed = iv_results
        self.progress(f"  {len(iv_passed)} symbols with IV data")

        if not iv_passed:
            return PipelineRunResult(
                config=self.config,
                symbols_scanned=len(symbols),
                symbols_discovered=len(discovered),
                symbols_validated=len(validated),
                errors=errors,
                elapsed_seconds=round(time.time() - start, 2),
            )

        # Layer 4: MC Simulation
        self.progress("Layer 4: Running MC simulation...")
        sim_symbols = list(iv_passed.keys())
        mc_results = self._run_simulation(sim_symbols, top_n, errors)
        self.progress(f"  Simulated {len(mc_results)} candidates")

        if not mc_results:
            return PipelineRunResult(
                config=self.config,
                symbols_scanned=len(symbols),
                symbols_discovered=len(discovered),
                symbols_validated=len(validated),
                errors=errors,
                elapsed_seconds=round(time.time() - start, 2),
            )

        # Layer 5: Conviction Score + Sizing
        self.progress("Layer 5: Scoring and sizing...")
        disc_map = {d.symbol: d for d in discovered}
        recommendations = self._score_and_size(mc_results, disc_map, validated, iv_passed)
        self.progress(f"  {len(recommendations)} recommendations above min conviction")

        # Rank by conviction descending, take top_n
        recommendations.sort(key=lambda r: r.conviction_score, reverse=True)
        recommendations = recommendations[:top_n]

        return PipelineRunResult(
            config=self.config,
            symbols_scanned=len(symbols),
            symbols_discovered=len(discovered),
            symbols_validated=len(validated),
            symbols_simulated=len(mc_results),
            recommendations=recommendations,
            errors=errors,
            elapsed_seconds=round(time.time() - start, 2),
        )

    # ── Layer 1: Signal Discovery ────────────────────────────────────────

    def _discover_signals(
        self, symbols: list[str], errors: list[str]
    ) -> list[SignalDiscoveryResult]:
        """Run all four screeners per symbol, keep those with signal > 0."""

        def _scan_one(symbol: str) -> SignalDiscoveryResult | None:
            dip_score = None
            pead_score = None
            mispricing_score = None
            smart_money_score = None
            scores: list[float] = []

            # Dip screener
            try:
                from advisor.confluence.dip_screener import check_dip_fundamental

                result = check_dip_fundamental(symbol)
                dip_score = result.dip_screener.overall_score if result.dip_screener else None
                if dip_score:
                    scores.append(float(_DIP_SCORE_MAP.get(dip_score, 0)))
            except Exception as e:
                logger.debug("Dip screener failed for %s: %s", symbol, e)

            # PEAD screener
            try:
                from advisor.confluence.pead_screener import check_pead_fundamental

                result = check_pead_fundamental(symbol)
                pead_score = result.pead_screener.overall_score if result.pead_screener else None
                if pead_score:
                    scores.append(float(_PEAD_SCORE_MAP.get(pead_score, 0)))
            except Exception as e:
                logger.debug("PEAD screener failed for %s: %s", symbol, e)

            # Mispricing screener
            try:
                from advisor.confluence.mispricing_screener import screen_mispricing

                result = screen_mispricing(symbol)
                mispricing_score = result.total_score
                scores.append(max(0.0, min(100.0, mispricing_score)))
            except Exception as e:
                logger.debug("Mispricing screener failed for %s: %s", symbol, e)

            # Smart money screener
            try:
                from advisor.confluence.smart_money_screener import screen_smart_money

                result = screen_smart_money(symbol)
                raw = result.total_score
                # Normalize [-35, +100] → [0, 100]
                smart_money_score = max(0.0, min(100.0, (raw + 35) / 135 * 100))
                scores.append(smart_money_score)
            except Exception as e:
                logger.debug("Smart money screener failed for %s: %s", symbol, e)

            best = max(scores) if scores else 0.0
            if best <= 0:
                return None

            return SignalDiscoveryResult(
                symbol=symbol,
                dip_score=dip_score,
                pead_score=pead_score,
                mispricing_score=round(mispricing_score, 2)
                if mispricing_score is not None
                else None,
                smart_money_score=round(smart_money_score, 2)
                if smart_money_score is not None
                else None,
                best_signal_score=round(best, 2),
            )

        results: list[SignalDiscoveryResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_scan_one, sym): sym for sym in symbols}
            for future in concurrent.futures.as_completed(futures):
                sym = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    errors.append(f"Signal discovery failed for {sym}: {e}")
                    logger.warning("Signal discovery failed for %s: %s", sym, e)

        results.sort(key=lambda r: r.best_signal_score, reverse=True)
        return results[:50]

    # ── Layer 2: Validation ──────────────────────────────────────────────

    def _validate_candidates(
        self, symbols: list[str], errors: list[str]
    ) -> dict[str, ValidationResult]:
        """Run confluence + alpha scorer per symbol."""

        def _validate_one(symbol: str) -> ValidationResult:
            verdict = "PASS"
            alpha = 0.0
            fundamental_safe = False

            try:
                from advisor.confluence.orchestrator import run_confluence

                conf = run_confluence(symbol, strategy_name="buy_the_dip", force_all=True)
                verdict = conf.verdict.value
                if conf.fundamental:
                    fundamental_safe = conf.fundamental.is_clear
            except Exception as e:
                logger.warning("Confluence failed for %s: %s", symbol, e)

            try:
                from advisor.confluence.alpha_scorer import compute_alpha

                alpha_result = compute_alpha(symbol, skip_layers={"sentiment"})
                alpha = alpha_result.alpha_score
            except Exception as e:
                logger.warning("Alpha scorer failed for %s: %s", symbol, e)

            return ValidationResult(
                symbol=symbol,
                confluence_verdict=verdict,
                alpha_score=round(alpha, 2),
                fundamental_safe=fundamental_safe,
            )

        validated: dict[str, ValidationResult] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_validate_one, sym): sym for sym in symbols}
            for future in concurrent.futures.as_completed(futures):
                sym = futures[future]
                try:
                    validated[sym] = future.result()
                except Exception as e:
                    errors.append(f"Validation failed for {sym}: {e}")
                    logger.warning("Validation failed for %s: %s", sym, e)

        return validated

    # ── Layer 3: IV Timing ───────────────────────────────────────────────

    def _check_iv_timing(self, symbols: list[str], errors: list[str]) -> dict[str, IVTimingResult]:
        """Compute IV percentile, term structure, and vol dynamics per symbol."""

        def _check_one(symbol: str) -> IVTimingResult:
            iv_rank = None
            iv_percentile = 0.0
            current_iv = 0.0
            term_structure = "flat"
            vol_mean_level = 0.0
            vol_mean_revert_speed = 0.0

            # IV percentile
            try:
                from advisor.market.iv_analysis import compute_iv_percentile

                iv_result = compute_iv_percentile(symbol)
                iv_rank = iv_result.iv_rank
                iv_percentile = iv_result.iv_percentile
                current_iv = iv_result.current_iv
            except Exception as e:
                logger.warning("IV percentile failed for %s: %s", symbol, e)

            # Term structure
            try:
                from advisor.market.iv_analysis import classify_term_structure

                ts_result = classify_term_structure(symbol)
                term_structure = ts_result.classification
            except Exception as e:
                logger.warning("Term structure failed for %s: %s", symbol, e)

            # Vol dynamics
            try:
                from advisor.simulator.calibration import (
                    _get_daily_returns,
                    estimate_vol_dynamics,
                )

                returns = _get_daily_returns(symbol)
                vol_dyn = estimate_vol_dynamics(returns)
                vol_mean_level = vol_dyn["vol_mean_level"]
                vol_mean_revert_speed = vol_dyn["vol_mean_revert_speed"]
            except Exception as e:
                logger.warning("Vol dynamics failed for %s: %s", symbol, e)

            # Vol direction logic
            if current_iv > 0 and vol_mean_level > 0:
                if current_iv > vol_mean_level * 1.1 and vol_mean_revert_speed > 1.0:
                    vol_direction = "mean_reverting_down"
                elif current_iv > vol_mean_level * 1.2:
                    vol_direction = "elevated"
                else:
                    vol_direction = "neutral"
            else:
                vol_direction = "neutral"

            # Composite iv_timing_score
            # IV rank component: 0-40 pts (linear scale)
            rank_val = iv_rank if iv_rank is not None else iv_percentile
            iv_rank_pts = min(40.0, rank_val / 100 * 40) if rank_val else 0.0

            # Term structure component: 0-30 pts
            ts_pts = {"contango": 30.0, "flat": 15.0, "backwardation": 0.0}.get(
                term_structure, 15.0
            )

            # Vol direction component: 0-30 pts
            vol_dir_pts = {
                "mean_reverting_down": 30.0,
                "elevated": 15.0,
                "neutral": 10.0,
            }.get(vol_direction, 10.0)

            iv_timing_score = min(100.0, iv_rank_pts + ts_pts + vol_dir_pts)

            # Fallback: if IV data completely failed, default to neutral (50)
            # so we don't penalize symbols just because the data source is down.
            iv_data_available = (iv_rank is not None) or iv_percentile > 0 or current_iv > 0.05
            if not iv_data_available:
                iv_timing_score = 50.0

            return IVTimingResult(
                symbol=symbol,
                iv_rank=iv_rank,
                iv_percentile=round(iv_percentile, 2),
                current_iv=round(current_iv, 4),
                term_structure=term_structure,
                vol_mean_level=round(vol_mean_level, 4),
                vol_mean_revert_speed=round(vol_mean_revert_speed, 4),
                vol_direction=vol_direction,
                iv_timing_score=round(iv_timing_score, 2),
            )

        results: dict[str, IVTimingResult] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_check_one, sym): sym for sym in symbols}
            for future in concurrent.futures.as_completed(futures):
                sym = futures[future]
                try:
                    results[sym] = future.result()
                except Exception as e:
                    errors.append(f"IV timing failed for {sym}: {e}")
                    logger.warning("IV timing failed for %s: %s", sym, e)

        return results

    # ── Layer 4: MC Simulation ───────────────────────────────────────────

    def _run_simulation(
        self,
        symbols: list[str],
        top_n: int,
        errors: list[str],
    ) -> dict[str, MCEdgeResult]:
        """Run MC simulation via SimulatorPipeline internals, preserving candidate pairing."""
        from advisor.simulator.candidates import scan_and_generate
        from advisor.simulator.db import SimulatorStore
        from advisor.simulator.models import SimConfig
        from advisor.simulator.pipeline import SimulatorPipeline

        config = SimConfig()
        store = SimulatorStore()
        pipeline = SimulatorPipeline(config=config, store=store, progress_callback=self.progress)

        try:
            # Step 1: Calibrate per-symbol engines
            engines, cal_params = pipeline._calibrate_per_symbol(symbols)

            # Step 2: Scan and generate candidates
            try:
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass

                if loop and loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        candidates = pool.submit(
                            asyncio.run,
                            scan_and_generate(symbols, config, store, self.progress),
                        ).result()
                else:
                    candidates = asyncio.run(
                        scan_and_generate(symbols, config, store, self.progress)
                    )
            except Exception as e:
                errors.append(f"Candidate scan failed: {e}")
                logger.error("Candidate scan failed: %s", e)
                return {}

            if not candidates:
                return {}

            # Step 3: Update engines with live IV
            pipeline._update_engines_with_live_iv(candidates, engines, cal_params)

            # Step 4: Quick sim on top candidates (by sell_score)
            pre_scored = candidates[:200]
            self.progress(
                f"Quick sim ({self.config.quick_paths:,} paths) on {len(pre_scored)} candidates..."
            )
            quick_results = pipeline._simulate_batch_per_symbol(
                pre_scored, engines, self.config.quick_paths, "Quick sim"
            )

            # Take top for deep sim
            quick_top = quick_results[:20]

            # Step 5: Deep sim on top candidates
            deep_candidates = [c for c, _ in quick_top]
            self.progress(
                f"Deep sim ({self.config.deep_paths:,} paths)"
                f" on {len(deep_candidates)} candidates..."
            )
            deep_results = pipeline._simulate_batch_per_symbol(
                deep_candidates, engines, self.config.deep_paths, "Deep sim"
            )

        except Exception as e:
            errors.append(f"MC simulation failed: {e}")
            logger.error("MC simulation failed: %s", e)
            return {}

        # Post-process: compute POP edge, keep best per symbol
        best_per_symbol: dict[str, MCEdgeResult] = {}
        for cand, result in deep_results:
            pop_edge = result.pop - cand.pop_estimate
            max_loss = (cand.width - cand.net_credit) * 100

            # MC edge score: POP edge (0-50) + EV/BP (0-50)
            # POP edge: +5% edge = 25pts, +10% = 50pts (capped)
            pop_edge_pts = min(50.0, max(0.0, pop_edge * 100 * 5))
            # EV/BP: 0.01 per dollar = 25pts, 0.02 = 50pts
            ev_bp_pts = min(50.0, max(0.0, result.ev_per_bp * 2500))
            mc_edge_score = min(100.0, pop_edge_pts + ev_bp_pts)

            mc = MCEdgeResult(
                symbol=cand.symbol,
                short_strike=cand.short_strike,
                long_strike=cand.long_strike,
                expiration=cand.expiration,
                dte=cand.dte,
                credit=cand.net_credit,
                max_loss=round(max_loss, 2),
                bp=cand.buying_power,
                mc_pop=round(result.pop, 4),
                market_pop=round(cand.pop_estimate, 4),
                pop_edge=round(pop_edge, 4),
                ev=round(result.ev, 2),
                ev_per_bp=round(result.ev_per_bp, 6),
                cvar_95=round(result.cvar_95, 2),
                stop_prob=round(result.stop_prob, 4),
                mc_edge_score=round(mc_edge_score, 2),
            )

            # Keep best per symbol by ev_per_bp
            existing = best_per_symbol.get(cand.symbol)
            if existing is None or mc.ev_per_bp > existing.ev_per_bp:
                best_per_symbol[cand.symbol] = mc

        return best_per_symbol

    # ── Layer 5: Conviction Score + Sizing ───────────────────────────────

    def _score_and_size(
        self,
        mc_results: dict[str, MCEdgeResult],
        disc_map: dict[str, SignalDiscoveryResult],
        validated: dict[str, ValidationResult],
        iv_passed: dict[str, IVTimingResult],
    ) -> list[TradeRecommendation]:
        """Compute conviction score, size positions, build recommendations."""
        recommendations: list[TradeRecommendation] = []

        for symbol, mc in mc_results.items():
            disc = disc_map.get(symbol)
            val = validated.get(symbol, ValidationResult(symbol=symbol))
            iv = iv_passed.get(symbol, IVTimingResult(symbol=symbol))

            # Signal strength: 0-20 pts
            signal_pts = (disc.best_signal_score / 100 * self.config.w_signal) if disc else 0.0

            # Fundamental safety: 0-20 pts (confluence 0-10 + alpha 0-10)
            verdict_score = _VERDICT_SCORE_MAP.get(val.confluence_verdict, 0)
            fund_pts = verdict_score / 100 * (
                self.config.w_fundamental / 2
            ) + val.alpha_score / 100 * (self.config.w_fundamental / 2)

            # IV environment: 0-20 pts
            iv_pts = iv.iv_timing_score / 100 * self.config.w_iv

            # MC edge: 0-25 pts
            mc_pts = mc.mc_edge_score / 100 * self.config.w_mc_edge

            # Sizing
            max_loss_per_contract = mc.max_loss
            bp_per_contract = mc.bp
            sizing = compute_position_size(max_loss_per_contract, bp_per_contract, self.config)

            # Sizing: 0-15 pts
            sizing_pts = sizing.sizing_score / 100 * self.config.w_sizing

            # Total conviction
            conviction = max(0.0, min(100.0, signal_pts + fund_pts + iv_pts + mc_pts + sizing_pts))
            conviction = round(conviction, 2)

            # Tier
            if conviction >= 75:
                tier = ConvictionTier.AUTO_ALERT
            elif conviction >= 50:
                tier = ConvictionTier.WATCH
            else:
                tier = ConvictionTier.SKIP

            # Filter by min conviction
            if conviction < self.config.min_conviction:
                continue

            # Build reasoning
            parts = []
            if disc:
                active_signals = []
                if disc.dip_score and disc.dip_score not in ("FAIL", "WEAK"):
                    active_signals.append(f"Dip:{disc.dip_score}")
                if disc.pead_score and disc.pead_score not in ("FAIL",):
                    active_signals.append(f"PEAD:{disc.pead_score}")
                if disc.mispricing_score and disc.mispricing_score > 40:
                    active_signals.append(f"Mispricing:{disc.mispricing_score:.0f}")
                if disc.smart_money_score and disc.smart_money_score > 40:
                    active_signals.append(f"SmartMoney:{disc.smart_money_score:.0f}")
                if active_signals:
                    parts.append(f"Signals: {', '.join(active_signals)}")

            parts.append(f"Confluence: {val.confluence_verdict}")

            if iv.iv_rank is not None:
                parts.append(f"IVR:{iv.iv_rank:.0f}")
            parts.append(f"IV:{iv.term_structure}")

            parts.append(f"POP:{mc.mc_pop:.1%} (edge {mc.pop_edge:+.1%})")
            parts.append(f"EV:${mc.ev:.2f}")

            if sizing.sizing_feasible:
                parts.append(f"Qty:{sizing.suggested_contracts} @ {sizing.risk_pct:.1f}% risk")

            reasoning = " | ".join(parts)

            recommendations.append(
                TradeRecommendation(
                    symbol=symbol,
                    short_strike=mc.short_strike,
                    long_strike=mc.long_strike,
                    expiration=mc.expiration,
                    dte=mc.dte,
                    credit=mc.credit,
                    max_loss=mc.max_loss,
                    bp=mc.bp,
                    conviction_score=conviction,
                    conviction_tier=tier,
                    signal_breakdown=SignalBreakdown(
                        signal_strength=round(signal_pts, 2),
                        fundamental_safety=round(fund_pts, 2),
                        iv_environment=round(iv_pts, 2),
                        mc_edge=round(mc_pts, 2),
                        sizing_feasibility=round(sizing_pts, 2),
                    ),
                    mc_pop=mc.mc_pop,
                    pop_edge=mc.pop_edge,
                    ev=mc.ev,
                    ev_per_bp=mc.ev_per_bp,
                    cvar_95=mc.cvar_95,
                    stop_prob=mc.stop_prob,
                    suggested_contracts=sizing.suggested_contracts,
                    risk_pct=sizing.risk_pct,
                    iv_rank=iv.iv_rank,
                    iv_percentile=iv.iv_percentile,
                    current_iv=iv.current_iv,
                    iv_timing_score=iv.iv_timing_score,
                    reasoning=reasoning,
                )
            )

        return recommendations
