"""Pipeline orchestrator — chains stages 1-6 into a complete strategy case."""

from __future__ import annotations

import logging
import time

from advisor.strategy_case.models import StrategyCase, StrategyCaseConfig

logger = logging.getLogger(__name__)


class StrategyCaseBuilder:
    """Orchestrates the 6-stage strategy case pipeline."""

    def __init__(
        self,
        config: StrategyCaseConfig | None = None,
        progress_callback: callable | None = None,
    ):
        self.config = config or StrategyCaseConfig()
        self.progress = progress_callback or (lambda msg: None)

    def build(self, symbol: str) -> StrategyCase:
        """Run the full pipeline for a single symbol."""
        symbol = symbol.upper()
        start = time.time()

        case = StrategyCase(symbol=symbol, config=self.config)

        # ── Stage 1: Scenario Detection ──────────────────────────────────
        self.progress("Stage 1: Detecting scenario...")
        try:
            from advisor.strategy_case.scenarios import detect_scenario

            case.scenario = detect_scenario(symbol)
            self.progress(
                f"  → {case.scenario.scenario_type.value} "
                f"({case.scenario.confidence:.0%} confidence)"
            )
        except Exception as e:
            case.errors.append(f"Scenario detection failed: {e}")
            logger.error(f"Stage 1 failed for {symbol}: {e}")
            case.elapsed_seconds = time.time() - start
            return case

        # ── Stage 2: Strategy Mapping ────────────────────────────────────
        self.progress("Stage 2: Mapping strategies...")
        try:
            from advisor.strategy_case.strategy_mapper import rank_strategies

            case.ranking = rank_strategies(case.scenario, self.config)
            if case.ranking.selected:
                self.progress(
                    f"  → {case.ranking.selected.strategy.value} "
                    f"(fit {case.ranking.selected.fit_score:.0f}/100)"
                )
        except Exception as e:
            case.errors.append(f"Strategy mapping failed: {e}")
            logger.error(f"Stage 2 failed for {symbol}: {e}")
            case.elapsed_seconds = time.time() - start
            return case

        # ── Stage 3: Fundamental Research (optional) ─────────────────────
        if self.config.enable_research:
            self.progress("Stage 3: Running fundamental research...")
            try:
                from advisor.strategy_case.research_bridge import run_research

                case.research = run_research(symbol)
                if case.research:
                    self.progress(f"  → Research verdict: {case.research.verdict}")
                else:
                    case.errors.append("Research returned no results")
            except Exception as e:
                case.errors.append(f"Research failed: {e}")
                logger.error(f"Stage 3 failed for {symbol}: {e}")
        else:
            self.progress("Stage 3: Research skipped (use --research to enable)")

        # ── Stage 4: Options Analysis ────────────────────────────────────
        self.progress("Stage 4: Analyzing options chains...")
        if case.ranking.selected:
            try:
                from advisor.strategy_case.options_analysis import analyze_options

                case.options = analyze_options(symbol, case.ranking.selected, self.config)
                n = len(case.options.recommendations)
                self.progress(f"  → Found {n} strike recommendation(s)")
            except Exception as e:
                case.errors.append(f"Options analysis failed: {e}")
                logger.error(f"Stage 4 failed for {symbol}: {e}")
        else:
            case.errors.append("No strategy selected — skipping options analysis")

        # ── Stage 5: Risk Assessment ─────────────────────────────────────
        if case.options and case.options.recommendations:
            top_rec = case.options.recommendations[0]
            if self.config.enable_mc:
                self.progress("Stage 5: Running MC simulation...")
                try:
                    from advisor.strategy_case.risk_assessment import assess_risk_mc

                    case.risk = assess_risk_mc(symbol, top_rec, self.config)
                    self.progress(
                        f"  → {case.risk.source.upper()} POP: {case.risk.pop:.1%}, "
                        f"EV: ${case.risk.ev:.2f}"
                    )
                except Exception as e:
                    case.errors.append(f"MC simulation failed: {e}")
                    logger.error(f"Stage 5 MC failed for {symbol}: {e}")
            else:
                self.progress("Stage 5: BSM risk estimate (use --mc for full simulation)")
                try:
                    from advisor.strategy_case.risk_assessment import assess_risk_bsm

                    case.risk = assess_risk_bsm(top_rec, self.config)
                    self.progress(f"  → BSM POP: {case.risk.pop:.1%}, " f"EV: ${case.risk.ev:.2f}")
                except Exception as e:
                    case.errors.append(f"BSM risk assessment failed: {e}")
                    logger.error(f"Stage 5 BSM failed for {symbol}: {e}")
        else:
            self.progress("Stage 5: Skipped (no strike recommendations)")

        # ── Stage 6: Synthesis ───────────────────────────────────────────
        self.progress("Stage 6: Synthesizing trade case...")
        if case.scenario and case.ranking and case.options:
            try:
                from advisor.strategy_case.synthesis import synthesize_case

                case.synthesis = synthesize_case(
                    symbol=symbol,
                    scenario=case.scenario,
                    ranking=case.ranking,
                    options=case.options,
                    risk=case.risk,
                    research=case.research,
                )
                self.progress(
                    f"  → Verdict: {case.synthesis.verdict.value} "
                    f"({case.synthesis.conviction_score:.0f}/100)"
                )
            except Exception as e:
                case.errors.append(f"Synthesis failed: {e}")
                logger.error(f"Stage 6 failed for {symbol}: {e}")
        else:
            self.progress("Stage 6: Skipped (missing required stage outputs)")

        case.elapsed_seconds = round(time.time() - start, 1)
        self.progress(f"Done in {case.elapsed_seconds:.1f}s")
        return case
