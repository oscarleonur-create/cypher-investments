"""Stage 3: Research bridge — connect to research_agent for fundamental analysis."""

from __future__ import annotations

import logging

from advisor.strategy_case.models import ResearchSummary

logger = logging.getLogger(__name__)


def run_research(symbol: str) -> ResearchSummary | None:
    """Run the research agent pipeline for a symbol.

    Returns a condensed ResearchSummary or None on failure.
    This is a cost-gated stage (~$0.10-0.50 per call).
    """
    try:
        from research_agent.config import ResearchConfig
        from research_agent.models import InputMode, ResearchInput
        from research_agent.pipeline import run

        config = ResearchConfig()
        input_data = ResearchInput(mode=InputMode.TICKER, value=symbol.upper())

        card = run(input_data, config)

        return ResearchSummary(
            verdict=card.verdict.value,
            bull_case=card.bull_case[:5],
            bear_case=card.bear_case[:5],
            catalyst_summary=card.catalyst.summary if card.catalyst else "",
            grounding_score=card.grounding_score,
            key_metrics_revenue_growth=card.key_metrics.revenue_growth or "Unknown",
            key_metrics_margins=card.key_metrics.margins or "Unknown",
            key_metrics_fcf=card.key_metrics.fcf or "Unknown",
            key_metrics_guidance=card.key_metrics.guidance_notes or "Unknown",
        )
    except Exception as e:
        logger.error(f"Research pipeline failed for {symbol}: {e}")
        return None


def run_research_batch(
    symbols: list[str], concurrency: int = 3
) -> dict[str, ResearchSummary | None]:
    """Run the research agent pipeline for multiple symbols in parallel.

    Returns a dict mapping each symbol to its ResearchSummary (or None on failure).
    """
    try:
        from research_agent.batch import run_batch
        from research_agent.config import ResearchConfig

        config = ResearchConfig()
        batch_result = run_batch(symbols, config, concurrency=concurrency)

        results: dict[str, ResearchSummary | None] = {}

        # Map completed cards
        for card in batch_result.cards:
            symbol = card.input.value.upper()
            results[symbol] = ResearchSummary(
                verdict=card.verdict.value,
                bull_case=card.bull_case[:5],
                bear_case=card.bear_case[:5],
                catalyst_summary=card.catalyst.summary if card.catalyst else "",
                grounding_score=card.grounding_score,
                key_metrics_revenue_growth=card.key_metrics.revenue_growth or "Unknown",
                key_metrics_margins=card.key_metrics.margins or "Unknown",
                key_metrics_fcf=card.key_metrics.fcf or "Unknown",
                key_metrics_guidance=card.key_metrics.guidance_notes or "Unknown",
            )

        # Mark failures as None
        for symbol in batch_result.failures:
            results[symbol] = None

        return results
    except Exception as e:
        logger.error(f"Batch research pipeline failed: {e}")
        return {s.upper(): None for s in symbols}
