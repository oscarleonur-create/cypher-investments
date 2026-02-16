"""Sentiment agent — scores recent news sentiment via research agent infrastructure.

Uses the same TavilyClient → SourceRegistry → ClaudeLLM pipeline as the
research agent, with a sentiment-scoring prompt and structured output.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from research_agent.config import ResearchConfig
from research_agent.evidence import SourceRegistry
from research_agent.llm import ClaudeLLM
from research_agent.search import SearchResult, TavilyClient
from research_agent.store import Store

from advisor.confluence.models import SentimentResult, SourceInfo

logger = logging.getLogger(__name__)


# ── Structured LLM response model ───────────────────────────────────────────


class _SentimentScore(BaseModel):
    """Structured LLM response for sentiment scoring."""

    score: float = Field(description="Overall sentiment score 0-100")
    positive_pct: float = Field(description="Percentage of positive articles 0-100")
    key_headlines: list[str] = Field(default_factory=list, description="Top 3-5 notable headlines")
    reasoning: str = ""


_SENTIMENT_SYSTEM_PROMPT = """\
You are a financial sentiment analyst. Given recent news articles about a stock,
score the overall sentiment.

Each article is prefixed with a source ID (e.g. [s1], [s2]) for citation tracking.

Provide:
- score: 0-100 overall sentiment (50 = neutral, >70 = positive, <30 = negative)
- positive_pct: percentage of articles with positive sentiment (0-100)
- key_headlines: the 3-5 most impactful headlines
- reasoning: brief explanation of the sentiment assessment

Be objective and evidence-based. Weight recent analyst opinions and earnings
reports more heavily than general market commentary."""


# ── Helpers (same pattern as research_agent.agent) ───────────────────────────


def _format_search_results(results: list[SearchResult], registry: SourceRegistry) -> str:
    """Format search results into a text block for LLM consumption.

    Uses the same [sN] citation format as the research agent pipeline.
    """
    parts = []
    for r in results:
        sid = registry.source_id_for_citation(r.url)
        parts.append(f"[{sid}] {r.title}\nURL: {r.url}\n{r.content}\n")
    return "\n---\n".join(parts)


# ── Main check ───────────────────────────────────────────────────────────────


def check_sentiment(symbol: str) -> SentimentResult:
    """Search for recent news and score sentiment using Claude.

    Wires up the research agent pipeline:
      Store → TavilyClient (search + cache) → SourceRegistry (citations) → ClaudeLLM

    Returns SentimentResult with score, positive percentage, headlines, and sources.
    """
    config = ResearchConfig()
    store = Store(config.db_path)
    search = TavilyClient(config, store)
    llm = ClaudeLLM(config)
    registry = SourceRegistry()

    try:
        # Run two searches: general news + analyst outlook
        news_results = search.search(f"{symbol} stock news latest", max_results=5)
        analyst_results = search.search(f"{symbol} stock analyst outlook rating", max_results=3)

        all_results = news_results + analyst_results

        if not all_results:
            return SentimentResult(
                score=50.0,
                positive_pct=50.0,
                key_headlines=[],
                sources=[],
                is_bullish=False,
            )

        # Register sources in the registry (dedup + tier classification)
        for r in all_results:
            registry.add(url=r.url, title=r.title, snippet=r.content)

        # Format with [sN] citation IDs for the LLM
        context = _format_search_results(all_results, registry)

        user_prompt = (
            f"Analyze the sentiment of these recent articles about {symbol}:\n\n" f"{context}"
        )

        result: _SentimentScore = llm.complete(
            system_prompt=_SENTIMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=_SentimentScore,
        )

        # Convert registry sources to SourceInfo for the result
        cited_sources = [
            SourceInfo(
                source_id=registry.get_id(s.url) or f"s{i}",
                url=s.url,
                title=s.title,
                tier=s.tier,
            )
            for i, s in enumerate(registry.all_sources(), 1)
        ]

        return SentimentResult(
            score=result.score,
            positive_pct=result.positive_pct,
            key_headlines=result.key_headlines[:5],
            sources=cited_sources,
            is_bullish=result.positive_pct > 70,
        )

    except Exception as e:
        logger.warning(f"Sentiment check failed for {symbol}: {e}")
        return SentimentResult(
            score=50.0,
            positive_pct=50.0,
            key_headlines=[f"Sentiment analysis unavailable: {e}"],
            sources=[],
            is_bullish=False,
        )
    finally:
        store.close()
