"""Earnings call transcript analysis — last 4 quarters.

Fetches transcript summaries via Tavily, then uses the LLM to extract
per-quarter tone, key topics, and management guidance.  Cross-quarter
tone trend is synthesised in a final LLM call.
"""

from __future__ import annotations

import logging
from datetime import date

from advisor.research.models import TranscriptAnalysis, TranscriptSummary, TranscriptTone

logger = logging.getLogger(__name__)


def build_transcripts(symbol: str, company_name: str = "") -> TranscriptAnalysis:
    """Return TranscriptAnalysis for the last 4 reported quarters."""
    name = company_name or symbol
    summaries, trend = _fetch_and_analyse(symbol, name)
    return TranscriptAnalysis(
        symbol=symbol.upper(),
        summaries=summaries,
        tone_trend=trend,
    )


# ── LLM extraction ───────────────────────────────────────────────────────────


def _fetch_and_analyse(symbol: str, name: str) -> tuple[list[TranscriptSummary], str]:
    try:
        from pydantic import BaseModel
        from research_agent.config import ResearchConfig
        from research_agent.llm import OpenRouterLLM
        from research_agent.search import TavilyClient
        from research_agent.store import Store

        config = ResearchConfig()
        if not config.openrouter_api_key or not config.tavily_api_key:
            return [], ""

        store = Store(config.db_path)
        searcher = TavilyClient(config, store)

        # Anchor the search on the current + previous calendar year so we don't
        # bias toward stale results. Hardcoding year strings (e.g. "2024 2025")
        # caused queries to keep surfacing 2025 transcripts well into 2026.
        today = date.today()
        year_window = f"{today.year - 1} {today.year}"
        query = (
            f"{name} {symbol} latest earnings call transcript {year_window} "
            f"most recent quarter management guidance analyst questions"
        )
        results = searcher.search(query, max_results=8)
        if not results:
            return [], ""

        context = "\n\n".join(f"[{r.title}]\n{r.content[:800]}" for r in results[:6])

        class QuarterOut(BaseModel):
            quarter: str
            earnings_date: str
            tone: str  # bullish | neutral | bearish
            key_topics: list[str]
            management_guidance: str
            analyst_concerns: str
            highlight_quote: str

        class TranscriptOut(BaseModel):
            quarters: list[QuarterOut]
            tone_trend: str

        llm = OpenRouterLLM(config)
        out = llm.complete(
            system_prompt=(
                "You are a buy-side analyst reviewing earnings call transcripts. "
                f"Today is {today.isoformat()}. Extract per-quarter summaries "
                "for the FOUR MOST RECENTLY REPORTED quarters present in the "
                "provided sources (the highest fiscal-year quarter wins, even "
                "if labelled FY2026 or later). Skip stale quarters if newer "
                "ones are present. Each quarter must include an `earnings_date` "
                "in ISO format (YYYY-MM-DD) when the source mentions a date. "
                "tone must be: bullish, neutral, or bearish. key_topics: 3-5 "
                "bullet strings. Base everything on the provided text."
            ),
            user_prompt=f"Company: {name} ({symbol})\n\n{context}",
            response_model=TranscriptOut,
        )

        summaries: list[TranscriptSummary] = []
        for q in out.quarters[:4]:
            tone_str = q.tone.strip().lower()
            try:
                tone = TranscriptTone(tone_str)
            except ValueError:
                tone = TranscriptTone.NEUTRAL
            summaries.append(
                TranscriptSummary(
                    quarter=q.quarter,
                    earnings_date=q.earnings_date,
                    tone=tone,
                    key_topics=q.key_topics[:5],
                    management_guidance=q.management_guidance,
                    analyst_concerns=q.analyst_concerns,
                    highlight_quote=q.highlight_quote,
                )
            )

        return summaries, out.tone_trend

    except Exception as exc:  # noqa: BLE001
        logger.warning("Transcript analysis failed for %s: %s", symbol, exc)
        return [], ""
