"""Anthropic Claude client with structured output and prompt templates."""

from __future__ import annotations

import json
import logging
from typing import TypeVar

import anthropic
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from research_agent.config import ResearchConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ClaudeLLM:
    """Anthropic Claude client with structured output support."""

    def __init__(self, config: ResearchConfig) -> None:
        self._config = config
        self._client = anthropic.Anthropic(
            api_key=config.anthropic_api_key,
            timeout=config.llm_timeout_seconds,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T] | None = None,
    ) -> str | T:
        """Call Claude and optionally parse response into a Pydantic model.

        If response_model is provided, the prompt includes the JSON schema and
        the response is parsed into that model.
        """
        if response_model is not None:
            schema = response_model.model_json_schema()
            system_prompt += (
                "\n\nYou MUST respond with valid JSON matching this schema:\n"
                f"```json\n{json.dumps(schema, indent=2)}\n```\n"
                "Return ONLY the JSON object, no other text."
            )

        message = self._client.messages.create(
            model=self._config.llm_model,
            max_tokens=self._config.llm_max_tokens,
            temperature=self._config.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = message.content[0].text

        if response_model is not None:
            # Strip markdown code fences if present
            cleaned = text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                # Remove first and last lines (```json and ```)
                lines = [ln for ln in lines[1:] if not ln.strip() == "```"]
                cleaned = "\n".join(lines)
            return response_model.model_validate_json(cleaned)

        return text


# ── Prompt templates ─────────────────────────────────────────────────────────

TRIGGER_DETECTION_PROMPT = """\
You are a financial analyst. Given search results about a stock ticker, identify the \
recent catalyst or trigger that caused a price dip.

Analyze the search results and determine:
1. Whether a clear trigger/catalyst was found
2. The type of trigger (earnings miss, guidance cut, sector rotation, macro event, etc.)
3. A concise summary of what happened
4. Relevant source URLs

Be factual and evidence-based. Only cite information present in the search results."""

DIP_CLASSIFICATION_PROMPT = """\
You are a financial analyst. Given evidence about a stock's recent price decline, \
classify the dip as one of:

- TEMPORARY: The decline is likely short-lived. Fundamentals are intact, the catalyst \
is a one-time event, or the market is overreacting.
- STRUCTURAL: The decline reflects a fundamental deterioration in the business. \
Competitive position weakened, secular headwinds, or broken growth story.
- UNCLEAR: Insufficient evidence to classify confidently. More data needed.

Provide your classification, a confidence score (0.0-1.0), and detailed reasoning. \
Base your analysis ONLY on the evidence provided."""

FACT_EXTRACTION_PROMPT = """\
You are a financial research analyst. Extract structured factual claims from the \
search results provided. For each claim, cite the source(s) that support it.

Focus on extracting facts in these categories:
- earnings_highlights: Key earnings data points (revenue, EPS, growth rates)
- guidance_changes: Forward guidance updates, revisions, or reaffirmations
- competitive_landscape: Market position, competitive threats, industry dynamics
- unit_economics: Per-unit metrics, margins, efficiency indicators
- balance_sheet: Cash, debt, liquidity, capital allocation
- valuation_comparison: Multiples, relative valuation, historical context
- bear_rebuttals: Counterarguments to bearish theses, positive catalysts

Each item must have a factual text and list of source IDs that support it. \
Only extract claims that are directly supported by the provided sources."""

CARD_SYNTHESIS_PROMPT = """\
You are a senior financial analyst. Synthesize the research evidence into an \
Opportunity Card for a "buy-the-dip" analysis.

Based on ALL the evidence provided, generate:
1. verdict: BUY_THE_DIP (strong case for buying), WATCH (wait for more data), \
or AVOID (fundamental problems)
2. bull_case: Top 3 reasons to buy (with source citations like [s1], [s2])
3. bear_case: Top 3 risks or reasons against (with source citations)
4. key_metrics: Revenue growth, margins, FCF, cash, debt, guidance notes
5. risks: Key risk factors
6. invalidation: Conditions that would invalidate the thesis
7. validation_checklist: Things to monitor over next 1-4 weeks
8. next_actions: Concrete next steps for the investor

Be balanced and evidence-based. Every claim should be traceable to a source."""
