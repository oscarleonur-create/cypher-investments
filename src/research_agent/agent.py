"""Iterative agent loop that drives the 4-step research pipeline."""

from __future__ import annotations

import logging
from datetime import date

from pydantic import BaseModel, Field

from research_agent.config import ResearchConfig
from research_agent.evidence import SourceRegistry
from research_agent.llm import (
    CARD_SYNTHESIS_PROMPT,
    DIP_CLASSIFICATION_PROMPT,
    FACT_EXTRACTION_PROMPT,
    TRIGGER_DETECTION_PROMPT,
    ClaudeLLM,
)
from research_agent.models import (
    AgentState,
    ClassificationResult,
    DipType,
    EvidenceItem,
    FactPack,
    OpportunityCard,
    TriggerResult,
    Verdict,
)
from research_agent.search import SearchResult, TavilyClient

logger = logging.getLogger(__name__)


# ── Structured response models for LLM output ───────────────────────────────


class _TriggerResponse(BaseModel):
    found: bool = False
    trigger_type: str = ""
    summary: str = ""
    source_urls: list[str] = Field(default_factory=list)


class _ClassificationResponse(BaseModel):
    dip_type: str = "UNCLEAR"
    confidence: float = 0.0
    reasoning: str = ""


class _EvidenceItemRaw(BaseModel):
    text: str = ""
    source_urls: list[str] = Field(default_factory=list)


class _FactExtractionResponse(BaseModel):
    earnings_highlights: list[_EvidenceItemRaw] = Field(default_factory=list)
    guidance_changes: list[_EvidenceItemRaw] = Field(default_factory=list)
    competitive_landscape: list[_EvidenceItemRaw] = Field(default_factory=list)
    unit_economics: list[_EvidenceItemRaw] = Field(default_factory=list)
    balance_sheet: list[_EvidenceItemRaw] = Field(default_factory=list)
    valuation_comparison: list[_EvidenceItemRaw] = Field(default_factory=list)
    bear_rebuttals: list[_EvidenceItemRaw] = Field(default_factory=list)


class _CardSynthesisResponse(BaseModel):
    verdict: str = "WATCH"
    bull_case: list[str] = Field(default_factory=list)
    bear_case: list[str] = Field(default_factory=list)
    key_metrics: dict = Field(default_factory=dict)
    risks: list[str] = Field(default_factory=list)
    invalidation: list[str] = Field(default_factory=list)
    validation_checklist: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


# ── Helper: format search results for LLM ────────────────────────────────────


def _format_search_results(results: list[SearchResult], registry: SourceRegistry) -> str:
    """Format search results into a text block for LLM consumption."""
    parts = []
    for r in results:
        sid = registry.source_id_for_citation(r.url)
        parts.append(f"[{sid}] {r.title}\nURL: {r.url}\n{r.content}\n")
    return "\n---\n".join(parts)


def _format_evidence_for_llm(state: AgentState, registry: SourceRegistry) -> str:
    """Format accumulated evidence for LLM synthesis."""
    parts = []
    if state.trigger:
        parts.append(f"## Trigger\n{state.trigger.summary}")
    if state.classification:
        parts.append(
            f"## Dip Classification\n"
            f"Type: {state.classification.dip_type}\n"
            f"Confidence: {state.classification.confidence}\n"
            f"Reasoning: {state.classification.reasoning}"
        )
    fp = state.fact_pack
    for field_name in FactPack.model_fields:
        items: list[EvidenceItem] = getattr(fp, field_name)
        if items:
            label = field_name.replace("_", " ").title()
            lines = [f"- {item.text} [{', '.join(item.source_ids)}]" for item in items]
            parts.append(f"## {label}\n" + "\n".join(lines))

    # Source index
    sources = registry.all_sources()
    if sources:
        source_lines = []
        for i, s in enumerate(sources, 1):
            source_lines.append(f"[s{i}] {s.title} ({s.url}) - Tier {s.tier}")
        parts.append("## Sources\n" + "\n".join(source_lines))

    return "\n\n".join(parts)


# ── Pipeline steps ────────────────────────────────────────────────────────────


def step1_detect_trigger(
    state: AgentState,
    search: TavilyClient,
    llm: ClaudeLLM,
    registry: SourceRegistry,
    config: ResearchConfig,
) -> None:
    """Search for the trigger/catalyst behind the price dip."""
    ticker = state.input.value.upper()
    queries = [
        f"{ticker} stock price drop reason {date.today().year}",
        f"{ticker} earnings catalyst decline recent",
    ]

    all_results: list[SearchResult] = []
    for q in queries[: config.max_queries_per_iteration]:
        if q not in state.queries_executed:
            results = search.search(q, max_results=config.max_urls_per_query)
            all_results.extend(results)
            state.queries_executed.append(q)

    if not all_results:
        state.trigger = TriggerResult(found=False, summary="No search results found")
        return

    # Register sources
    for r in all_results:
        registry.add(url=r.url, title=r.title, snippet=r.content)

    context = _format_search_results(all_results, registry)

    try:
        resp: _TriggerResponse = llm.complete(
            system_prompt=TRIGGER_DETECTION_PROMPT,
            user_prompt=f"Ticker: {ticker}\n\nSearch Results:\n{context}",
            response_model=_TriggerResponse,
        )
        state.trigger = TriggerResult(
            found=resp.found,
            trigger_type=resp.trigger_type,
            summary=resp.summary,
            links=resp.source_urls,
        )
    except Exception as e:
        logger.error("Trigger detection failed: %s", e)
        state.errors.append(f"trigger_detection: {e}")
        state.trigger = TriggerResult(found=False, summary=f"Analysis failed: {e}")


def step2_classify_dip(
    state: AgentState,
    llm: ClaudeLLM,
    registry: SourceRegistry,
) -> None:
    """Classify the dip as TEMPORARY, STRUCTURAL, or UNCLEAR."""
    evidence = _format_evidence_for_llm(state, registry)

    try:
        resp: _ClassificationResponse = llm.complete(
            system_prompt=DIP_CLASSIFICATION_PROMPT,
            user_prompt=f"Ticker: {state.input.value.upper()}\n\nEvidence:\n{evidence}",
            response_model=_ClassificationResponse,
        )
        dip_type = DipType.UNCLEAR
        for dt in DipType:
            if dt.value == resp.dip_type.upper():
                dip_type = dt
                break
        state.classification = ClassificationResult(
            dip_type=dip_type,
            confidence=resp.confidence,
            reasoning=resp.reasoning,
        )
    except Exception as e:
        logger.error("Dip classification failed: %s", e)
        state.errors.append(f"dip_classification: {e}")
        state.classification = ClassificationResult(
            dip_type=DipType.UNCLEAR,
            confidence=0.0,
            reasoning=f"Classification failed: {e}",
        )


def step3_research_facts(
    state: AgentState,
    search: TavilyClient,
    llm: ClaudeLLM,
    registry: SourceRegistry,
    config: ResearchConfig,
) -> None:
    """Generate targeted queries and extract structured facts."""
    ticker = state.input.value.upper()
    year = date.today().year

    # Targeted queries per category
    category_queries = {
        "earnings": f"{ticker} earnings results revenue EPS {year}",
        "guidance": f"{ticker} forward guidance outlook forecast {year}",
        "competitive": f"{ticker} competitive position market share industry",
        "balance_sheet": f"{ticker} balance sheet cash debt free cash flow",
        "valuation": f"{ticker} valuation PE ratio compared peers historical",
        "bear_case": f"{ticker} risks bear case concerns problems {year}",
    }

    all_results: list[SearchResult] = []
    queries_this_step = 0
    for _cat, q in category_queries.items():
        if queries_this_step >= config.max_queries_per_iteration:
            break
        if q not in state.queries_executed:
            results = search.search(q, max_results=config.max_urls_per_query)
            all_results.extend(results)
            state.queries_executed.append(q)
            queries_this_step += 1

    if not all_results:
        return

    # Register sources
    for r in all_results:
        registry.add(url=r.url, title=r.title, snippet=r.content)

    context = _format_search_results(all_results, registry)

    try:
        resp: _FactExtractionResponse = llm.complete(
            system_prompt=FACT_EXTRACTION_PROMPT,
            user_prompt=f"Ticker: {ticker}\n\nSearch Results:\n{context}",
            response_model=_FactExtractionResponse,
        )
        # Convert raw items to EvidenceItems with proper source IDs
        for field_name in FactPack.model_fields:
            raw_items: list[_EvidenceItemRaw] = getattr(resp, field_name, [])
            evidence_items = []
            for raw in raw_items:
                source_ids = []
                for url in raw.source_urls:
                    sid = registry.get_id(url)
                    if sid:
                        source_ids.append(sid)
                evidence_items.append(EvidenceItem(text=raw.text, source_ids=source_ids))
            existing = getattr(state.fact_pack, field_name)
            existing.extend(evidence_items)
    except Exception as e:
        logger.error("Fact extraction failed: %s", e)
        state.errors.append(f"fact_extraction: {e}")


def step4_generate_card(
    state: AgentState,
    llm: ClaudeLLM,
    registry: SourceRegistry,
) -> None:
    """Synthesize all evidence into an OpportunityCard."""
    evidence = _format_evidence_for_llm(state, registry)

    try:
        resp: _CardSynthesisResponse = llm.complete(
            system_prompt=CARD_SYNTHESIS_PROMPT,
            user_prompt=(
                f"Ticker: {state.input.value.upper()}\n\n"
                f"Research Evidence:\n{evidence}"
            ),
            response_model=_CardSynthesisResponse,
        )
        verdict = Verdict.WATCH
        for v in Verdict:
            if v.value == resp.verdict.upper():
                verdict = v
                break

        from research_agent.card import build_card

        state.card = build_card(state, registry, verdict, resp)
    except Exception as e:
        logger.error("Card synthesis failed: %s", e)
        state.errors.append(f"card_synthesis: {e}")


# ── Main agent loop ──────────────────────────────────────────────────────────


def run_loop(
    state: AgentState,
    search: TavilyClient,
    llm: ClaudeLLM,
    registry: SourceRegistry,
    config: ResearchConfig,
) -> OpportunityCard:
    """Run the iterative research loop.

    Steps:
      1. Detect trigger (what caused the dip?)
      2. Classify dip (temporary vs structural)
      3. Research facts (earnings, guidance, competitive, balance sheet, etc.)
      4. Generate card (synthesize into opportunity card)
    """
    for iteration in range(config.max_iterations):
        state.iteration = iteration + 1
        logger.info("Iteration %d/%d", state.iteration, config.max_iterations)

        if state.trigger is None:
            logger.info("Step 1: Detecting trigger...")
            step1_detect_trigger(state, search, llm, registry, config)
            continue

        if state.classification is None:
            logger.info("Step 2: Classifying dip...")
            step2_classify_dip(state, llm, registry)
            continue

        if state.fact_pack.total_items < config.min_evidence_items:
            logger.info(
                "Step 3: Researching facts (%d/%d evidence items)...",
                state.fact_pack.total_items,
                config.min_evidence_items,
            )
            step3_research_facts(state, search, llm, registry, config)
            continue

        logger.info("Step 4: Generating opportunity card...")
        step4_generate_card(state, llm, registry)
        if state.card:
            return state.card

    # If we exit the loop without a card, force generation with what we have
    if state.card is None:
        logger.info("Budget exhausted, generating partial card...")
        step4_generate_card(state, llm, registry)

    if state.card:
        return state.card

    # Last resort: build a minimal card
    from research_agent.card import build_partial_card

    return build_partial_card(state, registry)
