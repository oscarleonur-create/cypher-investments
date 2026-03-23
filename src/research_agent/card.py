"""OpportunityCard builder and Markdown renderer."""

from __future__ import annotations

from research_agent.evidence import SourceRegistry
from research_agent.models import (
    AgentState,
    BatchResult,
    Catalyst,
    DipType,
    FactPack,
    InputMode,
    KeyMetrics,
    OpportunityCard,
    Verdict,
)


def _compute_coverage_confidence(fact_pack: FactPack) -> dict[str, float]:
    """Compute per-category confidence scores (0.0-1.0) based on evidence counts."""
    confidence = {}
    for name in FactPack.model_fields:
        count = len(getattr(fact_pack, name))
        # 0 items -> 0.0, 1 item -> 0.5, 2+ items -> 1.0
        confidence[name] = min(count / 2.0, 1.0)
    return confidence


def build_card(
    state: AgentState,
    registry: SourceRegistry,
    verdict: Verdict,
    synthesis,  # _CardSynthesisResponse from agent.py
) -> OpportunityCard:
    """Assemble an OpportunityCard from agent state and LLM synthesis."""
    metrics_data = synthesis.key_metrics if isinstance(synthesis.key_metrics, dict) else {}

    # Prefer synthesis catalyst (richer, from full evidence) over step1 trigger summary
    catalyst_summary = synthesis.catalyst_summary or (
        state.trigger.summary if state.trigger else ""
    )
    catalyst_date = synthesis.catalyst_date or ""

    # Compute coverage confidence from fact_pack
    coverage = _compute_coverage_confidence(state.fact_pack)

    return OpportunityCard(
        id=state.input.run_id(),
        input=state.input,
        verdict=verdict,
        catalyst=Catalyst(
            summary=catalyst_summary,
            date=catalyst_date,
        ),
        dip_type=state.classification.dip_type if state.classification else DipType.UNCLEAR,
        bull_case=synthesis.bull_case[:3],
        bear_case=synthesis.bear_case[:3],
        key_metrics=KeyMetrics(
            revenue_growth=metrics_data.get("revenue_growth"),
            margins=metrics_data.get("margins"),
            fcf=metrics_data.get("fcf"),
            cash=metrics_data.get("cash"),
            debt=metrics_data.get("debt"),
            guidance_notes=metrics_data.get("guidance_notes"),
        ),
        risks=synthesis.risks,
        invalidation=synthesis.invalidation,
        validation_checklist=synthesis.validation_checklist,
        next_actions=synthesis.next_actions,
        sources=registry.all_sources(),
        coverage_confidence=coverage,
    )


def build_partial_card(state: AgentState, registry: SourceRegistry) -> OpportunityCard:
    """Build a minimal card from whatever state we have (fallback)."""
    return OpportunityCard(
        id=state.input.run_id(),
        input=state.input,
        verdict=Verdict.WATCH,
        catalyst=Catalyst(
            summary=state.trigger.summary if state.trigger else "Unable to determine",
        ),
        dip_type=state.classification.dip_type if state.classification else DipType.UNCLEAR,
        bull_case=[],
        bear_case=[],
        key_metrics=KeyMetrics(),
        risks=["Insufficient data collected - manual review recommended"],
        invalidation=[],
        validation_checklist=[],
        next_actions=["Re-run research with broader search terms"],
        sources=registry.all_sources(),
    )


def render_markdown(card: OpportunityCard) -> str:
    """Render an OpportunityCard as human-readable Markdown."""
    label = card.input.value.upper() if card.input.mode == InputMode.TICKER else card.input.value
    lines = [
        f"# Opportunity Card: {label}",
        "",
        f"**Setup**: {card.verdict.value} | **Dip Type**: {card.dip_type.value}",
        "",
    ]

    # What Changed
    lines.append("## What Changed")
    lines.append(card.catalyst.summary or "No catalyst identified.")
    lines.append("")

    # Bull Case
    if card.bull_case:
        lines.append("## Bull Case")
        for i, item in enumerate(card.bull_case, 1):
            lines.append(f"{i}. {item}")
        lines.append("")

    # Bear Case
    if card.bear_case:
        lines.append("## Bear Case")
        for i, item in enumerate(card.bear_case, 1):
            lines.append(f"{i}. {item}")
        lines.append("")

    # Key Metrics
    lines.append("## Key Metrics")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    metrics = card.key_metrics
    for label, value in [
        ("Revenue Growth", metrics.revenue_growth),
        ("Margins", metrics.margins),
        ("FCF", metrics.fcf),
        ("Cash", metrics.cash),
        ("Debt", metrics.debt),
    ]:
        lines.append(f"| {label} | {value or 'N/A'} |")
    if metrics.guidance_notes:
        lines.append(f"\n**Guidance**: {metrics.guidance_notes}")
    lines.append("")

    # Validation Checklist
    if card.validation_checklist:
        lines.append("## Validation Checklist (1-4 weeks)")
        for item in card.validation_checklist:
            lines.append(f"- [ ] {item}")
        lines.append("")

    # Risk Plan
    if card.invalidation:
        lines.append("## Risk Plan")
        lines.append("**Invalidation**: " + "; ".join(card.invalidation))
        lines.append("")

    # Risks
    if card.risks:
        lines.append("## Risks")
        for item in card.risks:
            lines.append(f"- {item}")
        lines.append("")

    # Next Actions
    if card.next_actions:
        lines.append("## Next Actions")
        for item in card.next_actions:
            lines.append(f"- {item}")
        lines.append("")

    # Evidence Coverage
    if card.coverage_confidence:
        lines.append("## Evidence Coverage")
        lines.append("| Category | Confidence |")
        lines.append("|----------|------------|")
        for cat, score in card.coverage_confidence.items():
            label = cat.replace("_", " ").title()
            if score >= 1.0:
                indicator = "Strong"
            elif score >= 0.5:
                indicator = "Partial"
            else:
                indicator = "Thin"
            lines.append(f"| {label} | {indicator} ({score:.0%}) |")
        lines.append("")

    # Sources
    if card.sources:
        lines.append("## Sources")
        for i, src in enumerate(card.sources, 1):
            tier_label = f"Tier {src.tier}"
            pub = f" ({src.publisher})" if src.publisher else ""
            lines.append(f"[{i}] {src.title}{pub} - {src.url} - {tier_label}")
        lines.append("")

    return "\n".join(lines)


def render_batch_summary(batch_result: BatchResult) -> str:
    """Render a batch result as a Markdown comparison table."""
    lines = [
        "# Batch Research Summary",
        "",
        f"**Batch ID**: {batch_result.batch_id}",
        f"**Completed**: {batch_result.total_completed}/{batch_result.total_requested} "
        f"| **Failed**: {batch_result.total_failed} "
        f"| **Elapsed**: {batch_result.elapsed_seconds:.1f}s",
        "",
    ]

    if batch_result.cards:
        lines.append("## Comparison")
        lines.append("| Ticker | Verdict | Dip Type | Revenue Growth | Catalyst |")
        lines.append("|--------|---------|----------|----------------|----------|")
        for card in batch_result.cards:
            ticker = (
                card.input.value.upper()
                if card.input.mode == InputMode.TICKER
                else card.input.value
            )
            verdict = card.verdict.value
            dip_type = card.dip_type.value
            rev_growth = card.key_metrics.revenue_growth or "N/A"
            catalyst = (
                card.catalyst.summary[:60] + "..."
                if len(card.catalyst.summary) > 60
                else card.catalyst.summary
            )
            lines.append(f"| {ticker} | {verdict} | {dip_type} | {rev_growth} | {catalyst} |")
        lines.append("")

    if batch_result.failures:
        lines.append("## Failures")
        for symbol, error in batch_result.failures.items():
            lines.append(f"- **{symbol}**: {error}")
        lines.append("")

    return "\n".join(lines)
