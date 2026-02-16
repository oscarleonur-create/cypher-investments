"""Domain models for the research agent."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ── Enums ────────────────────────────────────────────────────────────────────


class InputMode(StrEnum):
    TICKER = "ticker"
    SECTOR = "sector"
    THESIS = "thesis"


class Verdict(StrEnum):
    BUY_THE_DIP = "BUY_THE_DIP"
    WATCH = "WATCH"
    AVOID = "AVOID"


class DipType(StrEnum):
    TEMPORARY = "TEMPORARY"
    STRUCTURAL = "STRUCTURAL"
    UNCLEAR = "UNCLEAR"


# ── Core value objects ───────────────────────────────────────────────────────


class ResearchInput(BaseModel):
    mode: InputMode
    value: str

    def run_id(self) -> str:
        """Stable hash of input + date for deterministic IDs."""
        raw = f"{self.mode}:{self.value}:{date.today().isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


class Source(BaseModel):
    url: str
    title: str = ""
    publisher: str = ""
    tier: int = 3
    snippet: str = ""
    accessed_at: datetime = Field(default_factory=datetime.now)


class EvidenceItem(BaseModel):
    """A claim backed by source references (e.g. ['s1', 's3'])."""

    text: str
    source_ids: list[str] = Field(default_factory=list)


class Catalyst(BaseModel):
    summary: str
    date: str = ""  # YYYY-MM-DD or empty


class KeyMetrics(BaseModel):
    revenue_growth: str | None = None
    margins: str | None = None
    fcf: str | None = None
    cash: str | None = None
    debt: str | None = None
    guidance_notes: str | None = None


# ── Pipeline step results ────────────────────────────────────────────────────


class TriggerResult(BaseModel):
    found: bool = False
    trigger_type: str = ""
    summary: str = ""
    links: list[str] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    dip_type: DipType = DipType.UNCLEAR
    confidence: float = 0.0
    reasoning: str = ""


class FactPack(BaseModel):
    """Structured evidence collected across iterations."""

    earnings_highlights: list[EvidenceItem] = Field(default_factory=list)
    guidance_changes: list[EvidenceItem] = Field(default_factory=list)
    competitive_landscape: list[EvidenceItem] = Field(default_factory=list)
    unit_economics: list[EvidenceItem] = Field(default_factory=list)
    balance_sheet: list[EvidenceItem] = Field(default_factory=list)
    valuation_comparison: list[EvidenceItem] = Field(default_factory=list)
    bear_rebuttals: list[EvidenceItem] = Field(default_factory=list)

    @property
    def total_items(self) -> int:
        return sum(len(getattr(self, f)) for f in FactPack.model_fields)


# ── Final output ─────────────────────────────────────────────────────────────


class OpportunityCard(BaseModel):
    id: str
    input: ResearchInput
    verdict: Verdict = Verdict.WATCH
    catalyst: Catalyst = Field(default_factory=lambda: Catalyst(summary=""))
    dip_type: DipType = DipType.UNCLEAR
    bull_case: list[str] = Field(default_factory=list)
    bear_case: list[str] = Field(default_factory=list)
    key_metrics: KeyMetrics = Field(default_factory=KeyMetrics)
    risks: list[str] = Field(default_factory=list)
    invalidation: list[str] = Field(default_factory=list)
    validation_checklist: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


# ── Mutable agent state ─────────────────────────────────────────────────────


class AgentState(BaseModel):
    input: ResearchInput
    iteration: int = 0
    sources: list[Source] = Field(default_factory=list)
    trigger: TriggerResult | None = None
    classification: ClassificationResult | None = None
    fact_pack: FactPack = Field(default_factory=FactPack)
    card: OpportunityCard | None = None
    queries_executed: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
