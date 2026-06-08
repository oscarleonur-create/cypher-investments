"""Risk register builder — LLM-ranked by probability × impact.

Combines structural sources (red flags from statements, leverage) with
qualitative search evidence (Tavily) to produce a ranked risk list.
"""

from __future__ import annotations

import logging

from advisor.research.models import (
    CatalystRiskResult,
    RedFlagList,
    RiskItem,
    RiskSeverity,
)

logger = logging.getLogger(__name__)


def build_risks(
    symbol: str,
    company_name: str = "",
    red_flags: RedFlagList | None = None,
    catalyst_result: CatalystRiskResult | None = None,
) -> CatalystRiskResult:
    """Return a CatalystRiskResult with a ranked risk register.

    Starts from any existing CatalystRiskResult (to preserve catalysts),
    then appends/replaces the risks list.
    """
    name = company_name or symbol
    risks: list[RiskItem] = []

    # 1. Materialise red flags as quantitative risks
    if red_flags:
        risks.extend(_flags_to_risks(red_flags))

    # 2. LLM qualitative risks from Tavily search
    risks.extend(_llm_risks(symbol, name))

    # Sort by risk_score desc (prob × impact); unknowns go last
    risks.sort(
        key=lambda r: (r.risk_score is not None, r.risk_score or 0),
        reverse=True,
    )

    base = catalyst_result or CatalystRiskResult(symbol=symbol.upper())
    return CatalystRiskResult(
        symbol=base.symbol,
        catalysts=base.catalysts,
        risks=risks,
    )


# ── Red-flag → risk conversion ───────────────────────────────────────────────

_FLAG_RISK_MAP = {
    "DSO_DRIFT": ("Receivables deterioration", "execution", 0.4, 0.5, RiskSeverity.MEDIUM),
    "INVENTORY_DRIFT": (
        "Inventory build-up / demand risk",
        "execution",
        0.4,
        0.6,
        RiskSeverity.MEDIUM,
    ),
    "SHARE_DILUTION": ("Ongoing shareholder dilution", "financial", 0.8, 0.4, RiskSeverity.MEDIUM),
    "FCF_QUALITY_GAP": (
        "Earnings quality / accruals risk",
        "financial",
        0.5,
        0.7,
        RiskSeverity.HIGH,
    ),
    "LEVERAGE_STEP_UP": (
        "Leverage-driven financial distress",
        "financial",
        0.4,
        0.8,
        RiskSeverity.HIGH,
    ),
    "MARGIN_COMPRESSION": (
        "Structural margin pressure",
        "competitive",
        0.6,
        0.7,
        RiskSeverity.HIGH,
    ),
}


def _flags_to_risks(red_flags: RedFlagList) -> list[RiskItem]:
    risks: list[RiskItem] = []
    for flag in red_flags.flags:
        mapping = _FLAG_RISK_MAP.get(flag.code)
        if mapping is None:
            continue
        desc, category, prob, impact, severity = mapping
        risks.append(
            RiskItem(
                description=f"{desc}: {flag.detail}",
                category=category,
                probability=prob,
                impact=impact,
                severity=severity,
                source="financial_statements",
            )
        )
    return risks


# ── LLM qualitative risk extraction ─────────────────────────────────────────


def _llm_risks(symbol: str, name: str) -> list[RiskItem]:
    try:
        from pydantic import BaseModel
        from research_agent.config import ResearchConfig
        from research_agent.llm import OpenRouterLLM
        from research_agent.search import TavilyClient
        from research_agent.store import Store

        config = ResearchConfig()
        if not config.tavily_api_key:
            return []

        store = Store(config.db_path)
        searcher = TavilyClient(config, store)

        query = (
            f"{name} {symbol} key risks headwinds competition regulation "
            f"macro threat 2025 2026 analyst concern"
        )
        results = searcher.search(query, max_results=6)
        if not results:
            return []

        context = "\n\n".join(f"[{r.title}] {r.content[:600]}" for r in results[:4])

        class RiskList(BaseModel):
            risks: list[dict]

        llm = OpenRouterLLM(config)
        extracted = llm.complete(
            system_prompt=(
                "You are a financial analyst. Extract 5-8 key investment risks from the text. "
                "For each risk provide: description (string, 1-2 sentences), "
                "category (competitive|macro|regulatory|execution|financial), "
                "probability (0.0-1.0), impact (0.0-1.0). "
                "Base everything on the provided text. Do not invent data."
            ),
            user_prompt=f"Company: {name} ({symbol})\n\n{context}",
            response_model=RiskList,
        )
        items: list[RiskItem] = []
        for item in extracted.risks[:8]:
            prob = _safe_float(item.get("probability"))
            impact = _safe_float(item.get("impact"))
            risk_score = (prob or 0) * (impact or 0)
            severity = (
                RiskSeverity.HIGH
                if risk_score >= 0.3
                else RiskSeverity.MEDIUM
                if risk_score >= 0.15
                else RiskSeverity.LOW
            )
            items.append(
                RiskItem(
                    description=str(item.get("description", "")),
                    category=str(item.get("category", "other")),
                    probability=prob,
                    impact=impact,
                    severity=severity,
                    source="tavily",
                )
            )
        return items
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM risk extraction failed for %s: %s", symbol, exc)
        return []


def _safe_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f or f < 0 or f > 1) else f
    except (TypeError, ValueError):
        return None
