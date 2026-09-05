"""Catalyst calendar builder.

Sources (in order of reliability):
1. yfinance earnings calendar  → EARNINGS catalyst
2. Tavily search               → product launches, regulatory, contract wins
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from advisor.research.models import CatalystItem, CatalystRiskResult, CatalystType

logger = logging.getLogger(__name__)


def build_catalysts(symbol: str, company_name: str = "") -> CatalystRiskResult:
    """Return a CatalystRiskResult with upcoming catalysts (risks added in risks.py)."""
    name = company_name or symbol
    catalysts: list[CatalystItem] = []

    catalysts.extend(_earnings_catalysts(symbol))
    catalysts.extend(_news_catalysts(symbol, name))

    return CatalystRiskResult(symbol=symbol.upper(), catalysts=catalysts)


# ── Earnings from yfinance ───────────────────────────────────────────────────


def _earnings_catalysts(symbol: str) -> list[CatalystItem]:
    from advisor.data.yahoo import fetch_earnings_dates

    today = date.today()
    items: list[CatalystItem] = []
    for d in fetch_earnings_dates(symbol):
        is_near = d <= today + timedelta(days=90)
        items.append(
            CatalystItem(
                description=f"Earnings release expected {d.isoformat()}",
                catalyst_type=CatalystType.EARNINGS,
                expected_date=d.isoformat(),
                is_near_term=is_near,
                source="yfinance",
                probability=0.5,  # coin-flip on beat/miss at time of build
                direction="mixed",
                price_impact_pct=None,  # filled from options IV if available
            )
        )
    return items


# ── News catalysts via Tavily ────────────────────────────────────────────────


def _resolve_expected_date(
    raw: str, llm_is_near_term: bool, today: date
) -> tuple[str, bool] | None:
    """Sanity-check an LLM-extracted date string against the real current date.

    Search results are recency-filtered but describe events relative to
    *when the article was written*, not relative to today — an April 2026
    article calling an earnings release "upcoming" is stale by August 2026.
    The LLM has no reliable sense of the real current date on its own, so
    this recomputes is_near_term from actual date math and drops anything
    that has already happened, regardless of what the model claimed.

    Returns (expected_date, is_near_term), or None if the catalyst should be
    dropped as already in the past. Non-ISO strings (quarter labels like
    "Q3 FY2026") can't be verified this way, so the LLM's own is_near_term
    guess passes through unchanged — best-effort, not a complete fix for
    that class.
    """
    try:
        parsed = date.fromisoformat(raw[:10])
    except (ValueError, TypeError):
        return raw, llm_is_near_term  # unparseable — can't verify, trust the LLM's own guess
    if parsed < today:
        return None
    return raw, parsed <= today + timedelta(days=90)


def _news_catalysts(symbol: str, name: str) -> list[CatalystItem]:
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
            f"{name} {symbol} upcoming product launch regulatory decision "
            f"contract win catalyst 2025 2026"
        )
        results = searcher.search(query, max_results=6)
        if not results:
            return []

        context = "\n\n".join(f"[{r.title}] {r.content[:500]}" for r in results[:4])
        today = date.today()

        class CatalystList(BaseModel):
            catalysts: list[dict]

        llm = OpenRouterLLM(config)
        extracted = llm.complete(
            system_prompt=(
                f"You are a financial analyst. Today's date is {today.isoformat()}. "
                "Extract upcoming near-term catalysts. "
                "The source text may be an older article describing something that was "
                "'upcoming' when it was written but has already happened by today — "
                "only extract catalysts that are still ahead of today's date. If a "
                "described event's date has already passed, do not include it. "
                "For each return a JSON object with: "
                "description (string), "
                "catalyst_type (one of: EARNINGS, PRODUCT_LAUNCH, REGULATORY, CONTRACT, MACRO, "
                "OTHER), "
                "expected_date (YYYY-MM-DD or quarter string), "
                "is_near_term (bool, true = within 90 days of today), "
                "probability (float 0-1: estimated likelihood this event occurs in the next "
                "12 months; use 0.5 for genuinely uncertain, >0.7 for highly likely, "
                "<0.3 for speculative), "
                "direction (one of: bullish, bearish, mixed, neutral — impact on the stock "
                "if it occurs), "
                "price_impact_pct (float: estimated % stock price move magnitude if it occurs; "
                "unsigned — sign is implied by direction; e.g. 8.0 for an 8% expected move). "
                "Only include catalysts with a concrete timeline mentioned in the text."
            ),
            user_prompt=f"Company: {name} ({symbol})\n\n{context}",
            response_model=CatalystList,
        )
        items: list[CatalystItem] = []
        for item in extracted.catalysts[:8]:
            try:
                resolved = _resolve_expected_date(
                    str(item.get("expected_date", "")),
                    bool(item.get("is_near_term", True)),
                    today,
                )
                if resolved is None:
                    continue  # already in the past — drop regardless of what the LLM said
                expected_date, is_near_term = resolved

                raw_prob = item.get("probability")
                raw_impact = item.get("price_impact_pct")
                items.append(
                    CatalystItem(
                        description=str(item.get("description", "")),
                        catalyst_type=CatalystType(item.get("catalyst_type", "OTHER")),
                        expected_date=expected_date,
                        is_near_term=is_near_term,
                        source="tavily",
                        probability=float(raw_prob) if raw_prob is not None else None,
                        direction=str(item.get("direction", "neutral")),
                        price_impact_pct=float(raw_impact) if raw_impact is not None else None,
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        return items
    except Exception as exc:  # noqa: BLE001
        logger.warning("News catalyst extraction failed for %s: %s", symbol, exc)
        return []
