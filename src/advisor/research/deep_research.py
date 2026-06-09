"""Deep Research layer — white-paper-style, citation-backed business brief.

Gathers from three source classes and synthesises one cited narrative:
  1. SEC: the latest 10-K's "Item 1. Business" full text (verbatim quotes).
  2. News: recent contract / market-share coverage via Tavily.
  3. Website: the company's own site (products / customers / about) via a
     Tavily domain-restricted search.

Every material claim carries citation ids that resolve to a numbered
References bibliography. References are built from the *real* gathered URLs;
the LLM only emits source indices, and any index it did not receive is dropped
— so footnotes can never point at a hallucinated source. Verbatim management
quotes are validated against the actual filing text before they are kept.

Best-effort: returns an (almost) empty DeepResearch when keys/data are missing.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from advisor.research.models import (
    CustomerUseCase,
    DeepResearch,
    FilingQuote,
    FilingRef,
    FormType,
    RecentDevelopment,
    Reference,
    SecondOrderThesis,
    SourceType,
    SupplyChainPosition,
)

logger = logging.getLogger(__name__)

_BUSINESS_CHAR_CAP = 50_000
_SNIPPET_CHARS = 800


def build_deep_research(
    symbol: str,
    company_name: str = "",
    sector: str = "",
    industry: str = "",
    filings: list[FilingRef] | None = None,
) -> DeepResearch:
    """Return a cited DeepResearch brief, or an empty one on failure."""
    sym = symbol.upper()
    name = company_name or sym
    try:
        from research_agent.config import ResearchConfig
        from research_agent.llm import OpenRouterLLM
        from research_agent.search import TavilyClient
        from research_agent.store import Store

        config = ResearchConfig()
        if not config.openrouter_api_key or not config.tavily_api_key:
            return DeepResearch(symbol=sym)

        store = Store(config.db_path)
        searcher = TavilyClient(config, store)

        # ── Step 1: gather sources into one numbered registry (1-based ids) ──────
        sources: list[Reference] = []
        business_text = ""

        sec_ref = _latest_10k_ref(filings or [])
        if sec_ref is not None:
            business_text = _fetch_business_section(sec_ref)
            if business_text:
                sources.append(
                    Reference(
                        id=len(sources) + 1,
                        title=f"{name} {sec_ref.form.value}",
                        url=sec_ref.url,
                        source_type=SourceType.SEC_FILING,
                        published_date=sec_ref.filing_date.isoformat(),
                        detail=f"{sec_ref.form.value}, Item 1. Business",
                    )
                )
        sec_source_id = sources[0].id if sources and business_text else None

        snippet_by_url: dict[str, str] = {}

        news = searcher.search(
            f"{name} {sym} contract award order win market share competitors 2025 2026",
            max_results=8,
        )
        for r in news:
            if not r.url:
                continue
            snippet_by_url[r.url] = r.content
            sources.append(
                Reference(
                    id=len(sources) + 1,
                    title=r.title,
                    url=r.url,
                    source_type=SourceType.NEWS,
                    detail=_domain(r.url),
                )
            )

        domain = _company_domain(sym)
        if domain:
            site = searcher.search(
                f"{name} products customers applications about",
                domains=[domain],
                max_results=6,
            )
            for r in site:
                if not r.url:
                    continue
                snippet_by_url[r.url] = r.content
                sources.append(
                    Reference(
                        id=len(sources) + 1,
                        title=r.title,
                        url=r.url,
                        source_type=SourceType.WEBSITE,
                        detail=_domain(r.url),
                    )
                )

        if not sources:
            return DeepResearch(symbol=sym)

        # ── Step 2: synthesise via one structured LLM call ───────────────────────
        context = _build_context(business_text, snippet_by_url, sources)
        out = _llm_synthesise(OpenRouterLLM(config), name, sym, sector, industry, context)
        if out is None:
            return DeepResearch(symbol=sym)

        brief = _assemble(sym, out, sources, sec_source_id, business_text, sec_ref)
        return brief

    except Exception as exc:  # noqa: BLE001
        logger.warning("Deep research failed for %s: %s", symbol, exc)
        return DeepResearch(symbol=sym)


# ── SEC filing text ──────────────────────────────────────────────────────────


def _latest_10k_ref(filings: list[FilingRef]) -> FilingRef | None:
    tens = [f for f in filings if f.form == FormType.K10]
    if not tens:
        return None
    return max(tens, key=lambda f: f.filing_date)


def _fetch_business_section(ref: FilingRef) -> str:
    """Pull the 10-K and slice the 'Item 1. Business' section, capped."""
    try:
        from advisor.research.edgar import EdgarClient

        text = EdgarClient().get_filing_text(ref.accession_number, as_markdown=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_filing_text failed for %s: %s", ref.accession_number, exc)
        return ""
    if not text:
        return ""
    return _slice_business(text)[:_BUSINESS_CHAR_CAP]


def _slice_business(text: str) -> str:
    """Heuristically extract Item 1. Business (up to Item 1A / Item 2)."""
    start = re.search(r"item\s*1\.?\s*business", text, re.IGNORECASE)
    if not start:
        return text  # fall back to whole doc (will be capped by caller)
    body = text[start.start() :]
    end = re.search(r"item\s*1a\.?\s*risk|item\s*2\.?\s*propert", body, re.IGNORECASE)
    return body[: end.start()] if end else body


# ── LLM synthesis ────────────────────────────────────────────────────────────


def _build_context(
    business_text: str, snippet_by_url: dict[str, str], sources: list[Reference]
) -> str:
    """Render the numbered source registry the LLM cites by [id]."""
    parts: list[str] = []
    for ref in sources:
        if ref.source_type == SourceType.SEC_FILING:
            parts.append(
                f"[{ref.id}] ({ref.detail}, filed {ref.published_date}) {ref.url}\n"
                f"{business_text}"
            )
        else:
            snippet = (snippet_by_url.get(ref.url, "") or "")[:_SNIPPET_CHARS]
            label = "WEBSITE" if ref.source_type == SourceType.WEBSITE else "NEWS"
            parts.append(f"[{ref.id}] ({label}) {ref.title}\n{ref.url}\n{snippet}")
    return "\n\n".join(parts)


def _llm_synthesise(llm, name: str, sym: str, sector: str, industry: str, context: str):
    from pydantic import BaseModel

    class _Customer(BaseModel):
        customer: str
        use_case: str = ""
        program: str = ""
        citation_ids: list[int] = []

    class _Supply(BaseModel):
        market_share_pct: float | None = None
        share_basis: str = ""
        geographic_note: str = ""
        global_players: list[str] = []
        sole_source: bool | None = None
        position_note: str = ""
        citation_ids: list[int] = []

    class _Dev(BaseModel):
        date: str = ""
        headline: str = ""
        amount_usd: float | None = None
        citation_ids: list[int] = []

    class _Quote(BaseModel):
        quote: str

    class _SecondOrder(BaseModel):
        thesis: str = ""
        analogs: list[str] = []
        citation_ids: list[int] = []

    class _Out(BaseModel):
        abstract: str = ""
        what_they_do: str = ""
        customers: list[_Customer] = []
        supply_chain: _Supply | None = None
        recent_developments: list[_Dev] = []
        management_quotes: list[_Quote] = []
        second_order_thesis: _SecondOrder | None = None

    system_prompt = (
        "You are an equity analyst writing a concise, WHITE-PAPER-STYLE deep-research "
        "brief. You are given numbered sources [n]: a SEC 10-K 'Item 1. Business' "
        "section, recent news, and the company's own website.\n"
        "Rules:\n"
        "- Use ONLY the provided sources. Do not invent facts.\n"
        "- Attach `citation_ids` (the [n] numbers) to EVERY factual claim — customers, "
        "supply_chain, recent_developments, second_order_thesis. Never cite a number "
        "that was not provided.\n"
        "- customers: named customers mapped to the SPECIFIC use_case (what they use "
        "the product FOR) and program/product when stated.\n"
        "- supply_chain: market_share_pct (number) + share_basis, named global_players "
        "(competitors/producers), sole_source (true/false), position_note (e.g. the "
        "Western/US hedge angle).\n"
        "- recent_developments: dated contract wins, orders, raises, capacity expansions "
        "with amount_usd when stated.\n"
        "- management_quotes: copy 2-5 VERBATIM sentences from the 10-K Business text "
        "ONLY (exact wording, no paraphrase). Pick claims about leadership, "
        "sole-source status, customers, or moat.\n"
        "- second_order_thesis: a forward-looking, ADJACENT-MARKET optionality thesis "
        "with historical analogs. This is speculative inference — keep it clearly "
        "framed as a hypothesis.\n"
        "- abstract: a 2-4 sentence white-paper abstract. what_they_do: one precise line."
    )
    user_prompt = (
        f"Company: {name} ({sym})\nSector: {sector}\nIndustry: {industry}\n\n"
        f"SOURCES:\n{context}"
    )
    try:
        return llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=_Out,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Deep research LLM call failed for %s: %s", sym, exc)
        return None


# ── Assembly + citation integrity ────────────────────────────────────────────


def _assemble(
    sym: str,
    out,
    sources: list[Reference],
    sec_source_id: int | None,
    business_text: str,
    sec_ref: FilingRef | None,
) -> DeepResearch:
    valid_ids = {r.id for r in sources}

    def _clean(ids: list[int]) -> list[int]:
        seen: list[int] = []
        for i in ids or []:
            if i in valid_ids and i not in seen:
                seen.append(i)
        return seen

    customers = [
        CustomerUseCase(
            customer=c.customer,
            use_case=c.use_case,
            program=c.program,
            citation_ids=_clean(c.citation_ids),
        )
        for c in (out.customers or [])
        if c.customer
    ]

    supply = None
    if out.supply_chain is not None:
        s = out.supply_chain
        supply = SupplyChainPosition(
            market_share_pct=_safe_float(s.market_share_pct),
            share_basis=s.share_basis,
            geographic_note=s.geographic_note,
            global_players=s.global_players[:8],
            sole_source=s.sole_source,
            position_note=s.position_note,
            citation_ids=_clean(s.citation_ids),
        )

    developments = [
        RecentDevelopment(
            date=d.date,
            headline=d.headline,
            amount_usd=_safe_float(d.amount_usd),
            citation_ids=_clean(d.citation_ids),
        )
        for d in (out.recent_developments or [])
        if d.headline
    ]

    # Verbatim quotes: keep only those actually present in the filing text.
    quotes: list[FilingQuote] = []
    norm_body = _norm(business_text)
    for q in out.management_quotes or []:
        if not q.quote or not norm_body:
            continue
        if _norm(q.quote) not in norm_body:
            continue
        quotes.append(
            FilingQuote(
                quote=q.quote.strip(),
                form=sec_ref.form.value if sec_ref else "",
                filing_date=sec_ref.filing_date.isoformat() if sec_ref else "",
                accession_number=sec_ref.accession_number if sec_ref else "",
                url=sec_ref.url if sec_ref else "",
                citation_id=sec_source_id,
            )
        )

    second_order = None
    if out.second_order_thesis is not None and out.second_order_thesis.thesis:
        so = out.second_order_thesis
        second_order = SecondOrderThesis(
            thesis=so.thesis,
            analogs=so.analogs[:6],
            is_speculative=True,
            citation_ids=_clean(so.citation_ids),
        )

    # ── Keep only cited references; renumber 1..N; remap all citation ids ──────
    cited: set[int] = set()
    for c in customers:
        cited.update(c.citation_ids)
    if supply:
        cited.update(supply.citation_ids)
    for d in developments:
        cited.update(d.citation_ids)
    if second_order:
        cited.update(second_order.citation_ids)
    for q in quotes:
        if q.citation_id is not None:
            cited.add(q.citation_id)

    kept = [r for r in sources if r.id in cited]
    remap = {r.id: new_id for new_id, r in enumerate(kept, start=1)}
    references = [
        Reference(
            id=remap[r.id],
            title=r.title,
            url=r.url,
            source_type=r.source_type,
            published_date=r.published_date,
            detail=r.detail,
        )
        for r in kept
    ]

    def _remap(ids: list[int]) -> list[int]:
        return [remap[i] for i in ids if i in remap]

    for c in customers:
        c.citation_ids = _remap(c.citation_ids)
    if supply:
        supply.citation_ids = _remap(supply.citation_ids)
    for d in developments:
        d.citation_ids = _remap(d.citation_ids)
    if second_order:
        second_order.citation_ids = _remap(second_order.citation_ids)
    for q in quotes:
        q.citation_id = remap.get(q.citation_id) if q.citation_id is not None else None

    return DeepResearch(
        symbol=sym,
        abstract=out.abstract,
        what_they_do=out.what_they_do,
        customers=customers,
        supply_chain=supply,
        recent_developments=developments,
        management_quotes=quotes,
        second_order_thesis=second_order,
        references=references,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _norm(s: str) -> str:
    """Lowercase + collapse whitespace for verbatim-quote matching."""
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _safe_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:  # noqa: BLE001
        return ""


def _company_domain(symbol: str) -> str:
    """Best-effort company website domain from yfinance for site-restricted search."""
    try:
        import yfinance as yf

        site = (yf.Ticker(symbol).info or {}).get("website", "")
        return _domain(site) if site else ""
    except Exception:  # noqa: BLE001
        return ""
