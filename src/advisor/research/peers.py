"""Peer-set builder for comparable company analysis.

Strategy (no paid data required):
1. Fetch the subject's sector / industry / market-cap from yfinance.
2. Pull the S&P 500 universe (already cached in advisor.data.universe).
3. Filter candidates: same GICS sector, market-cap within 0.2x–5x of subject.
4. Optionally call the LLM to trim to the 5–8 most comparable names.
5. Cache the peer list for 90 days.

Callers can also pass explicit peers to skip steps 1-4.
"""

from __future__ import annotations

import logging
from typing import Any

from advisor.data.cache import DiskCache
from advisor.research.config import get_settings

logger = logging.getLogger(__name__)

_PEER_TTL_HOURS = 90 * 24  # 90 days


def build_peer_set(
    symbol: str,
    explicit_peers: list[str] | None = None,
    max_peers: int = 7,
    llm_refine: bool = True,
) -> list[str]:
    """Return a deduplicated list of ticker symbols suitable as comparables.

    If `explicit_peers` is provided they are used as-is (after dedup + upper).
    Otherwise the function auto-discovers peers from the S&P 500.
    """
    symbol = symbol.upper()

    if explicit_peers:
        peers = [p.upper() for p in explicit_peers if p.upper() != symbol]
        return peers[:max_peers]

    settings = get_settings()
    cache = DiskCache(cache_dir=settings.cache_dir, ttl_hours=_PEER_TTL_HOURS)
    cache_key = ("research", "peers", symbol, str(max_peers))
    cached = cache.get_json(*cache_key)
    if cached is not None:
        return cached["peers"]

    peers = _discover_peers(symbol, max_peers=max_peers * 3)  # wider net for LLM to trim
    if llm_refine and len(peers) > max_peers:
        peers = _llm_refine_peers(symbol, peers, max_peers)

    peers = peers[:max_peers]
    cache.set_json({"peers": peers}, *cache_key)
    return peers


def _discover_peers(symbol: str, max_peers: int = 20) -> list[str]:
    """Filter S&P 500 by sector + market-cap band."""
    import yfinance as yf

    from advisor.data.universe import fetch_sp500

    try:
        info = yf.Ticker(symbol).info
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance info failed for %s: %s", symbol, exc)
        return []

    sector = info.get("sector", "")
    market_cap = info.get("marketCap")

    try:
        sp500 = fetch_sp500()
        universe = [s.symbol for s in sp500]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch S&P 500 universe: %s", exc)
        universe = []

    if not sector or not universe:
        return []

    candidates: list[str] = []
    for ticker in universe:
        if ticker == symbol:
            continue
        try:
            peer_info = yf.Ticker(ticker).fast_info
            peer_sector = getattr(peer_info, "sector", None) or ""
            peer_cap = getattr(peer_info, "market_cap", None)
        except Exception:  # noqa: BLE001
            continue

        if peer_sector.lower() != sector.lower():
            continue
        if market_cap and peer_cap:
            ratio = peer_cap / market_cap
            if not (0.2 <= ratio <= 5.0):
                continue
        candidates.append(ticker)
        if len(candidates) >= max_peers:
            break

    return candidates


def _llm_refine_peers(symbol: str, candidates: list[str], max_peers: int) -> list[str]:
    """Ask the LLM to trim the candidate list to the most comparable names."""
    try:
        from pydantic import BaseModel
        from research_agent.config import ResearchConfig
        from research_agent.llm import OpenRouterLLM

        class PeerList(BaseModel):
            peers: list[str]

        config = ResearchConfig()
        if not config.openrouter_api_key:
            return candidates[:max_peers]

        llm = OpenRouterLLM(config)
        prompt = (
            f"You are a sell-side analyst. Given the subject company {symbol} and "
            f"this list of candidate peers from the same S&P 500 sector:\n"
            f"{', '.join(candidates)}\n\n"
            f"Select the {max_peers} most directly comparable companies — same business model, "
            f"similar product/service mix, overlapping customer base. "
            f"Exclude conglomerates or companies with very different revenue mixes. "
            f"Return only ticker symbols."
        )
        result = llm.complete(
            system_prompt="You are a financial analyst. Respond with JSON only.",
            user_prompt=prompt,
            response_model=PeerList,
        )
        refined = [p.upper() for p in result.peers if p.upper() != symbol]
        return refined[:max_peers]
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM peer refinement failed: %s", exc)
        return candidates[:max_peers]


def _get_ticker_info(ticker: str) -> dict[str, Any]:
    import yfinance as yf

    try:
        return yf.Ticker(ticker).info or {}
    except Exception:  # noqa: BLE001
        return {}
