"""Company network builder — maps Level-2 connections from a ResearchReport.

Level 1 (subject)  — full ResearchReport you already built.
Level 2 (direct)   — peers, public institutional holders, key competitors.
Level 3 (extended) — drill into any Level-2 node with:
                       `advisor research ticker <TICKER>`

Connection sources (all from cached data, no extra API calls needed):
  - Peers         : multiples.peers (yfinance snapshot already present)
  - Top holders   : ecosystem.holders.top_holders → mapped to tickers
  - Competitors   : industry.key_competitors (if tickers; else try yfinance search)
"""

from __future__ import annotations

import logging
from datetime import date

from advisor.research.models import (
    CompanyNetwork,
    NetworkNode,
    PeerSnapshot,
    RelationshipType,
    ResearchReport,
)

logger = logging.getLogger(__name__)

# Known public institutional holders: name fragment (lower) → ticker
_PUBLIC_HOLDER_MAP: dict[str, str] = {
    "blackrock": "BLK",
    "state street": "STT",
    "morgan stanley": "MS",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "berkshire": "BRK-B",
    "t. rowe price": "TROW",
    "t.rowe price": "TROW",
    "wells fargo": "WFC",
    "goldman sachs": "GS",
    "bank of america": "BAC",
    "citigroup": "C",
    "northern trust": "NTRS",
    "invesco": "IVZ",
    "franklin resources": "BEN",
    "charles schwab": "SCHW",
    "raymond james": "RJF",
    "ameriprise": "AMP",
    "principal financial": "PFG",
    "sun life": "SLF",
    "pnc financial": "PNC",
    "mufg": "MUFG",
    "capital group": None,  # private
    "vanguard": None,  # private
    "fmr": None,  # Fidelity, private
    "geode": None,  # private
    "ssga": "STT",  # State Street subsidiary
}


def build_network(symbol: str, report: ResearchReport | None = None) -> CompanyNetwork:
    """Build a Level-2 company network from an existing ResearchReport."""
    sym = symbol.upper()

    if report is None:
        report = _load_cached(sym)

    subject_snap = _snapshot(sym)
    nodes: list[NetworkNode] = []
    seen: set[str] = {sym}

    # ── Peers (from multiples) ────────────────────────────────────────────────
    if report and report.multiples:
        for peer in report.multiples.peers:
            if peer.symbol not in seen:
                seen.add(peer.symbol)
                nodes.append(
                    NetworkNode(
                        symbol=peer.symbol,
                        name=peer.name,
                        relationship=RelationshipType.PEER,
                        note=f"{peer.sector}" if peer.sector else "",
                        snapshot=peer,
                    )
                )

    # ── Competitors from industry (if tickers) ─────────────────────────────--
    # Map common company names to correct tickers (LLM may return partial names)
    _NAME_TO_TICKER: dict[str, str] = {
        "HP": "HPQ",  # HP Inc (not Helmerich & Payne)
        "ALPHABET": "GOOGL",
        "GOOGLE": "GOOGL",
        "META": "META",
        "AMAZON": "AMZN",
        "MICROSOFT": "MSFT",
        "SAMSUNG": "005930.KS",
        "XIAOMI": "1810.HK",
        "LENOVO": "0992.HK",
    }
    if report and report.industry and report.industry.key_competitors:
        for comp in report.industry.key_competitors:
            ticker = _NAME_TO_TICKER.get(comp.upper(), comp.upper())
            if ticker in seen or len(ticker) > 10 or " " in comp.upper():
                continue  # skip names that aren't clean tickers
            snap = _snapshot(ticker)
            if snap and snap.name:  # valid ticker
                seen.add(ticker)
                nodes.append(
                    NetworkNode(
                        symbol=ticker,
                        name=snap.name,
                        relationship=RelationshipType.COMPETITOR,
                        note="Key competitor",
                        snapshot=snap,
                    )
                )

    # ── Public institutional holders ─────────────────────────────────────────
    if report and report.ecosystem and report.ecosystem.holders:
        for holder in report.ecosystem.holders.top_holders[:15]:
            ticker = _holder_ticker(holder.name)
            if ticker is None or ticker in seen:
                continue
            seen.add(ticker)
            pct = f"{holder.pct_held:.1%}" if holder.pct_held else ""
            shares_str = _fmt_shares(holder.shares)
            note = f"{shares_str} shares" + (f" ({pct})" if pct else "")
            snap = _snapshot(ticker)
            nodes.append(
                NetworkNode(
                    symbol=ticker,
                    name=holder.name,
                    relationship=RelationshipType.TOP_HOLDER,
                    note=note,
                    snapshot=snap,
                )
            )

    return CompanyNetwork(
        subject_symbol=sym,
        subject_snapshot=subject_snap,
        nodes=nodes,
        as_of=date.today(),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_cached(symbol: str) -> ResearchReport | None:
    try:
        from advisor.research.config import get_settings
        from advisor.research.store import ResearchStore

        return ResearchStore(get_settings().db_path).load_latest_report(symbol)
    except Exception:  # noqa: BLE001
        return None


def _snapshot(symbol: str) -> PeerSnapshot | None:
    try:
        from advisor.research.valuation.multiples import _snapshot as _ms

        return _ms(symbol)
    except Exception:  # noqa: BLE001
        return None


def _holder_ticker(name: str) -> str | None:
    name_lower = name.lower()
    for fragment, ticker in _PUBLIC_HOLDER_MAP.items():
        if fragment in name_lower:
            return ticker
    return None


def _fmt_shares(shares: float | None) -> str:
    if shares is None:
        return "?"
    if shares >= 1e9:
        return f"{shares / 1e9:.2f}B"
    if shares >= 1e6:
        return f"{shares / 1e6:.1f}M"
    return f"{shares:,.0f}"
