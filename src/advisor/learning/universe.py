"""The replay universe: broad on purpose (user decision, 2026-09-25).

Replaying only the watchlist would test the rules on names already chosen
for having done well. A broad list of large, liquid US names across sectors
is the wider sample; the watchlist and held names are added and reported
apart.

It is still today's survivors: names delisted or acquired before today are
missing, which flatters any long-only rule. The replay report says so.
Symbols with no price data are dropped and listed, not silently skipped.
"""

from __future__ import annotations

BROAD: tuple[str, ...] = (
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "ORCL", "CRM", "ADBE",
    "AMD", "INTC", "CSCO", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT", "LRCX",
    "KLAC", "MU", "ADI", "MRVL", "SNPS", "CDNS", "PANW", "CRWD", "FTNT", "ANET",
    "DELL", "HPQ", "PLTR", "SHOP", "UBER", "ABNB", "NFLX", "DDOG", "SNOW",
    # Communication and consumer
    "DIS", "CMCSA", "T", "VZ", "TMUS", "NKE", "SBUX", "MCD", "HD", "LOW",
    "COST", "WMT", "TGT", "KO", "PEP", "PG", "CL", "MDLZ", "TSLA", "F", "GM",
    # Health care
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "GILD", "ISRG", "VRTX", "REGN", "BMY", "CVS", "MDT",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA",
    "PYPL", "COF",
    # Industrials, energy, materials, utilities
    "CAT", "DE", "BA", "GE", "HON", "UPS", "UNP", "LMT", "RTX", "XOM", "CVX",
    "COP", "SLB", "EOG", "LIN", "NEE", "DUK", "SO",
)  # fmt: skip


def replay_universe(book_symbols: list[str] | None = None) -> tuple[list[str], set[str]]:
    """(all symbols, the user's own subset). The user's names are kept even if not in BROAD."""
    own = {s.upper() for s in (book_symbols or [])}
    return sorted(set(BROAD) | own), own
