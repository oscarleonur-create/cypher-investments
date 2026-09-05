"""Sector-ETF reference map.

Salvaged from the removed ``advisor.ml.features``. Doubles as the sector leg of
the macro factor panel: the keys are the sector ETFs regressed against, and the
values give a cheap symbol -> sector fallback when a research report has no
sector classification.
"""

from __future__ import annotations

SECTOR_ETFS = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ADBE", "AMD", "INTC", "CSCO", "ORCL"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "XLV": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MDLZ", "CL", "MO", "KHC"],
    "XLI": ["HON", "UPS", "CAT", "GE", "BA", "RTX", "DE", "LMT", "MMM", "UNP"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
    "XLRE": ["PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB"],
    "XLC": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR"],
    "XLB": ["LIN", "APD", "SHW", "FCX", "ECL", "NEM", "DOW", "NUE", "DD", "VMC"],
}

_SYMBOL_TO_SECTOR: dict[str, str] | None = None


def get_sector_etf(symbol: str) -> str | None:
    """Return the sector ETF for ``symbol``, or ``None`` if unclassified."""
    global _SYMBOL_TO_SECTOR
    if _SYMBOL_TO_SECTOR is None:
        _SYMBOL_TO_SECTOR = {s: etf for etf, members in SECTOR_ETFS.items() for s in members}
    return _SYMBOL_TO_SECTOR.get(symbol.upper())
