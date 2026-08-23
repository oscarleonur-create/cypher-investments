"""Intraday equity scalping: a lightweight scanner over live candle data.

Pure-pandas signal engine (deliberately not Backtrader) fed by TastyTrade
candles with a yfinance fallback. See `scanner.ScalpScanner` for the entry point
and `strategies.SCALP_STRATEGIES` for the available setups.
"""

from advisor.scalping.models import ScalpAction, ScalpScanResult, ScalpSignal
from advisor.scalping.scanner import ScalpScanner
from advisor.scalping.strategies import SCALP_STRATEGIES

__all__ = [
    "ScalpAction",
    "ScalpScanResult",
    "ScalpSignal",
    "ScalpScanner",
    "SCALP_STRATEGIES",
]
