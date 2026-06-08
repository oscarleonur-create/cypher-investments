"""Shared safe-conversion helpers used across screeners and scanners.

Centralises ``safe_float``, ``safe_int``, and ``safe_info`` so that every
module uses the same defensive conversion logic instead of copy-pasting it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import yfinance as yf


def safe_float(val, default: float | None = None) -> float | None:
    """Convert *val* to ``float``, returning *default* on ``NaN``/``None``/error.

    Uses :func:`math.isfinite` so that ``inf`` / ``-inf`` / ``nan`` all map to
    *default*.
    """
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def safe_int(val, default: int = 0) -> int:
    """Convert *val* to ``int``, returning *default* on ``NaN``/``None``/error.

    The value is first converted to ``float`` (to handle numpy types and
    strings like ``"3.0"``) and then truncated to ``int``.
    """
    if val is None:
        return default
    try:
        f = float(val)
        return int(f) if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def safe_info(ticker: yf.Ticker) -> dict:
    """Safely access ``ticker.info``, returning an empty dict on failure."""
    try:
        return ticker.info or {}
    except Exception:
        return {}
