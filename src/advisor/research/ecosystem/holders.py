"""Institutional and insider ownership from yfinance."""

from __future__ import annotations

import logging
from datetime import date

from advisor.research.models import HolderSummary, InstitutionalHolder

logger = logging.getLogger(__name__)


def get_holders(symbol: str) -> HolderSummary:
    import yfinance as yf

    ticker = yf.Ticker(symbol.upper())
    holders: list[InstitutionalHolder] = []

    try:
        df = ticker.institutional_holders
        if df is not None and not df.empty:
            for _, row in df.head(25).iterrows():
                holders.append(
                    InstitutionalHolder(
                        name=str(row.get("Holder", "")),
                        shares=_safe_float(row.get("Shares")),
                        pct_held=_safe_float(row.get("% Out")),
                        value_usd=_safe_float(row.get("Value")),
                        date_reported=_safe_date(row.get("Date Reported")),
                    )
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("institutional_holders failed for %s: %s", symbol, exc)

    pct_inst: float | None = None
    pct_insider: float | None = None
    try:
        major = ticker.major_holders
        if major is not None and not major.empty:
            pct_inst = _safe_float(major.iloc[1, 0])
            pct_insider = _safe_float(major.iloc[0, 0])
    except Exception as exc:  # noqa: BLE001
        logger.warning("major_holders failed for %s: %s", symbol, exc)

    return HolderSummary(
        symbol=symbol.upper(),
        top_holders=holders,
        pct_institutional=pct_inst,
        pct_insider=pct_insider,
    )


def _safe_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _safe_date(v: object) -> date | None:
    if v is None:
        return None
    try:
        import pandas as pd

        ts = pd.Timestamp(v)
        return ts.date()
    except Exception:  # noqa: BLE001
        return None
