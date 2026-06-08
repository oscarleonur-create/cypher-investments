"""Sell-side analyst consensus — pulled entirely from yfinance.

No new API keys required.  All fields are best-effort; any missing data
leaves the corresponding attribute as None / 0 rather than raising.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import yfinance as yf

from advisor.research.models import AnalystConsensus

logger = logging.getLogger(__name__)


def build_consensus(symbol: str) -> AnalystConsensus:
    """Return AnalystConsensus for *symbol* from yfinance analyst data."""
    sym = symbol.upper()
    try:
        ticker = yf.Ticker(sym)
        info = ticker.info or {}
        return _parse(sym, ticker, info)
    except Exception as exc:  # noqa: BLE001
        logger.warning("consensus(%s) failed: %s", sym, exc)
        return AnalystConsensus(symbol=sym)


# ── Internal parsing ──────────────────────────────────────────────────────────


def _parse(sym: str, ticker: yf.Ticker, info: dict) -> AnalystConsensus:
    current_price = _f(info.get("currentPrice") or info.get("regularMarketPrice"))

    target_mean = _f(info.get("targetMeanPrice"))
    target_low = _f(info.get("targetLowPrice"))
    target_high = _f(info.get("targetHighPrice"))
    target_median = _f(info.get("targetMedianPrice"))

    consensus_upside: float | None = None
    if target_mean is not None and current_price and current_price > 0:
        consensus_upside = (target_mean / current_price) - 1

    n_analysts = int(info.get("numberOfAnalystOpinions") or 0)
    rec_mean = _f(info.get("recommendationMean"))
    rec_key = str(info.get("recommendationKey") or "")

    # Forward EPS estimates
    eps_cy: float | None = None
    eps_ny: float | None = None
    rev_cy: float | None = None
    rev_ny: float | None = None
    try:
        ee = ticker.earnings_estimate
        if ee is not None and not ee.empty:
            # rows indexed by period strings like "0q", "+1q", "0y", "+1y"
            if "0y" in ee.index:
                eps_cy = _f(ee.loc["0y", "avg"] if "avg" in ee.columns else None)
            if "+1y" in ee.index:
                eps_ny = _f(ee.loc["+1y", "avg"] if "avg" in ee.columns else None)
    except Exception:  # noqa: BLE001
        pass

    try:
        re = ticker.revenue_estimate
        if re is not None and not re.empty:
            if "0y" in re.index:
                rev_cy = _f(re.loc["0y", "avg"] if "avg" in re.columns else None)
            if "+1y" in re.index:
                rev_ny = _f(re.loc["+1y", "avg"] if "avg" in re.columns else None)
    except Exception:  # noqa: BLE001
        pass

    # EPS revision counts (30-day window)
    up_30d = 0
    down_30d = 0
    try:
        er = ticker.eps_revisions
        if er is not None and not er.empty:
            # columns: upLast7days, upLast30days, downLast30days, downLast90days
            # rows indexed by period strings
            for col in er.columns:
                lc = col.lower()
                if "uplast30" in lc or "up_last_30" in lc or col == "upLast30days":
                    up_30d = int(er[col].sum() or 0)
                if "downlast30" in lc or "down_last_30" in lc or col == "downLast30days":
                    down_30d = int(er[col].sum() or 0)
    except Exception:  # noqa: BLE001
        pass

    if up_30d > down_30d:
        revision_trend = "improving"
    elif down_30d > up_30d:
        revision_trend = "deteriorating"
    else:
        revision_trend = "flat"

    # Rating changes in last 30 days
    upgrades_30d = 0
    downgrades_30d = 0
    try:
        recs = ticker.recommendations
        if recs is not None and not recs.empty:
            cutoff = datetime.now() - timedelta(days=30)
            if recs.index.tz is not None:
                import pandas as pd

                cutoff = pd.Timestamp(cutoff, tz="UTC")
            recent = recs[recs.index >= cutoff]
            if "To Grade" in recent.columns and "Action" in recent.columns:
                for action in recent["Action"].str.lower():
                    if "upgrade" in action:
                        upgrades_30d += 1
                    elif "downgrade" in action:
                        downgrades_30d += 1
    except Exception:  # noqa: BLE001
        pass

    return AnalystConsensus(
        symbol=sym,
        n_analysts=n_analysts,
        recommendation_mean=rec_mean,
        recommendation_key=rec_key,
        target_price_low=target_low,
        target_price_mean=target_mean,
        target_price_median=target_median,
        target_price_high=target_high,
        current_price=current_price,
        consensus_upside_pct=consensus_upside,
        eps_estimate_current_year=eps_cy,
        eps_estimate_next_year=eps_ny,
        revenue_estimate_current_year=rev_cy,
        revenue_estimate_next_year=rev_ny,
        eps_revisions_up_30d=up_30d,
        eps_revisions_down_30d=down_30d,
        revision_trend=revision_trend,
        upgrades_30d=upgrades_30d,
        downgrades_30d=downgrades_30d,
    )


def _f(v: object) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)  # type: ignore[arg-type]
        return f if f == f else None  # filter NaN
    except (TypeError, ValueError):
        return None
