"""Management Quality Score — 0–100 composite from capital allocation signals.

Uses only already-fetched report data (RatioBundle, HolderSummary, InsiderSummary,
CashFlowPeriod) plus one yfinance field (compensationRisk) so no new API keys needed.

Scoring breakdown (100 pts total):
  25 pts — ROIC trend (improving/stable/declining)
  20 pts — Insider ownership % (skin in the game)
  15 pts — C-suite open-market buying in the last 12 months
  10 pts — Net insider buying direction (purchases > sales by value)
  15 pts — ISS compensation risk score (1=low → 10=high, inverted)
  15 pts — Capital return consistency (dividends + buybacks / FCF)
"""

from __future__ import annotations

import logging

from advisor.research.models import (
    HolderSummary,
    InsiderSummary,
    ManagementQualityScore,
    RatioBundle,
)

logger = logging.getLogger(__name__)


def build_management_quality(
    symbol: str,
    ratios: RatioBundle | None = None,
    holders: HolderSummary | None = None,
    insiders: InsiderSummary | None = None,
) -> ManagementQualityScore:
    mq = ManagementQualityScore(symbol=symbol.upper())

    # ── 1. ROIC trend (25 pts) ────────────────────────────────────────────────
    if ratios and len(ratios.periods) >= 2:
        roics = [p.roic for p in ratios.periods if p.roic is not None]
        if len(roics) >= 2:
            recent = roics[:2]
            older = roics[2:4] if len(roics) >= 4 else roics[2:]
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older) if older else recent_avg
            delta = recent_avg - older_avg
            if delta > 0.015:
                mq.roic_trend = "improving"
                mq.roic_points = 25
            elif delta < -0.015:
                mq.roic_trend = "declining"
                mq.roic_points = 5
            else:
                mq.roic_trend = "stable"
                mq.roic_points = 15
            # Bonus: high absolute ROIC (>15%) in recent period
            if recent_avg > 0.15 and mq.roic_points < 25:
                mq.roic_points = min(25, mq.roic_points + 5)

    # ── 2. Insider ownership (20 pts) ─────────────────────────────────────────
    if holders and holders.pct_insider is not None:
        pct = holders.pct_insider
        mq.insider_ownership_pct = pct
        if pct >= 0.10:
            mq.insider_ownership_points = 20
        elif pct >= 0.05:
            mq.insider_ownership_points = 14
        elif pct >= 0.02:
            mq.insider_ownership_points = 8
        else:
            mq.insider_ownership_points = 2

    # ── 3. C-suite open-market buying (15 pts) ────────────────────────────────
    if insiders:
        mq.c_suite_buying = insiders.c_suite_buying
        mq.c_suite_buying_points = 15 if insiders.c_suite_buying else 0

        # ── 4. Net insider buying direction (10 pts) ──────────────────────────
        mq.net_insider_buying = insiders.net_buying_usd >= 0
        mq.net_insider_buying_points = 10 if mq.net_insider_buying else 0

    # ── 5. ISS compensation risk (15 pts, inverted) ───────────────────────────
    comp_risk = _fetch_comp_risk(symbol)
    if comp_risk is not None:
        mq.compensation_risk_score = comp_risk
        # ISS scale: 1=low risk → 10=high risk; map linearly to 0–15
        mq.compensation_points = max(0, round(15 * (1 - (comp_risk - 1) / 9)))

    # ── 6. Capital return consistency (15 pts) ────────────────────────────────
    mq.capital_return_points = _score_capital_return(ratios)

    # ── Composite ─────────────────────────────────────────────────────────────
    mq.composite_score = float(
        mq.roic_points
        + mq.insider_ownership_points
        + mq.c_suite_buying_points
        + mq.net_insider_buying_points
        + mq.compensation_points
        + mq.capital_return_points
    )
    if mq.composite_score >= 70:
        mq.quality_tier = "HIGH"
    elif mq.composite_score >= 40:
        mq.quality_tier = "MEDIUM"
    else:
        mq.quality_tier = "LOW"

    return mq


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fetch_comp_risk(symbol: str) -> float | None:
    try:
        import yfinance as yf

        info = yf.Ticker(symbol.upper()).info or {}
        raw = info.get("compensationRisk") or info.get("auditRisk")
        if raw is not None:
            return float(raw)
    except Exception as exc:  # noqa: BLE001
        logger.debug("compensationRisk fetch failed for %s: %s", symbol, exc)
    return None


def _score_capital_return(ratios: RatioBundle | None) -> int:
    """Award points for consistent positive FCF with buybacks/dividends."""
    if ratios is None or not ratios.periods:
        return 5  # neutral when data missing

    positive_fcf_years = sum(
        1 for p in ratios.periods if p.fcf_margin is not None and p.fcf_margin > 0
    )
    pct_positive = positive_fcf_years / len(ratios.periods)

    if pct_positive >= 0.80:
        return 15
    elif pct_positive >= 0.60:
        return 10
    elif pct_positive >= 0.40:
        return 5
    return 0
