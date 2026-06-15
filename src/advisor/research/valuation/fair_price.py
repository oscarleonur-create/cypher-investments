"""Consolidated fair-price synthesis.

Blends the valuation estimates already produced elsewhere in the report — DCF
base case, peer-multiples-implied price, the Bayesian posterior median, and the
analyst consensus target — into one headline fair value with a low–high range
and a per-method breakdown. Pure compute (no LLM, no extra network beyond the
Bayesian Monte-Carlo), so it works even when the LLM layers fail.
"""

from __future__ import annotations

import logging
from statistics import median

from advisor.research.models import (
    FairPriceMethod,
    FairPriceResult,
    MultiplesResult,
    ResearchReport,
)

logger = logging.getLogger(__name__)

# Default weights by method; renormalized over whichever methods are present.
_WEIGHTS = {
    "dcf_base": 0.30,
    "multiples": 0.25,
    "bayesian_median": 0.25,
    "analyst_target": 0.20,
}
_LABELS = {
    "dcf_base": "DCF (base case)",
    "multiples": "Peer multiples",
    "bayesian_median": "Bayesian median",
    "analyst_target": "Analyst target",
}


def _positive(x: float | None) -> float | None:
    return x if isinstance(x, (int, float)) and x > 0 else None


def _multiples_implied_price(report: ResearchReport) -> float | None:
    """Per-share value implied by applying peer-median multiples to the subject.

    Uses EV/EBITDA, EV/Sales and trailing P/E where both the peer multiples and
    the matching subject metric are available, then returns the median of the
    per-multiple implied prices.
    """
    mult: MultiplesResult | None = report.multiples
    dcf = report.dcf
    if mult is None or dcf is None:
        return None

    shares = _positive(dcf.shares_outstanding)
    if shares is None:
        return None
    net_debt = dcf.net_debt or 0.0
    subject = mult.subject
    peers = mult.peers or []
    if not peers:
        return None

    def peer_median(attr: str) -> float | None:
        vals = [v for p in peers if (v := _positive(getattr(p, attr, None))) is not None]
        return median(vals) if vals else None

    estimates: list[float] = []

    # EV-based multiples → enterprise value → equity → per share.
    ev_ebitda = peer_median("ev_to_ebitda")
    if ev_ebitda and _positive(subject.ebitda_ttm):
        estimates.append((ev_ebitda * subject.ebitda_ttm - net_debt) / shares)

    ev_sales = peer_median("ev_to_sales")
    if ev_sales and _positive(subject.revenue_ttm):
        estimates.append((ev_sales * subject.revenue_ttm - net_debt) / shares)

    # Equity multiple: P/E on trailing net income → equity value → per share.
    pe = peer_median("pe_trailing")
    if pe and _positive(subject.net_income_ttm):
        estimates.append((pe * subject.net_income_ttm) / shares)

    positives = [e for e in estimates if e > 0]
    return median(positives) if positives else None


def _gather_methods(report: ResearchReport) -> dict[str, float]:
    """Collect each method's point estimate (only when usable)."""
    out: dict[str, float] = {}

    if report.dcf and report.dcf.base and _positive(report.dcf.base.implied_price):
        out["dcf_base"] = report.dcf.base.implied_price

    mult_price = _multiples_implied_price(report)
    if mult_price:
        out["multiples"] = mult_price

    try:
        from advisor.research.valuation.bayesian import build_bayesian_pricing

        bayes = build_bayesian_pricing(report)
        if bayes.meaningful and _positive(bayes.median_price):
            out["bayesian_median"] = bayes.median_price
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bayesian leg of fair price failed for %s: %s", report.symbol, exc)

    if report.consensus and _positive(report.consensus.target_price_mean):
        out["analyst_target"] = report.consensus.target_price_mean

    return out


def _range(report: ResearchReport, estimates: list[float]) -> tuple[float, float]:
    """Low/high band: spread of method estimates, widened by DCF and Bayesian tails."""
    lows = [min(estimates)]
    highs = [max(estimates)]
    dcf = report.dcf
    if dcf:
        if dcf.bear and _positive(dcf.bear.implied_price):
            lows.append(dcf.bear.implied_price)
        if dcf.bull and _positive(dcf.bull.implied_price):
            highs.append(dcf.bull.implied_price)
    return min(lows), max(highs)


def _confidence(estimates: list[float], blended: float) -> str:
    """HIGH when several methods agree tightly; LOW when few or dispersed."""
    n = len(estimates)
    if n < 2 or blended <= 0:
        return "LOW"
    spread = (max(estimates) - min(estimates)) / blended
    if n >= 3 and spread <= 0.25:
        return "HIGH"
    if spread <= 0.5:
        return "MEDIUM"
    return "LOW"


def build_fair_price(report: ResearchReport) -> FairPriceResult | None:
    """Blend available valuation methods into one fair value, or None if none work."""
    current = report.dcf.current_price if report.dcf else None
    current = _positive(current) or (report.consensus.current_price if report.consensus else None)
    if not _positive(current):
        return None

    raw = _gather_methods(report)
    if not raw:
        return None

    total_w = sum(_WEIGHTS[name] for name in raw)
    methods: list[FairPriceMethod] = []
    blended = 0.0
    for name, estimate in raw.items():
        weight = _WEIGHTS[name] / total_w
        blended += weight * estimate
        methods.append(
            FairPriceMethod(name=name, label=_LABELS[name], estimate=estimate, weight=weight)
        )
    methods.sort(key=lambda m: m.weight, reverse=True)

    estimates = list(raw.values())
    low, high = _range(report, estimates)
    note = "" if len(raw) > 1 else "Single-method estimate — treat as indicative only."
    fair_price = round(blended, 2)

    return FairPriceResult(
        symbol=report.symbol.upper(),
        current_price=float(current),
        fair_price=fair_price,
        low=round(low, 2),
        high=round(high, 2),
        upside_pct=fair_price / float(current) - 1.0,
        methods=methods,
        confidence=_confidence(estimates, blended),
        note=note,
    )
