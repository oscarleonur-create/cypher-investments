"""Portfolio shape analytics — concentration, valuation risk, sector rotation.

Pure functions over a :class:`PortfolioReview` (plus light market-data lookups
for sector rotation). No Streamlit dependency, so these are unit-testable and
reusable from the CLI. Weights come from the live dollar exposure captured in
``portfolio_review`` — if a review predates that change, ``total_market_value``
is 0 and the weight-based helpers return empty.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from advisor.research.portfolio_review import SECTOR_NAME_TO_ETF, PortfolioReview

logger = logging.getLogger(__name__)

# Below this much remaining DCF/Bayesian upside a name is "priced in".
_PRICED_IN_THRESHOLD = 0.05


# ── Concentration ────────────────────────────────────────────────────────────


def concentration_metrics(review: PortfolioReview) -> dict:
    """Top-weight, top-5 weight, HHI and effective N over holding weights.

    HHI is the Herfindahl–Hirschman index (Σ wᵢ²); effective N = 1/HHI is the
    number of equally-weighted positions that would give the same concentration.
    """
    weights = sorted(
        (p.weight for p in review.positions if p.weight > 0),
        reverse=True,
    )
    if not weights:
        return {
            "n_holdings": len(review.positions),
            "top_weight": 0.0,
            "top5_weight": 0.0,
            "hhi": 0.0,
            "effective_n": 0.0,
        }

    hhi = sum(w * w for w in weights)
    return {
        "n_holdings": len(weights),
        "top_weight": weights[0],
        "top5_weight": sum(weights[:5]),
        "hhi": hhi,
        "effective_n": (1.0 / hhi) if hhi > 0 else 0.0,
    }


def sector_breakdown(review: PortfolioReview) -> dict[str, float]:
    """Portfolio weight summed by sector ("Unknown" bucket for missing)."""
    out: dict[str, float] = {}
    for p in review.positions:
        if p.weight <= 0:
            continue
        sector = p.sector or "Unknown"
        out[sector] = out.get(sector, 0.0) + p.weight
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


# ── Valuation risk ───────────────────────────────────────────────────────────


def _valuation_flag(upside: float | None) -> str:
    """OVERPRICED / PRICED_IN / UPSIDE / UNKNOWN from a decimal upside."""
    if upside is None:
        return "UNKNOWN"
    if upside < 0:
        return "OVERPRICED"
    if upside < _PRICED_IN_THRESHOLD:
        return "PRICED_IN"
    return "UPSIDE"


def valuation_table(review: PortfolioReview) -> list[dict]:
    """Per-holding valuation risk row, sorted cheapest-upside first.

    ``upside`` prefers the DCF base case, falling back to the Bayesian posterior.
    Names with no research report surface as ``UNKNOWN``.
    """
    rows: list[dict] = []
    for p in review.positions:
        upside = p.base_upside if p.base_upside is not None else p.bayes_upside
        rows.append(
            {
                "symbol": p.symbol,
                "weight": p.weight,
                "base_upside": p.base_upside,
                "bayes_upside": p.bayes_upside,
                "bayes_prob_undervalued": p.bayes_prob_undervalued,
                "upside": upside,
                "flag": _valuation_flag(upside),
                "has_report": p.has_report,
            }
        )
    # Sort: known upside ascending (most overpriced first), unknowns last.
    rows.sort(key=lambda r: (r["upside"] is None, r["upside"] if r["upside"] is not None else 0.0))
    return rows


# ── Sector rotation ──────────────────────────────────────────────────────────


def _pct_change(close, lookback: int) -> float | None:
    """Trailing return over ``lookback`` trading rows of a Close series."""
    if close is None or len(close) <= lookback:
        return None
    last = float(close.iloc[-1])
    prior = float(close.iloc[-1 - lookback])
    if prior <= 0:
        return None
    return last / prior - 1.0


def sector_rotation(sectors_held: list[str]) -> dict[str, dict]:
    """Relative momentum of each held sector's ETF vs SPY.

    For every held sector that maps to a SPDR sector ETF, returns 1-month and
    3-month returns of the ETF **relative to SPY** (positive = leading the
    market). Returns an empty dict on any data failure rather than raising.
    """
    etfs = {s: SECTOR_NAME_TO_ETF[s] for s in sectors_held if s in SECTOR_NAME_TO_ETF}
    if not etfs:
        return {}

    tickers = sorted(set(etfs.values()) | {"SPY"})
    try:
        import yfinance as yf

        end = date.today() + timedelta(days=1)
        start = end - timedelta(days=140)  # ~95 trading days of cushion
        data = yf.download(
            tickers, start=str(start), end=str(end), progress=False, auto_adjust=True
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sector rotation download failed: %s", exc)
        return {}

    # yf.download returns a column MultiIndex (field, ticker) for >1 ticker.
    try:
        closes = data["Close"]
    except Exception:  # noqa: BLE001
        return {}

    def ret(ticker: str, lookback: int) -> float | None:
        if ticker not in closes:
            return None
        return _pct_change(closes[ticker].dropna(), lookback)

    spy_1m, spy_3m = ret("SPY", 21), ret("SPY", 63)

    out: dict[str, dict] = {}
    for sector, etf in etfs.items():
        etf_1m, etf_3m = ret(etf, 21), ret(etf, 63)
        rel_1m = (etf_1m - spy_1m) if (etf_1m is not None and spy_1m is not None) else None
        rel_3m = (etf_3m - spy_3m) if (etf_3m is not None and spy_3m is not None) else None
        out[sector] = {
            "etf": etf,
            "etf_return_1m": etf_1m,
            "etf_return_3m": etf_3m,
            "rel_1m": rel_1m,
            "rel_3m": rel_3m,
            "leading": (rel_3m is not None and rel_3m > 0),
        }
    return out
