"""The company's own free-cash-flow margins, to show required growth as a range.

Implied expectations turn a price into required revenue growth through an
assumed steady-state free-cash-flow margin, and the answer depends on that
assumption more than on anything else. On 2026-09-25 META "required":

    -0.4%/yr at its FY2023–FY2025 median margin of 32.7%
    +2.3%/yr at the generic 25%
    +5.7%/yr at its trailing twelve-month margin of 18.0%

Six points from one assumption. No single margin is right: the median lags a
capex cycle (flattering META, whose margin fell, and punishing AMD, whose
margin rose), the trailing figure is one volatile year, and the generic 25%
fits nobody (AMZN "required" -6.0% at 25% against a real -0.3%).

So, by the user's decision (2026-09-25), no margin is picked. The weekly
valuation stores the company's own two readings beside the generic scenario,
and required growth is shown as the range they span, each with its source.
The snapshot's base case stays generic, so the user's thesis rules — SPCX's
"required growth must not exceed 25%" — keep testing the same quantity.

A margin at or below zero cannot anchor a steady state (a company burning
cash says nothing about what it will convert later): it is recorded, and left
out of the range with the reason.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)

YEARS = 3
OCF_CONCEPTS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
CAPEX_CONCEPTS = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")

_YEAR = re.compile(r"CY(\d{4})$")


@dataclass(frozen=True)
class MarginResult:
    margin: float | None  # None when it cannot be computed
    label: str  # e.g. "FY2023–FY2025 median", or why there is none
    yearly: tuple[tuple[date, float], ...] = field(default=())


def annual_values(rows: list[dict]) -> dict[date, float]:
    """Annual (fiscal-year) values by period end, from framed rows. Pure."""
    out: dict[date, float] = {}
    for row in rows:
        if not _YEAR.fullmatch(str(row.get("frame") or "")):
            continue
        try:
            out[date.fromisoformat(row["end"])] = float(row["val"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _merged(series: list[dict[date, float]]) -> dict[date, float]:
    merged: dict[date, float] = {}
    for s in series:
        for k, v in s.items():
            merged.setdefault(k, v)
    return merged


def median_margin(
    ocf: list[dict[date, float]],
    capex: list[dict[date, float]],
    revenue: list[dict[date, float]],
    *,
    years: int = YEARS,
) -> MarginResult:
    """Median FCF margin over the last ``years`` fiscal years. Pure."""
    o, c, r = _merged(ocf), _merged(capex), _merged(revenue)
    common = sorted(d for d in o if d in c and d in r and r[d] > 0)[-years:]
    if len(common) < 2:
        return MarginResult(None, "fewer than two fiscal years of cash-flow data")
    # Capex is a payment; its sign is normalized so an entity reporting it
    # negative is not credited with the outflow.
    yearly = tuple((d, (o[d] - abs(c[d])) / r[d]) for d in common)
    median = statistics.median(m for _, m in yearly)
    return MarginResult(median, f"FY{common[0].year}–FY{common[-1].year} median", yearly)


def load_median_margin(symbol: str) -> MarginResult:
    """Live: SEC annual frames for cash flow, capex and revenue. Never raises."""
    from advisor.news.edgar import company_for
    from advisor.valuation.history import REVENUE_CONCEPTS, _concept_rows

    try:
        cik = getattr(company_for(symbol), "cik", None)
        if not cik:
            return MarginResult(None, "no SEC filer found")
        ocf = [annual_values(_concept_rows(int(cik), c)) for c in OCF_CONCEPTS]
        capex = [annual_values(_concept_rows(int(cik), c)) for c in CAPEX_CONCEPTS]
        revenue = [annual_values(_concept_rows(int(cik), c)) for c in REVENUE_CONCEPTS]
    except Exception as exc:  # noqa: BLE001
        logger.warning("margins: SEC cash-flow series unavailable for %s: %s", symbol, exc)
        return MarginResult(None, "cash-flow series unavailable")
    return median_margin(ocf, capex, revenue)


def load_trailing_margin(symbol: str) -> MarginResult:
    """Trailing four quarters of free cash flow over revenue (yfinance). Never raises."""
    try:
        import yfinance as yf

        t = yf.Ticker(symbol)
        fcf = t.quarterly_cashflow.loc["Free Cash Flow"].dropna().sort_index().iloc[-4:]
        rev = t.quarterly_income_stmt.loc["Total Revenue"].dropna().sort_index().iloc[-4:]
    except Exception as exc:  # noqa: BLE001
        logger.info("margins: no trailing FCF margin for %s: %s", symbol, exc)
        return MarginResult(None, "trailing cash flow unavailable")
    if len(fcf) < 4 or len(rev) < 4 or float(rev.sum()) <= 0:
        return MarginResult(None, "fewer than four quarters of cash flow")
    return MarginResult(
        float(fcf.sum()) / float(rev.sum()), f"trailing 4 quarters to {fcf.index[-1].date()}"
    )


class MarginReading:
    """Required growth at one margin: what the range is made of."""

    __slots__ = ("margin", "label", "required")

    def __init__(self, margin: float, label: str, required: float | None) -> None:
        self.margin, self.label, self.required = margin, label, required


def required_range(snapshot, price: float | None = None) -> tuple[list[MarginReading], list[str]]:
    """Required growth at the generic, trailing and median margins. Pure.

    Returns (readings with a usable margin, notes on the ones left out). The
    terminal multiple and horizon are the snapshot's base case; only the
    margin varies, because the margin is the assumption that moves the answer.
    """
    base = snapshot.base_case() if snapshot is not None else None
    if base is None:
        return [], []
    price = price or snapshot.price
    runrate = snapshot.revenue_runrate
    ev = snapshot.shares_outstanding * price - (snapshot.net_cash or 0.0)

    def required(margin: float) -> float | None:
        if not runrate or runrate <= 0 or ev <= 0 or margin <= 0:
            return None
        revenue = ev / (base.terminal_multiple * margin)
        return (revenue / runrate) ** (1 / base.years) - 1

    readings = [MarginReading(base.fcf_margin, "generic", required(base.fcf_margin))]
    notes = []
    for margin, label in (
        (snapshot.margin_trailing, snapshot.margin_trailing_label or "trailing"),
        (snapshot.margin_median, snapshot.margin_median_label or "median"),
    ):
        if margin is None:
            notes.append(f"own {label}: unavailable")
        elif margin <= 0:
            notes.append(f"own {label} margin {margin:+.1%}: burning cash, no steady state to read")
        else:
            readings.append(MarginReading(margin, f"own {label}", required(margin)))
    return [r for r in readings if r.required is not None], notes
