"""What a valuation is made of, and what it refuses to guess.

This module never produces a "fair value". It produces the arithmetic in the
other direction: given the price the market is charging, what growth and
margin would the business have to deliver to justify it. That number is a
fact, not an opinion, and it is falsifiable — which makes it something a
thesis can be written against and the daemon can monitor.

Every field here is either read from a filing or computed from fields that
were. Nothing is estimated, and a missing input produces a missing output
rather than a default.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import ClassVar

from pydantic import BaseModel, Field

from advisor.daemon.market_calendar import now_et


class Fundamentals(BaseModel):
    """Consolidated figures pulled from one filing's XBRL.

    "Consolidated" is the whole difficulty. XBRL reports the same concept many
    times over — once per segment, per instrument, per class — and the naive
    read picks a slice and calls it the total. SPCX's Q2 revenue appears
    sixteen times in the same period; only one of those facts is
    undimensioned, and it is the only one that means $7,814M.
    """

    symbol: str
    source_accession: str
    period_end: date
    period_start: date | None = None
    fiscal_period: str | None = None

    revenue: float | None = None  # the period's consolidated revenue
    # The same-length period a year earlier, from the same filing's
    # comparative column. None when the filing carries no comparative (a
    # first report after an IPO) — never estimated.
    prior_revenue: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    cash: float | None = None
    marketable_securities: float | None = None
    total_debt: float | None = None
    shares_outstanding: float | None = None  # summed across classes

    retrieved_at: datetime = Field(default_factory=now_et)
    missing: list[str] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Whether a valuation can be computed at all from this."""
        return not self.missing

    @property
    def net_cash(self) -> float | None:
        if self.cash is None:
            return None
        return self.cash + (self.marketable_securities or 0.0) - (self.total_debt or 0.0)

    @property
    def revenue_yoy(self) -> float | None:
        """Growth over the comparable period a year earlier, or None."""
        if not self.revenue or not self.prior_revenue or self.prior_revenue <= 0:
            return None
        return self.revenue / self.prior_revenue - 1

    @property
    def revenue_runrate(self) -> float | None:
        """The period's revenue annualised.

        A quarter is multiplied by four. This is a run-rate, not a forecast,
        and it is wrong for a seasonal business — which is why the period is
        carried alongside it rather than discarded.
        """
        if self.revenue is None:
            return None
        return self.revenue * 4 if self.fiscal_period != "FY" else self.revenue


class ImpliedExpectations(BaseModel):
    """What the current price requires, expressed as growth the business must
    deliver.

    The inputs are deliberately visible. "The price implies 26% revenue growth
    for a decade" means nothing without the terminal multiple and margin that
    produced it, so those travel with the answer and the caller can move them.
    """

    terminal_multiple: float  # EV / FCF at the end of the horizon
    fcf_margin: float  # steady-state free cash flow margin
    years: int
    required_fcf: float
    required_revenue: float
    implied_cagr: float

    def describe(self) -> str:
        return (
            f"{self.implied_cagr:.1%} revenue CAGR for {self.years} years to "
            f"${self.required_revenue / 1e9:,.0f}bn, at {self.fcf_margin:.0%} FCF "
            f"margin and {self.terminal_multiple:g}x terminal"
        )


class ValuationSnapshot(BaseModel):
    """One symbol's valuation at one moment, with its inputs attached."""

    symbol: str
    asof: date
    price: float
    shares_outstanding: float
    market_cap: float
    net_cash: float | None
    enterprise_value: float
    revenue_runrate: float | None
    ev_to_revenue: float | None
    source_accession: str
    period_end: date
    # Year-over-year growth of the period's revenue, from the filing's own
    # comparative column. What the business is delivering, set beside what the
    # price requires. None when there is no comparative.
    revenue_yoy: float | None = None
    scenarios: list[ImpliedExpectations] = Field(default_factory=list)
    # The company's own free-cash-flow margins, recorded beside the generic
    # scenarios so required growth can be shown as a range across them
    # (``valuation.margins``). The scenarios themselves stay generic.
    margin_trailing: float | None = None
    margin_trailing_label: str = ""
    margin_median: float | None = None
    margin_median_label: str = ""
    computed_at: datetime = Field(default_factory=now_et)

    # Past this, the filing behind the valuation is old enough that the
    # run-rate may describe a different business. NBIS files annually as a
    # foreign private issuer, so its figures are routinely two quarters old.
    STALE_AFTER_DAYS: ClassVar[int] = 120

    def period_age_days(self, today: date | None = None) -> int:
        return ((today or date.today()) - self.period_end).days

    def is_stale(self, today: date | None = None) -> bool:
        """Whether the filing behind this is old enough to mislead."""
        return self.period_age_days(today) > self.STALE_AFTER_DAYS

    def base_case(self) -> ImpliedExpectations | None:
        """The middle scenario — the one quoted when only one number fits."""
        if not self.scenarios:
            return None
        return sorted(self.scenarios, key=lambda s: s.implied_cagr)[len(self.scenarios) // 2]
