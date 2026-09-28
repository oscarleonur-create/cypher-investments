"""What a valuation is made of, and what it refuses to guess.

The headline is the arithmetic in the backward direction: given the price the
market is charging, what growth and margin would the business have to deliver
to justify it. That number is a fact, not an opinion, and it is falsifiable —
which makes it something a thesis can be written against and the daemon can
monitor.

Beside it, by user decision (2026-09-27), sits a value range: bear, base and
bull per share from the company's own filed margins, with every assumption
attached. It is an opinion and is labelled as one; it is refused, not
defaulted, when the filings offer no positive margin to build it on.

Every input is either read from a filing or computed from fields that were.
A missing input produces a missing output rather than a default.
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
    def period_days(self) -> int | None:
        """Days the period covers, or None when it cannot be known — including
        a start on or after the end, which describes no period at all."""
        if self.period_start is None or self.period_start >= self.period_end:
            return None
        return (self.period_end - self.period_start).days + 1

    @property
    def revenue_runrate(self) -> float | None:
        """The period's revenue annualised.

        A quarter is multiplied by four and a year taken as it is. This is a
        run-rate, not a forecast, and it is wrong for a seasonal business —
        which is why the period is carried alongside it rather than discarded.

        A period that is neither is scaled by its own length. WOLF's FY2026
        10-K covers nine months, because fresh-start accounting on leaving
        bankruptcy split the year; read as a year it understated revenue by a
        quarter.
        """
        if self.revenue is None:
            return None
        return self.revenue * self._annualiser()[0]

    def _annualiser(self) -> tuple[float, str]:
        """(multiplier, label). The period's own length decides when it is
        known; the form type only when it is not."""
        days = self.period_days
        if days is None:
            if self.fiscal_period == "FY":
                return 1.0, f"fiscal year to {self.period_end}"
            return 4.0, f"quarter to {self.period_end} × 4"
        if 80 <= days <= 100:
            return 4.0, f"quarter to {self.period_end} × 4"
        if 350 <= days <= 380:
            return 1.0, f"year to {self.period_end}"
        return 365 / days, f"{days}-day period to {self.period_end}, annualised"

    def runrate_label(self) -> str:
        return self._annualiser()[1]


class ImpliedExpectations(BaseModel):
    """What the current price requires, expressed as growth the business must
    deliver.

    The inputs are deliberately visible. "The price implies 26% revenue growth
    for a decade" means nothing without the margin, discount rate and terminal
    that produced it, so those travel with the answer and the caller can move
    them. ``discount_rate`` is None on a reading from before the engine
    discounted (``ValuationSnapshot.method == "undiscounted"``).
    """

    terminal_multiple: float  # EV / FCF at the end of the horizon
    fcf_margin: float  # steady-state free cash flow margin
    years: int
    required_fcf: float
    required_revenue: float
    implied_cagr: float
    discount_rate: float | None = None
    terminal_growth: float | None = None

    def describe(self) -> str:
        head = (
            f"{self.implied_cagr:.1%} revenue CAGR for {self.years} years to "
            f"${self.required_revenue / 1e9:,.0f}bn, at {self.fcf_margin:.0%} FCF margin"
        )
        if self.discount_rate is None:
            return f"{head} and {self.terminal_multiple:g}x terminal"
        return (
            f"{head}, discounted at {self.discount_rate:.0%} with "
            f"{self.terminal_growth or 0:.0%} terminal growth"
        )


class OwnMargin(BaseModel):
    """One of the company's own steady-state margin readings.

    ``kind`` is ``fcf`` (free cash flow over the latest span the filings
    support), ``median`` (median FCF margin over recent fiscal years),
    ``nopat`` (operating margin after tax) or ``yahoo`` (a foreign issuer's
    trailing FCF, where the SEC has no series).
    """

    kind: str
    value: float
    label: str


class ScenarioValue(BaseModel):
    """One scenario of the value range, with the assumptions that produced it."""

    name: str  # bear | base | bull
    value_per_share: float
    upside: float  # value / price − 1
    growth_start: float  # today's growth, held and then faded
    held_years: int = 0  # years today's growth is held before fading
    growth_early: float  # average growth, years 1–3
    growth_late: float  # average growth, years 4–10
    target_margin: float
    margin_label: str
    discount_rate: float
    terminal_growth: float
    enterprise_value: float
    terminal_share: float | None = None  # of value, in the perpetuity


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
    ev_to_revenue: float | None  # over ``revenue_base`` (run-rate before "dcf")
    source_accession: str
    period_end: date
    # Growth the business is delivering, set beside what the price requires:
    # trailing twelve months against the twelve before where the filings
    # allow it, else the latest period against the same one a year earlier.
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

    # How the scenarios were computed. Rows stored before 2026-09-27 carry no
    # method and read as "undiscounted"; the engine writes "dcf". Two readings
    # of different methods are never compared as a move.
    method: str = "undiscounted"
    price_source: str = ""
    discount_rate: float | None = None
    terminal_growth: float | None = None
    revenue_base: float | None = None  # what the projection starts from
    revenue_base_label: str = ""
    revenue_growth_label: str = ""
    start_margin: float | None = None  # today's FCF margin, where the fade starts
    start_margin_label: str = ""
    own_margins: list[OwnMargin] = Field(default_factory=list)
    # The value range: bear, base and bull per share from the company's own
    # margins — an opinion, labelled as one. Empty with ``value_refused`` set
    # when the filings cannot support it.
    value: list[ScenarioValue] = Field(default_factory=list)
    value_refused: str | None = None
    # The steady-state FCF margin the price requires at the growth the
    # business is delivering, faded to terminal.
    implied_margin: float | None = None
    implied_margin_label: str = ""
    notes: list[str] = Field(default_factory=list)

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

    def value_scenario(self, name: str) -> ScenarioValue | None:
        return next((v for v in self.value if v.name == name), None)

    def assumptions_text(self) -> str:
        """How the requirement was computed, in one clause."""
        if self.method != "dcf" or self.discount_rate is None:
            base = self.base_case()
            multiple = f"{base.terminal_multiple:g}x FCF terminal" if base else "generic terminal"
            return f"{multiple}, undiscounted"
        return (
            f"discounted at {self.discount_rate:.0%}, {self.terminal_growth or 0:.0%} terminal "
            f"growth, from {self.revenue_base_label or 'the latest revenue'}"
        )
