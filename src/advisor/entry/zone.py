"""The entry zone, as growth: the price below which it asks for less than is delivered.

``valuation.implied`` answers "what growth does this price require?". This
runs the same arithmetic the other way — "what price requires exactly this
growth?" — so a zone can be drawn in the one unit the project trusts:

    at $516.13 (2026-09-13) AMD required +11.2%/yr for a decade; it
    delivered +50.1% year on year. The zone top is the price at which the
    requirement equals the lower of delivered growth and the consensus.

The user accepted this framing on 2026-09-25 knowing what it is: the closest
the system comes to a fair value. It stays arithmetic and falsifiable because
its assumptions are printed beside it — the scenario (terminal multiple, free
cash flow margin, horizon) and which reference growth drew the line. Change
the assumption and the line moves; the proposal says so rather than hiding it.

**What it cannot tell:** whether the reference growth will last. For a
company growing 50% a year the zone is almost always "in", because a decade
at 11% asks far less than this year's 50%. The zone measures how demanding a
price is, not how durable a business is; the reading has to weigh durability.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel, Field

from advisor.valuation.models import ValuationSnapshot

LADDER_STEPS: tuple[float, ...] = (0.0, -0.05, -0.10, -0.15, -0.20)


class Scenario(BaseModel):
    terminal_multiple: float
    fcf_margin: float
    years: int


class Rung(BaseModel):
    price: float
    change: float  # vs the current price
    required: float  # 10-year revenue CAGR the price requires


class Zone(BaseModel):
    price: float  # the price everything is measured at
    required: float  # what that price requires
    scenario: Scenario
    delivered: float | None = None  # revenue YoY, latest filing
    delivered_source: str = ""
    consensus: float | None = None  # annualised growth, run-rate -> last consensus year
    consensus_label: str = ""
    reference: float | None = None  # the lower of the two: the growth the line is drawn at
    reference_label: str = ""
    top: float | None = None  # below this price, required <= reference
    ladder: list[Rung] = Field(default_factory=list)
    stale: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def in_zone(self) -> bool | None:
        if self.top is None:
            return None
        return self.price <= self.top

    @property
    def distance(self) -> float | None:
        """How far the price is above the zone top (negative: already inside)."""
        if self.top is None or self.top <= 0:
            return None
        return self.price / self.top - 1


def required_at(snapshot: ValuationSnapshot, price: float, scenario: Scenario) -> float | None:
    """Revenue CAGR a price requires, from a stored snapshot's balance sheet."""
    runrate = snapshot.revenue_runrate
    if not runrate or runrate <= 0 or price <= 0 or snapshot.shares_outstanding <= 0:
        return None
    ev = snapshot.shares_outstanding * price - (snapshot.net_cash or 0.0)
    if ev <= 0:
        return None
    revenue = ev / (scenario.terminal_multiple * scenario.fcf_margin)
    return (revenue / runrate) ** (1 / scenario.years) - 1


def price_for(snapshot: ValuationSnapshot, growth: float, scenario: Scenario) -> float | None:
    """The price at which the snapshot requires exactly ``growth``. The inverse."""
    runrate = snapshot.revenue_runrate
    if not runrate or runrate <= 0 or snapshot.shares_outstanding <= 0 or growth <= -1:
        return None
    revenue = runrate * (1 + growth) ** scenario.years
    ev = revenue * scenario.terminal_multiple * scenario.fcf_margin
    price = (ev + (snapshot.net_cash or 0.0)) / snapshot.shares_outstanding
    return price if price > 0 else None


def consensus_growth(snapshot: ValuationSnapshot, consensus) -> tuple[float | None, str]:
    """Annualised growth from the run-rate to the furthest consensus year.

    One number comparable to a CAGR, rather than the provider's single-year
    growth, whose base year may not be the run-rate the valuation uses.
    """
    if consensus is None or not consensus.years or not snapshot.revenue_runrate:
        return None, ""
    last = consensus.years[-1]
    years = (last.fiscal_year_end - snapshot.period_end) / timedelta(days=365.25)
    if years <= 0.25 or last.avg <= 0:
        return None, ""
    growth = (last.avg / snapshot.revenue_runrate) ** (1 / years) - 1
    analysts = f", {last.analysts} analysts" if last.analysts else ""
    return growth, f"consensus {last.label} (${last.avg / 1e9:,.1f}bn{analysts})"


def build_zone(
    snapshot: ValuationSnapshot | None, price: float | None, consensus=None
) -> Zone | None:
    """The zone at ``price``. None when there is no valuation to draw it from."""
    if snapshot is None or not price or price <= 0:
        return None
    base = snapshot.base_case()
    if base is None:
        return None
    scenario = Scenario(
        terminal_multiple=base.terminal_multiple, fcf_margin=base.fcf_margin, years=base.years
    )
    required = required_at(snapshot, price, scenario)
    if required is None:
        return None

    zone = Zone(price=price, required=required, scenario=scenario, stale=snapshot.is_stale())
    if snapshot.revenue_yoy is not None:
        zone.delivered = snapshot.revenue_yoy
        zone.delivered_source = f"revenue YoY, period ending {snapshot.period_end}"
    zone.consensus, zone.consensus_label = consensus_growth(snapshot, consensus)

    refs = [
        (g, label)
        for g, label in (
            (zone.delivered, "delivered"),
            (zone.consensus, "consensus"),
        )
        if g is not None
    ]
    if refs:
        # The lower of the two: a price is in the zone only when it asks for
        # less than both what the company delivers and what analysts expect.
        zone.reference, zone.reference_label = min(refs, key=lambda r: r[0])
        zone.top = price_for(snapshot, zone.reference, scenario)
    else:
        zone.notes.append("no delivered growth or consensus to draw a zone against")

    for step in LADDER_STEPS:
        p = price * (1 + step)
        r = required_at(snapshot, p, scenario)
        if r is not None:
            zone.ladder.append(Rung(price=p, change=step, required=r))
    if zone.stale:
        zone.notes.append(f"the filing behind this is stale (period ending {snapshot.period_end})")
    return zone
