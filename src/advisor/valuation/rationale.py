"""One valuation, as the frontend shows it: four cases and the reasoning.

The workstation used to show five pricing panels that disagreed with one
another — a blended "fair price", a DCF bar chart, a Bayesian what-if, peer
multiples and a portfolio "valuation risk" chart built on analyst targets. By
user decision (2026-09-28) they are replaced by one card: the value range and
a rationale for it, built here from the engine's snapshot.

Four cases, and the fourth is not an opinion:

- **bear / base / bull** — the company's own filed margins (lowest, median,
  highest) with today's growth held 0, 3 or 5 years before fading.
- **market** — the price itself, and what it assumes: the same growth path as
  the base case and the steady-state margin that makes the value equal the
  price. It is never used as a scenario — a case built on the price's own
  assumption would value the business at the price, by construction.

The rationale is deterministic. Every sentence is filled from the snapshot,
so every number in it traces to a filing, the broker's close or a stated
assumption; nothing is written by a model.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from advisor.valuation.dcf import GROWTH_BOUNDS, HELD_YEARS, clamp
from advisor.valuation.models import OwnMargin, ValuationSnapshot


class Case(BaseModel):
    name: str  # bear | base | bull | market
    value_per_share: float
    upside: float  # value / price − 1; zero for the market case
    growth: float | None = None  # today's growth, bounded
    held_years: int | None = None
    margin: float | None = None  # steady-state FCF margin
    margin_label: str = ""


class Requirement(BaseModel):
    margin: float
    label: str
    growth: float  # constant revenue growth for ten years the price requires


class PriceRange(BaseModel):
    symbol: str
    asof: date
    price: float
    price_source: str = ""
    live: bool = False  # computed on request, not read from the weekly job
    stale: bool = False
    period_end: date
    market_cap: float
    net_cash: float | None
    enterprise_value: float
    revenue_base: float | None
    revenue_base_label: str = ""
    ev_to_revenue: float | None
    growth: float | None
    growth_label: str = ""
    own_margins: list[OwnMargin] = Field(default_factory=list)
    cases: list[Case] = Field(default_factory=list)
    refused: str | None = None
    requires: list[Requirement] = Field(default_factory=list)
    verdict: str | None = None  # above_bull | above_base | above_bear | below_bear
    rationale: list[str] = Field(default_factory=list)
    assumptions: str = ""
    notes: list[str] = Field(default_factory=list)


def _bn(value: float) -> str:
    return f"${value / 1e9:,.1f}bn"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _market_case(snapshot: ValuationSnapshot) -> Case | None:
    if snapshot.implied_margin is None or snapshot.revenue_yoy is None:
        return None
    return Case(
        name="market",
        value_per_share=snapshot.price,
        upside=0.0,
        growth=clamp(snapshot.revenue_yoy, GROWTH_BOUNDS),
        held_years=HELD_YEARS["base"],
        margin=snapshot.implied_margin,
        margin_label="what the price assumes",
    )


def _verdict(price: float, cases: dict[str, Case]) -> str | None:
    bear, base, bull = cases.get("bear"), cases.get("base"), cases.get("bull")
    if not (bear and base and bull):
        return None
    if price > bull.value_per_share:
        return "above_bull"
    if price > base.value_per_share:
        return "above_base"
    if price >= bear.value_per_share:
        return "above_bear"
    return "below_bear"


_VERDICT_TEXT = {
    "above_bull": (
        "The price is above even the bull case: owning it is a bet on economics the "
        "company has not yet reported."
    ),
    "above_base": (
        "The price sits between the base and bull cases: it needs today's growth to "
        "last and margins toward the best the company has shown."
    ),
    "above_bear": (
        "The price sits between the bear and base cases: it asks no more than the "
        "company's median economics."
    ),
    "below_bear": "The price is below even the bear case.",
}


def _margin_reading(implied: float, own: list[OwnMargin]) -> str:
    positive = sorted(m.value for m in own if m.value > 0)
    if not positive:
        if not own:
            return "The filings carry no margin to compare it with."
        shown = "; ".join(f"{m.value:+.1%} {m.label}" for m in own)
        return f"Every margin it has filed is negative ({shown}): none supports it yet."
    lo, hi = positive[0], positive[-1]
    mid = (
        positive[len(positive) // 2]
        if len(positive) % 2
        else (positive[len(positive) // 2 - 1] + positive[len(positive) // 2]) / 2
    )
    span = f"{lo:.1%}" if lo == hi else f"{lo:.1%} to {hi:.1%}"
    if implied > hi:
        tail = f"the price asks for more than it has ever reported ({implied:.1%} vs {hi:.1%})"
    elif implied > mid:
        tail = "the price asks for more than its median but within what it has shown"
    elif implied >= lo:
        tail = "the price asks for no more than its median"
    else:
        tail = "the price asks for less than even its lowest"
    return f"Its own filed margins run {span}: {tail}."


def _case_phrase(case: Case) -> str:
    held = (
        "growth fading from year one"
        if not case.held_years
        else f"growth held {case.held_years} years"
    )
    return f"{_money(case.value_per_share)} ({case.name}: {held}, {case.margin:.1%} margin)"


def price_range(
    snapshot: ValuationSnapshot, *, live: bool = False, today: date | None = None
) -> PriceRange:
    """The card for one snapshot. Pure."""
    from advisor.valuation.margins import required_range

    cases = [
        Case(
            name=v.name,
            value_per_share=v.value_per_share,
            upside=v.upside,
            growth=v.growth_start,
            held_years=v.held_years,
            margin=v.target_margin,
            margin_label=v.margin_label,
        )
        for v in snapshot.value
    ]
    market = _market_case(snapshot)
    if market is not None:
        cases.append(market)
    by_name = {c.name: c for c in cases}
    readings, _ = required_range(snapshot)

    out = PriceRange(
        symbol=snapshot.symbol,
        asof=snapshot.asof,
        price=snapshot.price,
        price_source=snapshot.price_source,
        live=live,
        stale=snapshot.is_stale(today),
        period_end=snapshot.period_end,
        market_cap=snapshot.market_cap,
        net_cash=snapshot.net_cash,
        enterprise_value=snapshot.enterprise_value,
        revenue_base=snapshot.revenue_base,
        revenue_base_label=snapshot.revenue_base_label,
        ev_to_revenue=snapshot.ev_to_revenue,
        growth=snapshot.revenue_yoy,
        growth_label=snapshot.revenue_growth_label,
        own_margins=list(snapshot.own_margins),
        cases=cases,
        refused=snapshot.value_refused,
        requires=[Requirement(margin=r.margin, label=r.label, growth=r.required) for r in readings],
        verdict=_verdict(snapshot.price, by_name),
        assumptions=snapshot.assumptions_text(),
        notes=list(snapshot.notes),
    )
    out.rationale = _rationale(snapshot, out, market, by_name)
    return out


def _rationale(snapshot, out: PriceRange, market: Case | None, by_name) -> list[str]:
    lines = []
    cash = snapshot.net_cash or 0.0
    pays = (
        f"At {_money(snapshot.price)} the market values the business at "
        f"{_bn(snapshot.enterprise_value)}"
        + (f", net of {_bn(abs(cash))} {'cash' if cash >= 0 else 'debt'}" if cash else "")
    )
    if snapshot.revenue_base and snapshot.ev_to_revenue:
        pays += f": {snapshot.ev_to_revenue:.1f}× its {_bn(snapshot.revenue_base)} of revenue"
        if snapshot.revenue_yoy is not None:
            pays += f", which grew {snapshot.revenue_yoy:+.1%}"
    lines.append(pays + ".")

    if market is not None:
        lines.append(
            f"That price earns a {snapshot.discount_rate or 0:.0%} return if growth of "
            f"{market.growth:+.1%} holds {market.held_years} years and fades to "
            f"{snapshot.terminal_growth or 0:.0%} by year ten, and the business settles "
            f"at a {market.margin:.1%} free-cash-flow margin."
        )
        lines.append(_margin_reading(market.margin, snapshot.own_margins))
    elif snapshot.revenue_yoy is None:
        lines.append(
            "Without a year-over-year revenue comparison, the margin the price assumes "
            "cannot be read."
        )
    elif snapshot.implied_margin_label:
        lines.append(f"At that growth, {snapshot.implied_margin_label}.")

    bear, base, bull = (by_name.get(n) for n in ("bear", "base", "bull"))
    if bear and base and bull:
        lines.append(
            "On its own margins the business is worth "
            f"{_case_phrase(bear)}, {_case_phrase(base)} and {_case_phrase(bull)}."
        )
        lines.append(_VERDICT_TEXT[out.verdict])
    elif snapshot.value_refused:
        lines.append(f"No value range: {snapshot.value_refused}.")
    if out.stale:
        lines.append(
            f"The balance sheet is from {snapshot.period_end}; the figures may describe "
            "a different business."
        )
    return lines
