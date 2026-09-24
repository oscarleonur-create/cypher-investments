"""The numbers a holder needs before any sentence about a position.

A first reading of SPCX was four sentences of prose and called the position
"at risk" because the price requires 25.9% growth a year and the holder's
line is 25%. It had no room for the two numbers that change that conclusion:
the business grew 91.9% in its last quarter, and the consensus for next year
is +142%. If that consensus is met, the rest of the decade needs about 14% a
year. The holder asked, fairly, what use a card is without numbers.

So the numbers come first, and they are deterministic: each row is read from
a stored filing, valuation, estimate or claim, says where it came from, and
none is written by a model. Two rules carried over from the valuation module:

- **What a price requires, never what a business is worth.** Consensus is a
  third party's estimate, shown with its analyst count and range, and used
  only in arithmetic of the form "if this is met, then the rest must be".
- **A row that cannot be filled says so.** A threshold nothing can check is
  "can't be checked" with the reason, never a pass.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, Field

from advisor.daemon.mechanics import MechanicsLimits
from advisor.daemon.store import DaemonStore
from advisor.thesis.match import evaluate_claim
from advisor.thesis.models import Comparator
from advisor.thesis.reachability import claim_reachability
from advisor.thesis.state import evaluate_against_state
from advisor.valuation.consensus import load_consensus, remaining_cagr

Status = Literal["OK", "TRIPPED", "UNCHECKABLE"]


class Row(BaseModel):
    label: str
    value: str
    detail: str = ""
    source: str = ""
    status: Status | None = None
    claim: str | None = None  # the holder's words, for a threshold row


class Scorecard(BaseModel):
    symbol: str
    expectations: list[Row] = Field(default_factory=list)
    thresholds: list[Row] = Field(default_factory=list)


def _money(value: float) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.1f}bn"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.0f}M"
    return f"${value:,.0f}"


def _pct(value: float, *, sign: bool = False) -> str:
    return f"{value * 100:{'+' if sign else ''}.1f}%"


# Trigger fields as a holder would name them, and whether they are fractions.
_FIELDS: dict[str, tuple[str, bool]] = {
    "implied_cagr": ("Required growth", True),
    "dilution_pct": ("Dilution", True),
    "offering_pct_of_cap": ("Offering size, % of cap", True),
    "weight": ("Position weight", True),
    "unrealized_pct": ("Return vs entry", True),
    "residual_z": ("Unexplained move (z)", False),
}


def _fmt(field: str, value: float) -> str:
    return _pct(value) if _FIELDS.get(field, ("", True))[1] else f"{value:.2f}"


def _expectations(store: DaemonStore, symbol: str, consensus_loader) -> list[Row]:
    rows: list[Row] = []
    valuation = store.load_latest_valuation(symbol)
    base = valuation.base_case() if valuation else None
    if valuation is None or base is None:
        return [Row(label="Price requires", value="no valuation", detail="nothing stored yet")]

    stale = " · stale filing" if valuation.is_stale() else ""
    rows.append(
        Row(
            label="Price requires",
            value=f"{_pct(base.implied_cagr)}/yr × {base.years}y",
            detail=(
                f"to {_money(base.required_revenue)} revenue · at ${valuation.price:,.2f} on "
                f"{valuation.asof} · {base.terminal_multiple:g}x FCF, {base.fcf_margin:.0%} margin"
            ),
            source=f"valuation {valuation.asof}{stale}",
        )
    )
    if valuation.revenue_runrate:
        rows.append(
            Row(
                label="Run-rate now",
                value=_money(valuation.revenue_runrate),
                detail=f"period ending {valuation.period_end}, annualised",
                source=f"XBRL {valuation.source_accession}",
            )
        )
    if valuation.revenue_yoy is not None:
        rows.append(
            Row(
                label="Growing",
                value=f"{_pct(valuation.revenue_yoy, sign=True)} YoY",
                detail=f"period ending {valuation.period_end} vs the same period a year earlier",
                source=f"XBRL {valuation.source_accession}",
            )
        )

    consensus = consensus_loader(store, symbol)
    if consensus is None:
        rows.append(Row(label="Consensus", value="unavailable", source="yfinance"))
        return rows
    for estimate in consensus.years:
        detail = []
        if estimate.growth is not None:
            detail.append(f"{_pct(estimate.growth, sign=True)} vs prior year")
        if estimate.analysts:
            detail.append(f"{estimate.analysts} analysts")
        if estimate.low is not None and estimate.high is not None:
            detail.append(f"{_money(estimate.low)}–{_money(estimate.high)}")
        rows.append(
            Row(
                label=f"Consensus {estimate.label}",
                value=_money(estimate.avg),
                detail=" · ".join(detail),
                source=f"{consensus.source} {consensus.asof.date()}",
            )
        )

    horizon_end = valuation.period_end + timedelta(days=round(base.years * 365.25))
    last = consensus.years[-1] if consensus.years else None
    remaining = remaining_cagr(base.required_revenue, horizon_end, last) if last else None
    if last and remaining:
        cagr, years = remaining
        rows.append(
            Row(
                label=f"If {last.label} holds",
                value=f"{_pct(cagr)}/yr",
                detail=(
                    f"needed for the remaining {years:.1f} years to reach "
                    f"{_money(base.required_revenue)} by {horizon_end.year}"
                ),
                source="arithmetic on the two rows above",
            )
        )
    return rows


def _threshold_rows(store: DaemonStore, symbol: str) -> list[Row]:
    rows: list[Row] = []
    book = store.load_latest_book()
    events = store.recent_events(symbol=symbol, limit=1000)
    earliest = min((e.ts for e in events), default=None)

    for claim in store.load_claims(symbol):
        trigger = claim.trigger
        if trigger.field is None or trigger.threshold is None:
            continue  # untestable or HAPPENS-only claims are not numbers
        name, _ = _FIELDS.get(trigger.field, (trigger.field, True))
        op = "≤" if trigger.comparator is Comparator.ABOVE else "≥"
        label = f"{name} {op} {_fmt(trigger.field, trigger.threshold)}"
        base = dict(label=label, claim=claim.text)

        reach = claim_reachability(store, symbol, claim)
        if reach.blocked:
            rows.append(
                Row(**base, value="can't be checked", detail=reach.reason, status="UNCHECKABLE")
            )
            continue

        standing = evaluate_against_state(store, symbol, claim, book) if book else None
        if standing is not None and standing.observed is not None:
            rows.append(
                Row(
                    **base,
                    value=_fmt(trigger.field, standing.observed),
                    detail="current state",
                    status="TRIPPED" if standing.tripped else "OK",
                )
            )
            continue

        # An event-type claim: the latest event that tested it, if any.
        tested = next(
            (
                (event, result)
                for event in events
                if (result := evaluate_claim(claim, event)) is not None
            ),
            None,
        )
        if tested is None:
            since = f"since {earliest.date()}" if earliest else "nothing archived yet"
            rows.append(
                Row(**base, value="none", detail=f"no qualifying event {since}", status="OK")
            )
            continue
        event, result = tested
        observed = result.observed
        rows.append(
            Row(
                **base,
                value=_fmt(trigger.field, observed) if observed is not None else "occurred",
                detail=f"latest: {event.kind.replace('_', ' ').lower()} on {event.ts.date()}",
                status="TRIPPED" if result.tripped else "OK",
            )
        )

    # The book's own limit, not the holder's claim — shown because it is the
    # other line a position can cross.
    if book and book.net_liq:
        held = [p for p in book.positions if p.underlying.upper() == symbol]
        if held:
            weight = sum(p.signed_notional for p in held) / book.net_liq
            limit = MechanicsLimits().concentration_pct
            rows.append(
                Row(
                    label=f"Position weight ≤ {_pct(limit)}",
                    value=_pct(weight),
                    detail=f"book limit · as of {book.as_of.date()}",
                    status="TRIPPED" if weight > limit else "OK",
                )
            )
    return rows


def build_scorecard(
    store: DaemonStore, symbol: str, *, consensus_loader=load_consensus
) -> Scorecard:
    symbol = symbol.upper()
    return Scorecard(
        symbol=symbol,
        expectations=_expectations(store, symbol, consensus_loader),
        thresholds=_threshold_rows(store, symbol),
    )


def scorecard_facts(card: Scorecard) -> list[str]:
    """Each row as one line of text, for the reading's fact list."""
    lines = []
    for row in card.expectations:
        lines.append(f"{row.label}: {row.value} ({row.detail}) [{row.source}]".replace(" ()", ""))
    for row in card.thresholds:
        verdict = {"OK": "within", "TRIPPED": "BREACHED", "UNCHECKABLE": "cannot be checked"}.get(
            row.status or "", ""
        )
        lines.append(f"Threshold {row.label}: {row.value}, {verdict} ({row.detail})")
    return lines
