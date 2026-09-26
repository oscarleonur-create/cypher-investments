"""An entry proposal: what to do with one watched name today, sized and bounded.

Built from the daily sheet, the book, and (optionally) the model's reading.
Every rule here is explicit so every proposal can be tracked and a rule that
keeps losing can be found and changed. The user's decisions (2026-09-25):
the zone is relative (P/S at or below its own two-year median), the risk
budget is 2–3% of net liq per entry, and the watchlist's names are dual —
traded short-term and held long — so a proposal carries two legs.

**Position leg** (months). Proposed when the name is in zone *and* something
happened today: it crossed into the zone after at least five sessions above
it, it fell 2σ or more while in the zone, or the scanner recorded a setup
on it. In zone with nothing happening
is IN_ZONE: an acceptable price, no reason to act *today*.

- Stop: two weeks of 2σ daily moves, 2·σ·√10, kept between 8% and 25%.
- Risk: 2% of net liq; 3% if P/S is in its cheapest quarter of two years.
- Review a trim above the price at its two-year 80th percentile P/S.

**Trade leg** (hours to the next session). Proposed only on a scanner setup
(A, B or C) today — the one short-term edge the user's own history measured.

- Stop: 1.5 daily σ below entry.
- Time stop: out by the next session's close (held overnight trades paid on
  net, measured 2026-09-23; longer was never measured).
- Risk: 2% of net liq.

**Thesis.** Where the user has written a long-term thesis and none of its
rules is broken or standing, the position leg gets one more point of risk
and the total cap rises to 4%. Where a rule is broken or standing, the
proposal waits: the user's own rule says something is wrong.

**Limits.** Total risk at most 3% of net liq (4% with an intact thesis). A
position leg may not take the name past 20% of the book (the book's own
concentration limit).

**Blockers** turn a proposal into WAIT: a tier-A event on the name today
(read it first), or a model reading of AT_RISK. A CONSTRUCTIVE reading adds
nothing: the reading leans on the scorecard's required growth, which moves
six points with the assumed margin (META -0.4% to +5.7% on 2026-09-25), too
fragile to size a position on (user decision, 2026-09-25). AT_RISK still
blocks, because waiting is the cautious direction.

Nothing here is a view of what the company is worth. The zone is a
comparison with the name's own history; the stops are its own volatility;
the size is the user's own budget.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from advisor.entry.ruleset import entry_rules
from advisor.entry.sheet import Sheet
from advisor.learning.rules import Origin, RuleStamp

# User decision, 2026-09-25: base risk 2-3% of net liq per entry (raised from
# 1-3% after 1% sized AMZN/CRDO at one share and META at none on a ~$8k book).
TRADE_RISK = 0.02
POSITION_RISK = 0.02
POSITION_RISK_CHEAP = 0.03
CHEAP_PERCENTILE = 0.25  # "cheap": P/S in the cheapest quarter of its two years
MAX_TOTAL_RISK = 0.03
# User decision, 2026-09-25: more risk where a long-term thesis is written and
# intact (no rule broken or standing) — one point more, up to 4% in total.
THESIS_BONUS = 0.01
MAX_TOTAL_RISK_THESIS = 0.04
BOOK_LIMIT = 0.20  # MechanicsLimits.concentration_pct
POSITION_STOP_MIN, POSITION_STOP_MAX = 0.08, 0.25
# The position stop is two weeks of 2σ daily moves: 2·σ·√10.
POSITION_STOP_SIGMAS = 2.0
POSITION_STOP_SESSIONS = 10
TRADE_STOP_SIGMAS = 1.5
DIP_SIGMAS = 2.0
# An entry into the zone counts only after this many sessions above it, so a
# price hovering at the median does not fire ENTER every other day.
ENTRY_CONFIRM_SESSIONS = 5


class Action(StrEnum):
    ENTER = "ENTER"  # not held; at least one leg qualifies
    ADD = "ADD"  # held; the position leg qualifies within the book limit
    IN_ZONE = "IN_ZONE"  # acceptable price, nothing happened today
    WAIT = "WAIT"  # a leg would qualify, but something must be read first
    NONE = "NONE"  # out of zone and no setup
    CANNOT_SAY = "CANNOT_SAY"  # no price, or neither a zone nor a setup to judge by


class Leg(BaseModel):
    horizon: str  # "trade" | "position"
    entry: float
    stop: float
    stop_basis: str
    risk_pct: float
    shares: int
    notional: float
    exit_rules: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Reason(BaseModel):
    text: str
    source: str  # where the fact comes from


class Proposal(BaseModel):
    symbol: str
    session: date
    built_at: datetime
    action: Action
    price: float | None = None
    triggers: list[str] = Field(default_factory=list)
    reasons: list[Reason] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    legs: list[Leg] = Field(default_factory=list)
    stance: str | None = None  # the model reading's stance, when one was read
    reading: list[str] = Field(default_factory=list)  # its sentences
    gaps: list[str] = Field(default_factory=list)
    net_liq: float | None = None
    # What it was decided on, as numbers, and by which rules: what a review
    # groups by and a replay compares against. None on pre-registry rows.
    features: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    rules: RuleStamp | None = None
    origin: Origin = Origin.LIVE
    outcomes: dict[str, float | None] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.session.isoformat()}:{self.symbol}:{self.action.value}"

    @property
    def risk_pct(self) -> float:
        return sum(leg.risk_pct for leg in self.legs)


def position_stop_pct(sigma: float | None) -> float | None:
    if not sigma or sigma <= 0:
        return None
    pct = POSITION_STOP_SIGMAS * sigma * math.sqrt(POSITION_STOP_SESSIONS)
    return min(max(pct, POSITION_STOP_MIN), POSITION_STOP_MAX)


def _size(net_liq: float | None, risk: float, entry: float, stop: float) -> tuple[int, float]:
    if not net_liq or net_liq <= 0 or entry <= stop:
        return 0, 0.0
    shares = math.floor(net_liq * risk / (entry - stop))
    return max(shares, 0), max(shares, 0) * entry


def triggers_for(sheet: Sheet) -> list[str]:
    out = []
    if sheet.candidates:
        out.append(f"scanner setup today: {', '.join(sheet.candidates)}")
    z, prev = sheet.zone, sheet.zone_prev
    if (
        z is not None
        and z.in_zone
        and prev is not None
        and not prev.in_zone
        and z.sessions_above >= ENTRY_CONFIRM_SESSIONS
    ):
        out.append(
            f"entered the zone today after {z.sessions_above} sessions above it: "
            f"P/S {z.ps_now:.2f}x ≤ 2y median {z.median:.2f}x (was {prev.ps_now:.2f}x)"
        )
    m = sheet.move
    if z is not None and z.in_zone and m is not None and m.z is not None and m.z <= -DIP_SIGMAS:
        out.append(f"fell {m.day:+.1%} today ({m.z:+.1f}σ) inside the zone")
    return out


def features_of(sheet: Sheet) -> dict[str, float | int | bool | str | None]:
    """The inputs a proposal is decided on, as numbers. Pure.

    The reasons tell the user the same things in prose; these are what a
    review groups by and a replay compares. Only what the sheet held — nothing
    derived from the decision itself, so a feature can never explain the
    outcome by restating the action.
    """
    m, z, prev, c = sheet.move, sheet.zone, sheet.zone_prev, sheet.context
    setups = sorted({cid.split(":")[1] for cid in sheet.candidates if cid.count(":") >= 2})
    return {
        "price": m.price if m else None,
        "day": m.day if m else None,
        "d5": m.d5 if m else None,
        "d20": m.d20 if m else None,
        "sigma": m.sigma if m else None,
        "move_z": m.z if m else None,
        "in_zone": z.in_zone if z else None,
        "prev_in_zone": prev.in_zone if prev else None,
        "ps": z.ps_now if z else None,
        "ps_median": z.median if z else None,
        "ps_percentile": z.percentile if z else None,
        "zone_distance": z.distance if z else None,
        "sessions_above": z.sessions_above if z else None,
        "zone_observations": z.observations if z else None,
        "setups": ",".join(setups) or None,
        "events_today": len(sheet.events_today),
        "tier_a_today": sum(e.tier == "A" for e in sheet.events_today),
        "events_week": sheet.events_week,
        "held": sheet.holding is not None,
        "weight": sheet.holding.weight if sheet.holding else 0.0,
        "thesis": sheet.thesis,
        "delivered_growth": c.delivered if c else None,
        "consensus_growth": c.consensus if c else None,
        "required_low": c.low if c else None,
        "required_high": c.high if c else None,
    }


def build_proposal(
    sheet: Sheet,
    *,
    net_liq: float | None,
    reading=None,
) -> Proposal:
    """One proposal from one sheet. Pure: the reading, if any, is passed in."""
    m, z = sheet.move, sheet.zone
    p = Proposal(
        symbol=sheet.symbol,
        session=sheet.built_at.date(),
        built_at=sheet.built_at,
        action=Action.CANNOT_SAY,
        price=m.price if m else None,
        gaps=list(sheet.gaps),
        net_liq=net_liq,
        features=features_of(sheet),
        rules=entry_rules(),
    )
    if reading is not None and getattr(reading, "stance", None) is not None:
        p.stance = reading.stance.value
        p.reading = [s.text for s in reading.sentences]
    if m is None:
        p.blockers.append("no price")
        return p
    if z is None and not sheet.candidates:
        return p  # CANNOT_SAY: no zone to judge a position, no setup for a trade

    p.triggers = triggers_for(sheet)

    # Reasons, each with its source.
    if z is not None:
        where = "at or below" if z.in_zone else f"{z.distance:+.1%} above"
        p.reasons.append(
            Reason(
                text=(
                    f"P/S {z.ps_now:.1f}x, {where} its 2-year median {z.median:.1f}x "
                    f"(percentile {z.percentile:.0%}); zone top ${z.top:,.2f}"
                    + (f" [short history: {z.observations} sessions]" if z.short else "")
                ),
                source=f"{z.source} revenue and diluted shares as known each day; "
                "yfinance closes",
            )
        )
        if z.short:
            p.gaps.append(
                f"zone from {z.observations} sessions (under a year): shown, not used to "
                "open a position"
            )
    c = sheet.context
    if c is not None and c.delivered is not None:
        p.reasons.append(
            Reason(
                text=f"revenue growing {c.delivered:+.1%} YoY"
                + (f"; {c.consensus_label} implies {c.consensus:+.1%}/yr" if c.consensus else ""),
                source="latest filing XBRL; yfinance consensus",
            )
        )
    if sheet.events_today:
        p.reasons.append(
            Reason(
                text=f"{len(sheet.events_today)} event(s) since the previous close",
                source="event stream",
            )
        )

    # Blockers.
    if any(e.tier == "A" for e in sheet.events_today):
        p.blockers.append("a tier-A event on this name today: read it before entering")
    if p.stance == "AT_RISK":
        p.blockers.append("the reading of the recent facts is AT_RISK")
    if sheet.thesis == "broken":
        p.blockers.append(
            "a rule of your thesis is broken or standing: " + "; ".join(sheet.thesis_broken[:3])
        )
    elif sheet.thesis == "intact":
        p.reasons.append(
            Reason(
                text="your long-term thesis is written and intact",
                source="thesis claims (action card)",
            )
        )
    cap = MAX_TOTAL_RISK_THESIS if sheet.thesis == "intact" else MAX_TOTAL_RISK

    weight = sheet.holding.weight if sheet.holding else 0.0
    held = sheet.holding is not None
    sigma = m.sigma
    sized = bool(net_liq and net_liq > 0)

    def unfilled_note(leg: Leg) -> None:
        # Whether to enter does not depend on whether the book can size it.
        # A leg that rounds to zero shares is still proposed, with the reason.
        if not sized or leg.shares > 0:
            return
        one = leg.entry - leg.stop
        budget = (net_liq or 0) * leg.risk_pct
        leg.notes.append(
            f"one share risks ${one:,.0f}, above the {leg.risk_pct:.0%} budget of "
            f"${budget:,.0f}: the smallest position exceeds it"
        )

    # Position leg.
    position_ok = z is not None and z.in_zone and not z.short and bool(p.triggers)
    if position_ok:
        stop_pct = position_stop_pct(sigma)
        if stop_pct is None:
            p.gaps.append("no volatility estimate: position stop cannot be set")
        else:
            risk = POSITION_RISK_CHEAP if z.percentile <= CHEAP_PERCENTILE else POSITION_RISK
            if sheet.thesis == "intact":
                risk += THESIS_BONUS
            risk = min(risk, cap)
            stop = m.price * (1 - stop_pct)
            shares, notional = _size(net_liq, risk, m.price, stop)
            leg = Leg(
                horizon="position",
                entry=m.price,
                stop=stop,
                stop_basis=(
                    f"{POSITION_STOP_SIGMAS:g}·σ·√{POSITION_STOP_SESSIONS} = {stop_pct:.1%} "
                    f"below entry (σ {sigma:.2%}/day), kept in "
                    f"{POSITION_STOP_MIN * 100:.0f}–{POSITION_STOP_MAX:.0%}"
                ),
                risk_pct=risk,
                shares=shares,
                notional=notional,
                exit_rules=[
                    f"stop at ${stop:,.2f}",
                    f"review a trim above ${z.p80_price:,.2f} "
                    "(P/S at its 2-year 80th percentile)",
                ],
            )
            at_limit = held and weight >= BOOK_LIMIT
            if sized and not at_limit:
                room = max(BOOK_LIMIT - weight, 0.0) * net_liq
                if room < m.price:
                    # Not at the limit: one share is simply larger than the room
                    # left under it (a $2,000 share on a $7,966 book).
                    leg.notes.append(
                        f"one share (${m.price:,.0f}) is more than the ${room:,.0f} left "
                        "under the 20% book limit"
                    )
                    leg.shares, leg.notional = 0, 0.0
                elif notional > room:
                    capped = math.floor(room / m.price)
                    leg.notes.append(
                        f"capped from {shares} to {capped} shares by the 20% book limit"
                    )
                    leg.shares, leg.notional = capped, capped * m.price
            if at_limit:
                p.blockers.append(f"already {weight:.1%} of the book, at the 20% limit")
            else:
                unfilled_note(leg)
                p.legs.append(leg)

    # Trade leg.
    if sheet.candidates and sigma:
        stop = m.price * (1 - TRADE_STOP_SIGMAS * sigma)
        used = sum(leg.risk_pct for leg in p.legs)
        risk = min(TRADE_RISK, cap - used)
        if risk > 0:
            shares, notional = _size(net_liq, risk, m.price, stop)
            leg = Leg(
                horizon="trade",
                entry=m.price,
                stop=stop,
                stop_basis=f"{TRADE_STOP_SIGMAS}·σ = {TRADE_STOP_SIGMAS * sigma:.1%} below entry",
                risk_pct=risk,
                shares=shares,
                notional=notional,
                exit_rules=[f"stop at ${stop:,.2f}", "out by the next session's close"],
            )
            unfilled_note(leg)
            p.legs.append(leg)
        else:
            p.gaps.append(f"no risk budget left for a trade leg (the {cap:.0%} cap is used)")
    elif sheet.candidates:
        p.gaps.append("no volatility estimate: trade stop cannot be set")

    if not sized:
        p.gaps.append("net liq unknown: legs are not sized")

    # Action.
    if p.legs and p.blockers:
        p.action = Action.WAIT
    elif p.legs:
        p.action = Action.ADD if held else Action.ENTER
    elif z is not None and z.in_zone:
        p.action = Action.WAIT if p.blockers else Action.IN_ZONE
    elif z is not None or sheet.candidates:
        p.action = Action.NONE
    return p
