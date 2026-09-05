"""Position mechanics: deterministic checks over the book.

These are the cheapest and highest-signal events in the system — pure
arithmetic over data already fetched, no news, no LLM, no paid feed. Two
families:

- **Diff events** — what changed since the last snapshot: positions opened,
  closed or resized, and thresholds *crossed*.
- **State events** — standing conditions that are true right now (too
  concentrated, deeply underwater).

Thresholds are **edge-triggered, never level-triggered**. This distinction was
forced by live data: measured as a level, a book held through a drawdown fires
a stop alert on every position on every sweep, forever. A stop is only news at
the moment price crosses it, so crossings are computed against the previous
snapshot and standing losses are reported once a week in the digest instead.

Thresholds live in ``MechanicsLimits`` and are the numbers the user was asked
to confirm; every one of them is a starting default, not a backtested edge.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from advisor.daemon.book import BookSnapshot, Position
from advisor.daemon.models import Event, EventSource, EventTier

logger = logging.getLogger(__name__)


class MechanicsLimits(BaseModel):
    """Tunable trigger levels. Conservative starting points, not backtested."""

    equity_stop_pct: float = -0.08  # -8% from entry
    profit_target_pct: float = 0.25  # +25% from entry on equity
    concentration_pct: float = 0.20  # one underlying > 20% of net liq
    size_change_pct: float = 0.05  # ignore quantity drift below 5%
    drawdown_warn_pct: float = -0.20  # deep-loss review threshold


# ── Diff events: what changed since the last snapshot ────────────────────────


def diff_events(
    previous: BookSnapshot | None,
    current: BookSnapshot,
    *,
    limits: MechanicsLimits | None = None,
) -> list[Event]:
    """Events describing how the book changed between two snapshots.

    The first ever run has no baseline; it emits nothing rather than reporting
    every existing holding as newly opened.
    """
    limits = limits or MechanicsLimits()
    if previous is None:
        return []

    events: list[Event] = []
    before, after = previous.by_key(), current.by_key()
    session = current.as_of.date().isoformat()

    for key, position in after.items():
        if key not in before:
            events.append(
                _event(
                    "POSITION_OPENED",
                    EventTier.B,
                    position,
                    dedup_key=f"{key}:{session}:open",
                    quantity=position.quantity,
                    price=position.price,
                    notional=round(position.notional, 2),
                )
            )
            continue

        prior = before[key]
        if prior.quantity == position.quantity:
            continue
        # Ignore trivial drift (fractional share reinvestment, rounding).
        base = abs(prior.quantity) or 1.0
        if abs(position.quantity - prior.quantity) / base < limits.size_change_pct:
            continue
        events.append(
            _event(
                "POSITION_SIZE_CHANGED",
                EventTier.B,
                position,
                dedup_key=f"{key}:{session}:{prior.quantity}->{position.quantity}",
                from_quantity=prior.quantity,
                to_quantity=position.quantity,
                direction="increased"
                if abs(position.quantity) > abs(prior.quantity)
                else "reduced",
            )
        )

    for key, prior in before.items():
        if key not in after:
            events.append(
                _event(
                    "POSITION_CLOSED",
                    EventTier.B,
                    prior,
                    dedup_key=f"{key}:{session}:close",
                    quantity=prior.quantity,
                    realized_pct=round(prior.unrealized_pct, 4),
                )
            )

    return events


# ── State events: what is true right now ─────────────────────────────────────


def state_events(book: BookSnapshot, *, limits: MechanicsLimits | None = None) -> list[Event]:
    """Standing conditions — never Tier A.

    Nothing here is time-critical by construction: these are facts that stay
    true for days, so they belong in a digest. Threshold *crossings* are in
    :func:`diff_events`.

    Deep drawdowns dedup by ISO week, not by day: being 36% underwater is worth
    a weekly reminder, not a daily one.
    """
    limits = limits or MechanicsLimits()
    events: list[Event] = []
    year, week, _ = book.as_of.isocalendar()
    period = f"{year}W{week:02d}"
    session = book.as_of.date().isoformat()

    for position in book.positions:
        if position.is_option or not position.avg_open_price or not position.price:
            continue
        pct = position.unrealized_pct
        if pct <= limits.drawdown_warn_pct:
            events.append(
                _event(
                    "DEEP_DRAWDOWN",
                    EventTier.B,
                    position,
                    dedup_key=f"{position.key()}:{period}:drawdown",
                    unrealized_pct=round(pct, 4),
                    threshold=limits.drawdown_warn_pct,
                    entry=position.avg_open_price,
                    price=position.price,
                    unrealized_usd=round(position.unrealized_pnl, 2),
                )
            )

    # Concentration is a book-level property, measured per underlying across
    # accounts rather than per position.
    if book.net_liq > 0:
        for underlying, notional in book.exposure_by_underlying().items():
            weight = notional / book.net_liq
            if weight >= limits.concentration_pct:
                events.append(
                    Event(
                        source=EventSource.COMPUTED,
                        kind="CONCENTRATION_WARNING",
                        tier=EventTier.B,
                        symbol=underlying,
                        dedup_key=f"{underlying}:{session}:concentration",
                        payload={
                            "weight": round(weight, 4),
                            "threshold": limits.concentration_pct,
                            "notional": round(notional, 2),
                            "net_liq": round(book.net_liq, 2),
                        },
                    )
                )

    return events


def crossing_events(
    previous: BookSnapshot,
    current: BookSnapshot,
    *,
    limits: MechanicsLimits | None = None,
) -> list[Event]:
    """Threshold crossings between two snapshots — the only Tier A source here.

    Fires when a position moves from one side of a level to the other. A
    position already through the level stays silent, which is what keeps a
    drawn-down book from screaming on every sweep.
    """
    limits = limits or MechanicsLimits()
    events: list[Event] = []
    before, after = previous.by_key(), current.by_key()
    session = current.as_of.date().isoformat()

    for key, position in after.items():
        prior = before.get(key)
        if prior is None or position.is_option:
            continue
        if not (prior.avg_open_price and prior.price and position.price):
            continue

        was, now = prior.unrealized_pct, position.unrealized_pct

        if was > limits.equity_stop_pct >= now:
            events.append(
                _event(
                    "STOP_BREACHED",
                    EventTier.A,
                    position,
                    dedup_key=f"{key}:{session}:stop",
                    unrealized_pct=round(now, 4),
                    previous_pct=round(was, 4),
                    threshold=limits.equity_stop_pct,
                    entry=position.avg_open_price,
                    price=position.price,
                    unrealized_usd=round(position.unrealized_pnl, 2),
                )
            )
        elif was < limits.profit_target_pct <= now:
            events.append(
                _event(
                    "PROFIT_TARGET_HIT",
                    EventTier.B,
                    position,
                    dedup_key=f"{key}:{session}:target",
                    unrealized_pct=round(now, 4),
                    previous_pct=round(was, 4),
                    threshold=limits.profit_target_pct,
                    unrealized_usd=round(position.unrealized_pnl, 2),
                )
            )

    return events


def _event(kind: str, tier: EventTier, position: Position, *, dedup_key: str, **payload) -> Event:
    return Event(
        source=EventSource.COMPUTED,
        kind=kind,
        tier=tier,
        symbol=position.underlying,
        dedup_key=dedup_key,
        payload={"account": position.account, "instrument": position.instrument, **payload},
    )
