"""Keeping implied expectations current, and saying when they move.

The valuation gap is not a number you compute once. Revenue changes each
quarter, the price changes every day, and the growth the price requires moves
with both. Recomputing it turns a one-off analysis into a monitored quantity —
which is the only form in which it can be part of a thesis.

Events are **edge-triggered**, like everything else in this daemon. A price
that drifts does not deserve a notification; a price that has moved enough to
change what the business must deliver by two percentage points of decade-long
growth does. This repo learned that lesson expensively in phase 2a, when
level-triggered thresholds fired nine interrupts at once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.valuation.implied import build_snapshot
from advisor.valuation.models import ValuationSnapshot

logger = logging.getLogger(__name__)

# Two percentage points of required decade CAGR. Below this the price has
# drifted; above it, what the business must become has genuinely changed.
MATERIAL_CAGR_SHIFT = 0.02


@dataclass
class ValuationResult:
    snapshots: list[ValuationSnapshot] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"{len(self.snapshots)} valued"]
        if self.events:
            parts.append(f"{len(self.events)} shift(s)")
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        return ", ".join(parts)


def shift_event(previous: ValuationSnapshot | None, current: ValuationSnapshot) -> Event | None:
    """A Tier B event when the required growth has moved materially.

    Tier B, never A: a valuation is a standing condition, not an action with a
    deadline. If it breaks a stated invalidation, the thesis layer is what
    says so — the tier of this event is not where that judgement belongs.
    """
    base = current.base_case()
    if base is None:
        return None
    if previous is None:
        return None  # nothing to compare against; the first reading is not news

    prior = previous.base_case()
    if prior is None:
        return None

    delta = base.implied_cagr - prior.implied_cagr
    if abs(delta) < MATERIAL_CAGR_SHIFT:
        return None

    return Event(
        source=EventSource.COMPUTED,
        kind="IMPLIED_EXPECTATIONS_SHIFT",
        tier=EventTier.B,
        symbol=current.symbol,
        dedup_key=f"{current.symbol}:{current.asof.isoformat()}",
        payload={
            "implied_cagr": round(base.implied_cagr, 4),
            "previous_implied_cagr": round(prior.implied_cagr, 4),
            "change": round(delta, 4),
            "direction": "harder" if delta > 0 else "easier",
            "ev_to_revenue": round(current.ev_to_revenue, 2) if current.ev_to_revenue else None,
            "enterprise_value": current.enterprise_value,
            "revenue_runrate": current.revenue_runrate,
            "price": current.price,
            "required_revenue": base.required_revenue,
            "terminal_multiple": base.terminal_multiple,
            "fcf_margin": base.fcf_margin,
            "years": base.years,
            "as_of_filing": current.source_accession,
            "period_end": current.period_end.isoformat(),
        },
    )


async def refresh_valuations(
    store: DaemonStore, symbols: list[str], prices: dict[str, float], *, asof: date | None = None
) -> ValuationResult:
    """Recompute implied expectations for each symbol and emit any shifts."""
    from advisor.valuation.fundamentals import latest_fundamentals

    result = ValuationResult()
    today = asof or date.today()

    for symbol in symbols:
        price = prices.get(symbol.upper())
        if not price:
            result.skipped[symbol] = "no price"
            continue
        try:
            fundamentals = latest_fundamentals(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("valuation: %s failed: %s", symbol, exc)
            result.skipped[symbol] = str(exc)[:120]
            continue
        if fundamentals is None:
            result.skipped[symbol] = "no usable filing"
            continue

        snapshot = build_snapshot(fundamentals, price, asof=today)
        if snapshot is None:
            result.skipped[symbol] = f"missing {', '.join(fundamentals.missing)}"
            continue

        previous = store.load_latest_valuation(symbol, before=today)
        store.save_valuation(snapshot)
        result.snapshots.append(snapshot)

        event = shift_event(previous, snapshot)
        if event is not None and store.emit(event):
            result.events.append(event)

    return result
