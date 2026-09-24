"""Which candidates the user took, which they didn't, and why.

The scanner records every candidate and what its price did next. That alone
answers "does the setup work". It cannot answer the question the user trades
with: **"are my own choices any good?"** — are the ones I pass on worse than
the ones I take, and which of my reasons for passing cost me money.

Two sources, never guessed:

- **Taken** comes from the broker. An opening fill in either account, on the
  candidate's symbol and session, marks it taken, with the fill price. The
  user never has to log a trade they made.
- **Not taken, and why** comes from the user (`advisor scan skip`). A
  candidate with no fill and no reason stays *untagged* — reported as such,
  never assumed to be a deliberate pass.

Decisions are append-only. A skip recorded at 10:00 and a fill found at
16:20 are both kept; the fill, being newer, is the one that stands.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

from advisor.daemon import market_calendar as mc
from advisor.scanner.models import Candidate, DecisionSource, TradeDecision

logger = logging.getLogger(__name__)

OPENING_ACTIONS = frozenset({"Buy to Open", "Sell to Open"})


@dataclass(frozen=True)
class Fill:
    underlying: str
    symbol: str
    action: str
    quantity: float
    price: float
    executed_at: datetime  # tz-aware
    account: str
    instrument: str  # "Equity" or "Equity Option"


def is_long_entry(f: Fill) -> bool:
    """All three setups are long. Buying shares or buying an option counts.

    Selling to open (a put sale is also a bullish bet) is left out on purpose:
    the user's history has five spreads and no naked sales, and a spread's
    legs would otherwise each count as a separate entry.
    """
    return f.action == "Buy to Open"


def match(candidates: list[Candidate], fills: list[Fill]) -> list[TradeDecision]:
    """A broker-sourced TAKEN decision for each candidate with a long fill.

    Matched on underlying symbol and session date. Timing within the session
    is not required: buying at 09:31 a gap the scanner recorded at 09:35 is
    still taking the setup.
    """
    out = []
    for c in candidates:
        mine = [
            f
            for f in fills
            if f.underlying.upper() == c.symbol.upper()
            and mc.to_et(f.executed_at).date() == c.session
            and is_long_entry(f)
        ]
        if not mine:
            continue
        first = min(mine, key=lambda f: f.executed_at)
        qty = sum(f.quantity for f in mine if f.instrument == first.instrument)
        out.append(
            TradeDecision(
                candidate_id=c.id,
                taken=True,
                source=DecisionSource.BROKER,
                fill_price=first.price,
                fill_at=first.executed_at,
                quantity=qty,
                account=first.account,
                note=first.instrument if first.instrument != "Equity" else "",
            )
        )
    return out


def sync_fills(
    store,
    sessions: list[date],
    *,
    fetch: Callable[[date, date], list[Fill]] | None = None,
) -> int:
    """Record broker fills against those sessions' candidates. Idempotent.

    Returns how many new TAKEN decisions were written. A candidate whose
    standing decision is already a broker fill is not written again.
    """
    if not sessions:
        return 0
    fetch = fetch or fetch_fills
    candidates = [c for day in sessions for c in store.list(session=day, limit=5000)]
    if not candidates:
        return 0
    fills = fetch(min(sessions), max(sessions))
    standing = store.latest_decisions()
    written = 0
    for d in match(candidates, fills):
        prior = standing.get(d.candidate_id)
        if prior is not None and prior.taken and prior.source is DecisionSource.BROKER:
            continue
        store.record_decision(d)
        written += 1
    return written


def _text(raw) -> str:
    """An SDK field as its plain value.

    The SDK returns enums: ``str(OrderAction.BUY_TO_OPEN)`` is
    "OrderAction.BUY_TO_OPEN", not "Buy to Open". The first live run compared
    against ``str()`` and matched none of the user's September trades.
    """
    return str(getattr(raw, "value", raw) or "")


def fill_from_transaction(t, account: str) -> Fill | None:
    """A broker transaction as a Fill, or None if it is not an opening trade."""
    action = _text(getattr(t, "action", None))
    if _text(getattr(t, "transaction_type", None)) != "Trade" or action not in OPENING_ACTIONS:
        return None
    return Fill(
        underlying=str(t.underlying_symbol or t.symbol or ""),
        symbol=str(t.symbol or ""),
        action=action,
        quantity=float(t.quantity or 0),
        price=float(t.price or 0),
        executed_at=t.executed_at,
        account=account,
        instrument=_text(getattr(t, "instrument_type", None)),
    )


def fetch_fills(start: date, end: date) -> list[Fill]:
    """Opening trades in every account between two dates. [] on any failure."""

    async def _get():
        from tastytrade import Account

        from advisor.market.tastytrade_client import get_session

        session = await get_session()
        out = []
        for account in await Account.get(session):
            txs = await account.get_history(
                session, start_date=start, end_date=end, page_offset=None
            )
            for t in txs:
                f = fill_from_transaction(t, account.account_number)
                if f is not None:
                    out.append(f)
        return out

    try:
        return asyncio.run(_get())
    except Exception as exc:  # noqa: BLE001
        logger.warning("journal: broker fills unavailable: %s", exc)
        return []


# ── Review: what passing on things has cost, by setup and by reason ────────


def status(c: Candidate, decision: TradeDecision | None) -> str:
    if decision is None:
        return "untagged"
    if decision.taken:
        return "taken"
    return f"skipped:{decision.reason.value if decision.reason else 'other'}"


REVIEW_HORIZONS = ("close", "next_close", "d5", "d10", "d20")


def review(candidates: list[Candidate], decisions: dict[str, TradeDecision]) -> list[dict]:
    """Outcome statistics per (setup, status). Aggregates only, never a ranking."""
    import statistics
    from collections import defaultdict

    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in candidates:
        groups[(c.setup.value, status(c, decisions.get(c.id)))].append(c)
    rows = []
    for (setup, st), members in sorted(groups.items()):
        row: dict = {"setup": setup, "status": st, "n": len(members)}
        for key in REVIEW_HORIZONS:
            vals = [c.outcomes[key] for c in members if c.outcomes.get(key) is not None]
            row[key] = {
                "n": len(vals),
                "mean": statistics.fmean(vals) if vals else None,
                "positive": sum(v > 0 for v in vals) / len(vals) if vals else None,
            }
        rows.append(row)
    return rows
