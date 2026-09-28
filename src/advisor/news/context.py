"""What the news agent knows about a name before it reads the news. Deterministic.

A judgment without context is a judgment of the headline. The agent could not
say whether a $52M insider sale is large (it is 0.003% of SpaceX), whether the
position it touches is 15% of the book and 4.7% from its stop, whether the
market had already moved on it, or whether "new" was new. Each line here is
computed from the store and the price history — no model — and numbered, so
the agent can cite it and the gate can check every number against it.

- **Position**: held or not, weight, cost, unrealized, the exit stop from cost.
- **Size**: market cap, revenue run-rate and growth, from the latest valuation.
- **Market**: per item, the close-to-close move on the session it could first
  trade on, in the name's own daily sigma — did the market already read it?
- **History**: the name's filings and material events of the last 30 days, and
  the headlines already seen, so "new" is judged against something.
- **Thesis**: the user's claims, with their kind.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from datetime import date, datetime, timedelta

from advisor.daemon import market_calendar as mc

HISTORY_DAYS = 30
SIGMA_SESSIONS = 60
MAX_HISTORY_LINES = 12
# The book's own alarms: the position line already states where the position
# stands. Shown as history, "stop breached" (the fixed -8% alarm) led the agent
# to write "the stop breached" for CBRS, 26.8% above its real exit stop.
_MECHANICS = frozenset(
    {"STOP_BREACHED", "DEEP_DRAWDOWN", "CONCENTRATION_WARNING", "STOP_RECOVERED"}
)


def _money(x: float) -> str:
    a = abs(x)
    if a >= 1e12:
        return f"${x / 1e12:.2f}T"
    if a >= 1e9:
        return f"${x / 1e9:.2f}bn"
    if a >= 1e6:
        return f"${x / 1e6:.1f}M"
    return f"${x:,.0f}"


def position_line(store, symbol: str, sigma: float | None) -> str:
    book = store.load_latest_book()
    if book is None:
        return "Position: unknown (no book snapshot)."
    held = [p for p in book.positions if p.underlying.upper() == symbol and not p.is_option]
    if not held:
        return "Position: not held (watchlist)."
    qty = sum(p.quantity for p in held)
    basis = sum(p.cost_basis for p in held)
    price = held[0].price
    parts = [f"Position: held, {qty:,.0f} shares"]
    if book.net_liq:
        parts.append(f"{sum(p.notional for p in held) / book.net_liq * 100:.1f}% of the book")
    if basis and qty:
        cost = basis / qty
        parts.append(f"average cost ${cost:,.2f}, price ${price:,.2f} ({price / cost - 1:+.1%})")
        from advisor.entry.proposal import position_stop_pct

        pct = position_stop_pct(sigma)
        if pct is not None:
            stop = cost * (1 - pct)
            where = "already below it" if price <= stop else f"{price / stop - 1:.1%} above it"
            parts.append(f"exit stop ${stop:,.2f} ({pct:.0%} below cost), {where}")
    return ", ".join(parts) + f"; as of {mc.to_et(book.as_of).date()}."


def size_line(store, symbol: str) -> str | None:
    snap = store.load_latest_valuation(symbol)
    if snap is None:
        return None
    parts = [f"Company size ({snap.asof}): market cap {_money(snap.market_cap)}"]
    if snap.revenue_runrate:
        parts.append(f"revenue run-rate {_money(snap.revenue_runrate)}")
    if snap.revenue_yoy is not None:
        parts.append(f"revenue growth {snap.revenue_yoy:+.1%} YoY")
    parts.append(f"{snap.shares_outstanding / 1e6:,.1f}M shares outstanding")
    return ", ".join(parts) + "."


def sigma_of(closes: list[tuple[date, float]]) -> float | None:
    """Daily sigma over the last SIGMA_SESSIONS sessions."""
    px = [c for _, c in closes if c and c > 0][-(SIGMA_SESSIONS + 1) :]
    rets = [math.log(b / a) for a, b in zip(px, px[1:])]
    return statistics.pstdev(rets) if len(rets) >= 20 else None


def market_move(
    published: datetime, closes: list[tuple[date, float]], sigma: float | None
) -> tuple[str, float | None, float | None]:
    """(line, move, z): the first session the item could trade on, close to close. Pure."""
    days = [d for d, c in closes if c and c > 0]
    px = {d: c for d, c in closes if c and c > 0}
    pub = mc.to_et(published)
    after = [
        i
        for i, d in enumerate(days)
        if datetime.combine(d, mc.session_close(d), tzinfo=mc.MARKET_TZ) >= pub
    ]
    if not after or after[0] == 0:
        return "Market: no session has traded on it yet.", None, None
    i = after[0]
    move = px[days[i]] / px[days[i - 1]] - 1
    z = move / sigma if sigma else None
    zs = f" ({z:+.1f} sigma of its usual daily move)" if z is not None else ""
    line = f"Market: {days[i].isoformat()}, the first session after it, closed {move:+.1%}{zs}."
    return line, move, z


def market_line(published: datetime, closes: list[tuple[date, float]], sigma: float | None):
    return market_move(published, closes, sigma)[0]


def history_lines(store, symbol: str, now: datetime, *, exclude: set[str]) -> list[str]:
    """Filings and material events, then headlines already seen, newest first.

    ``exclude``: the URLs and lower-cased titles of the items being judged. The
    same story under a second URL is the item itself, not earlier coverage —
    shown as history, the model would call its own item a rehash.
    """
    from advisor.daemon.summarize import summarize

    since = now - timedelta(days=HISTORY_DAYS)
    out = []
    for e in store.recent_events(symbol=symbol, since=since, limit=500):
        if e.tier.value not in ("A", "B") or e.kind.startswith("NEWS") or e.kind in _MECHANICS:
            continue
        line = summarize(e) or e.kind.replace("_", " ").lower()
        out.append(f"{mc.to_et(e.ts).date()} {e.kind.replace('_', ' ').lower()}: {line}"[:220])
    for it in store.source_items_between(symbol, since, now):
        if it.tier.value == "PRIMARY" or it.url in exclude:
            continue
        if it.title.strip().lower() not in exclude:
            out.append(f"{mc.to_et(it.published_at).date()} headline: {it.title[:150]}")
    out.sort(reverse=True)
    return out[:MAX_HISTORY_LINES]


def name_context(
    store, symbol: str, now: datetime, closes: list[tuple[date, float]], *, exclude=frozenset()
) -> list[str]:
    """The numbered context lines for one name (C1, C2, ...), without the per-item market."""
    symbol = symbol.upper()
    lines = [position_line(store, symbol, sigma_of(closes))]
    size = size_line(store, symbol)
    if size:
        lines.append(size)
    hist = history_lines(store, symbol, now, exclude=set(exclude))
    if hist:
        lines.append("Earlier on this name (last 30 days): " + " | ".join(hist))
    return lines


def closes_for(symbol: str, loader: Callable[[str], list[tuple[date, float]]] | None = None):
    if loader is None:
        from advisor.entry.sheet import daily_closes as loader
    try:
        return loader(symbol) or []
    except Exception:  # noqa: BLE001
        return []
