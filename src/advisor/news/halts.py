"""Trading halts from the exchange: a primary fact, read for an exit.

A report that a company is in trouble needs corroboration; the exchange
stopping its shares does not. Nasdaq Trader publishes every current US halt,
Nasdaq- and NYSE-listed alike, as one free RSS feed with the reason code the
exchange assigned. The user's rule (2026-09-27): a halt that says the
company's standing is in question is an EXIT on the exchange's word alone;
a halt pending news is a REVIEW until trading resumes.

The feed lists every halt still in force — on 2026-09-27 the oldest dated
from 2019 — plus the ones that ended in the last few days. It is polled every
few minutes and each halt on a watched name is kept as an event; resumption
arrives as a second event on the same halt. A halt still in force counts
however old it is: a position does not stop being frozen because a fortnight
passed.

Codes read (Nasdaq Trader's own table); anything else — volatility pauses,
IPOs, market-wide breakers — is archived as context and never calls anything:

- ``T12`` additional information requested by the exchange
- ``H4`` non-compliance, ``H9`` not current in filings, ``H10`` SEC trading
  suspension, ``H11`` regulatory concern
- ``T1`` news pending, ``T6`` extraordinary market activity
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel

from advisor.daemon import market_calendar as mc
from advisor.daemon.models import Event, EventSource, EventTier

logger = logging.getLogger(__name__)

FEED_URL = "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"
_NS = "{http://www.nasdaqtrader.com/}"

# The company's standing is in question: an EXIT while halted.
EXIT_CODES = {
    "T12": "halted: the exchange requested more information from the company",
    "H4": "halted for non-compliance with listing requirements",
    "H9": "halted: not current in its required filings",
    "H10": "trading suspended by the SEC",
    "H11": "halted for a regulatory concern",
}
# Something is about to be said: a REVIEW until trading resumes.
REVIEW_CODES = {
    "T1": "halted, news pending",
    "T6": "halted for extraordinary market activity",
}


class Halt(BaseModel):
    symbol: str
    name: str = ""
    market: str = ""
    code: str
    halted_at: datetime  # America/New_York
    resumed_at: datetime | None = None

    @property
    def key(self) -> str:
        return f"halt:{self.symbol}:{self.halted_at.isoformat()}:{self.code}"

    @property
    def grade(self) -> str | None:
        if self.code in EXIT_CODES:
            return "EXIT"
        if self.code in REVIEW_CODES:
            return "REVIEW"
        return None

    @property
    def label(self) -> str:
        return EXIT_CODES.get(self.code) or REVIEW_CODES.get(self.code) or f"halted ({self.code})"


def _stamp(day: str | None, clock: str | None) -> datetime | None:
    """'09/25/2026' and '19:50:00.000' (ET) to an aware datetime; None if either is blank."""
    day, clock = (day or "").strip(), (clock or "").strip()
    if not day or not clock:
        return None
    for fmt in ("%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(f"{day} {clock}", fmt).replace(tzinfo=mc.MARKET_TZ)
        except ValueError:
            continue
    logger.info("halts: unparseable time %r %r", day, clock)
    return None


def _resumed(day: str, clock: str) -> datetime | None:
    """When trading resumed. A resumption date with an unreadable time still ends the halt,
    at the start of that day: a halt left standing by a format change would keep an EXIT."""
    if not day.strip():
        return None
    return _stamp(day, clock) or _stamp(day, "00:00:00")


def parse(xml: str) -> list[Halt]:
    """Every halt in the feed. Pure. A row without a symbol, code or time is skipped."""
    # The live feed opens with a byte-order mark; as bytes, the parser drops it.
    root = ET.fromstring(xml.encode("utf-8"))
    out = []
    for item in root.iter("item"):

        def f(tag: str, _item=item) -> str:
            return (_item.findtext(f"{_NS}{tag}") or "").strip()

        symbol, code = f("IssueSymbol").upper(), f("ReasonCode").upper()
        halted = _stamp(f("HaltDate"), f("HaltTime"))
        if not symbol or not code or halted is None:
            continue
        out.append(
            Halt(
                symbol=symbol,
                name=f("IssueName"),
                market=f("Market"),
                code=code,
                halted_at=halted,
                resumed_at=_resumed(f("ResumptionDate"), f("ResumptionTradeTime")),
            )
        )
    return out


def _http_get(url: str) -> str:
    import httpx

    r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0 advisor"}, timeout=20)
    r.raise_for_status()
    return r.text


def fetch(get: Callable[[str], str] = _http_get) -> list[Halt]:
    """The current feed. Raises on a network or parse failure: the caller records it."""
    return parse(get(FEED_URL))


def halt_events(halts: list[Halt], watched: dict[str, bool]) -> list[Event]:
    """Events for the halts of watched names; ``watched`` maps symbol to held.

    A halt that grades an exit on a held name is tier A; any other graded
    halt is tier B; the rest is context. A resumption is its own event, so a
    halt seen before and after it is stored once and ended once.
    """
    events = []
    for h in halts:
        if h.symbol not in watched:
            continue
        held = watched[h.symbol]
        tier = (
            EventTier.A if h.grade == "EXIT" and held else EventTier.B if h.grade else EventTier.C
        )
        payload = {
            "code": h.code,
            "label": h.label,
            "grade": h.grade,
            "market": h.market,
            "name": h.name,
            "halted_at": h.halted_at.isoformat(),
            "source": "Nasdaq Trader trade halts",
            "url": FEED_URL,
        }
        events.append(
            Event(
                ts=h.halted_at,
                source=EventSource.EXCHANGE,
                kind="TRADING_HALT",
                tier=tier,
                symbol=h.symbol,
                dedup_key=h.key,
                payload=payload,
            )
        )
        if h.resumed_at is not None:
            events.append(
                Event(
                    ts=h.resumed_at,
                    source=EventSource.EXCHANGE,
                    kind="TRADING_RESUMED",
                    tier=EventTier.C,
                    symbol=h.symbol,
                    dedup_key=f"{h.key}:resumed",
                    payload={**payload, "resumed_at": h.resumed_at.isoformat()},
                )
            )
    return events


def active_halts(store, symbol: str, since: datetime) -> list[dict]:
    """The name's halts still in force, of any age, and those since ``since`` that ended.

    Newest first, each with ``resumed_at`` if trading resumed.
    """
    rows = store.recent_events(symbol=symbol, kinds=("TRADING_HALT", "TRADING_RESUMED"), limit=500)

    def halt_id(p: dict) -> tuple:  # the store keeps only a hash of dedup_key
        return (p.get("halted_at"), p.get("code"))

    resumed = {
        halt_id(e.payload or {}): e.payload or {} for e in rows if e.kind == "TRADING_RESUMED"
    }
    out = []
    for e in rows:
        if e.kind != "TRADING_HALT":
            continue
        p = e.payload or {}
        end = resumed.get(halt_id(p), {})
        if end.get("resumed_at") and e.ts < since:
            continue  # ended, and long enough ago to be history
        out.append(
            {
                **p,
                "resumed_at": end.get("resumed_at"),
                "resumed_inferred": bool(end.get("inferred")),
            }
        )
    out.sort(key=lambda h: h.get("halted_at") or "", reverse=True)
    return out


def ended_halts(store, halts: list[Halt], watched, now: datetime) -> list[Event]:
    """Stored halts still in force that the feed no longer lists: they ended unseen.

    The feed lists every halt in force, so one that left it is over even if
    its resumption was never seen (the laptop asleep for days). Without this
    an EXIT would stand forever on a name trading again. An empty feed proves
    nothing, so it ends nothing.
    """
    if not halts:
        return []
    in_feed = {(h.symbol, h.halted_at.isoformat(), h.code) for h in halts}
    events = []
    for symbol in watched:
        for p in active_halts(store, symbol, now):
            if p.get("resumed_at") or (symbol, p.get("halted_at"), p.get("code")) in in_feed:
                continue
            events.append(
                Event(
                    ts=now,
                    source=EventSource.EXCHANGE,
                    kind="TRADING_RESUMED",
                    tier=EventTier.C,
                    symbol=symbol,
                    dedup_key=f"halt:{symbol}:{p.get('halted_at')}:{p.get('code')}:resumed",
                    payload={
                        **{k: v for k, v in p.items() if k != "resumed_inferred"},
                        "resumed_at": now.isoformat(),
                        "inferred": True,
                    },
                )
            )
    return events
