"""A company in a few lines: the state of the business, its price and your position, today.

The ticker page carried everything about a name — the reading, the price
range, the filings, the research — and nothing that said, at the top, where
the company stands (user, 2026-09-29: "no tenemos un resumen al principio que
dé un outline de la empresa"). This is that outline: a handful of bullets,
each one line, each citing where it comes from and when.

Assembled deterministically from what the system already keeps: the
company's health from its filings (``entry.health``), the latest proposal
on file (zone, move, call, stop, results date), the news agent's judgments of
the week, the book, and a one-line description of the business. No model
writes it and nothing in it is new: it is a summary of the page below it.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Bullet(BaseModel):
    topic: str  # Business, Growth, Cash, Concerns, Valuation, News, Position, Call
    text: str
    source: str
    tone: str = "neutral"  # pos | neg | warn | neutral: how it reads for the holder


class Overview(BaseModel):
    symbol: str
    as_of: datetime
    bullets: list[Bullet] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)  # what could not be said, and why


def _pct(x: float, sign: bool = True) -> str:
    return f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%"


def _money(x: float) -> str:
    return f"${x / 1e9:,.2f}bn" if abs(x) >= 1e9 else f"${x / 1e6:,.1f}M"


# ── Each topic, pure ─────────────────────────────────────────────────────


def business_bullet(text: str | None, source: str) -> Bullet | None:
    if not text:
        return None
    return Bullet(topic="Business", text=text.strip(), source=source)


def health_bullets(h) -> list[Bullet]:
    """Growth, cash and balance sheet, and the concerns, from ``entry.health.Health``."""
    if h is None:
        return []
    src = h.source_label()
    out = []
    trend = h.trend()
    if h.revenue_ttm is not None:
        text = f"Revenue {_money(h.revenue_ttm)} over 12 months"
        if h.growth is not None:
            text += f", {_pct(h.growth)} a year"
            if trend and trend != "shrinking" and h.growth_year is not None:
                text += f", {trend} (from {_pct(h.growth_year)} a year before)"
            elif trend:
                text += f", {trend}"
        elif h.quarter_growth is not None:
            text += f"; last quarter {_pct(h.quarter_growth)} on a year earlier"
        tone = {"accelerating": "pos", "decelerating": "warn", "shrinking": "neg"}.get(
            trend or "", "neutral"
        )
        out.append(Bullet(topic="Growth", text=text + ".", source=src, tone=tone))

    if h.financial:
        cash = "A financial company: free cash flow and net cash do not measure it"
        tone = "neutral"
    else:
        parts = []
        if h.fcf_margin is not None:
            part = f"free cash flow {_pct(h.fcf_margin, sign=False)} of revenue"
            median = h._median()
            if median is not None:
                part += f" (3-year median {_pct(median, sign=False)})"
            parts.append(part)
        if h.net_cash is not None:
            kind = "net cash" if h.net_cash >= 0 else "net debt"
            part = f"{kind} {_money(abs(h.net_cash))}"
            if h.market_cap:
                part += f" ({_pct(abs(h.net_cash) / h.market_cap, sign=False)} of market cap)"
            parts.append(part)
        if h.dilution is not None:
            parts.append(f"shares {_pct(h.dilution)} a year")
        cash = "; ".join(parts)
        cash = cash[:1].upper() + cash[1:] if cash else ""
        burning = h.fcf_margin is not None and h.fcf_margin < 0
        tone = "neg" if burning else ("warn" if h.concerns() else "neutral")
    if cash:
        out.append(Bullet(topic="Cash", text=cash + ".", source=src, tone=tone))

    concerns = h.concerns()
    out.append(
        Bullet(
            topic="Filings",
            text=(
                f"Concerns in the filings to {h.period_end}: {', '.join(concerns)}."
                if concerns
                else f"No concern flagged in the filings to {h.period_end}."
            ),
            source=src,
            tone="warn" if concerns else "pos",
        )
    )
    return out


def valuation_bullet(p: dict | None) -> Bullet | None:
    """Where the price sits against the name's own P/S history, from the latest proposal."""
    f = (p or {}).get("features") or {}
    ps, median, pctile = f.get("ps"), f.get("ps_median"), f.get("ps_percentile")
    if ps is None or median is None:
        return None
    text = f"P/S {ps:.1f}x against its own 2-year median {median:.1f}x"
    if pctile is not None:
        text += f", at the {round(pctile * 100):.0f}th percentile of its last 2 years"
    if f.get("in_zone"):
        text += ": in its zone"
        tone = "pos"
    else:
        text += ": above its zone"
        tone = "warn" if pctile is not None and pctile >= 0.8 else "neutral"
    return Bullet(
        topic="Valuation",
        text=text + ".",
        source=f"entry engine, {str(p.get('built_at', ''))[:16].replace('T', ' ')}",
        tone=tone,
    )


def news_bullet(judgments: list, days: int) -> Bullet:
    """The most material news the agent judged about the company this week."""
    if not judgments:
        return Bullet(
            topic="News",
            text=f"No news about the company judged in the last {days} days.",
            source="news agent",
        )
    j = judgments[0]
    direction = j.direction.value.lower()
    text = f"{j.published_at.date()}, {direction} ({j.materiality.value.lower()}): {j.title}"
    if len(judgments) > 1:
        text += f" — and {len(judgments) - 1} more this week"
    material = j.materiality.value in ("HIGH", "MEDIUM")
    tone = {"negative": "neg", "positive": "pos"}.get(direction, "neutral")
    if not material:
        tone = "neutral"  # a minor item does not colour the outline
    return Bullet(topic="News", text=text[:300], source=f"{j.provider} (news agent)", tone=tone)


def position_bullet(held: list, net_liq: float | None, as_of) -> Bullet:
    if not held:
        return Bullet(topic="Position", text="Not held.", source="TastyTrade positions")
    quantity = sum(p.quantity for p in held)
    basis = sum(p.cost_basis for p in held)
    pnl = sum(p.unrealized_pnl for p in held)
    signed = sum(p.signed_notional for p in held)
    text = f"You hold {quantity:,.0f} share{'' if abs(quantity) == 1 else 's'}"
    if net_liq:
        text += f", {_pct(signed / net_liq, sign=False)} of net liq"
    tone = "neutral"
    if basis:
        average = basis / abs(quantity)
        text += f"; {_pct(pnl / basis)} (${pnl:,.0f}) from your average ${average:,.2f}"
        tone = "pos" if pnl >= 0 else "neg"
    return Bullet(
        topic="Position", text=text + ".", source=f"TastyTrade positions, {as_of}", tone=tone
    )


_CALL_TONE = {"EXIT": "neg", "TRIM": "warn", "REVIEW": "warn", "ENTER": "pos", "ADD": "pos"}


def call_bullet(p: dict | None) -> Bullet | None:
    """The system's call on file, its first reason, the stop and the next results date."""
    if not p:
        return None
    action = p.get("action", "")
    reasons = [r.get("text", "") for r in p.get("reasons") or []]
    why = next((r.split(": ", 1)[1] for r in reasons if r.startswith(f"Why {action}: ")), None)
    exits = p.get("exits") or []
    if why is None and exits:
        why = exits[0].get("why") or None
    if why is None and p.get("triggers"):
        why = p["triggers"][0]
    if why:
        # Scanner candidates are stored by id ("2026-09-28:C:MDB"): name the setup.
        why = re.sub(r"\d{4}-\d{2}-\d{2}:([A-Z]):[A-Z.\-]+", r"setup \1", why)
    text = f"{action}" + (f": {why}" if why else "")
    results = next((r for r in reasons if r.startswith("next results")), None)
    if results:
        text += f"; {results}"
    return Bullet(
        topic="Call",
        text=text[:300] + ".",
        source=f"entry engine, {str(p.get('built_at', ''))[:16].replace('T', ' ')}",
        tone=_CALL_TONE.get(action, "neutral"),
    )


def assemble(
    symbol: str,
    now: datetime,
    *,
    business: tuple[str | None, str] = (None, ""),
    health=None,
    proposal: dict | None = None,
    judgments: list | None = None,
    news_days: int = 7,
    held: list | None = None,
    net_liq: float | None = None,
    book_as_of=None,
) -> Overview:
    """The overview from its parts. Pure: every input is already loaded."""
    ov = Overview(symbol=symbol.upper(), as_of=now)
    for bullet in (
        business_bullet(*business),
        *health_bullets(health),
        valuation_bullet(proposal),
        news_bullet(judgments or [], news_days),
        position_bullet(held or [], net_liq, book_as_of),
        call_bullet(proposal),
    ):
        if bullet is not None:
            ov.bullets.append(bullet)
    if health is None:
        ov.gaps.append("no health from the filings: the SEC and Yahoo had nothing for it")
    if proposal is None:
        ov.gaps.append("no proposal on file: evaluate the name to place its price and call")
    return ov


# ── Live ─────────────────────────────────────────────────────────────────

_PROFILES: dict[tuple[str, object], tuple[str | None, str]] = {}


def business_line(db_path, symbol: str, today) -> tuple[str | None, str]:
    """One line on what the company does: the deep research's, else Yahoo's profile."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT created_at, json_extract(report_json, '$.deep_research.what_they_do') "
            "FROM research_reports WHERE symbol = ? ORDER BY created_at DESC",
            (symbol,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    for created, text in rows:
        if text:
            return text, f"deep research, {str(created)[:10]}"
    key = (symbol, today)
    if key not in _PROFILES:
        _PROFILES[key] = _yahoo_profile(symbol)
    return _PROFILES[key]


def first_sentence(text: str, limit: int = 240) -> str:
    """The first sentence of a profile, cut at a word if it runs long. Pure."""
    text = " ".join(text.split())
    # A period ends a sentence when a capital follows it — not "Inc." mid-name.
    m = re.search(r"(?<!\b[A-Z])(?<!\bInc)(?<!\bCorp)(?<!\bCo)(?<!\bLtd)\.\s+(?=[A-Z])", text)
    sentence = text[: m.start() + 1] if m else text
    if len(sentence) > limit:
        sentence = sentence[:limit].rsplit(" ", 1)[0] + "…"
    return sentence


def _yahoo_profile(symbol: str) -> tuple[str | None, str]:
    try:
        import yfinance as yf

        summary = (yf.Ticker(symbol).info or {}).get("longBusinessSummary")
    except Exception as exc:  # noqa: BLE001
        logger.info("overview: no Yahoo profile for %s: %s", symbol, exc)
        return None, ""
    if not summary:
        return None, ""
    return first_sentence(summary), "Yahoo company profile"


def load_overview(store, symbol: str, now: datetime, *, profile=None) -> Overview:
    """The overview for ``symbol`` from the store (health refreshed once a day)."""
    from advisor.breadth.position import latest_evaluation
    from advisor.entry.health import refresh_health
    from advisor.entry.run import NEWS_DAYS, judged_news

    sym = symbol.upper()
    db_path = store.db_path
    health = proposal = None
    judgments: list = []
    try:
        health = refresh_health(db_path, sym, now)
    except Exception as exc:  # noqa: BLE001
        logger.warning("overview: health failed for %s: %s", sym, exc)
    try:
        proposal = latest_evaluation(db_path, sym)
    except Exception as exc:  # noqa: BLE001
        logger.warning("overview: no proposal for %s: %s", sym, exc)
    try:
        judgments = judged_news(db_path, sym, now)
    except Exception as exc:  # noqa: BLE001
        logger.warning("overview: no news for %s: %s", sym, exc)
    business = (profile or business_line)(db_path, sym, now.date())
    book = store.load_latest_book()
    held = [p for p in book.positions if p.underlying.upper() == sym] if book else []
    return assemble(
        sym,
        now,
        business=business,
        health=health,
        proposal=proposal,
        judgments=judgments,
        news_days=NEWS_DAYS,
        held=held,
        net_liq=book.net_liq if book else None,
        book_as_of=book.as_of.date() if book else None,
    )
