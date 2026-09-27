"""The news agent: one structured judgment per news item, recorded and measured.

User decision (2026-09-27): news needs an agent of its own that judges each
item — what it is about, whether it is good or bad *for the business*, how
material and how new it is, and which claim of the user's thesis it touches.
This replaces the earlier decision not to score sentiment, with the
condition the user chose: **context and measurement only.** A judgment
changes no action. It is shown, stored, and scored against what the price
did next (excess over the name's own drift), so that after enough sessions
the record says whether its "negative, material" calls anticipate anything.
Only then may the user give it a say in a decision.

What keeps a judgment honest:

- **It reads only what was ingested.** No searches of its own: the items are
  the ones already archived (``source_items``) for held names, the watchlist
  and Swing, so the Tavily budget is untouched.
- **It must quote.** Every judgment carries a verbatim span of the item that
  supports it; a quote the item does not contain voids the judgment.
- **No number it cannot show.** A number in the explanation must appear in
  the item (thousands separators ignored, the lesson of "38,311.8 m²").
- **Thesis links are to claims that exist.** An unknown claim id is dropped.
- **Off-topic is neutral.** An item the model says is not about the company
  keeps no direction, materiality or thesis link.
- **Versioned.** Each judgment carries the model and the hash of the
  instructions; a new prompt re-judges, and the two are never pooled.

Direction is about the business, not the tone of the headline: "shares slide
after upgrade" is price commentary about a positive analyst action.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import statistics
from collections.abc import Callable
from datetime import date, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field

from advisor.daemon import market_calendar as mc

logger = logging.getLogger(__name__)

# Items published this many days back are judged; older ones are left alone.
JUDGE_WINDOW_DAYS = 7
# Items per model call; a name with more is judged in several calls.
BATCH = 10
# Characters of an item's summary shown to the model.
SUMMARY_CHARS = 1200
# The item tiers judged: news and the company's own filings with a lead.
JUDGED_TIERS = ("AGGREGATOR", "UNTAGGED", "PRIMARY")
# Sessions after the judgment over which the price is scored.
HORIZONS = {"d1": 1, "d5": 5, "d20": 20}


class EventType(StrEnum):
    RESULTS = "RESULTS"
    GUIDANCE = "GUIDANCE"
    CUSTOMER = "CUSTOMER"  # a customer, contract, order or partnership
    PRODUCT = "PRODUCT"
    REGULATORY = "REGULATORY"
    LEGAL = "LEGAL"
    FINANCING = "FINANCING"  # equity or debt raised, buybacks, dividends
    DEAL = "DEAL"  # M&A, spin-offs, stakes
    MANAGEMENT = "MANAGEMENT"
    INSIDER = "INSIDER"
    ANALYST = "ANALYST"  # ratings, targets, estimates
    SECTOR = "SECTOR"  # the industry or the economy, not the company itself
    PRICE = "PRICE"  # the stock's move, with no new fact about the business
    OTHER = "OTHER"


class Direction(StrEnum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    MIXED = "MIXED"
    NEUTRAL = "NEUTRAL"


class Materiality(StrEnum):
    HIGH = "HIGH"  # could change revenue, margins, solvency or the share count
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Novelty(StrEnum):
    NEW = "NEW"  # a fact not reported before
    FOLLOW_UP = "FOLLOW_UP"  # new detail on a known story
    REHASH = "REHASH"  # a known story told again


class Basis(StrEnum):
    COMPANY = "COMPANY"  # the company said it: a filing, a release, a call
    REPORTED = "REPORTED"  # a journalist reports a fact, with a source
    OPINION = "OPINION"  # an analyst's or a writer's view
    RUMOR = "RUMOR"  # unnamed sources, speculation


class ClaimLink(BaseModel):
    claim_id: str
    effect: str  # SUPPORTS | CONTRADICTS: the item is evidence for or against the claim's text
    kind: str | None = None  # the claim's kind, filled by the gate from the thesis

    @property
    def against_thesis(self) -> bool:
        """Evidence a kill condition or risk is coming true, or that a driver is not."""
        if self.effect.upper() == "SUPPORTS":
            return self.kind in THREATS
        return self.kind not in THREATS


# Claims whose text states what would go wrong: evidence FOR them is bad news.
THREATS = frozenset({"INVALIDATION", "RISK"})


class ItemCall(BaseModel):
    """What the model returns for one item."""

    id: str
    about_company: bool
    event_type: EventType
    direction: Direction
    materiality: Materiality
    novelty: Novelty
    basis: Basis
    claims: list[ClaimLink] = Field(default_factory=list)
    quote: str  # verbatim words of the item that carry the judgment
    why: str  # one sentence: why this direction and materiality


class Draft(BaseModel):
    judgments: list[ItemCall]


class Judgment(BaseModel):
    """One item, judged: what the model said, where it came from, what followed."""

    key: str  # the item's dedup key
    symbol: str
    published_at: datetime
    title: str
    provider: str
    tier: str
    url: str | None = None
    about_company: bool
    event_type: EventType
    direction: Direction
    materiality: Materiality
    novelty: Novelty
    basis: Basis
    claims: list[ClaimLink] = Field(default_factory=list)
    quote: str
    why: str
    model: str | None = None
    prompt_version: str
    judged_at: datetime
    problems: list[str] = Field(default_factory=list)  # what the gate removed
    # Close-to-close returns: "d0" from the last close before publication to
    # the first close after the judgment (what the market had already done);
    # "d1".."d20" from that close on (what acting on the judgment would see).
    outcomes: dict[str, float | None] = Field(default_factory=dict)

    @property
    def group(self) -> str:
        if not self.about_company:
            return "off-topic"
        return f"{self.direction.value}/{self.materiality.value}"


SYSTEM_PROMPT = """\
You judge news items about one listed company for a long-term holder.

For EACH numbered item return one judgment:
- about_company: is the item really about this company (not only a list or \
the sector that mentions it)?
- event_type: what happened (RESULTS, GUIDANCE, CUSTOMER, PRODUCT, \
REGULATORY, LEGAL, FINANCING, DEAL, MANAGEMENT, INSIDER, ANALYST, SECTOR, \
PRICE, OTHER). PRICE is a story about the share price with no new fact about \
the business.
- direction: is the FACT good or bad for the BUSINESS (revenue, margins, \
solvency, share count, competitive position)? Judge the fact, not the tone \
of the headline or the share-price move. POSITIVE, NEGATIVE, MIXED or NEUTRAL.
- materiality: HIGH only if it could change revenue, margins, solvency or the \
share count in a way a holder must act on; MEDIUM if it adds real information; \
LOW otherwise (opinion pieces, lists, price recaps are LOW).
- novelty: NEW fact, FOLLOW_UP on a known story, or REHASH of old news.
- basis: COMPANY (the company itself said it), REPORTED (journalist reporting \
a fact), OPINION, or RUMOR.
- claims: only if the item bears on one of the holder's thesis claims listed \
below, the claim id and SUPPORTS (the item is evidence the claim's text is \
true) or CONTRADICTS (evidence it is false). A claim marked INVALIDATION or \
RISK describes what would go wrong. Otherwise an empty list.
- quote: copy, word for word, the few words of the item that your judgment \
rests on.
- why: one sentence. Use only numbers that appear in the item.

Do not invent facts. If the item says too little, judge it NEUTRAL and LOW."""


def prompt_version() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]


# ── The items ─────────────────────────────────────────────────────────────


def _text(item) -> str:
    return f"{item.title} {item.summary or ''}"


def items_to_judge(store, conn, symbol: str, now: datetime) -> list:
    """The name's archived items of the last JUDGE_WINDOW_DAYS not judged under this prompt."""
    since = now - timedelta(days=JUDGE_WINDOW_DAYS)
    done = NewsJudgmentStore(conn).keys(symbol, prompt_version())
    out, seen = [], set()
    for item in store.source_items_between(symbol, since, now + timedelta(days=1)):
        if item.tier.value not in JUDGED_TIERS:
            continue
        if item.tier.value == "PRIMARY" and not item.summary:
            continue  # a filing with no lead has nothing to judge but its form
        title = item.title.strip().lower()
        if item.dedup_key() in done or title in seen:
            continue
        seen.add(title)
        out.append(item)
    return out


def user_prompt(symbol: str, company: str | None, items: list, claims: list) -> str:
    lines = [f"Company: {company or symbol} ({symbol})", ""]
    if claims:
        lines.append("The holder's thesis claims:")
        lines += [f"{c.id} [{c.kind.value}]: {c.text}" for c in claims if c.id]
        lines.append("")
    lines.append("Items:")
    for n, item in enumerate(items, 1):
        summary = (item.summary or "").strip().replace("\n", " ")[:SUMMARY_CHARS]
        lines.append(
            f"N{n} [{item.published_at.date().isoformat()}] ({item.provider}, "
            f"{item.doc_type or 'NEWS'}) {item.title}" + (f"\n    {summary}" if summary else "")
        )
    return "\n".join(lines)


# ── The gate ──────────────────────────────────────────────────────────────

_NUM = re.compile(r"\d[\d,]*\.?\d*")


_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "…": "..."})
# A quote may stitch fragments of the item together with an ellipsis.
_ELLIPSIS = re.compile(r"\[\s*\.\.\.\s*\]|\.\.\.")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_QUOTES)).strip().lower()


def quoted(quote: str, text: str) -> bool:
    """Every fragment of ``quote`` (split at ellipses) appears in ``text``."""
    body = _squash(text)
    parts = [_squash(f).strip(" \"'.,;:") for f in _ELLIPSIS.split(quote.translate(_QUOTES))]
    parts = [f for f in parts if f]
    return bool(parts) and sum(len(f) for f in parts) >= 3 and all(f in body for f in parts)


def _close_to(x: str, known: set[str]) -> bool:
    """``x`` is a number of the item, or that number rounded (within 1%)."""
    try:
        v = float(x)
    except ValueError:
        return False
    for k in known:
        try:
            w = float(k)
        except ValueError:
            continue
        if w and abs(v - w) / abs(w) <= 0.01:
            return True
    return False


def _numbers(text: str) -> set[str]:
    out = set()
    for m in _NUM.findall(text):
        n = m.replace(",", "").rstrip(".")
        if n:
            out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


def check(call: ItemCall, item, claim_ids) -> tuple[ItemCall | None, list[str]]:
    """Keep what the item supports. Pure. (None, problems) when the judgment is void."""
    problems = []
    text = _text(item)
    if not quoted(call.quote, text):
        return None, [f"{call.id}: the quote is not in the item"]
    known = _numbers(text)
    stray = {n for n in _numbers(call.why) - known if not _close_to(n, known)}
    if stray:
        return None, [f"{call.id}: 'why' uses {sorted(stray)}, not in the item"]
    kinds = claim_ids if isinstance(claim_ids, dict) else dict.fromkeys(claim_ids)
    kept = [
        c.model_copy(update={"effect": c.effect.upper(), "kind": kinds[c.claim_id]})
        for c in call.claims
        if c.claim_id in kinds and c.effect.upper() in ("SUPPORTS", "CONTRADICTS")
    ]
    if len(kept) < len(call.claims):
        problems.append(f"{call.id}: dropped claim links to unknown claims")
    call = call.model_copy(update={"claims": kept})
    if not call.about_company:
        call = call.model_copy(
            update={
                "direction": Direction.NEUTRAL,
                "materiality": Materiality.LOW,
                "claims": [],
            }
        )
    return call, problems


# ── Judging ───────────────────────────────────────────────────────────────

Complete = Callable[[str, str], Draft]


def _openrouter() -> tuple[Complete, str] | None:
    from research_agent.config import ResearchConfig
    from research_agent.llm import OpenRouterLLM

    config = ResearchConfig()
    if not config.openrouter_api_key:
        return None
    llm = OpenRouterLLM(config)
    return (lambda s, u: llm.complete(s, u, response_model=Draft)), config.llm_model


def judge_symbol(
    store,
    conn,
    symbol: str,
    now: datetime,
    *,
    company: str | None = None,
    complete: Complete | None = None,
    model: str | None = None,
) -> tuple[list[Judgment], list[str]]:
    """Judge and store every unjudged item of ``symbol``. Returns (judgments, problems)."""
    symbol = symbol.upper()
    items = items_to_judge(store, conn, symbol, now)
    if not items:
        return [], []
    if complete is None:
        configured = _openrouter()
        if configured is None:
            return [], [f"{symbol}: no model configured"]
        complete, model = configured
    claims = [c for c in store.load_claims(symbol) if c.id]
    claim_ids = {c.id: c.kind.value for c in claims}
    out, problems = [], []
    judgments = NewsJudgmentStore(conn)
    for start in range(0, len(items), BATCH):
        batch = items[start : start + BATCH]
        try:
            draft = complete(SYSTEM_PROMPT, user_prompt(symbol, company, batch, claims))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{symbol}: model failed: {exc}")
            continue
        by_id = {c.id: c for c in draft.judgments}
        for n, item in enumerate(batch, 1):
            call = by_id.get(f"N{n}")
            if call is None:
                problems.append(f"{symbol} N{n}: not judged by the model")
                continue
            kept, why = check(call, item, claim_ids)
            problems += [f"{symbol} {p}" for p in why]
            if kept is None:
                continue
            j = Judgment(
                key=item.dedup_key(),
                symbol=symbol,
                published_at=item.published_at,
                title=item.title,
                provider=item.provider,
                tier=item.tier.value,
                url=item.url,
                model=model,
                prompt_version=prompt_version(),
                judged_at=mc.now_et(),
                problems=why,
                **kept.model_dump(exclude={"id"}),
            )
            judgments.add(j)
            out.append(j)
    return out, problems


def judge_all(
    store, conn, symbols: list[str], now: datetime, *, names=None, **kw
) -> tuple[list[Judgment], list[str]]:
    out, problems = [], []
    for sym in symbols:
        company = None
        if names is not None:
            try:
                company = names(sym)
            except Exception:  # noqa: BLE001
                company = None
        js, ps = judge_symbol(store, conn, sym, now, company=company, **kw)
        out += js
        problems += ps
    return out, problems


# ── Storage ───────────────────────────────────────────────────────────────

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS news_judgments (
    key            TEXT NOT NULL,            -- the item's dedup key
    prompt_version TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    published_at   TEXT NOT NULL,
    payload_json   TEXT NOT NULL,            -- news.judge.Judgment
    judged_at      TEXT NOT NULL,
    PRIMARY KEY (key, prompt_version)
);
CREATE INDEX IF NOT EXISTS idx_news_judgments_symbol ON news_judgments(symbol, published_at);
"""


class NewsJudgmentStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        self._conn = conn
        conn.executescript(_SCHEMA)

    def add(self, j: Judgment) -> bool:
        """Store once per item and prompt; a re-run is a no-op."""
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO news_judgments (key, prompt_version, symbol, published_at, "
            "payload_json, judged_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                j.key,
                j.prompt_version,
                j.symbol,
                j.published_at.isoformat(),
                j.model_dump_json(),
                j.judged_at.isoformat(),
            ),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update(self, j: Judgment) -> None:
        self._conn.execute(
            "UPDATE news_judgments SET payload_json = ? WHERE key = ? AND prompt_version = ?",
            (j.model_dump_json(), j.key, j.prompt_version),
        )
        self._conn.commit()

    def keys(self, symbol: str, version: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT key FROM news_judgments WHERE symbol = ? AND prompt_version = ?",
            (symbol.upper(), version),
        ).fetchall()
        return {r["key"] for r in rows}

    def list(
        self,
        *,
        symbol: str | None = None,
        since: datetime | None = None,
        version: str | None = None,
    ) -> list[Judgment]:
        clauses, args = [], []
        if symbol:
            clauses.append("symbol = ?")
            args.append(symbol.upper())
        if version:
            clauses.append("prompt_version = ?")
            args.append(version)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT payload_json FROM news_judgments {where} ORDER BY published_at DESC", args
        ).fetchall()
        out = [Judgment.model_validate_json(r["payload_json"]) for r in rows]
        if since is not None:
            out = [j for j in out if mc.to_et(j.published_at) >= mc.to_et(since)]
        return out


# ── What followed ─────────────────────────────────────────────────────────


def outcomes_for(j: Judgment, closes: dict[date, float]) -> dict[str, float | None]:
    """Returns after the judgment, from the first close it could have been acted on. Pure.

    The entry is the first close after both publication and judgment; "d0" is
    what the price did from the last close before publication to that entry.
    """
    days = sorted(d for d, px in closes.items() if px and px > 0)
    ready = max(mc.to_et(j.published_at), mc.to_et(j.judged_at))
    after = [
        i
        for i, d in enumerate(days)
        if datetime.combine(d, mc.session_close(d), tzinfo=mc.MARKET_TZ) >= ready
    ]
    if not after:
        return {}
    e = after[0]
    pub = mc.to_et(j.published_at)
    before = [
        i
        for i, d in enumerate(days)
        if datetime.combine(d, mc.session_close(d), tzinfo=mc.MARKET_TZ) < pub
    ]
    out: dict[str, float | None] = {}
    if before:
        out["d0"] = closes[days[e]] / closes[days[before[-1]]] - 1
    for name, k in HORIZONS.items():
        if e + k < len(days):
            out[name] = closes[days[e + k]] / closes[days[e]] - 1
    return out


def fill_outcomes(conn, closes_fn: Callable[[str], dict[date, float] | None]) -> int:
    """Fill every judgment's missing horizons from daily closes. Returns judgments updated."""
    store = NewsJudgmentStore(conn)
    by_symbol: dict[str, list[Judgment]] = {}
    for j in store.list():
        if any(j.outcomes.get(h) is None for h in HORIZONS):
            by_symbol.setdefault(j.symbol, []).append(j)
    n = 0
    for sym, js in by_symbol.items():
        closes = closes_fn(sym)
        if not closes:
            continue
        for j in js:
            new = outcomes_for(j, closes)
            if new and new != {k: v for k, v in j.outcomes.items() if k in new}:
                j.outcomes = {**j.outcomes, **new}
                store.update(j)
                n += 1
    return n


# ── Are its calls worth anything? ─────────────────────────────────────────


def evaluate(judgments: list[Judgment], baselines) -> list[dict]:
    """Excess over each name's own drift per (group, horizon), with the evaluator's intervals.

    A NEGATIVE/HIGH call that "works" has an interval *below* zero. Nothing is
    called an edge without enough independent windows (the evaluator's rule).
    """
    from advisor.learning.evaluate import blocks_of, cluster_ci, verdict

    groups: dict[str, list[Judgment]] = {}
    for j in judgments:
        groups.setdefault(j.group, []).append(j)
    rows = []
    for group, members in sorted(groups.items()):
        for h, k in HORIZONS.items():
            excess = []
            for j in members:
                r = j.outcomes.get(h)
                b = baselines.get(j.symbol, k)
                if r is None or b is None:
                    continue
                excess.append((mc.session_of(j.published_at), r - b))
            if not excess:
                continue
            windows = len(blocks_of(excess, k))
            ci = cluster_ci(excess, block=k) if windows >= 10 else None
            v, why = verdict(len(excess), windows, ci)
            rows.append(
                {
                    "group": group,
                    "horizon": h,
                    "n": len(excess),
                    "windows": windows,
                    "excess": statistics.fmean(x for _, x in excess),
                    "ci": list(ci) if ci else None,
                    "verdict": v.value,
                    "reason": why,
                }
            )
    return rows


def summary(judgments: list[Judgment]) -> dict:
    """Counts a board row shows: material calls by direction."""
    about = [j for j in judgments if j.about_company]
    material = [j for j in about if j.materiality is not Materiality.LOW]
    return {
        "items": len(judgments),
        "about": len(about),
        "positive": sum(j.direction is Direction.POSITIVE for j in material),
        "negative": sum(j.direction is Direction.NEGATIVE for j in material),
        "mixed": sum(j.direction is Direction.MIXED for j in material),
        "high": sum(j.materiality is Materiality.HIGH for j in about),
        "against_thesis": sum(any(c.against_thesis for c in j.claims) for j in about),
        "for_thesis": sum(
            any(not c.against_thesis for c in j.claims)
            and not any(c.against_thesis for c in j.claims)
            for j in about
        ),
    }
