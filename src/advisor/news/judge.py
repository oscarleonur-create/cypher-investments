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
  and Swing, so the Tavily budget is untouched. News is judged only once
  ``news.verify`` has accepted its date, and at the checked date.
- **It must quote.** Every judgment carries a verbatim span of the item that
  supports it; a quote the item does not contain voids the judgment.
- **It reads the article and knows the position** (2026-09-28, the user found
  the first judgments "escueto"; 17 of 21 had been judged from a headline and
  a teaser). A thin item is read from the publisher's page (``news.article``);
  each name comes with a computed context (``news.context``): the position
  and its exit stop, the company's size, what the price did the first session
  after each item, and the last 30 days of filings and headlines. Each
  judgment says what happened, how big it is against the company, why it
  matters for this holder and what would confirm it; a weekly synthesis per
  name weighs them. It describes; it never recommends.
- **No number it cannot show.** A number the judgment writes must be in the
  item or the context, a rounding of one, or arithmetic on two (a share of
  shares outstanding); product names ("800G") and periods ("Q3") are not
  quantities. A stray number voids the judgment — or, in "watch", only the
  watch: a made-up lock-up date must not survive, the rest of the call may.
- **The market's read is computed**, not asked: the first session's move in
  the name's own sigma (``read_market``).
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
# A name never judged under the current prompt starts from this far back, so
# the first reading of a position is a month of its news, not a week.
FIRST_RUN_DAYS = 30
# Items per model call: each can carry a full article now, so fewer per call.
BATCH = 5
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


class MarketRead(StrEnum):
    """What the price had done by the first session after the item. Computed, not asked.

    The model called a +5.1% (0.8 sigma) day after a NEGATIVE lock-up story
    "moved with" it; this is arithmetic on the market line, so the code does it.
    """

    MOVED_WITH = "MOVED_WITH"  # beyond MOVE_SIGMAS, in the news' direction
    MOVED_AGAINST = "MOVED_AGAINST"  # beyond MOVE_SIGMAS, the other way
    MOVED = "MOVED"  # beyond MOVE_SIGMAS, news with no single direction
    QUIET = "QUIET"  # within its usual daily range
    UNKNOWN = "UNKNOWN"  # no session has traded on it yet, or no volatility estimate


# A first-session move this many of the name's daily sigmas is the market reading it.
MOVE_SIGMAS = 1.0


def read_market(direction: Direction, move: float | None, z: float | None) -> MarketRead:
    """The market's read of an item from its first session's move. Pure."""
    if move is None or z is None:
        return MarketRead.UNKNOWN
    if abs(z) < MOVE_SIGMAS:
        return MarketRead.QUIET
    if direction is Direction.POSITIVE:
        return MarketRead.MOVED_WITH if move > 0 else MarketRead.MOVED_AGAINST
    if direction is Direction.NEGATIVE:
        return MarketRead.MOVED_WITH if move < 0 else MarketRead.MOVED_AGAINST
    return MarketRead.MOVED


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
    what: str = ""  # what happened, in one or two sentences, with its numbers
    magnitude: str = ""  # how big it is against the company or the position
    why: str  # why it matters (or not) for this holder: thesis, position, size
    watch: str = ""  # what would confirm or refute it, and when
    market_read: MarketRead = MarketRead.UNKNOWN


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
    what: str = ""
    magnitude: str = ""
    watch: str = ""
    market_read: MarketRead = MarketRead.UNKNOWN
    # How the item was read: "article" (the publisher's page), "feed" (the
    # feed's own text) or "headline" (nothing more could be read).
    read_from: str = "feed"
    market: str | None = None  # the market line it was shown
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
You are the news analyst for a long-term holder of one listed company. You \
are given what is known about the company and the holder's position (lines \
C1, C2, ...), the holder's thesis claims if any, and the news items (N1, \
N2, ...), each with its text and what the share price did on the first \
session after it.

For EACH item return one judgment:
- about_company: is the item really about this company, or only a list, a \
comparison or a sector piece that mentions it?
- event_type: RESULTS, GUIDANCE, CUSTOMER (a customer, contract, order or \
partnership), PRODUCT, REGULATORY, LEGAL, FINANCING (equity or debt raised, \
buybacks), DEAL, MANAGEMENT, INSIDER (insiders or lock-ups; not a fund \
buying), ANALYST (ratings, targets, estimates, funds' positions), SECTOR, \
PRICE (a story about the share price with no new business fact), OTHER.
- direction: is the FACT good or bad for the BUSINESS (revenue, margins, \
solvency, share count, competitive position)? Judge the fact, not the tone \
of the headline or the share-price move. POSITIVE, NEGATIVE, MIXED, NEUTRAL.
- materiality: HIGH only if it could change revenue, margins, solvency or the \
share count enough that a holder must act; MEDIUM if it adds real \
information; LOW otherwise (opinion, lists, price recaps).
- novelty: NEW fact, FOLLOW_UP on a known story, or REHASH. Judge it against \
"Earlier on this name" in the context.
- basis: COMPANY (the company said it), REPORTED (a journalist reports a \
fact), OPINION, or RUMOR.
- claims: only if the item bears on a thesis claim, its id and SUPPORTS (the \
item is evidence the claim's text is true) or CONTRADICTS. INVALIDATION and \
RISK claims describe what would go wrong.
- quote: copy, word for word, the few words of the item your judgment rests on.
- what: what happened, in one or two sentences, with the item's own numbers \
(amounts, dates, percentages). Not the headline restated.
- magnitude: how big it is AGAINST THE COMPANY (market cap, revenue, shares \
outstanding in the context) or the holder's position — e.g. "14.6M shares, \
about 5% of shares outstanding" — or "not quantifiable from the item".
- why: two or three sentences for THIS holder: what it changes for the \
business and for the thesis or position (weight, distance to the stop). Say \
plainly when it changes nothing.
- watch: the concrete, dated or checkable thing that would confirm or refute \
it (a filing, a results date, a lock-up date, a price level from the context).

The context lines are facts about the holder's position and the company: \
never contradict them (C1 states where the price stands against the exit \
stop). Describe and weigh; do NOT recommend buying, selling, holding, \
trimming or tightening anything — the holder's own rules decide that.

Use only numbers that appear in the item, the context lines or the item's \
market line; derive a ratio only from two of those. Do not invent facts. If \
the item says too little (read from the headline only), judge it NEUTRAL and \
LOW and say so."""


def prompt_version() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]


# ── The items ─────────────────────────────────────────────────────────────


def _text(item) -> str:
    return f"{item.title} {item.summary or ''}"


def checked_dates(store, symbol: str, since: datetime) -> dict[str, datetime]:
    """URL -> checked publication date, for this name's news whose date ``news.verify`` accepted.

    The check is recorded on the news event (``set_news_verification``), not
    on the archived item, so the two are joined by URL.
    """
    from advisor.news.verify import usable

    out = {}
    for e in store.recent_events(symbol=symbol, since=since - timedelta(days=2), limit=2000):
        p = e.payload or {}
        if p.get("url") and usable(p):
            out[p["url"]] = e.ts
    return out


def items_to_judge(store, conn, symbol: str, now: datetime) -> list:
    """The name's archived items of the last JUDGE_WINDOW_DAYS not judged under this prompt.

    News is judged only once its date has been checked (user decision,
    2026-09-27: an unverified date may support nothing), and is dated at the
    checked publication. A filing is the regulator's own record and needs no check.
    """
    done = NewsJudgmentStore(conn).keys(symbol, prompt_version())
    since = now - timedelta(days=JUDGE_WINDOW_DAYS if done else FIRST_RUN_DAYS)
    checked = checked_dates(store, symbol, since)
    out, seen = [], set()
    for item in store.source_items_between(symbol, since - timedelta(days=2), now):
        if item.tier.value not in JUDGED_TIERS:
            continue
        if item.tier.value == "PRIMARY":
            if not item.summary:
                continue  # a filing with no lead has nothing to judge but its form
        elif item.url in checked:
            item = item.model_copy(update={"published_at": checked[item.url]})
        else:
            continue  # its date is unchecked or failed the check
        if item.published_at < since:
            continue
        title = item.title.strip().lower()
        if item.dedup_key() in done or title in seen:
            continue
        seen.add(title)
        out.append(item)
    return out


class Entry(BaseModel):
    """One item as the model is shown it: its text, how it was read, what the price did."""

    model_config = {"arbitrary_types_allowed": True}

    item: object
    text: str
    read_from: str
    market: str
    move: float | None = None  # first-session close-to-close move
    z: float | None = None  # the same, in the name's daily sigma

    @property
    def full_text(self) -> str:
        return f"{self.item.title} {self.text}"


_READ_LABEL = {
    "article": "full article",
    "feed": "feed summary only",
    "headline": "HEADLINE ONLY - nothing more could be read",
}


def user_prompt(
    symbol: str, company: str | None, entries: list, claims: list, context: list[str] = ()
) -> str:
    lines = [f"Company: {company or symbol} ({symbol})", ""]
    if context:
        lines += [f"C{n}: {c}" for n, c in enumerate(context, 1)]
        lines.append("")
    if claims:
        lines.append("The holder's thesis claims:")
        lines += [f"{c.id} [{c.kind.value}]: {c.text}" for c in claims if c.id]
        lines.append("")
    lines.append("Items:")
    for n, e in enumerate(entries, 1):
        it = e.item
        lines.append(
            f"N{n} [{it.published_at.date().isoformat()}] ({it.provider}, "
            f"{it.doc_type or 'NEWS'}; {_READ_LABEL[e.read_from]}) {it.title}\n"
            f"    {e.market}" + (f"\n    Text: {e.text.strip()}" if e.text.strip() else "")
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


# Names, not quantities: periods ("Q3", "H2", "FY2027"), the SEC's form names
# ("10-Q", "8-K", "Form 4", "13G", "424B5" — "watch the next 10-Q" dropped nine
# watches in the first live run), and the prompt's own line ids ("C2", "N3").
_LABELS = re.compile(
    r"\b(?:Q[1-4]|H[12]|[FC]Y\s?'?\d{2,4})\b"
    r"|\b(?:10-[KQ]|8-K|6-K|20-F|40-F|S-[134]|424B\d|13[DFG]|Form\s?\d{1,3}|144)\b"
    r"|\b[CNJF]\d{1,2}\b",
    re.IGNORECASE,
)
# A number written as a share or a multiple: "6.1%", "about 9x", "3 times".
_SCALED = re.compile(r"(\d[\d,]*\.?\d*)\s*(?:%|x\b|times\b)", re.IGNORECASE)
# A number fused to letters ("800G", "1.6T", "224G") is a product name: it is
# known only if the same token is.
_TOKEN = re.compile(r"\b\d[\d.,]*[A-Za-z]+\b|\b[A-Za-z]+\d[\w.]*\b")


def _numbers(text: str) -> set[str]:
    out = set()
    for m in _NUM.findall(_LABELS.sub(" ", text)):
        n = m.replace(",", "").rstrip(".")
        if n:
            out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(_LABELS.sub(" ", text))}


def _floats(nums: set[str]) -> list[float]:
    out = []
    for n in nums:
        try:
            f = abs(float(n))
        except ValueError:
            continue
        if f and not (1900 <= f <= 2100 and f == int(f)):  # a year is not a size
            out.append(f)
    return out


def _derivable(x: str, item_nums: set[str], context_nums: set[str]) -> bool:
    """``x`` is a share or multiple of an item number against a context number (1.5%).

    The prompt asks for size against the company — "14.6M shares, about 6% of
    the 237.6M outstanding" — which is one number from the item over one from
    the context. Nothing wider: over every pair of every known number, almost
    any figure is "derivable" (an invented "99 waves" passed that way).
    """
    try:
        v = abs(float(x))
    except ValueError:
        return False
    for a in _floats(item_nums):
        for b in _floats(context_nums):
            for c in (a / b * 100, b / a * 100, a / b, b / a):
                if c and abs(v - c) / c <= 0.015:
                    return True
    return False


def _stray(
    fields: dict[str, str],
    known: set[str],
    tokens: set[str] = frozenset(),
    *,
    item_nums: set[str] = frozenset(),
    context_nums: set[str] = frozenset(),
):
    """Numbers in each field that are not known, a rounding of one, or (for a share or
    a multiple only) an item number against a context number."""
    out = {}
    for name, value in fields.items():
        text = value
        for t in _tokens(value):
            if t in tokens:  # a product name the item or context uses: not a quantity
                text = re.sub(re.escape(t), " ", text, flags=re.IGNORECASE)
        scaled = _numbers(" ".join(_SCALED.findall(_LABELS.sub(" ", text))))
        bad = sorted(
            n
            for n in _numbers(text) - known
            if not _close_to(n, known)
            and not (n in scaled and _derivable(n, item_nums, context_nums))
        )
        if bad:
            out[name] = bad
    return out


def check(
    call: ItemCall, item, claim_ids, *, known_text: str = ""
) -> tuple[ItemCall | None, list[str]]:
    """Keep what the item supports. Pure. (None, problems) when the judgment is void.

    ``item`` is the text the model read (a SourceItem, or the full text of an
    Entry). The quote must be in it; every number the judgment writes must be
    in it, in ``known_text`` (the context and market lines), or a rounding of one.
    """
    problems = []
    text = item if isinstance(item, str) else _text(item)
    if not quoted(call.quote, text):
        return None, [f"{call.id}: the quote is not in the item"]
    item_nums, context_nums = _numbers(text), _numbers(known_text)
    known = item_nums | context_nums
    tokens = _tokens(text) | _tokens(known_text)
    sizes = {"item_nums": item_nums, "context_nums": context_nums}
    fields = {"what": call.what, "magnitude": call.magnitude, "why": call.why}
    stray = _stray(fields, known, tokens, **sizes)
    if stray:
        where = "; ".join(f"'{k}' uses {v}" for k, v in stray.items())
        return None, [f"{call.id}: {where}, not in the item or its context"]
    # What to watch is forward-looking: a date or level the item does not give
    # (a lock-up day it made up) loses the watch, not the whole judgment.
    watch = _stray({"watch": call.watch}, known, tokens, **sizes) if call.watch else {}
    if watch:
        bad = watch["watch"]
        problems.append(f"{call.id}: 'watch' dropped, it uses {bad} not in the item or context")
        call = call.model_copy(update={"watch": ""})
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


def _openrouter() -> tuple[Complete, Callable, str] | None:
    """(judge, summarize, model) over the configured model; None when none is."""
    from research_agent.config import ResearchConfig
    from research_agent.llm import OpenRouterLLM

    config = ResearchConfig()
    if not config.openrouter_api_key:
        return None
    llm = OpenRouterLLM(config)
    return (
        (lambda s, u: llm.complete(s, u, response_model=Draft)),
        (lambda s, u: llm.complete(s, u, response_model=SummaryDraft)),
        config.llm_model,
    )


def judge_symbol(
    store,
    conn,
    symbol: str,
    now: datetime,
    *,
    company: str | None = None,
    complete: Complete | None = None,
    summarize: Callable | None = None,
    model: str | None = None,
    closes: Callable | None = None,
    fetch=None,
) -> tuple[list[Judgment], list[str]]:
    """Judge and store every unjudged item of ``symbol``, then its weekly synthesis.

    ``closes(symbol)`` -> [(date, close)] for the market lines (default: yfinance);
    ``fetch(url)`` -> html for thin items (default: the publisher's page). A test
    that passes ``complete`` and no ``summarize`` gets no synthesis.
    """
    from advisor.news.article import ArticleCache, text_for
    from advisor.news.context import closes_for, market_move, name_context, sigma_of

    symbol = symbol.upper()
    items = items_to_judge(store, conn, symbol, now)
    if not items:
        return [], []
    if complete is None:
        configured = _openrouter()
        if configured is None:
            return [], [f"{symbol}: no model configured"]
        complete, summarize, model = configured
    history = closes_for(symbol, closes)
    sigma = sigma_of(history)
    cache = ArticleCache(conn)
    entries = []
    for it in items:
        text, read_from = text_for(it, cache, fetch=fetch)
        line, move, z = market_move(it.published_at, history, sigma)
        entries.append(Entry(item=it, text=text, read_from=read_from, market=line, move=move, z=z))
    judged = {i.url for i in items} | {i.title.strip().lower() for i in items}
    context = name_context(store, symbol, now, history, exclude=judged)
    claims = [c for c in store.load_claims(symbol) if c.id]
    claim_ids = {c.id: c.kind.value for c in claims}
    out, problems = [], []
    judgments = NewsJudgmentStore(conn)
    for start in range(0, len(entries), BATCH):
        batch = entries[start : start + BATCH]
        try:
            draft = complete(SYSTEM_PROMPT, user_prompt(symbol, company, batch, claims, context))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{symbol}: model failed: {exc}")
            continue
        by_id = {c.id: c for c in draft.judgments}
        for n, e in enumerate(batch, 1):
            call = by_id.get(f"N{n}")
            if call is None:
                problems.append(f"{symbol} N{n}: not judged by the model")
                continue
            known = "\n".join([*context, e.market, *(c.text for c in claims)])
            kept, why = check(call, e.full_text, claim_ids, known_text=known)
            problems += [f"{symbol} {p}" for p in why]
            if kept is None:
                continue
            it = e.item
            kept = kept.model_copy(update={"market_read": read_market(kept.direction, e.move, e.z)})
            j = Judgment(
                key=it.dedup_key(),
                symbol=symbol,
                published_at=it.published_at,
                title=it.title,
                provider=it.provider,
                tier=it.tier.value,
                url=it.url,
                read_from=e.read_from,
                market=e.market,
                model=model,
                prompt_version=prompt_version(),
                judged_at=mc.now_et(),
                problems=why,
                **kept.model_dump(exclude={"id"}),
            )
            judgments.add(j)
            out.append(j)
    if out and summarize is not None:
        _, ps = synthesize(conn, symbol, now, context, summarize=summarize, model=model)
        problems += ps
    return out, problems


# ── The week, per name ────────────────────────────────────────────────────


class SummaryDraft(BaseModel):
    """What the model returns for a name's week of judged news."""

    net: Direction
    headline: str  # one line: the week for this holder
    text: str  # three to five sentences
    thesis: str = ""  # what the week does to the thesis or the position; "" if nothing
    watch: list[str] = Field(default_factory=list)  # the next things to check


class NameSummary(BaseModel):
    symbol: str
    day: date
    net: Direction
    headline: str
    text: str
    thesis: str = ""
    watch: list[str] = Field(default_factory=list)
    items: int  # judgments it was written from
    model: str | None = None
    prompt_version: str
    generated_at: datetime
    problems: list[str] = Field(default_factory=list)


SUMMARY_PROMPT = """\
You write the weekly news synthesis for a long-term holder of one company, \
from the judgments of the individual items (J1, J2, ...) and the context \
(C1, C2, ...). Weigh the items by materiality and basis: a company filing \
outweighs an opinion piece; off-topic items do not count.

Return: net (POSITIVE, NEGATIVE, MIXED, NEUTRAL for the business over the \
week); headline (one line); text (three to five sentences: what actually \
changed, what is noise, and how big the changes are against the company); \
thesis (what the week does to the holder's thesis claims or position — its \
weight and distance to the stop — or "" when nothing); watch (up to three \
concrete next things to check, with dates where the items give them).

The context lines are facts about the position and the company: never \
contradict them. Describe and weigh; do NOT recommend buying, selling, \
holding, trimming or tightening anything — the holder's own rules decide that.

Use only numbers that appear in the judgments or the context. If nothing \
material happened, say so in one sentence."""


def summary_version() -> str:
    return hashlib.sha256((SYSTEM_PROMPT + SUMMARY_PROMPT).encode()).hexdigest()[:12]


def _judgment_line(n: int, j: Judgment) -> str:
    if not j.about_company:
        return f"J{n} [{j.published_at.date()}] off-topic: {j.title}"
    links = ", ".join(f"{'against' if c.against_thesis else 'for'} {c.kind} {c.claim_id}"
                      for c in j.claims)  # fmt: skip
    return (
        f"J{n} [{j.published_at.date()}] {j.direction.value} {j.materiality.value} "
        f"{j.event_type.value} ({j.basis.value}, {j.novelty.value}; market "
        f"{j.market_read.value}){' thesis: ' + links if links else ''} | {j.title}\n"
        f"    quote: {j.quote}\n"
        f"    what: {j.what}\n    magnitude: {j.magnitude}\n    why: {j.why}\n"
        f"    watch: {j.watch}"
    )


def synthesize(
    conn,
    symbol: str,
    now: datetime,
    context: list[str],
    *,
    summarize: Callable,
    model: str | None = None,
) -> tuple[NameSummary | None, list[str]]:
    """The week's synthesis for ``symbol`` from its stored judgments. Gated like an item."""
    store = NewsJudgmentStore(conn)
    week = store.list(
        symbol=symbol, since=now - timedelta(days=JUDGE_WINDOW_DAYS), version=prompt_version()
    )
    if not any(j.about_company for j in week):
        return None, []
    lines = [_judgment_line(n, j) for n, j in enumerate(week, 1)]
    user = "\n".join(
        [f"Company: {symbol}", *[f"C{n}: {c}" for n, c in enumerate(context, 1)], "", *lines]
    )
    try:
        draft = summarize(SUMMARY_PROMPT, user)
    except Exception as exc:  # noqa: BLE001
        return None, [f"{symbol}: synthesis failed: {exc}"]
    known = _numbers(user)
    fields = {"headline": draft.headline, "text": draft.text, "thesis": draft.thesis,
              **{f"watch{i}": w for i, w in enumerate(draft.watch)}}  # fmt: skip
    stray = _stray(fields, known, _tokens(user))
    if stray:
        where = "; ".join(f"'{k}' uses {v}" for k, v in stray.items())
        return None, [f"{symbol} synthesis: {where}, not in the judgments or context"]
    summary = NameSummary(
        symbol=symbol,
        day=now.date(),
        items=len(week),
        model=model,
        prompt_version=summary_version(),
        generated_at=mc.now_et(),
        **draft.model_dump(),
    )
    store.save_summary(summary)
    return summary, []


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
CREATE TABLE IF NOT EXISTS news_summaries (
    symbol         TEXT NOT NULL,
    day            TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    payload_json   TEXT NOT NULL,            -- news.judge.NameSummary
    PRIMARY KEY (symbol, day, prompt_version)
);
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

    def save_summary(self, summary: "NameSummary") -> None:
        """One synthesis per name, day and prompt: the day's last one wins."""
        self._conn.execute(
            "INSERT OR REPLACE INTO news_summaries (symbol, day, prompt_version, payload_json) "
            "VALUES (?, ?, ?, ?)",
            (summary.symbol, summary.day.isoformat(), summary.prompt_version,
             summary.model_dump_json()),
        )  # fmt: skip
        self._conn.commit()

    def latest_summary(self, symbol: str, version: str | None = None) -> "NameSummary | None":
        version = version or summary_version()
        row = self._conn.execute(
            "SELECT payload_json FROM news_summaries WHERE symbol = ? AND prompt_version = ? "
            "ORDER BY day DESC LIMIT 1",
            (symbol.upper(), version),
        ).fetchone()
        return NameSummary.model_validate_json(row["payload_json"]) if row else None

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
