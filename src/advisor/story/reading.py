"""A per-ticker reading: what the recent facts, taken together, imply.

The event stream answers "what happened". A holder also wants "so what": AAOI
raised an at-the-market facility of up to $600 million and, within fifteen
days, committed to factory leases in Houston and Ningbo and bought its Blue
Ridge site. Each row is true alone; the reading is that the company is funding
a capacity build with shareholder dilution and now has to execute.

This is phase 7 (narration) brought forward for one surface, under phase 7's
rules:

- **The facts are assembled deterministically** from the store — position,
  events with their leads, implied expectations, the user's own claims — and
  numbered. No network, no model.
- **The model writes prose over those facts and nothing else.** Every sentence
  cites the facts it rests on, and every number in it must appear in a cited
  fact. The gate is arithmetic, not trust.
- **A reading that fails the gate is not shown.** The model is retried once
  with the gate's objections; if it fails again the result says so and the
  facts are still there to read. There is no fallback to unchecked prose.
- **Never a fair value or a price target.** This project reports what a price
  requires, never what a business is worth.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Callable

from pydantic import BaseModel, Field

from advisor.daemon.market_calendar import now_et, to_et
from advisor.daemon.models import Event, EventTier
from advisor.daemon.store import DaemonStore
from advisor.daemon.summarize import summarize
from advisor.story.scorecard import Scorecard, build_scorecard, scorecard_facts

logger = logging.getLogger(__name__)

WINDOW_DAYS = 45
# The numbers are in the scorecard; prose only connects them.
MAX_SENTENCES = 2
MAX_EVENT_FACTS = 20
# Angles are searched daily. With one shared cap, a day of Grok 4.7 coverage
# pushed the Benzinga story on SPCX's share unlock out of the reading; each
# kind of news now has its own room, and no single angle can take all of it.
MAX_CONTEXT_FACTS = 5
MAX_ANGLE_FACTS = 5
MAX_FACTS_PER_ANGLE = 3

# Position mechanics that re-fire while a condition persists. Only the latest
# of each says anything: SPCX carried eleven CONCENTRATION_WARNING rows, one a
# day, which is one fact. The names are the literals in daemon/mechanics.py;
# a test runs the real emitter so this set cannot drift from it again.
_STANDING_KINDS = {"DEEP_DRAWDOWN", "STOP_BREACHED", "CONCENTRATION_WARNING"}

# Third-party reporting: pulled to explain a move, or found through an angle.
_NEWS_KINDS = {"NEWS_CONTEXT", "NEWS_ANGLE"}


class Stance(StrEnum):
    """Where the facts leave the position, as the model reads them."""

    CONSTRUCTIVE = "CONSTRUCTIVE"
    NEUTRAL = "NEUTRAL"
    CAUTIOUS = "CAUTIOUS"
    AT_RISK = "AT_RISK"


class Fact(BaseModel):
    id: str  # "F3"
    kind: str  # POSITION | VALUATION | EVENT | NEWS | CLAIM
    text: str
    date: str | None = None
    url: str | None = None
    source: str | None = None  # NEWS only: the publisher, for attribution


class Sentence(BaseModel):
    text: str
    facts: list[str] = Field(default_factory=list)


class Draft(BaseModel):
    """What the model is asked to return."""

    stance: Stance
    sentences: list[Sentence]


class ReadingStatus(StrEnum):
    OK = "OK"
    REJECTED = "REJECTED"  # the model's text failed the gate twice
    NO_FACTS = "NO_FACTS"  # nothing in the window to read
    UNAVAILABLE = "UNAVAILABLE"  # no model configured, or it errored


class Reading(BaseModel):
    symbol: str
    status: ReadingStatus
    stance: Stance | None = None
    sentences: list[Sentence] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    facts_hash: str = ""
    model: str | None = None
    # Hash of the instructions the model was given. A reading written under
    # another prompt is another reading: never served from cache as this one,
    # and never pooled with it when stances are scored against outcomes.
    prompt_version: str | None = None
    problems: list[str] = Field(default_factory=list)
    rejected_draft: list[Sentence] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=now_et)
    window_days: int = WINDOW_DAYS
    # Rebuilt on every request, so the numbers shown are current even when
    # the sentences come from cache. The sentences cite the facts they were
    # written from, which carry their own dates.
    scorecard: Scorecard | None = None


# ── Facts ────────────────────────────────────────────────────────────────


def _position_fact(store: DaemonStore, symbol: str) -> str | None:
    book = store.load_latest_book()
    if book is None:
        return None
    held = [p for p in book.positions if p.underlying.upper() == symbol]
    if not held:
        return f"Not held in the book as of {to_et(book.as_of).date()}."
    quantity = sum(p.quantity for p in held)
    signed = sum(p.signed_notional for p in held)
    basis = sum(p.cost_basis for p in held)
    pnl = sum(p.unrealized_pnl for p in held)
    parts = [f"Held: {quantity:,.0f} shares"]
    if book.net_liq:
        parts.append(f"{signed / book.net_liq * 100:.1f}% of net liq")
    if basis:
        parts.append(f"unrealized {pnl / basis * 100:+.1f}% (${pnl:,.0f})")
        parts.append(f"average entry ${basis / abs(quantity):,.2f}, price ${held[0].price:,.2f}")
    return ", ".join(parts) + f"; as of {to_et(book.as_of).date()}."


def _valuation_fact(store: DaemonStore, symbol: str) -> str | None:
    snapshot = store.load_latest_valuation(symbol)
    if snapshot is None:
        return None
    base = snapshot.base_case()
    if base is None:
        return None
    # Dated and tied to its own price. Written as "At $152.64 ... the price
    # requires", the model read $152.64 as a level today's $147.60 was
    # approaching, rather than the price the requirement was computed at.
    text = (
        f"Last valuation, computed on {snapshot.asof} at that day's price of "
        f"${snapshot.price:,.2f} (market cap ${snapshot.market_cap / 1e9:,.2f}bn): that price "
        f"requires {base.describe()}."
    )
    if snapshot.is_stale():
        text += f" Stale: based on figures for the period ending {snapshot.period_end}."
    return text


def _event_text(event: Event) -> str:
    payload = event.payload or {}
    kind = event.kind.replace("FILING_", "").replace("_", " ").lower()
    parts = [f"{kind} (tier {event.tier.value}"]
    if payload.get("form"):
        parts[0] += f", {payload['form']}"
    if payload.get("provider") and event.kind in _NEWS_KINDS:
        parts[0] += f", third-party commentary from {payload['provider']}"
    if payload.get("angle"):
        parts[0] += f", found through the holder's tracked angle '{payload['angle']}'"
    parts[0] += ")"
    if str(payload.get("form", "")).startswith("144"):
        # A live draft wrote "an insider sale was recorded" from a Form 144.
        parts.append("notice of a proposed insider sale — not a completed sale")
    line = summarize(event)
    if line:
        parts.append(line)
    if payload.get("offering_usd") and "from time to time" in str(payload.get("quote", "")).lower():
        parts.append("(a maximum, sold at the market over time — not an amount already raised)")
    if payload.get("lead"):
        parts.append(f"— {payload['lead']}")
    return " ".join(parts)


def gather_facts(
    store: DaemonStore,
    symbol: str,
    *,
    days: int = WINDOW_DAYS,
    scorecard: Scorecard | None = None,
) -> list[Fact]:
    """Every fact the reading may use, numbered, from stored rows only."""
    symbol = symbol.upper()
    facts: list[Fact] = []

    def add(kind: str, text: str, *, date=None, url=None, source=None) -> None:
        fact_id = f"F{len(facts) + 1}"
        facts.append(Fact(id=fact_id, kind=kind, text=text, date=date, url=url, source=source))

    position = _position_fact(store, symbol)
    if position:
        add("POSITION", position)
    if scorecard is not None:
        # The scorecard supersedes the single valuation line: the requirement
        # now sits beside actual growth, consensus and the holder's lines.
        for line in scorecard_facts(scorecard):
            add("SCORECARD", line)
    else:
        valuation = _valuation_fact(store, symbol)
        if valuation:
            add("VALUATION", valuation)

    since = now_et() - timedelta(days=days)
    seen_standing: set[str] = set()
    context = 0
    per_angle: dict[str, int] = {}
    events = []
    for event in store.recent_events(symbol=symbol, since=since, limit=200):
        if event.kind in _STANDING_KINDS:
            if event.kind in seen_standing:
                continue
            seen_standing.add(event.kind)
        if event.kind == "NEWS_CONTEXT":
            context += 1
            if context > MAX_CONTEXT_FACTS:
                continue
        if event.kind == "NEWS_ANGLE":
            angle = str((event.payload or {}).get("angle", ""))
            if (
                per_angle.get(angle, 0) >= MAX_FACTS_PER_ANGLE
                or sum(per_angle.values()) >= MAX_ANGLE_FACTS
            ):
                continue
            per_angle[angle] = per_angle.get(angle, 0) + 1
        events.append(event)
    # Tier A and B first, then context, newest first within each.
    events.sort(key=lambda e: (e.tier is EventTier.C, -e.ts.timestamp()))
    for event in sorted(events[:MAX_EVENT_FACTS], key=lambda e: e.ts, reverse=True):
        payload = event.payload or {}
        occurred = payload.get("accepted_at") or payload.get("published_at") or event.ts.isoformat()
        # Third-party commentary is its own kind: the gate requires any sentence
        # resting on it to say who is talking.
        news = event.kind in _NEWS_KINDS
        add(
            "NEWS" if news else "EVENT",
            _event_text(event),
            date=str(occurred)[:10],
            url=payload.get("url"),
            source=payload.get("provider") if news else None,
        )

    for claim in store.load_claims(symbol):
        add("CLAIM", f"The holder's own thesis claim ({claim.kind.value.lower()}): {claim.text}")
    return facts


def _coarse(fact: Fact) -> str:
    """A fact's text as the cache key sees it.

    The position line carries a live price, so it is keyed only on whether the
    name is held; how far it has moved is judged by ``position_moved`` against
    the cached reading itself. Bucketing was tried first and thrashes at the
    edges: -22.6% and -22.1% straddle a 5pp boundary.
    """
    if fact.kind == "SCORECARD" and fact.text.startswith("Threshold Position weight"):
        # Moves with every price tick; only which side of the limit it is on
        # is worth a new reading.
        return re.sub(r":.*?,\s*", ": ", fact.text).split(" (")[0]
    if fact.kind != "POSITION":
        return fact.text
    return "held" if fact.text.startswith("Held") else "not held"


def facts_hash(facts: list[Fact]) -> str:
    """Identity of a fact set; the reading is regenerated only when this changes."""
    blob = json.dumps([[f.id, f.kind, _coarse(f)] for f in facts])
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


# A move smaller than this since the reading was written does not warrant a
# new one: the reading shows its own dated position line, so it stays honest.
REFRESH_WEIGHT_PP = 1.0
REFRESH_RETURN_PP = 5.0


def _position_numbers(facts: list[Fact]) -> tuple[float | None, float | None]:
    text = next((f.text for f in facts if f.kind == "POSITION"), "")
    weight = re.search(r"([+-]?\d+(?:\.\d+)?)% of net liq", text)
    ret = re.search(r"unrealized ([+-]?\d+(?:\.\d+)?)%", text)
    return (
        float(weight.group(1)) if weight else None,
        float(ret.group(1)) if ret else None,
    )


def position_moved(cached: list[Fact], current: list[Fact]) -> bool:
    """Whether the position has moved enough since ``cached`` to read again."""
    (w0, r0), (w1, r1) = _position_numbers(cached), _position_numbers(current)
    if (w0 is None) != (w1 is None) or (r0 is None) != (r1 is None):
        return True
    # Rounded: 16.4 - 15.4 is 0.9999999999999982, and exactly one point must count.
    return (w0 is not None and round(abs(w1 - w0), 6) >= REFRESH_WEIGHT_PP) or (
        r0 is not None and round(abs(r1 - r0), 6) >= REFRESH_RETURN_PP
    )


# ── The gate ─────────────────────────────────────────────────────────────

# A form name or item code is a label, not a quantity.
_LABELS = re.compile(
    r"\b(?:\d{1,2}-[KQ](?:/A)?|10-[KQ]|20-F|424B\d|S-\d(?:ASR)?|Form \d+|Item \d+\.\d+|Q[1-4]|"
    r"[FH][1-9]\d*)\b",
    re.IGNORECASE,
)
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|ene|abr|ago|dic)[a-z]*\.?"
_DATES = re.compile(
    rf"\b{_MONTHS}\s+\d{{1,2}}(?:,\s*\d{{4}})?|\b\d{{1,2}}\s+(?:de\s+)?{_MONTHS}(?:\s+(?:de\s+)?\d{{4}})?|"
    r"\b\d{4}-\d{2}-\d{2}\b|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(
    # Space-grouped thousands first ("38 311,8"), else digits and separators.
    r"(?P<num>\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)*)\s*"
    r"(?P<unit>%|mil millones|bill[oó]n(?:es)?|billions?|trillions?|millones|mill[oó]n|"
    r"millions?|bn|tn|mm|[bmkt]\b)?",
    re.IGNORECASE,
)
# Spanish counts in the long scale: un billón is 10^12, what English calls a
# trillion. Mapping "billones" to 10^9 rejected a correct "$2 billones" for
# Fool's "$2 trillion" market cap.
_SCALES = {
    "trillion": 1e12, "trillions": 1e12, "tn": 1e12, "t": 1e12,
    "billón": 1e12, "billon": 1e12, "billones": 1e12,
    "mil millones": 1e9, "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9,
    "millones": 1e6, "millon": 1e6, "millón": 1e6, "million": 1e6, "millions": 1e6,
    "mm": 1e6, "m": 1e6,
    "k": 1e3,
}  # fmt: skip

_FORBIDDEN = re.compile(
    r"fair value|price target|target price|intrinsic value|undervalued|overvalued|"
    r"valor (?:justo|razonable|intr[ií]nseco)|precio objetivo|"
    r"(?:sobre|infra|sub)valorad[ao]",
    re.IGNORECASE,
)


def _readings(raw: str) -> list[tuple[float, int]]:
    """Every way ``raw`` can be read, as (value, decimals).

    The model is told to write "1,234.5", and writes Spanish anyway: the
    Ningbo lease came back as "38 311,8" and "38.311,8". Where the last
    separator is followed by exactly three digits it may be a thousands mark
    or a decimal point, so both readings are returned and the gate accepts
    the number if either is grounded. Ambiguity widens the net only between
    readings of the same digits; it never lets an absent figure through.
    """
    digits = re.sub(r"[ \u00a0\u202f]", "", raw)
    seps = [c for c in digits if c in ".,"]
    if not seps:
        return [(float(digits), 0)]

    def as_decimal(mark: str) -> tuple[float, int]:
        other = "," if mark == "." else "."
        whole, _, frac = digits.replace(other, "").rpartition(mark)
        return float(f"{whole.replace(mark, '')}.{frac}"), len(frac)

    last = seps[-1]
    tail = digits.rsplit(last, 1)[1]
    grouped = float(digits.replace(".", "").replace(",", "")), 0
    if len(set(seps)) == 2 or len(seps) == 1 and len(tail) != 3:
        return [as_decimal(last)]
    if len(seps) > 1:  # "1,234,567" or "1.234.567": one mark, repeated, is grouping
        return [grouped]
    return [grouped, as_decimal(last)]


Quantity = tuple[float, int, str, float]  # value as written, decimals, kind, scale


def numbers_in(text: str) -> list[list[Quantity]]:
    """Every quantity in ``text``, dates and form names excluded.

    Each entry lists the ways one written number can be read (see
    ``_readings``). ``kind`` is "pct" or "amount". An amount keeps the scale it
    was written at ("$6.0 million" is 6.0 at 1e6), so rounding is judged where
    the writer rounded rather than against the raw figure.
    """
    text = _DATES.sub(" ", _LABELS.sub(" ", text))
    out: list[list[Quantity]] = []
    for match in _NUMBER.finditer(text):
        unit = (match.group("unit") or "").lower()
        kind, scale = ("pct", 1.0) if unit == "%" else ("amount", _SCALES.get(unit, 1.0))
        out.append([(v, d, kind, scale) for v, d in _readings(match.group("num"))])
    return out


def _grounded(number: Quantity, sources: list[Quantity]) -> bool:
    value, decimals, kind, scale = number
    if kind == "amount" and scale == 1.0 and decimals == 0 and value <= 10:
        # A small bare integer is a count the model made by counting cited
        # facts ("three leases", "2 filings"). Counting is not recomputing.
        return True
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    for src_value, _src_dec, src_kind, src_scale in sources:
        if src_kind != kind:
            continue
        # Sign is direction, and prose carries it in words: "fell 21.5%".
        if abs(abs(src_value * src_scale / scale) - value) <= tolerance:
            return True
    return False


# "[F1, F3]" or "(F4, F6)" written into the prose; the ids travel separately.
_INLINE_CITES = re.compile(r"\s*[\[(]\s*F\d+(?:\s*[,;]\s*F\d+)*\s*[\])]")


def tidy(draft: Draft) -> Draft:
    """Remove fact ids the model wrote into the prose despite being told not to."""
    return Draft(
        stance=draft.stance,
        sentences=[
            Sentence(text=_INLINE_CITES.sub("", s.text).strip(), facts=s.facts)
            for s in draft.sentences
        ],
    )


# Words that describe a crossed line as an uncrossed one. Three live drafts in
# a row did it: "rozando el umbral" for 25.9% against 25%, "roza el límite"
# for 20.6% against 20%.
_SOFTENING = re.compile(
    r"\b(roza\w*|cerca del?|cercan\w*|se acerca\w*|acerc\w*|casi|aproxim\w*|al borde|"
    r"near(ly|s|ing)?|approach\w*|close to|almost|edging|brush\w*)\b",
    re.IGNORECASE,
)

# Words that put a claim in someone else's mouth.
_ATTRIBUTION = re.compile(
    r"\b(seg[uú]n|de acuerdo con|reporta\w*|informa\w*|afirma\w*|sostiene\w*|se[nñ]ala\w*|"
    r"dice\w*|apunta\w*|atribuye\w*|prensa|medios|analistas|comentario\w*|"
    r"according to|report\w*|writes|wrote|says|said|claims?|per|commentary|analysts?)\b",
    re.IGNORECASE,
)


def _publisher(source: str | None) -> str | None:
    """'www.benzinga.com' -> 'benzinga'; 'Motley Fool' -> 'fool'."""
    if not source:
        return None
    words = re.sub(r"^www\.|\.(com|net|org|co|au|uk)\b", "", source.lower()).split()
    return words[-1] if words else None


def _attributed(text: str, cited: list[Fact]) -> bool:
    if _ATTRIBUTION.search(text):
        return True
    names = {_publisher(f.source) for f in cited if f.kind == "NEWS"} - {None}
    return any(name in text.lower() for name in names)


def check(draft: Draft, facts: list[Fact]) -> list[str]:
    """Everything wrong with ``draft``; empty means it may be shown."""
    by_id = {f.id: f for f in facts}
    problems: list[str] = []
    if not 1 <= len(draft.sentences) <= MAX_SENTENCES:
        problems.append(f"write 1 to {MAX_SENTENCES} sentences, not {len(draft.sentences)}")
    for index, sentence in enumerate(draft.sentences, 1):
        cited = [by_id[i] for i in sentence.facts if i in by_id]
        unknown = [i for i in sentence.facts if i not in by_id]
        if unknown:
            problems.append(f"sentence {index} cites facts that do not exist: {unknown}")
        if not cited:
            problems.append(f"sentence {index} cites no fact")
            continue
        if any(f.kind == "NEWS" for f in cited) and not _attributed(sentence.text, cited):
            problems.append(
                f"sentence {index} rests on third-party commentary "
                f"{[f.id for f in cited if f.kind == 'NEWS']} without saying who said it; "
                f"attribute it (e.g. 'según Benzinga')"
            )
        breached = [f.id for f in cited if f.kind == "SCORECARD" and "BREACHED" in f.text]
        if breached and (soft := _SOFTENING.search(sentence.text)):
            problems.append(
                f"sentence {index} calls a breached threshold {breached} "
                f"{soft.group(0)!r}; it is past the line, say so"
            )
        if _FORBIDDEN.search(sentence.text):
            problems.append(
                f"sentence {index} states a valuation opinion "
                f"({_FORBIDDEN.search(sentence.text).group(0)!r}); report what the price requires"
            )
        sources = [q for fact in cited for readings in numbers_in(fact.text) for q in readings]
        for readings in numbers_in(sentence.text):
            if not any(_grounded(q, sources) for q in readings):
                number = readings[0]
                shown = f"{number[0]:g}{'%' if number[2] == 'pct' else ''}"
                problems.append(
                    f"sentence {index} uses {shown}, which is in none of its cited facts "
                    f"{[f.id for f in cited]}"
                )
    return problems


# ── The model ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are writing a short reading of one stock position for its holder. You are given \
numbered facts. Say what they imply TOGETHER — the pattern across them, and what the \
company now has to deliver — not a list of the facts.

Rules, all enforced by a checker that rejects your answer if broken:
1. Use ONLY the facts given. No outside knowledge, no guesses about the future stated \
as fact.
2. Every sentence lists the ids of the facts it rests on.
3. Every number you write must appear in a fact that sentence cites. Keep the number's \
unit and write decimals with a point. You may round (21.53% -> 21.5%), never recompute.
4. Never state a fair value, price target, or that the stock is over/undervalued. You \
may say what the price requires.
5. NEWS facts are third-party commentary: a sentence that cites one must say who said it \
("según Benzinga", "reported by"). Never state it as fact. The holder's thesis claims \
are the holder's beliefs; you may test facts against them.
6. Do not recommend trades.
7. An offering, facility or agreement size is what MAY be sold or spent. Never say \
it was sold, raised or spent unless a fact says so.
8. Keep each number with its own subject and date. A figure computed at one price \
belongs to that price; do not pair it with a price from another fact.
9. No words that grade the price: not "only", "just", "cheap", "expensive", "modest".
   A threshold marked BREACHED is past its line: never "near", "approaching", "roza".
10. 1 or 2 sentences. The holder sees the SCORECARD facts as a table right above your \
text: never restate its rows. Say what they mean together — which number changes how \
another should be read — or what the events add that the table cannot show.
11. Put fact ids only in the "facts" list, never in the sentence text.

stance: CONSTRUCTIVE (facts support the position), NEUTRAL, CAUTIOUS (facts raise the \
bar), AT_RISK (facts cut against the position or its thesis). Choose from the facts, \
not from price alone.
"""


def _user_prompt(symbol: str, facts: list[Fact], language: str, objections: list[str]) -> str:
    lines = [f"Symbol: {symbol}", f"Write the sentences in {language}.", "", "Facts:"]
    for fact in facts:
        dated = f" [{fact.date}]" if fact.date else ""
        lines.append(f"{fact.id} ({fact.kind}){dated}: {fact.text}")
    if objections:
        lines += ["", "Your previous answer was rejected by the checker:"]
        lines += [f"- {o}" for o in objections]
        lines.append("Fix every point.")
    return "\n".join(lines)


Complete = Callable[[str, str], Draft]


def _openrouter() -> tuple[Complete, str] | None:
    """The app's configured model, or None when no key is set."""
    from research_agent.config import ResearchConfig
    from research_agent.llm import OpenRouterLLM

    config = ResearchConfig()
    if not config.openrouter_api_key:
        return None
    llm = OpenRouterLLM(config)
    return (lambda system, user: llm.complete(system, user, response_model=Draft)), config.llm_model


def _prompt_version() -> str:
    """Twelve hex of a hash over the system prompt and the user-prompt template."""
    import hashlib
    import inspect

    body = SYSTEM_PROMPT + "\n--\n" + inspect.getsource(_user_prompt)
    return hashlib.sha256(body.encode()).hexdigest()[:12]


PROMPT_VERSION = _prompt_version()


def read_symbol(
    store: DaemonStore,
    symbol: str,
    *,
    days: int = WINDOW_DAYS,
    language: str = "Spanish",
    refresh: bool = False,
    complete: Complete | None = None,
    model: str | None = None,
    consensus_loader=None,
) -> Reading:
    """The reading for ``symbol``, from cache unless its facts have changed."""
    symbol = symbol.upper()
    loader = {"consensus_loader": consensus_loader} if consensus_loader else {}
    scorecard = build_scorecard(store, symbol, **loader)
    facts = gather_facts(store, symbol, days=days, scorecard=scorecard)
    digest = facts_hash(facts)
    base = dict(
        symbol=symbol, facts=facts, facts_hash=digest, window_days=days, scorecard=scorecard
    )

    if not any(f.kind in ("EVENT", "NEWS") for f in facts):
        return Reading(status=ReadingStatus.NO_FACTS, **base)

    if not refresh:
        cached = store.load_reading(symbol, digest)
        if cached is not None:
            reading = Reading.model_validate_json(cached)
            # Readings cached before prompts were stamped carry None and are
            # rewritten once under the current prompt.
            same_prompt = reading.prompt_version == PROMPT_VERSION
            if same_prompt and not position_moved(reading.facts, facts):
                return reading.model_copy(update={"scorecard": scorecard})

    if complete is None:
        configured = _openrouter()
        if configured is None:
            return Reading(
                status=ReadingStatus.UNAVAILABLE,
                problems=["no LLM configured (RESEARCH_AGENT_OPENROUTER_API_KEY)"],
                **base,
            )
        complete, model = configured

    objections: list[str] = []
    draft: Draft | None = None
    for _attempt in range(2):
        try:
            draft = tidy(complete(SYSTEM_PROMPT, _user_prompt(symbol, facts, language, objections)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("reading: model call failed for %s: %s", symbol, exc)
            return Reading(
                status=ReadingStatus.UNAVAILABLE,
                problems=[str(exc)],
                model=model,
                prompt_version=PROMPT_VERSION,
                **base,
            )
        objections = check(draft, facts)
        if not objections:
            break

    if objections:
        # Kept for audit, never displayed as the reading.
        reading = Reading(
            status=ReadingStatus.REJECTED,
            problems=objections,
            rejected_draft=draft.sentences,
            model=model,
            prompt_version=PROMPT_VERSION,
            **base,
        )
    else:
        reading = Reading(
            status=ReadingStatus.OK,
            stance=draft.stance,
            sentences=draft.sentences,
            model=model,
            prompt_version=PROMPT_VERSION,
            **base,
        )
    store.save_reading(symbol, digest, reading.model_dump_json())
    return reading
