"""Actionables: what to do now, on the book and on what is tracked. Verbs only.

The user, 2026-10-04: "Necesito un tab de accionables. Los accionables son
sobre nuestras acciones o sobre el tracking que estamos haciendo, los
accionables deben tener uno o dos bullets. Y son solo acciones." The same
day: a REVIEW counts (as DECIDE), TRADE ONLY picks count, English, and a buy
carries its size and stop.

Nothing here makes a new call. Each actionable restates one the system
already made under its own rules, in one line and one or two bullets:

    SELL    held: the proposal's EXIT (its stop from cost, a filing, a report, a halt)
    DECIDE  held: a REVIEW (a thesis rule broke, P/S rich, a filing or report to answer);
            results due within the entry guard's sessions: hold through them or cut before;
            results just reported: keep or sell, with the bar before and after them;
            a group of held names that move as one bet, before a dated macro event
            within the entry guard's sessions: hedge it or carry it through
    READ    a sized buy that waits only on reading today's tier-A event
    BUY     a watched name's ENTER or ADD; a pick on its first day, before that
            session closes (the entry the replays measured), whose verdict is
            ENTER or TRADE ONLY

What is left out is left out on purpose: a TRIM (the user removed the 20%
concentration trim, 2026-10-04; one recorded before still reads TRIM and is
not shown), HOLD, NONE, IN_ZONE (an acceptable
price, no reason to act today), a WAIT with nothing sized behind it or that
waits on results, a pick past its first session (the plan replay measured
later entries at about zero) and a pick verdict of WAIT, UNPROVEN or CAN'T
VALUE. Calls on names no longer held (TE's EXIT after it was sold) are dropped
against the book.

**Results.** A held name's results are the one date its price is tested
against the business, so they are a decision with a deadline: hold through
them or cut before. The item states what the price requires, what the
business delivered and what the consensus expects — and, if the consensus is
met, what the years after it must still deliver. Arithmetic on stored
numbers, never a call on which way the results go: the answer is the
holder's.

After the results, for ``AFTER_RESULTS_SESSIONS`` sessions, the same name
asks again with the evidence they brought: the price and what it requires,
before and after; the consensus, before and after; and whether the new
quarter is in the filings read yet. "Before" is the last call made before the
results day — and only a date that call still listed counts, so a date the
calendar moved is never mistaken for results that happened.

**Groups.** Names that correlate as one bet (``risk.book``) are decided as
one, and only before a dated macro event (``macro.calendar``): a deadline is
what makes it an action. The item says what the group weighs, what share of
the book's risk it carries, and the hedge measured for it — named, never
staged. It is listed under ``BOOK``, not a ticker.

**Answers.** A SELL or BUY closes on its own when the book shows it
done; DONE quiets it for the grace the decisions module allows. KEEP on a
thesis DECIDE records a decision on the claims themselves — the same answer
the thesis card takes, which is what stops the proposal from asking. KEEP on
any other DECIDE is acknowledged at the reading it was made at (P/S
percentile for a rich price), so it returns only if that worsens materially.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta

from pydantic import BaseModel, Field

from advisor.entry.proposal import Proposal, current_params, position_stop_pct

ORDER = {"SELL": 0, "DECIDE": 1, "READ": 2, "BUY": 3}
PROPOSAL_DAYS = 10  # how far back proposals are read; only the newest session's are used
READ_BLOCKER = "a tier-A event on this name today"
PICK_VERDICTS = ("ENTER", "TRADE ONLY")
AFTER_RESULTS_SESSIONS = 3  # sessions after the results day the reading is asked for
BOOK = "BOOK"  # the symbol a book-level actionable is listed and answered under
# An answer, and the decision verdict it is recorded as.
ANSWERS = {"KEEP": "ACKNOWLEDGED", "DONE": "ACTED", "SKIP": "DISMISSED"}


class Actionable(BaseModel):
    id: str  # symbol:subject — stable across rebuilds of the same situation
    verb: str  # SELL | DECIDE | READ | BUY
    symbol: str
    section: str  # "book" (held) | "tracking"
    title: str
    bullets: list[str]
    answers: list[str]  # KEEP | DONE | SKIP
    asof: datetime
    source: str
    shares: float | None = None
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    weight: float | None = None  # of the book, held names: orders the list
    # What an answer records: {"kind": "CLAIM", "ids": [...]} or
    # {"kind": "ACTIONABLE", "id": ..., "observed": ..., "worse_is": ...}.
    subject: dict = Field(default_factory=dict)


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _shares(n: float) -> str:
    return f"{n:g} share{'' if n == 1 else 's'}"


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


class Holding(BaseModel):
    symbol: str
    quantity: float
    cost: float | None
    price: float | None
    notional: float


def holdings(book) -> dict[str, Holding]:
    """Long equity positions per symbol, accounts summed. Pure."""
    out: dict[str, Holding] = {}
    if book is None:
        return out
    by: dict[str, list] = {}
    for pos in book.positions:
        if pos.quantity > 0 and not pos.is_option:
            by.setdefault(pos.underlying.upper(), []).append(pos)
    for sym, ps in by.items():
        qty = sum(p.quantity for p in ps)
        basis = sum(p.cost_basis for p in ps)
        out[sym] = Holding(
            symbol=sym,
            quantity=qty,
            cost=basis / qty if qty else None,
            price=ps[0].price or None,
            notional=sum(p.notional for p in ps),
        )
    return out


def newest_session(proposals: list[Proposal]) -> dict[str, Proposal]:
    """The newest proposal per symbol, from the newest session on file only. Pure.

    A call from an older session is not current: TE's EXIT of 09-29 still
    read as the latest word on a name sold since.
    """
    if not proposals:
        return {}
    last = max(p.session for p in proposals)
    out: dict[str, Proposal] = {}
    for p in proposals:
        if p.session != last:
            continue
        cur = out.get(p.symbol.upper())
        if cur is None or p.built_at > cur.built_at:
            out[p.symbol.upper()] = p
    return out


def _source(p: Proposal) -> str:
    # A proposal is recorded once per session and action, so an hourly job
    # that says REVIEW again keeps the first row: the time is when it was first said.
    return (
        f"entry call of {p.session.strftime('%m-%d')}, first made {p.built_at.strftime('%H:%M')} ET"
    )


def _position_line(h: Holding, sigma) -> str | None:
    """Where the holding stands against its cost and its stop. Pure."""
    if not h.cost or not h.price:
        return None
    line = f"{h.price / h.cost - 1:+.1%} from your cost {_money(h.cost)}"
    pct = position_stop_pct(sigma)
    if pct is not None:
        line += f"; its stop is {_money(h.cost * (1 - pct))}"
    return line


def _stop_bullet(p: Proposal, h: Holding) -> str:
    price = p.price or h.price
    pct = position_stop_pct((p.features or {}).get("sigma"))
    if not (price and h.cost and pct is not None):
        return "past its stop from your cost"
    return (
        f"{_money(price)} is past its stop {_money(h.cost * (1 - pct))}: "
        f"{price / h.cost - 1:+.1%} from your cost {_money(h.cost)}"
    )


def _sell(p: Proposal, h: Holding, calls: list[dict]) -> Actionable:
    bullets = [
        _stop_bullet(p, h) if c.get("rule") == "stop" else c.get("why", "")
        for c in calls
        if c.get("action") == "EXIT"
    ][:2]
    price = h.price or p.price
    worth = f" (~{_money(h.quantity * price)})" if price else ""
    rules = "+".join(sorted({c.get("rule", "") for c in calls if c.get("action") == "EXIT"}))
    return Actionable(
        id=f"{h.symbol}:SELL:{rules}",
        verb="SELL",
        symbol=h.symbol,
        section="book",
        title=f"SELL {h.symbol} — all {_shares(h.quantity)}{worth}",
        bullets=bullets,
        answers=["DONE"],
        asof=p.built_at,
        source=_source(p),
        shares=h.quantity,
        subject={
            "kind": "ACTIONABLE",
            "id": f"act:SELL:{rules}",
            "observed": None,
            "worse_is": "NEITHER",
        },  # fmt: skip
    )


REVIEW_TITLES = {
    "thesis": "keep or sell: a rule of your thesis broke",
    "rich": "trim or keep: priced rich against its own two years",
    "filing": "keep or sell: a filing to answer",
    "halt": "keep or sell: a trading halt",
}


def _decide(p: Proposal, h: Holding, call: dict, claims: list[tuple[str, str]]) -> Actionable:
    rule = call.get("rule", "")
    position = _position_line(h, (p.features or {}).get("sigma"))
    title = REVIEW_TITLES.get(rule) or (
        "keep or sell: a news report to answer" if rule.startswith("news") else "keep or sell"
    )
    if rule == "thesis":
        texts = [e["text"] for e in call.get("evidence") or [] if e.get("text")]
        by_text = {t.strip(): cid for cid, t in claims}
        ids = [by_text[t.strip()] for t in texts if t.strip() in by_text]
        first = f'Your rule: "{texts[0]}"' if texts else call.get("why", "")
        if len(texts) > 1:
            first += f" (+{len(texts) - 1} more)"
        bullets = [first]
        if ids and len(ids) == len(texts):
            subject = {"kind": "CLAIM", "ids": ids}
        else:  # the claim's text changed since: answer the situation instead
            subject = {"kind": "ACTIONABLE", "id": f"act:DECIDE:thesis:{_hash('|'.join(texts))}",
                       "observed": None, "worse_is": "NEITHER"}  # fmt: skip
        key = "thesis"
    elif rule == "rich":
        # The percentile already says how rare the price is; the day count repeats it.
        bullets = [
            re.sub(r": it has been this expensive on \d+% of days;", ";", call.get("why", ""))
        ]
        pct = (p.features or {}).get("ps_percentile")
        subject = {"kind": "ACTIONABLE", "id": "act:DECIDE:rich", "observed": pct, "worse_is": "UP"}
        key = "rich"
    else:
        why = call.get("why", "")
        bullets = [why]
        key = f"{rule}:{_hash(why)}"
        subject = {"kind": "ACTIONABLE", "id": f"act:DECIDE:{key}", "observed": None,
                   "worse_is": "NEITHER"}  # fmt: skip
    if position:
        bullets.append(position)
    return Actionable(
        id=f"{h.symbol}:DECIDE:{key}",
        verb="DECIDE",
        symbol=h.symbol,
        section="book",
        title=f"DECIDE {h.symbol} — {title}",
        bullets=bullets[:2],
        answers=["KEEP"],
        asof=p.built_at,
        source=_source(p),
        subject=subject,
    )


def _pct_range(low: float, high: float) -> str:
    return f"{low:.1%}" if high - low < 0.0005 else f"{low:.1%} to {high:.1%}"


def _when(n: int) -> str:
    return "today" if n == 0 else "next session" if n == 1 else f"in {n} sessions"


def remaining_after(
    valuation, consensus, price: float | None
) -> tuple[str, int, float, float] | None:
    """If the last consensus year is met: (its label, the horizon's year, the
    lowest and highest growth a year the years after it still need). Pure.

    The same arithmetic as the scorecard's "If FY holds" row, at ``price``,
    across the margins the company offers. None when any input is missing.
    """
    from advisor.valuation.consensus import remaining_cagr
    from advisor.valuation.margins import required_range

    base = valuation.base_case() if valuation is not None else None
    if base is None or consensus is None or not consensus.years:
        return None
    start = valuation.revenue_base or valuation.revenue_runrate
    if not start or start <= 0:
        return None
    last = consensus.years[-1]
    horizon_end = valuation.period_end + timedelta(days=round(base.years * 365.25))
    readings, _ = required_range(valuation, price)
    got = [
        remaining_cagr(start * (1 + r.required) ** base.years, horizon_end, last) for r in readings
    ]
    cagrs = sorted(g[0] for g in got if g)
    if not cagrs:
        return None
    return last.label, horizon_end.year, cagrs[0], cagrs[-1]


def results_due(p: Proposal, today: date) -> tuple[date, int] | None:
    """The proposal's next results date and the sessions to it, when within the
    entry guard's window from ``today``. Pure.

    The same window that stops a new entry stops being a time to deliberate:
    a date already past (the proposal was built before it) is not due.
    """
    from advisor.entry.sheet import sessions_until

    raw = (p.features or {}).get("next_earnings")
    if not raw:
        return None
    try:
        day = date.fromisoformat(str(raw))
    except ValueError:
        return None
    if day < today:
        return None
    n = sessions_until(today, day)
    return (day, n) if n <= current_params().earnings_guard_sessions else None


def _results(
    p: Proposal, h: Holding, due: tuple[date, int], valuation=None, consensus=None
) -> Actionable:
    from advisor.valuation.dcf import YEARS

    day, n = due
    f = p.features or {}
    low, high = f.get("required_low"), f.get("required_high")
    price = p.price or h.price
    years = (
        valuation.base_case().years if valuation is not None and valuation.base_case() else YEARS
    )
    head = f"Results {day.strftime('%a %m-%d')}, {_when(n)}"
    if low is not None and high is not None:
        at = f"at {_money(price)} " if price else ""
        first = f"{head}: {at}the price requires {_pct_range(low, high)} a year for {years} years"
    else:
        first = f"{head}: no valuation on file, so what the price requires cannot be stated"
    parts = []
    if f.get("delivered_growth") is not None:
        parts.append(f"revenue grew {f['delivered_growth']:+.1%} last reported")
    if f.get("consensus_growth") is not None:
        label = consensus.years[-1].label if consensus is not None and consensus.years else ""
        parts.append(
            f"consensus {f['consensus_growth']:+.1%} a year" + (f" to {label}" if label else "")
        )
    rest = remaining_after(valuation, consensus, price)
    if rest is not None:
        label, end, lo, hi = rest
        if hi < 0:  # met, the estimate alone pays for the price at every margin
            parts.append(f"if {label} is met, revenue could shrink {-hi:.1%} a year to {end}")
        else:
            parts.append(f"if {label} is met, the years to {end} need {_pct_range(lo, hi)} a year")
    bullets = [first]
    if parts:
        bullets.append("; ".join(parts)[:1].upper() + "; ".join(parts)[1:])
    return Actionable(
        id=f"{h.symbol}:DECIDE:results:{day.isoformat()}",
        verb="DECIDE",
        symbol=h.symbol,
        section="book",
        title=f"DECIDE {h.symbol} — hold through results on {day.strftime('%m-%d')}, or cut before",
        bullets=bullets,
        answers=["KEEP"],
        asof=p.built_at,
        source=_source(p) + "; results date from the yfinance calendar",
        # Kept at one bar, it asks again only if the price runs to a materially higher one.
        subject={
            "kind": "ACTIONABLE",
            "id": f"act:DECIDE:results:{day.isoformat()}",
            "observed": high,
            "worse_is": "UP" if high is not None else "NEITHER",
        },
    )


def reported(history: list[Proposal], today: date) -> tuple[date, Proposal, int] | None:
    """The results a name reported within ``AFTER_RESULTS_SESSIONS`` before ``today``:
    (their day, the last call before that day, sessions since). Pure.

    ``history``: one symbol's proposals, any sessions. A date counts only if
    the last call built before it still listed it: one the calendar moved
    later was never a report.
    """
    from advisor.entry.sheet import sessions_until

    best = None
    for raw in {(p.features or {}).get("next_earnings") for p in history}:
        try:
            day = date.fromisoformat(str(raw))
        except ValueError:
            continue
        if day >= today:
            continue
        n = sessions_until(day, today)
        if not 1 <= n <= AFTER_RESULTS_SESSIONS:
            continue
        prior = [p for p in history if p.session < day]
        if not prior or not any(p.session > day for p in history):
            continue  # nothing called before it, or nothing since: no before and after
        last = max(prior, key=lambda p: (p.session, p.built_at))
        if (last.features or {}).get("next_earnings") != day.isoformat():
            continue  # rescheduled before it came
        if best is None or day > best[0]:
            best = (day, last, n)
    return best


def _read_on(raw) -> date | None:
    """The ET date of a stored ISO timestamp, or None. Pure."""
    from advisor.daemon import market_calendar as mc

    try:
        return mc.to_et(datetime.fromisoformat(str(raw))).date() if raw else None
    except ValueError:
        return None


def _range_of(f: dict) -> str | None:
    low, high = f.get("required_low"), f.get("required_high")
    return _pct_range(low, high) if low is not None and high is not None else None


def _after(p: Proposal, h: Holding, day: date, before: Proposal, n: int) -> Actionable:
    f, b = p.features or {}, before.features or {}
    since = f"since {before.session.strftime('%m-%d')}"
    first = f"Results of {day.strftime('%a %m-%d')}"
    if p.price and before.price:
        first += (
            f": {_money(before.price)} → {_money(p.price)} "
            f"({p.price / before.price - 1:+.1%}) {since}"
        )
    now, was = _range_of(f), _range_of(b)
    if now and was:
        first += f"; the price requires {now} a year (was {was})"
    elif now:
        first += f"; the price requires {now} a year"
    parts = []
    c_now, c_was = f.get("consensus_growth"), b.get("consensus_growth")
    read = _read_on(f.get("consensus_asof"))
    if c_now is None:
        pass
    elif read is None:
        parts.append(f"consensus {c_now:+.1%} a year (when it was read is not recorded)")
    elif read <= day:
        # The call read a cached estimate from before the results: no revision to show.
        parts.append(
            f"consensus {c_now:+.1%} a year, read {read.strftime('%m-%d')}: "
            "not re-read since the results"
        )
    elif c_was is not None:
        parts.append(f"consensus {c_was:+.1%} → {c_now:+.1%} a year")
    else:
        parts.append(f"consensus {c_now:+.1%} a year")
    d_now, d_was = f.get("delivered_growth"), b.get("delivered_growth")
    if d_now is not None and d_was is not None and abs(d_now - d_was) >= 0.0005:
        parts.append(f"revenue grew {d_now:+.1%} in the new filing (was {d_was:+.1%})")
    elif d_now is not None:
        parts.append("the new quarter is not in the filings read yet")
    bullets = [first]
    if parts:
        text = "; ".join(parts)
        bullets.append(text[:1].upper() + text[1:])
    return Actionable(
        id=f"{h.symbol}:DECIDE:reported:{day.isoformat()}",
        verb="DECIDE",
        symbol=h.symbol,
        section="book",
        title=f"DECIDE {h.symbol} — keep or sell after its results of {day.strftime('%m-%d')}",
        bullets=bullets,
        answers=["KEEP"],
        asof=p.built_at,
        source=f"{_source(p)} against the call of {before.session.strftime('%m-%d')}",
        subject={
            "kind": "ACTIONABLE",
            "id": f"act:DECIDE:reported:{day.isoformat()}",
            "observed": f.get("required_high"),
            "worse_is": "UP" if f.get("required_high") is not None else "NEITHER",
        },
    )


def group_decisions(risk, today: date) -> list[Actionable]:
    """A DECIDE per correlated group before each macro event within the window. Pure.

    ``risk``: the stored ``risk.book.BookRisk``, or None.
    """
    from advisor.macro.calendar import upcoming

    if risk is None or not risk.groups:
        return []
    out = []
    for day, label, n in upcoming(today, current_params().earnings_guard_sessions):
        when = "today" if n == 0 else "next session" if n == 1 else f"in {n} sessions"
        for g in risk.groups:
            top = max(
                (x for x in risk.names if x.symbol in g.members), key=lambda x: x.risk_share
            ).symbol
            first = (
                f"{g.label} correlate {g.correlation:.2f} on average: {g.weight:.1%} of net liq "
                f"carrying {g.risk_share:.0%} of the book's risk; a 2σ two-week move is "
                f"~${g.two_week:,.0f}"
            )
            if g.hedge is not None:
                h = g.hedge
                second = (
                    f"Hedge: short {_shares(h.shares)} of {h.etf} (~${h.shares * h.price:,.0f}) "
                    f"or {h.etf} puts on that notional; {h.etf} explains {h.r2:.0%} of the "
                    "group's daily moves"
                )
            else:
                second = f"No hedge: {g.hedge_note}"
            out.append(
                Actionable(
                    id=f"{BOOK}:DECIDE:group:{top}:{day.isoformat()}",
                    verb="DECIDE",
                    symbol=BOOK,
                    section="book",
                    title=(
                        f"DECIDE {BOOK} — {g.label} as one bet: hedge or carry it through "
                        f"{label} on {day.strftime('%m-%d')} ({when})"
                    ),
                    bullets=[first, second],
                    answers=["KEEP"],
                    asof=risk.asof,
                    source=(
                        f"book risk of {risk.asof.strftime('%m-%d %H:%M')} ET, "
                        f"{risk.sessions} sessions of daily returns"
                    ),
                    weight=g.weight,
                    # Kept at one share of the risk, it asks again if the group grows into more.
                    subject={
                        "kind": "ACTIONABLE",
                        "id": f"act:DECIDE:group:{top}:{day.isoformat()}",
                        "observed": g.risk_share,
                        "worse_is": "UP",
                    },
                )
            )
    return out


def _leg(p: Proposal):
    legs = sorted(p.legs, key=lambda g: g.horizon != "position")
    return legs[0] if legs else None


def _event_reason(p: Proposal) -> str | None:
    for r in p.reasons:
        if "tier A" in r.source or "8-K" in r.source:
            return r.text
    return None


def _read(p: Proposal, held: bool) -> Actionable | None:
    leg = _leg(p)
    if leg is None or leg.shares <= 0 or not p.blockers:
        return None
    if not all(b.startswith(READ_BLOCKER) for b in p.blockers):
        return None  # waits on results, a reading, a stale input: nothing to do but wait
    sym = p.symbol.upper()
    return Actionable(
        id=f"{sym}:READ:{p.session.isoformat()}",
        verb="READ",
        symbol=sym,
        section="book" if held else "tracking",
        title=f"READ {sym} before {'adding' if held else 'buying'} — today's tier-A event",
        bullets=[
            _event_reason(p) or p.blockers[0],
            f"then: buy {_shares(leg.shares)} near {_money(leg.entry)}, stop {_money(leg.stop)}",
        ],
        answers=["DONE"],
        asof=p.built_at,
        source=_source(p),
        shares=leg.shares,
        entry=leg.entry,
        stop=leg.stop,
        subject={
            "kind": "ACTIONABLE",
            "id": f"act:READ:{p.session.isoformat()}",
            "observed": None,
            "worse_is": "NEITHER",
        },  # fmt: skip
    )


def _buy(p: Proposal, held: bool) -> Actionable | None:
    leg = _leg(p)
    if leg is None or leg.shares <= 0:
        return None
    sym = p.symbol.upper()
    why = next((t for t in p.triggers), None) or (p.reasons[0].text if p.reasons else "")
    risk = max(leg.entry - leg.stop, 0.0) * leg.shares
    second = f"risks {_money(risk)} ({leg.risk_pct:.1%} of net liq) to the stop"
    if leg.horizon == "trade":
        second += "; a trade: out by the next session's close"
    elif leg.target:
        second += f"; a trim is reviewed at {_money(leg.target)}"
    verb = "add" if held else "buy"
    return Actionable(
        id=f"{sym}:BUY:{p.session.isoformat()}",
        verb="BUY",
        symbol=sym,
        section="book" if held else "tracking",
        title=(
            f"BUY {sym} — {verb} {_shares(leg.shares)} near {_money(leg.entry)}, "
            f"stop {_money(leg.stop)}"
        ),
        bullets=[why[:1].upper() + why[1:], second] if why else [second],
        answers=["DONE", "SKIP"],
        asof=p.built_at,
        source=_source(p),
        shares=leg.shares,
        entry=leg.entry,
        stop=leg.stop,
        target=leg.target,
        subject={
            "kind": "ACTIONABLE",
            "id": f"act:BUY:{p.session.isoformat()}",
            "observed": None,
            "worse_is": "NEITHER",
        },  # fmt: skip
    )


def _added_since(p: Proposal, h: Holding) -> bool:
    """Whether a held name has grown since an ADD was proposed on it. Pure."""
    w, nl, price = (p.features or {}).get("weight"), p.net_liq, p.price
    if not (w and nl and price):
        return False
    return h.quantity > w * nl / price + 0.5


def _record(evidence: str | None) -> str | None:
    """The measured record without its interval: the verdict word carries it. Pure."""
    if not evidence:
        return None
    text = re.sub(r"^Measured: ", "", evidence)
    text = re.sub(r" \(95% interval[^)]*\)", "", text)
    # The verdict module's survivorship warning, in five words.
    return re.sub(
        r"\. Names like these drop out of the history.*$", "; survivorship flatters it.", text
    )


def pick_buys(picks: dict, held: set[str], now: datetime) -> list[Actionable]:
    """A pick's first day, before that session closes, verdict ENTER or TRADE ONLY. Pure.

    Day 0 is the entry both replays measured; entering 5–15 sessions in
    measured about zero, and a pick whose day has closed is already later
    than that.
    """
    from advisor.daemon import market_calendar as mc

    day = picks.get("day")
    if not day:
        return []
    d = date.fromisoformat(day)
    et = mc.to_et(now)
    if et.date() != d or et.time() >= mc.session_close(d) or not mc.is_trading_day(d):
        return []
    out = []
    for p in picks.get("picks") or []:
        sym = p["symbol"].upper()
        plan, v = p.get("plan") or {}, p.get("verdict") or {}
        size = (plan.get("size") or {}).get("shares") or 0
        if sym in held or plan.get("stage") != "fresh" or v.get("action") not in PICK_VERDICTS:
            continue
        if not plan.get("ok") or size <= 0 or not plan.get("stop"):
            continue
        entry, stop = plan["entry"], plan["stop"]
        by = f" · by the {mc.session_close(d).strftime('%H:%M')} ET close"
        if v["action"] == "ENTER":
            first = (
                f"Under its base value {_money(v['entry'])}; target {_money(v['target'])} (bull)"
            )
            title = f"BUY {sym} — {_shares(size)} near {_money(entry)}, stop {_money(stop)}{by}"
        else:
            head = re.sub(
                r"^TRADE ONLY — at \$[\d,.]+ the price", "The price", v.get("headline") or ""
            )
            first = head.split(" Nothing in the filings")[0].rstrip(".")
            first = first.replace("the best the company has filed", "the best it has filed")
            first = first.replace("the company has never filed", "it has never filed")
            title = (
                f"BUY {sym} — trade only: {_shares(size)} near {_money(entry)}, "
                f"stop {_money(stop)}{by}"
            )
        if v.get("caveat"):
            first += f" — {v['caveat'].split(':')[0]}: check the business"
        bullets = [first]
        record = _record(v.get("evidence"))
        if record:
            bullets.append(record)
        out.append(
            Actionable(
                id=f"{sym}:BUY:pick:{day}",
                verb="BUY",
                symbol=sym,
                section="tracking",
                title=title,
                bullets=bullets,
                answers=["DONE", "SKIP"],
                asof=datetime.fromisoformat(p.get("asof") or picks.get("built_at")),
                source=f"picks {day}"
                + (" (provisional, live prices)" if p.get("provisional") else ""),  # fmt: skip
                shares=size,
                entry=entry,
                stop=stop,
                target=v.get("target"),
                subject={
                    "kind": "ACTIONABLE",
                    "id": f"act:BUY:pick:{day}",
                    "observed": None,
                    "worse_is": "NEITHER",
                },  # fmt: skip
            )
        )
    return out


def build(
    proposals: list[Proposal],
    book,
    picks: dict | None,
    claims: dict[str, list[tuple[str, str]]],
    now: datetime,
    expectations: dict[str, tuple] | None = None,
) -> list[Actionable]:
    """Every actionable, in order: SELL, DECIDE, READ, BUY. Pure.

    ``claims``: per symbol, the thesis claims as (id, text), to answer a
    broken rule on the claim itself. ``expectations``: per symbol, the stored
    (valuation, consensus) a results DECIDE does its arithmetic on.
    """
    from advisor.daemon import market_calendar as mc

    today = mc.to_et(now).date()
    expectations = expectations or {}
    held = holdings(book)
    net_liq = book.net_liq if book is not None else 0.0
    by_symbol: dict[str, list[Proposal]] = {}
    for p in proposals:
        by_symbol.setdefault(p.symbol.upper(), []).append(p)
    out: list[Actionable] = []
    for sym, p in newest_session(proposals).items():
        h = held.get(sym)
        action = p.action.value
        if h is not None:
            calls = list(p.exits or [])
            if any(c.get("action") == "EXIT" for c in calls):
                out.append(_sell(p, h, calls))
                continue
            for c in calls:
                if c.get("action") == "REVIEW":
                    out.append(_decide(p, h, c, claims.get(sym, [])))
            due = results_due(p, today)
            if due is not None:
                out.append(_results(p, h, due, *expectations.get(sym, (None, None))))
            done = reported(by_symbol.get(sym, []), today)
            if done is not None:
                out.append(_after(p, h, *done))
            if action == "ADD" and not _added_since(p, h):
                item = _buy(p, held=True)
            elif action == "WAIT":
                item = _read(p, held=True)
            else:
                item = None
        elif action == "ENTER":
            item = _buy(p, held=False)
        elif action == "WAIT":
            item = _read(p, held=False)
        else:
            item = None  # EXIT/REVIEW/HOLD on a name no longer held, NONE, IN_ZONE
        if item is not None:
            out.append(item)
    for item in out:
        h = held.get(item.symbol)
        if h is not None and net_liq:
            item.weight = h.notional / net_liq
    if picks:
        out.extend(pick_buys(picks, set(held), now))
    return ordered(out)


def ordered(items: list[Actionable]) -> list[Actionable]:
    """SELL, DECIDE, READ, BUY; the book first; the larger weight first. Pure."""
    return sorted(
        items, key=lambda a: (ORDER[a.verb], a.section != "book", -(a.weight or 0.0), a.symbol)
    )


def answered(item: Actionable, decided: dict, today: date) -> str | None:
    """Why an actionable is quiet, or None when it stands. Pure.

    ``decided``: the newest decision per subject id for the item's symbol.
    """
    from advisor.action.decisions import evaluate

    s = item.subject
    if s.get("kind") == "CLAIM":
        ds = [decided.get(cid) for cid in s.get("ids") or []]
        if ds and all(ds):
            last = max(ds, key=lambda d: (d.decided_at, bool(d.note)))
            return f"you kept it on {last.decided_at.date()}" + (
                f": {last.note}" if last.note else ""
            )
        return None
    d = decided.get(s.get("id"))
    if d is None:
        return None
    sup = evaluate(d, current=s.get("observed"), today=today)
    if not sup.quiet:
        return None
    return sup.reason + (f": {d.note}" if d.note else "")


def decision_for(item: Actionable, answer: str, note: str = "") -> list:
    """The decisions an answer records. Raises ValueError on an answer the item does not take."""
    from advisor.action.decisions import Decision, Direction, SubjectKind, Verdict

    answer = answer.upper()
    if answer not in item.answers:
        raise ValueError(f"{item.id} takes {item.answers}, not {answer!r}")
    if answer == "KEEP" and not note.strip():
        raise ValueError("KEEP needs your reason: it is what the rule stops asking for")
    verdict = Verdict(ANSWERS[answer])
    s = item.subject
    if s.get("kind") == "CLAIM":
        return [
            Decision(
                symbol=item.symbol,
                subject_kind=SubjectKind.CLAIM,
                subject_id=cid,
                verdict=verdict,
                note=note.strip(),
            )  # fmt: skip
            for cid in s["ids"]
        ]
    return [
        Decision(
            symbol=item.symbol,
            subject_kind=SubjectKind.ACTIONABLE,
            subject_id=s["id"],
            verdict=verdict,
            note=note.strip(),
            observed=s.get("observed"),
            worse_is=Direction(s.get("worse_is") or "NEITHER"),
        )
    ]


def load(db_path, now: datetime) -> dict:
    """The actionables over the stores in ``db_path``, and the ones already answered."""
    from pathlib import Path

    from advisor.daemon import market_calendar as mc
    from advisor.daemon.store import DaemonStore
    from advisor.entry.store import EntryStore
    from advisor.thesis.repo import load_thesis

    db_path = Path(db_path)
    entries, daemon = EntryStore(db_path), DaemonStore(db_path)
    try:
        proposals = entries.list(since=now.date() - timedelta(days=PROPOSAL_DAYS))
        book = daemon.load_latest_book()
        claims: dict[str, list[tuple[str, str]]] = {}
        for sym, p in newest_session(proposals).items():
            if any(c.get("rule") == "thesis" for c in p.exits or []):
                thesis = load_thesis(daemon, sym)
                claims[sym] = [(c.id, c.text) for c in (thesis.claims if thesis else []) if c.id]
        expectations = {}
        for sym, p in newest_session(proposals).items():
            if results_due(p, mc.to_et(now).date()) is not None:
                expectations[sym] = (daemon.load_latest_valuation(sym), _consensus(daemon, sym))
        items = build(proposals, book, _picks(db_path), claims, now, expectations)
        from advisor.risk.book import latest_risk

        today = mc.to_et(now).date()
        items = ordered(items + group_decisions(latest_risk(db_path, today), today))
        decided = {sym: daemon.latest_decisions(sym) for sym in {a.symbol for a in items}}
    finally:
        entries.close()
        daemon.close()
    live, quiet = [], []
    for a in items:
        why = answered(a, decided.get(a.symbol, {}), now.date())
        if why is None:
            live.append(a)
        else:
            quiet.append({"id": a.id, "title": a.title, "why": why})
    return {
        "asof": now.isoformat(),
        "book_asof": book.as_of.isoformat() if book is not None else None,
        "items": [a.model_dump(mode="json") for a in live],
        "answered": quiet,
    }


def _picks(db_path) -> dict | None:
    from advisor.breadth.picks import latest_picks
    from advisor.breadth.store import BreadthStore, breadth_path

    path = breadth_path(db_path)
    if not path.exists():
        return None
    with BreadthStore(path) as store:
        return latest_picks(store, history=1)


def _consensus(store, symbol: str):
    """The stored consensus, however old: the tab never fetches (the daemon's jobs do)."""
    from advisor.valuation.consensus import Consensus

    raw = store.load_consensus(symbol)
    return Consensus.model_validate_json(raw) if raw else None
