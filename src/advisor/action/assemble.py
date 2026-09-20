"""Building an action card from what is stored.

Order matters here, and it is the opposite of what feels natural. The evidence
is graded **first**, and a blocking gap short-circuits everything else: a card
that always produces an answer is a card that will eventually produce a
confident wrong one.

Only then are the user's own claims tested, and only a claim the user wrote
can produce urgency. The system never decides that something is bad; it
reports that a condition the user defined in advance has arrived.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from advisor.action.models import (
    ACTION_TEXT,
    ActionCard,
    ActionKind,
    ClaimVerdict,
    Evidence,
    EvidenceItem,
    TriggerRef,
)
from advisor.daemon.book import BookSnapshot
from advisor.daemon.market_calendar import now_et
from advisor.daemon.models import Event
from advisor.daemon.store import DaemonStore
from advisor.daemon.summarize import summarize

logger = logging.getLogger(__name__)

# How far back an event still counts as something to act on. Past this it is
# history: informative in a story, not a reason to look today.
RECENT_DAYS = 7

# A position snapshot older than this no longer describes what is held.
POSITION_STALE_DAYS = 3

# Kinds that merit a look even when no written rule covers them.
MATERIAL_KINDS = frozenset(
    {
        "STOP_BREACHED",
        "FILING_DILUTION",
        "FILING_RESTATEMENT",
        "FILING_AUDITOR_CHANGE",
        "FILING_DELISTING",
        "FILING_MERGER",
        "FILING_LATE_FILING",
        "FILING_ACTIVIST_STAKE",
        "FILING_RESULTS",
        "RESIDUAL_DIVERGENCE",
        "INSIDER_SELLING_CLUSTER",
        "INSIDER_BUYING_CLUSTER",
    }
)


def _evidence(
    store: DaemonStore, symbol: str, book: BookSnapshot, today: date
) -> tuple[Evidence, str]:
    """Grade every input the card would rest on, before it rests on any."""
    evidence = Evidence()
    position_note = ""

    held = [p for p in book.positions if p.underlying.upper() == symbol]
    snapshot_age = (now_et() - book.as_of).days
    if snapshot_age > POSITION_STALE_DAYS:
        evidence.items.append(
            EvidenceItem(
                name="position",
                asof=book.as_of.date(),
                detail=f"the book snapshot is {snapshot_age} days old",
                ok=False,
                blocking=True,
            )
        )
    elif not held:
        evidence.items.append(
            EvidenceItem(
                name="position",
                asof=book.as_of.date(),
                detail=f"{symbol} is not held",
                ok=False,
                blocking=True,
            )
        )
    else:
        position = held[0]
        weight = position.signed_notional / book.net_liq if book.net_liq else 0.0
        position_note = (
            f"{position.quantity:+,.0f} @ {position.avg_open_price:,.2f} entry, "
            f"now {position.price:,.2f} — {position.unrealized_pct * 100:+.1f}%, "
            f"{weight * 100:.1f}% of the book"
        )
        evidence.items.append(
            EvidenceItem(name="position", asof=book.as_of.date(), detail=position_note)
        )

    # A failed cross-source check is disqualifying: the advisor said itself
    # that it could not trust these numbers, and acting on them anyway would
    # make that warning decorative.
    failures = [
        e
        for e in store.recent_events(limit=200)
        if e.kind == "DATA_QUALITY_FAILURE"
        and e.ts.date() >= today - timedelta(days=1)
        and symbol in (e.payload.get("symbols") or [])
    ]
    if failures:
        evidence.items.append(
            EvidenceItem(
                name="data quality",
                asof=failures[0].ts.date(),
                detail=f"{failures[0].payload.get('check')} failed for {symbol}",
                ok=False,
                blocking=True,
            )
        )
    else:
        evidence.items.append(
            EvidenceItem(name="data quality", asof=today, detail="checks passing")
        )

    # The remaining inputs narrow the answer rather than blocking it: the card
    # can still be honest about a name the factor model cannot see, as long as
    # it says so.
    sensitivity = store.load_sensitivity(symbol)
    evidence.items.append(
        EvidenceItem(
            name="factor model",
            asof=sensitivity.asof if sensitivity else None,
            detail=(
                f"R² {sensitivity.r2:.2f}, residual {sensitivity.resid_vol * 100:.1f}%/day"
                if sensitivity
                else "no estimate — too little price history, so macro cannot "
                "explain or exonerate a move here"
            ),
            ok=sensitivity is not None,
        )
    )

    valuation = store.load_latest_valuation(symbol)
    evidence.items.append(
        EvidenceItem(
            name="valuation",
            asof=valuation.period_end if valuation else None,
            detail=(
                f"EV/revenue {valuation.ev_to_revenue:,.1f}x, price requires "
                f"{valuation.base_case().implied_cagr * 100:.1f}% growth for a decade"
                if valuation and valuation.base_case()
                else "not valued — no usable filing"
            ),
            ok=valuation is not None,
        )
    )

    return evidence, position_note


def _triggers(store: DaemonStore, symbol: str, today: date) -> list[Event]:
    """Events for this symbol recent enough to be about today."""
    cutoff = now_et() - timedelta(days=RECENT_DAYS)
    return [
        e
        for e in store.recent_events(limit=400)
        if (e.symbol or "").upper() == symbol and e.ts >= cutoff and e.tier.value in {"A", "B"}
    ]


def _claims(
    store: DaemonStore, symbol: str, events: list[Event], book: BookSnapshot
) -> list[ClaimVerdict]:
    """Where each written rule stands — against what happened, and against
    what is true now."""
    from advisor.thesis.match import evaluate_claim
    from advisor.thesis.repo import ThesisReadError, load_thesis
    from advisor.thesis.state import evaluate_against_state

    try:
        thesis = load_thesis(store, symbol)
    except ThesisReadError:
        return []
    if thesis is None or not thesis.claims:
        return []

    verdicts: list[ClaimVerdict] = []
    for claim in thesis.claims:
        blocked = thesis.blocked.get(claim.id or "")
        if blocked:
            verdicts.append(
                ClaimVerdict(
                    text=claim.text, kind=claim.kind.value, status="UNREACHABLE", note=blocked
                )
            )
            continue
        if not claim.monitored:
            verdicts.append(
                ClaimVerdict(
                    text=claim.text,
                    kind=claim.kind.value,
                    status="UNREACHABLE",
                    note="no trigger — nothing will ever check this",
                )
            )
            continue

        results = [r for r in (evaluate_claim(claim, e) for e in events) if r is not None]
        tripped = [r for r in results if r.tripped]

        # An event tripping a rule is news. The current state violating it is
        # a condition that was true yesterday too. Both matter and they are
        # not the same thing: a rule can be violated for weeks without any
        # event firing, because the event stream deliberately reports changes.
        standing = evaluate_against_state(store, symbol, claim, book)

        if tripped:
            verdicts.append(
                ClaimVerdict(
                    text=claim.text, kind=claim.kind.value, status="BROKEN", note=tripped[0].note
                )
            )
        elif standing is not None and standing.tripped:
            verdicts.append(
                ClaimVerdict(
                    text=claim.text, kind=claim.kind.value, status="STANDING", note=standing.note
                )
            )
        elif results or standing is not None:
            verdicts.append(
                ClaimVerdict(
                    text=claim.text,
                    kind=claim.kind.value,
                    status="INTACT",
                    note=(results[0] if results else standing).note,
                )
            )
        else:
            verdicts.append(
                ClaimVerdict(
                    text=claim.text,
                    kind=claim.kind.value,
                    status="UNTESTED",
                    note="nothing this week tested it, and it has no standing form",
                )
            )
    return verdicts


def build_card(store: DaemonStore, symbol: str, book: BookSnapshot, *, today: date | None = None):
    """Assemble one ticker's card. Never raises; refuses rather than guesses."""
    symbol = symbol.upper()
    today = today or date.today()
    evidence, position_note = _evidence(store, symbol, book, today)

    card = ActionCard(
        symbol=symbol,
        action=ActionKind.CANNOT_SAY,
        headline="",
        evidence=evidence,
        position_note=position_note,
    )

    # Evidence first. A blocked card says why and stops; proposing a next step
    # on data the system has already flagged would make the flag decorative.
    if not evidence.trustworthy:
        card.headline = ACTION_TEXT[ActionKind.CANNOT_SAY]
        card.because = [i.detail for i in evidence.blockers]
        card.what_would_sharpen_this = [
            "run `advisor daemon reconcile` and resolve what it reports",
        ]
        return card

    events = _triggers(store, symbol, today)
    card.triggers = [
        TriggerRef(
            kind=e.kind,
            tier=e.tier.value,
            when=e.ts,
            detail=summarize(e),
            url=(e.payload or {}).get("url"),
        )
        for e in events
    ]
    card.claims = _claims(store, symbol, events, book)

    broken = [c for c in card.claims if c.status == "BROKEN"]
    standing = [c for c in card.claims if c.status == "STANDING"]
    material = [e for e in events if e.kind in MATERIAL_KINDS]

    if broken:
        card.action = ActionKind.REVIEW_NOW
        card.headline = f"{len(broken)} rule(s) you wrote have broken"
        card.because = [f"{c.text} — {c.note}" for c in broken]
        # The deadline is the point of a Tier A: something has to be decided
        # before the market next opens on it.
        card.deadline = today + timedelta(days=1)
    elif standing:
        # Never REVIEW_NOW. A standing violation would be urgent every day for
        # as long as it lasts, which is precisely how an alert channel stops
        # being read — the lesson phase 2a paid for.
        card.action = ActionKind.REVIEW
        card.headline = f"{len(standing)} rule(s) you wrote are violated right now"
        card.because = [f"{c.text} — {c.note}" for c in standing]
    elif material:
        card.action = ActionKind.REVIEW
        card.headline = f"{len(material)} material event(s), no written rule covers them"
        card.because = [
            f"{e.kind.replace('_', ' ').lower()} — {summarize(e)}" for e in material[:4]
        ]
    elif not card.claims:
        card.action = ActionKind.WRITE_THESIS
        card.headline = "nothing written down, so nothing can be tested"
        card.because = [
            f"{len(events)} event(s) this week and no claim to judge them against"
            if events
            else "no claims, so every event here is filed and none is evaluated"
        ]
    else:
        card.action = ActionKind.HOLD
        card.headline = ACTION_TEXT[ActionKind.HOLD]
        intact = [c for c in card.claims if c.status == "INTACT"]
        card.because = (
            [f"{c.text} — {c.note}" for c in intact[:3]]
            if intact
            else ["no event this week tested any of your claims"]
        )

    card.what_would_sharpen_this = _suggestions(card, evidence)

    from advisor.action.rationale import build_rationale

    card.rationale = build_rationale(store, symbol, book, card).model_dump(mode="json")
    return card


def _suggestions(card: ActionCard, evidence: Evidence) -> list[str]:
    """What the user could do to let the system say more next time.

    Deliberately about the system's own blind spots, not about the position.
    """
    out: list[str] = []
    unreachable = [c for c in card.claims if c.status == "UNREACHABLE"]
    if not card.claims:
        out.append(
            f'write one invalidation: `advisor thesis add {card.symbol} "..." '
            "--on FILING_DILUTION --field dilution_pct --above 0.05`"
        )
    elif unreachable:
        out.append(
            f"{len(unreachable)} of your claims can never fire — " "`advisor thesis list` shows why"
        )
    for gap in evidence.gaps:
        if gap.name == "factor model":
            out.append(
                "macro cannot see this name yet; a move here has nothing to be judged against"
            )
        elif gap.name == "valuation":
            out.append("no valuation stored — run `advisor daemon once --job valuation`")
    return out


def build_all(store: DaemonStore, book: BookSnapshot, *, today: date | None = None):
    """Every held symbol's card, most urgent first, then by weight."""
    cards = [build_card(store, s, book, today=today) for s in book.symbols]
    from advisor.action.models import ACTION_RANK

    weights = {
        p.underlying.upper(): abs(p.signed_notional / book.net_liq) if book.net_liq else 0.0
        for p in book.positions
    }
    cards.sort(key=lambda c: (ACTION_RANK[c.action], -weights.get(c.symbol, 0.0)))
    return cards
