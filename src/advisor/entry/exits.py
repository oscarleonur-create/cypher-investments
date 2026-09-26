"""Exit calls for a held name: EXIT, TRIM or REVIEW, each with its rationale.

The entry engine said when to buy; nothing said when to sell. The book's
alarms fired (``STOP_BREACHED`` at a fixed -8%, ``DEEP_DRAWDOWN`` at -20%,
``CONCENTRATION_WARNING`` nine days running on SPCX) but none carried a
size, a reason the user could check, or a record to learn from. The user's
decisions (2026-09-26):

- **The stop is measured from the purchase price**, at the name's own
  volatility: 2·σ·√10, kept in 8–25% — the same stop the entry leg sets.
  A loss past it is an EXIT.
- **A primary filing that ends the investment case is an EXIT**: 8-K item
  1.03 (bankruptcy), 2.04 (an obligation accelerated — a default), 3.01
  (delisting notice), 4.02 (the financials cannot be relied upon). No
  interpretation: the company said it. A change of auditor (4.01) is a
  REVIEW.
- **A broken thesis rule is a REVIEW**, not an EXIT: the user answers it —
  keeps the position with a reason, or sells.
- **P/S at or above its two-year 80th percentile is a REVIEW** with the
  numbers, not an automatic trim.
- **Above the 20% book limit is a TRIM** back to it: the book's own rule.

Every call carries why, the evidence with its source, and what would change
it. A call without a rationale is not a call (``EntryStore.add`` refuses it).

Not here yet: news the SEC taxonomy does not map (a report of a possible
bankruptcy before any filing). That needs the model's reading and a
corroboration rule, and comes next.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from advisor.entry.proposal import BOOK_LIMIT, Reason, position_stop_pct

# The strongest call wins the proposal's action.
SEVERITY = {"EXIT": 3, "TRIM": 2, "REVIEW": 1}

RICH_PERCENTILE = 0.80

# 8-K items (and the classifier kinds foreign filers map to) that end the case.
EXIT_ITEMS = {
    "1.03": "filed for bankruptcy or receivership",
    "2.04": "reported an accelerated obligation (a default event)",
    "3.01": "received a delisting notice or failed a listing rule",
    "4.02": "said its previously issued financials cannot be relied upon",
}
EXIT_KINDS = {
    "FILING_BANKRUPTCY": "filed for bankruptcy or receivership",
    "FILING_DELISTING": "received a delisting notice or failed a listing rule",
    "FILING_RESTATEMENT": "said its previously issued financials cannot be relied upon",
}
REVIEW_ITEMS = {"4.01": "changed its certifying accountant"}
REVIEW_KINDS = {"FILING_AUDITOR_CHANGE": "changed its certifying accountant"}


class ExitCall(BaseModel):
    action: str  # EXIT | TRIM | REVIEW
    rule: str  # stop | filing | thesis | rich | concentration
    why: str
    evidence: list[Reason] = Field(default_factory=list)
    would_change: str
    shares: int | None = None  # to sell; None where the call does not size (REVIEW)


def strongest(calls: list[ExitCall]) -> str | None:
    return max((c.action for c in calls), key=SEVERITY.__getitem__, default=None)


def _filing_calls(sheet) -> list[ExitCall]:
    calls: list[ExitCall] = []
    seen: set[tuple[str, str]] = set()
    for e in sheet.filings:
        for code in e.items:
            if code in EXIT_ITEMS and ("exit", code) not in seen:
                seen.add(("exit", code))
                calls.append(_filing_call("EXIT", EXIT_ITEMS[code], f"8-K item {code}", e, sheet))
            elif code in REVIEW_ITEMS and ("review", code) not in seen:
                seen.add(("review", code))
                calls.append(
                    _filing_call("REVIEW", REVIEW_ITEMS[code], f"8-K item {code}", e, sheet)
                )
        if not e.items:  # foreign filers: the classifier's kind is all there is
            if e.kind in EXIT_KINDS and ("exit", e.kind) not in seen:
                seen.add(("exit", e.kind))
                calls.append(_filing_call("EXIT", EXIT_KINDS[e.kind], e.kind, e, sheet))
            elif e.kind in REVIEW_KINDS and ("review", e.kind) not in seen:
                seen.add(("review", e.kind))
                calls.append(_filing_call("REVIEW", REVIEW_KINDS[e.kind], e.kind, e, sheet))
    return calls


def _filing_call(action, what, label, e, sheet) -> ExitCall:
    qty = int(sheet.holding.quantity) if action == "EXIT" else None
    return ExitCall(
        action=action,
        rule="filing",
        why=f"{sheet.symbol} {what} ({label}, filed {e.ts.date().isoformat()})",
        evidence=[Reason(text=e.text[:240], source=f"SEC EDGAR, {label}")],
        would_change=(
            "nothing in the price: the company itself filed it. Only a later filing "
            "that withdraws or cures it"
            if action == "EXIT"
            else "the reason for the change, in the filing or the next 10-Q, "
            "showing no disagreement with the auditor"
        ),
        shares=qty,
    )


def _news_call(sheet) -> ExitCall | None:
    """EXIT (unconfirmed) on an exit-grade report from 2+ outlets; REVIEW on one."""
    from advisor.entry.distress import MIN_OUTLETS_FOR_EXIT, DistressReading, Verdict

    if not sheet.distress:
        return None
    r = DistressReading.model_validate(sheet.distress)
    if r.verdict is not Verdict.EXIT_GRADE or not r.cited:
        return None
    n = len(r.outlets)
    exit_ = n >= MIN_OUTLETS_FOR_EXIT
    confirmed = any(c.rule == "filing" and c.action == "EXIT" for c in _filing_calls(sheet))
    status = "confirmed by a filing" if confirmed else "no SEC filing confirms it yet"
    why = (
        f"{n} independent outlet{'s' if n != 1 else ''} report {r.label} "
        f"({', '.join(r.outlets)}); {status}"
        + ("" if exit_ else f": one outlet is a REVIEW, {MIN_OUTLETS_FOR_EXIT} make an EXIT")
        + (f". Reading: {r.reason}" if r.reason else "")
    )
    return ExitCall(
        action="EXIT" if exit_ else "REVIEW",
        rule="news" if confirmed else "news (unconfirmed)",
        why=why,
        evidence=[
            Reason(
                text=f"{i.date} {i.title}",
                source=i.provider + (f" {i.url}" if i.url else ""),
            )
            for i in r.cited[:5]
        ],
        would_change=(
            "a filing or company statement that denies it, or financing that removes "
            "the risk; until then it asks for two weeks"
        ),
        shares=int(sheet.holding.quantity) if exit_ else None,
    )


def exit_calls(sheet, *, net_liq: float | None) -> tuple[list[ExitCall], list[Reason], list[str]]:
    """Calls, HOLD reasons and gaps for a held long. Pure over the sheet.

    Returns (calls, position reasons, gaps). The position reasons state where
    the holding stands against its stop whether or not anything fires, so a
    HOLD has a rationale too.
    """
    h, m, z = sheet.holding, sheet.move, sheet.zone
    calls: list[ExitCall] = []
    reasons: list[Reason] = []
    gaps: list[str] = []
    if h is None or m is None:
        return calls, reasons, gaps
    if h.quantity <= 0:
        gaps.append("short position: exits are modeled for longs only")
        return calls, reasons, gaps

    # Stop, from the purchase price.
    stop_pct = position_stop_pct(m.sigma)
    loss = m.price / h.cost - 1 if h.cost and h.cost > 0 else None
    if loss is None:
        gaps.append("no cost basis: the stop from the purchase price cannot be measured")
    elif stop_pct is None:
        gaps.append("no volatility estimate: the position stop cannot be set")
        reasons.append(
            Reason(text=f"{loss:+.1%} from your average cost ${h.cost:,.2f}", source="broker")
        )
    else:
        stop_price = h.cost * (1 - stop_pct)
        where = (
            f"{loss:+.1%} from your average cost ${h.cost:,.2f}; its volatility stop is "
            f"{stop_pct:.1%} below cost, at ${stop_price:,.2f} "
            f"(2·σ·√10, σ {m.sigma:.2%}/day, kept in 8–25%)"
        )
        if loss <= -stop_pct:
            calls.append(
                ExitCall(
                    action="EXIT",
                    rule="stop",
                    why=f"past its stop: {where}",
                    evidence=[
                        Reason(text=f"price ${m.price:,.2f}", source="yfinance close"),
                        Reason(
                            text=f"{h.quantity:g} shares at ${h.cost:,.2f} average",
                            source="TastyTrade positions",
                        ),
                    ],
                    would_change=(
                        "nothing after the fact: the stop is the rule you set at entry. "
                        "A reason to hold on belongs in a thesis rule, written before the loss"
                    ),
                    shares=int(h.quantity),
                )
            )
        else:
            reasons.append(Reason(text=where, source="TastyTrade positions; yfinance closes"))

    # Filings that end the investment case.
    calls.extend(_filing_calls(sheet))

    # News the taxonomy does not map, read by the model, counted without it.
    news = _news_call(sheet)
    if news is not None:
        calls.append(news)

    # The user's own thesis.
    if sheet.thesis == "broken":
        rules = "; ".join(sheet.thesis_broken[:3])
        calls.append(
            ExitCall(
                action="REVIEW",
                rule="thesis",
                why=f"a rule of your thesis is broken or standing: {rules}",
                evidence=[
                    Reason(text=r, source="your thesis claims") for r in sheet.thesis_broken[:3]
                ],
                would_change=(
                    "your answer: record why you keep it (the rule stops asking), "
                    "or sell because the rule you wrote says the case is gone"
                ),
            )
        )
    elif sheet.thesis == "intact":
        reasons.append(
            Reason(text="your thesis is written and intact", source="your thesis claims")
        )

    # Expensive against its own history.
    if z is not None and z.percentile >= RICH_PERCENTILE:
        calls.append(
            ExitCall(
                action="REVIEW",
                rule="rich",
                why=(
                    f"P/S {z.ps_now:.1f}x is at percentile {z.percentile:.0%} of its own two "
                    f"years (median {z.median:.1f}x): it has been this expensive on "
                    f"{1 - z.percentile:.0%} of days; the 80th-percentile price is "
                    f"${z.p80_price:,.2f}" + (" [short history]" if z.short else "")
                ),
                evidence=[
                    Reason(
                        text=f"{z.observations} sessions, {z.window_start}..{z.window_end}",
                        source=f"{z.source}; yfinance closes",
                    )
                ],
                would_change=(
                    "a quarter whose revenue brings the multiple back under its 80th "
                    "percentile, or a reason the business now deserves more than it did"
                ),
            )
        )

    # The book's concentration limit.
    if net_liq and net_liq > 0 and h.weight > BOOK_LIMIT:
        excess = (h.weight - BOOK_LIMIT) * net_liq
        shares = min(math.ceil(excess / m.price), int(h.quantity))
        after = (h.weight * net_liq - shares * m.price) / net_liq
        calls.append(
            ExitCall(
                action="TRIM",
                rule="concentration",
                why=(
                    f"{h.weight:.1%} of the book against the 20% limit; selling {shares} "
                    f"(${shares * m.price:,.0f}) brings it to {after:.1%}"
                ),
                evidence=[
                    Reason(text=f"net liq ${net_liq:,.2f}", source="TastyTrade balances"),
                ],
                would_change="a price fall or a larger book that brings the weight under 20%",
                shares=shares,
            )
        )

    if sheet.next_earnings is not None:
        reasons.append(
            Reason(
                text=f"next results {sheet.next_earnings.isoformat()} "
                f"(in {sheet.earnings_in} sessions)",
                source="yfinance calendar",
            )
        )
    return calls, reasons, gaps
