"""Exit calls for a held name: EXIT or REVIEW, each with its rationale.

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
- ~~Above the 20% book limit is a TRIM~~ — removed by the user (2026-10-04): the
  weight of a name is no longer a reason to sell it.

Every call carries why, the evidence with its source, and what would change
it. A call without a rationale is not a call (``EntryStore.add`` refuses it).

- **News the taxonomy does not map** (``entry.distress``): an exit-grade
  report from two or more outlets is an EXIT labeled unconfirmed, one outlet
  a REVIEW; the company's own website saying it is an EXIT on its own
  (2026-09-27).
- **An exchange halt that questions the company's standing** (T12, H4, H9,
  H10, H11; ``news.halts``) is an EXIT on the exchange's word while it
  lasts, and a REVIEW once trading resumes; a halt pending news (T1, T6) is
  a REVIEW until it resumes (2026-09-27).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from advisor.entry.proposal import Reason, position_stop_pct

# The strongest call wins the proposal's action.
# No TRIM: the user removed the 20% concentration trim (2026-10-04, "quitemos esa
# regla"). Action.TRIM stays so older proposals still read.
SEVERITY = {"EXIT": 3, "REVIEW": 1}

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
    action: str  # EXIT | REVIEW (TRIM on proposals recorded before 2026-10-04)
    rule: str  # stop | filing | halt | news | thesis | rich
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


def _halt_calls(sheet) -> list[ExitCall]:
    """The name's exchange halts: EXIT while a standing halt lasts, REVIEW otherwise."""
    from advisor.news.halts import EXIT_CODES, REVIEW_CODES

    calls: list[ExitCall] = []
    seen: set[str] = set()
    for h in sheet.halts:  # newest first: one call per code
        code = str(h.get("code") or "")
        if code in seen or code not in EXIT_CODES | REVIEW_CODES:
            continue
        seen.add(code)
        resumed = h.get("resumed_at")
        if code in REVIEW_CODES and resumed:
            continue  # the news it waited for is out; the distress sweep reads it
        exit_ = code in EXIT_CODES and not resumed
        # The feed's time is when the exchange listed the halt, not when trading
        # stopped: 15 of 19 halts on 2026-09-27 read 19:50, SEC suspensions
        # issued that morning among them. The day is reliable; the hour is not.
        stamp = str(h.get("halted_at") or "")
        since = f"{stamp[:10]}, listed {stamp[11:16]} ET"
        label = EXIT_CODES.get(code) or REVIEW_CODES[code]
        why = f"{sheet.symbol} {label} ({code} on {h.get('market') or 'its exchange'}, {since})"
        if code == "T1":
            why += (
                "; Nasdaq also lists T1 halts ahead of corporate actions: all seven on "
                "2026-09-25 preceded a reverse split, so read the latest filing"
            )
        if exit_:
            why += "; it cannot be sold while halted: sell when trading resumes"
        elif resumed:
            when = str(resumed)[:16].replace("T", " ")
            why += (
                f"; no longer in the exchange's halt list by {when} ET"
                if h.get("resumed_inferred")
                else f"; trading resumed {when} ET"
            ) + ", and why the exchange stopped it is the question now"
        else:
            why += "; the news comes out when trading resumes"
        evidence = [
            Reason(
                text=f"{code} halt, {since}"
                + (f", resumed {str(resumed)[:16].replace('T', ' ')} ET" if resumed else ""),
                source=f"Nasdaq Trader trade halts {h.get('url') or ''}".strip(),
            )
        ]
        if sheet.filings:
            # The company's own word is the halt's likely reason (MCTA, EFTY:
            # "On November 11, 2025 … the SEC … suspension").
            latest = max(sheet.filings, key=lambda f: f.ts)
            evidence.append(
                Reason(
                    text=f"latest filing {latest.ts.date().isoformat()}: {latest.text[:200]}",
                    source="SEC EDGAR",
                )
            )
        calls.append(
            ExitCall(
                action="EXIT" if exit_ else "REVIEW",
                rule="halt",
                why=why,
                evidence=evidence,
                would_change=(
                    "the exchange lifting it with the company's explanation: the halt is "
                    "the exchange's word, not a report"
                    if code in EXIT_CODES
                    else "the news released when trading resumes"
                ),
                shares=int(sheet.holding.quantity) if exit_ else None,
            )
        )
    return calls


def _news_call(sheet) -> ExitCall | None:
    """EXIT on an exit-grade report from 2+ outlets or the company's own site; REVIEW on one."""
    from advisor.entry.distress import MIN_OUTLETS_FOR_EXIT, DistressReading, Verdict

    if not sheet.distress:
        return None
    r = DistressReading.model_validate(sheet.distress)
    if r.verdict is not Verdict.EXIT_GRADE or not r.cited:
        return None
    n = len(r.outlets)
    own = r.company_said
    exit_ = own or n >= MIN_OUTLETS_FOR_EXIT
    confirmed = any(c.rule == "filing" and c.action == "EXIT" for c in _filing_calls(sheet))
    status = "confirmed by a filing" if confirmed else "no SEC filing confirms it yet"
    if own:
        site = next(i.provider for i in r.cited if i.issuer)
        why = f"the company itself reports {r.label} on its own website ({site}); {status}"
    else:
        why = (
            f"{n} independent outlet{'s' if n != 1 else ''} report {r.label} "
            f"({', '.join(r.outlets)}); {status}"
            + ("" if exit_ else f": one outlet is a REVIEW, {MIN_OUTLETS_FOR_EXIT} make an EXIT")
        )
    why += f". Reading: {r.reason}" if r.reason else ""
    return ExitCall(
        action="EXIT" if exit_ else "REVIEW",
        rule="news" if confirmed else "news (company statement)" if own else "news (unconfirmed)",
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

    # The exchange stopping the shares.
    calls.extend(_halt_calls(sheet))

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

    if sheet.next_earnings is not None:
        reasons.append(
            Reason(
                text=f"next results {sheet.next_earnings.isoformat()} "
                f"(in {sheet.earnings_in} sessions)",
                source="yfinance calendar",
            )
        )
    return calls, reasons, gaps
