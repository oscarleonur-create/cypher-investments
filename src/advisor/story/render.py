"""Rendering a story as text. Templates only — no model, ever.

This is the artefact Telegram will send in phase 7, so it has to read well
without prose generation. The ordering is deliberate: what happened, what it
cost you, whether the market explains it, who else saw it. A slot that could
not be filled prints its reason rather than disappearing — a story silently
missing its position line is indistinguishable from one where you held
nothing.
"""

from __future__ import annotations

from advisor.story.models import VERDICT_TEXT, Confidence, Story, Verdict


def _money(value: float | None, *, sign: bool = False) -> str:
    if value is None:
        return "—"
    prefix = "+" if sign and value > 0 else ""
    return f"{prefix}{value:,.2f}"


def _pct(value: float | None, *, sign: bool = False, digits: int = 2) -> str:
    if value is None:
        return "—"
    prefix = "+" if sign and value > 0 else ""
    return f"{prefix}{value * 100:.{digits}f}%"


def headline(story: Story) -> str:
    """One line: the symbol, what happened, and the number that matters."""
    anchor = story.anchor
    facts = anchor.facts
    if "dilution_pct" in facts and "offering_usd" in facts:
        return (
            f"{story.symbol} — ${facts['offering_usd']:,.0f} offering, "
            f"{facts['dilution_pct'] * 100:.1f}% dilution"
        )
    if story.reaction.pct_move is not None:
        return f"{story.symbol} — {anchor.headline} ({_pct(story.reaction.pct_move, sign=True)})"
    return f"{story.symbol} — {anchor.headline}"


def render(story: Story, *, width: int = 78) -> str:
    """The full story as plain text."""
    a, pos, rx, at, cor, th = (
        story.anchor,
        story.position,
        story.reaction,
        story.attribution,
        story.corroboration,
        story.thesis,
    )
    lines: list[str] = [
        "=" * width,
        f"  {headline(story)}",
        "=" * width,
        "",
        f"WHAT HAPPENED   {a.occurred_at.strftime('%a %d %b %Y, %H:%M ET')}"
        + ("  (learned later)" if a.backfilled else ""),
        f"                [{a.tier.value}] {a.headline}",
    ]
    if a.facts.get("form"):
        items = a.facts.get("items") or []
        lines.append(f"                {a.facts['form']}" + (f"  items {items}" if items else ""))
    if a.facts.get("offering_usd"):
        sized = f"                ${a.facts['offering_usd']:,.0f}"
        if a.facts.get("dilution_pct") and a.facts.get("market_cap"):
            sized += (
                f" = {a.facts['dilution_pct'] * 100:.1f}% of a "
                f"${a.facts['market_cap']:,.0f} market cap"
            )
        lines.append(sized)
    lines.append(f"                source: {a.source}")
    if a.quote:
        lines += ["", f'EVIDENCE        "{a.quote[:220].strip()}…"']
    if a.url:
        lines.append(f"                {a.url}")

    lines += ["", "YOUR POSITION   "]
    if pos.confidence is Confidence.UNAVAILABLE:
        lines[-1] += f"unknown — {pos.note}"
    elif not pos.held:
        lines[-1] += f"not held{f' — {pos.note}' if pos.note else ''}"
    else:
        lines[-1] += f"{pos.quantity:+,.0f} shares @ {_money(pos.avg_open_price)} entry"
        if pos.weight_of_net_liq is not None:
            lines.append(
                f"                {_pct(pos.weight_of_net_liq)} of net liq "
                f"({_money(pos.net_liq)})"
            )
        stamp = pos.snapshot_asof.date() if pos.snapshot_asof else "?"
        lines.append(
            f"                position as of {stamp}"
            + ("" if pos.covers_event else "  ⚠ postdates the event")
        )
        if pos.note:
            lines.append(f"                {pos.note}")

    lines += ["", "WHAT IT COST    "]
    if rx.confidence is Confidence.UNAVAILABLE:
        lines[-1] += f"unknown — {rx.note}"
    else:
        priced = (
            " (next session — the event landed after the close)" if rx.priced_next_session else ""
        )
        lines[-1] += f"{rx.session}{priced}"
        lines.append(
            f"                {_money(rx.before)} → {_money(rx.after)}  "
            f"({_pct(rx.pct_move, sign=True)})"
        )
        if rx.dollars is not None:
            lines.append(
                f"                {_money(rx.dollars, sign=True)} on the position"
                + (
                    f" = {_pct(rx.pct_of_book, sign=True)} of the book"
                    if rx.pct_of_book is not None
                    else ""
                )
            )

    lines += ["", "WAS IT MACRO?   "]
    if at.confidence is Confidence.UNAVAILABLE:
        lines[-1] += f"unknown — {at.note}"
    else:
        lines[-1] += VERDICT_TEXT[at.verdict]
        lines.append(
            f"                macro expected {_pct(at.expected_return, sign=True)}, "
            f"actual {_pct(at.actual_return, sign=True)}"
        )
        lines.append(
            f"                residual z {at.residual_z:+.2f} against this name's "
            f"{_pct(at.resid_vol)}/day residual vol"
        )
        # Never let the narrative imply an alert that did not fire.
        if at.verdict is Verdict.PARTLY_SPECIFIC:
            lines.append(
                "                below the 2.0 alert threshold — a day like this is "
                "not unusual for this name"
            )
        if at.r2 is not None:
            lines.append(f"                the factor model explains {at.r2:.0%} of its variance")

    lines += [
        "",
        f"CORROBORATION   {len(cor.items)} independent item(s) in a {cor.window_days}-day window"
        f"  ({cor.primary_count} primary, {cor.aggregator_count} reported, "
        f"{cor.untagged_count} context)",
    ]
    for item in cor.items[:4]:
        lines.append(f"   [{item['tier']:10}] {item['title'][:58]}")

    lines += ["", "THESIS          "]
    if th.exists:
        lines[-1] += f"{th.title}  (conviction {th.conviction}, {th.status})"
    else:
        lines[-1] += th.note or "none"

    if story.unavailable_slots:
        lines += ["", f"NOT ESTABLISHED {', '.join(story.unavailable_slots)}"]

    return "\n".join(lines)
