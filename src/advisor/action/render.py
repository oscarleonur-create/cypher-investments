"""Rendering a card. Templates only — no model.

The order is the argument: what you hold, what happened, where your own rules
stand, and only then what follows. Evidence is printed last and always, so a
card that proposed something on thin data cannot hide the fact.
"""

from __future__ import annotations

from advisor.action.models import ACTION_TEXT, ActionCard, ActionKind

_MARK = {
    "BROKEN": "✗",
    "INTACT": "✓",
    "UNTESTED": "·",
    "UNREACHABLE": "!",
}


def render(card: ActionCard, *, width: int = 78) -> str:
    lines: list[str] = [
        "=" * width,
        f"  {card.symbol} — {card.action.value.replace('_', ' ').lower()}",
        "=" * width,
        "",
        f"WHAT TO DO      {card.headline}",
    ]
    if card.deadline:
        lines.append(f"                before {card.deadline}")

    if card.because:
        lines.append("")
        lines.append("BECAUSE")
        for reason in card.because:
            lines.append(f"   • {reason}")

    if card.position_note:
        lines += ["", f"POSITION        {card.position_note}"]

    if card.triggers:
        lines += ["", f"WHAT HAPPENED   ({len(card.triggers)} in the last week)"]
        for trigger in card.triggers[:6]:
            kind = trigger.kind.replace("_", " ").lower()
            lines.append(f"   [{trigger.tier}] {trigger.when.strftime('%m-%d %H:%M')}  {kind}")
            if trigger.detail:
                lines.append(f"        {trigger.detail}")

    if card.claims:
        broken = sum(1 for c in card.claims if c.status == "BROKEN")
        lines += ["", f"YOUR RULES      {len(card.claims)} written, {broken} broken"]
        for claim in card.claims:
            lines.append(f"   {_MARK[claim.status]} {claim.text[:64]}")
            if claim.note:
                lines.append(f"     {claim.note[:70]}")
    else:
        lines += ["", "YOUR RULES      none written for this name"]

    lines += ["", "EVIDENCE"]
    for item in card.evidence.items:
        mark = "!" if item.blocking else ("·" if not item.ok else " ")
        stamp = f"{item.asof}" if item.asof else "—"
        lines.append(f"   {mark} {item.name:14} {stamp:12} {item.detail[:44]}")

    if card.what_would_sharpen_this:
        lines += ["", "TO SHARPEN THIS"]
        for hint in card.what_would_sharpen_this:
            lines.append(f"   → {hint}")

    lines += [
        "",
        "This reports that conditions you defined in advance have or have not",
        "arrived. It holds no view of its own and proposes no trade.",
    ]
    return "\n".join(lines)


def one_line(card: ActionCard) -> str:
    """A single row for a whole-book listing."""
    mark = {
        ActionKind.REVIEW_NOW: "!!",
        ActionKind.REVIEW: " !",
        ActionKind.WRITE_THESIS: " ?",
        ActionKind.HOLD: "  ",
        ActionKind.CANNOT_SAY: " ×",
    }[card.action]
    return f"{mark} {card.symbol:6} {card.action.value:14} {card.headline[:52]}"


def summary_note(cards: list[ActionCard]) -> str:
    """What the whole book adds up to."""
    counts: dict[ActionKind, int] = {}
    for card in cards:
        counts[card.action] = counts.get(card.action, 0) + 1
    parts = [
        f"{n} {ACTION_TEXT[kind].split('—')[0].strip()}"
        for kind, n in sorted(counts.items(), key=lambda kv: kv[0].value)
    ]
    return "  ·  ".join(parts)
