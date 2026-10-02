"""A pick's verdict: where its price sits against the company's own value, and what to do.

User, 2026-09-30: *"quiero unos accionables claros. Posible entry, por esta
tesis, el valor de la empresa está por debajo de lo que esperamos, debería
estar en X valor"* — then, on the value test, *"si dale"*.

The value is the live price-range card's (``valuation.rationale.price_range``:
bear / base / bull from the company's own filed margins), priced at the
pick's own price. The verdict follows the test that measured it
(``value_replay``, 3 years of pick records valued as of their day): picks
priced at or below their base value beat matched peers by +2.3% to +2.7%
over 20 sessions; those above it did not beat them at all. So:

- **ENTER** — at or below the base value: entry up to the base, target the bull;
- **WAIT** — above it: the entry is the base value, and how far away it is;
- **TRADE ONLY** — no range, and the price asks a margin the company has never
  filed: nothing in the filings supports the price;
- **UNPROVEN** — no range, the price asks less than the one margin on file;
- **CAN'T VALUE** — the filings could not be read.

Every verdict quotes the measured record of its bucket (``evidence``), and a
company whose business a margin-based value misreads (``MISREAD_SIC``) is
flagged before anyone acts on the number.

Nothing here is new judgement: the value is the card's, the buckets are the
test's, and the entry and target are the card's own base and bull.
"""

from __future__ import annotations

import logging
from datetime import date

logger = logging.getLogger(__name__)

# Businesses a value from filed operating and free-cash-flow margins misreads
# (SIC ranges, inclusive). AGX on 2026-09-30 was "below even the bear case"
# on a 45.9% FCF margin made of customer prepayments.
MISREAD_SIC: tuple[tuple[int, int, str], ...] = (
    (1500, 1799, "a contractor: cash flow swings with customer prepayments, revenue with projects"),
    (4400, 4499, "a shipper: earnings follow charter rates, not a steady margin"),
    (6000, 6411, "a bank or insurer: its cash flow is not free cash flow"),
    (6500, 6599, "a real-estate company: value sits in land and property, not a margin"),
    (6798, 6798, "a REIT: valued on its properties and payout, not a margin"),
)

ACTIONS = {
    "below_base": "ENTER",
    "above_base": "WAIT",
    "no_support": "TRADE ONLY",
    "one_reading": "UNPROVEN",
    "no_data": "CAN'T VALUE",
}

_CARDS: dict[tuple[str, date], tuple[object | None, str]] = {}


def misread(sic: int | None) -> str | None:
    """Why a margin-based value misreads this business, or None. Pure."""
    if sic is None:
        return None
    for lo, hi, why in MISREAD_SIC:
        if lo <= sic <= hi:
            return why
    return None


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def evidence_line(cell: dict | None) -> str | None:
    """The measured record of a bucket, in one sentence. Pure."""
    if not cell or not cell.get("n"):
        return None
    ci = cell.get("ci")
    return (
        f"Measured: {cell['n']:,} past picks like this beat peers by {_pct(cell['excess'])} "
        f"over 20 sessions"
        + (f" (95% interval {_pct(ci[0])} … {_pct(ci[1])})" if ci else "")
        + f" — {cell.get('verdict')}, {cell.get('years', 3)}-year replay."
    )


def pick_verdict(
    price: float,
    card: dict | None,
    *,
    sic: int | None = None,
    evidence: dict[str, dict] | None = None,
    why_missing: str = "",
) -> dict:
    """The verdict for one pick from its value card (``PriceRange`` as a dict). Pure."""
    from advisor.breadth.value_replay import bucket

    if card is None:
        return {
            "action": ACTIONS["no_data"],
            "bucket": "no_data",
            "headline": "CAN'T VALUE: " + (why_missing or "the filings could not be read") + ".",
            "lines": [],
            "evidence": None,
            "caveat": misread(sic),
        }
    cases = {c["name"]: c for c in card.get("cases") or []}
    own = [m["value"] for m in card.get("own_margins") or [] if m.get("value") is not None]
    market = cases.get("market")
    b = bucket(
        {
            "verdict": card.get("verdict"),
            "market_margin": market["margin"] if market else None,
            "best_margin": max(own) if own else None,
        }
    )
    base = cases.get("base", {}).get("value_per_share")
    bear = cases.get("bear", {}).get("value_per_share")
    bull = cases.get("bull", {}).get("value_per_share")
    lines: list[str] = []
    out = {"action": ACTIONS[b], "bucket": b, "entry": None, "target": None, "value": base}
    if b == "below_base":
        out.update(entry=base, target=bull)
        headline = (
            f"ENTER up to {_money(base)} (its base value) — at {_money(price)} it is "
            f"{_pct(price / base - 1)} against it. Target {_money(bull)} (bull)."
        )
        if bear and price < bear:
            lines.append(
                f"Below even the bear case ({_money(bear)}): the cheapest reading of its own "
                "margins. Check the business before trusting a gap this wide."
            )
    elif b == "above_base":
        out.update(entry=base, target=bull)
        headline = (
            f"WAIT — entry at {_money(base)} (its base value); at {_money(price)} it is "
            f"{_pct(price / base - 1)} above it. Target {_money(bull)} (bull)."
        )
    elif b == "no_support":
        need = market["margin"] if market else None
        best = max(own) if own else None
        headline = (
            f"TRADE ONLY — at {_money(price)} the price asks a "
            f"{need * 100:.1f}% free-cash-flow margin"
            + (
                f"; the best the company has filed is {best * 100:.1f}%."
                if best is not None and best > 0
                else "; the company has never filed a positive one."
            )
            + " Nothing in the filings supports the price: no value to enter at."
        )
    elif b == "one_reading":
        need = market["margin"] if market else None
        headline = (
            f"UNPROVEN — no value range (one margin on file, {max(own) * 100:.1f}%); the "
            f"price asks {need * 100:.1f}%, less than that one reading."
        )
    else:
        headline = "CAN'T VALUE: " + (card.get("refused") or "no margin to value it on") + "."
    if b in ("below_base", "above_base") and bear and base and bull:
        lines.append(
            f"Value from its own filings: bear {_money(bear)} · base {_money(base)} · "
            f"bull {_money(bull)} ({card.get('assumptions', '')})."
        )
    for text in card.get("rationale") or []:
        if text.startswith(("That price earns", "Its own filed margins")):
            lines.append(text)
    record = evidence_line((evidence or {}).get(b))
    if record and b == "no_support":
        # Companies with no positive margin are the ones that fail and leave
        # the listings most often, and the replay's universe is today's.
        record += (
            " Names like these drop out of the history most often when they fail, so this "
            "record flatters them most."
        )
    out.update(headline=headline, lines=lines, evidence=record, caveat=misread(sic))
    return out


def figures_for(symbol: str, price: float, today: date) -> tuple[object | None, str]:
    """The company's filings, read once per symbol and day; never raises.

    Filings change a few times a year and intraday picks rebuild every 30
    minutes: only the price moves, and the snapshot is re-priced from these.
    """
    from advisor.valuation.figures import load_figures

    key = (symbol.upper(), today)
    if key not in _CARDS:
        try:
            _CARDS[key] = (load_figures(symbol, price, price_source="pick price"), "")
        except Exception as exc:  # noqa: BLE001
            logger.info("verdict: no figures for %s: %s", symbol, exc)
            _CARDS[key] = (None, f"filings unavailable ({exc.__class__.__name__})")
    return _CARDS[key]


def value_card(symbol: str, price: float, today: date) -> tuple[dict | None, str]:
    """The live price-range card priced at ``price``; filings cached for the day.

    Returns ``(card, why_missing)``. Never raises: a pick without a value is
    still a pick, said as such.
    """
    from advisor.valuation.implied import build_snapshot
    from advisor.valuation.rationale import price_range

    figures, why = figures_for(symbol, price, today)
    if figures is None:
        return None, why
    snap = build_snapshot(figures, price, asof=today)
    if snap is None:
        return None, "no price, share count, balance sheet or revenue in the filings"
    return price_range(snap, live=True, today=today).model_dump(mode="json"), ""


def evidence(store) -> dict[str, dict]:
    """The latest value test's 20-session cell per bucket for group 2+ (every pick is one)."""
    from advisor.breadth.value_replay import latest_value_runs

    runs = latest_value_runs(store)
    if not runs:
        return {}
    run = runs[0]  # the longest window
    return {
        c["bucket"]: {**c, "years": run.get("years")}
        for c in run.get("cells", [])
        if c.get("group") == "2+" and c.get("horizon") == "d20"
    }
