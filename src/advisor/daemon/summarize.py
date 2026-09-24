"""One line saying what an event means, in numbers.

A Tier A event is defined by being actionable, and `STOP_BREACHED` on its own
is not: it names a position and a rule without saying how far past the rule it
went, or from where. The frontend already renders this; the CLI listed the
kind and stopped, so `advisor daemon events --tier A` — the command you would
actually run in the morning — showed six rows and not a single number.

Every branch reads keys the emitter is known to write, and anything missing is
simply left out rather than guessed at or printed as None.
"""

from __future__ import annotations


def _pct(value: object, *, sign: bool = True) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    prefix = "+" if sign and value > 0 else ""
    return f"{prefix}{value * 100:.1f}%"


def _money(value: object) -> str | None:
    """Dollars, with the sign outside: -$34.86 rather than $-34.86."""
    if not isinstance(value, (int, float)):
        return None
    sign = "-" if value < 0 else ""
    amount = abs(value)
    if amount >= 1e9:
        return f"{sign}${amount / 1e9:,.2f}bn"
    if amount >= 1e6:
        return f"{sign}${amount / 1e6:,.0f}M"
    return f"{sign}${amount:,.2f}"


def summarize(event) -> str:
    """A compact, numeric reading of one event. Empty when there is nothing to add."""
    payload = event.payload or {}
    parts: list[str] = []

    # Position crossings and standing conditions.
    if "unrealized_pct" in payload:
        move = _pct(payload.get("unrealized_pct"))
        entry, price = payload.get("entry"), payload.get("price")
        if isinstance(entry, (int, float)) and isinstance(price, (int, float)):
            parts.append(f"{entry:,.2f} → {price:,.2f}")
        if move:
            parts.append(move)
        threshold = _pct(payload.get("threshold"))
        if threshold:
            parts.append(f"vs {threshold}")
        usd = _money(payload.get("unrealized_usd"))
        if usd:
            parts.append(f"({usd})")

    # A position closing. The realised return is the point of the event and
    # was the one thing not rendered — "BE position closed" said nothing
    # about whether it was closed at a gain or a loss.
    elif "realized_pct" in payload:
        quantity = payload.get("quantity")
        if isinstance(quantity, (int, float)):
            parts.append(f"{abs(quantity):,.0f} shares")
        realised = _pct(payload.get("realized_pct"))
        if realised:
            parts.append(f"closed at {realised}")

    # A position opening.
    elif "notional" in payload and "quantity" in payload:
        quantity, price = payload.get("quantity"), payload.get("price")
        if isinstance(quantity, (int, float)) and isinstance(price, (int, float)):
            parts.append(f"{quantity:+,.0f} @ {price:,.2f}")
        notional = _money(payload.get("notional"))
        if notional:
            parts.append(f"= {notional}")

    # A position resized.
    elif "from_quantity" in payload:
        was, now = payload.get("from_quantity"), payload.get("to_quantity")
        if isinstance(was, (int, float)) and isinstance(now, (int, float)):
            parts.append(f"{was:+,.0f} → {now:+,.0f} shares")
        direction = payload.get("direction")
        if direction:
            parts.append(f"({direction})")

    # Concentration.
    elif "weight" in payload:
        weight = _pct(payload.get("weight"), sign=False)
        if weight:
            parts.append(f"{weight} of the book")
        limit = _pct(payload.get("threshold"), sign=False)
        if limit:
            parts.append(f"vs {limit}")

    # Offerings, sized.
    elif "offering_usd" in payload:
        amount = _money(payload.get("offering_usd"))
        if amount:
            parts.append(amount)
        share = payload.get("dilution_pct") or payload.get("offering_pct_of_cap")
        pct = _pct(share, sign=False)
        if pct:
            label = "dilution" if "dilution_pct" in payload else "of market cap"
            parts.append(f"= {pct} {label}")
        if payload.get("preliminary"):
            parts.append("(preliminary)")

    # Factor shocks.
    elif "expected_book_move" in payload:
        factor, z = payload.get("factor"), payload.get("z")
        if factor:
            parts.append(str(factor))
        if isinstance(z, (int, float)):
            parts.append(f"z {z:+.2f}")
        book = _pct(payload.get("expected_book_move"))
        if book:
            parts.append(f"→ book {book}")

    # Residual divergence.
    elif "residual_z" in payload:
        actual, expected = _pct(payload.get("actual_return")), _pct(payload.get("expected_return"))
        if actual and expected:
            parts.append(f"{actual} vs {expected} expected")
        z = payload.get("residual_z")
        if isinstance(z, (int, float)):
            parts.append(f"z {z:+.2f}")

    # Insider clusters.
    elif "insider_count" in payload:
        count, side = payload.get("insider_count"), str(payload.get("side", "")).lower()
        if count:
            parts.append(f"{count} insiders {side}")
        total = _money(payload.get("total_value"))
        if total:
            parts.append(total)

    # Implied expectations.
    elif "implied_cagr" in payload:
        was, now = (
            _pct(payload.get("previous_implied_cagr"), sign=False),
            _pct(payload.get("implied_cagr"), sign=False),
        )
        if was and now:
            parts.append(f"required growth {was} → {now}")
            # Naming the cause, because an unchanged price whose required
            # growth halved has not repriced — it has been re-measured.
            driver = payload.get("driver")
            if driver == "FIGURES":
                parts.append("on newer figures, not a price move")
            elif driver == "BOTH":
                parts.append("on newer figures and a price move")

    # News pulled to explain something. The headline is the whole content of
    # the event; without it six of these render as six identical blank rows.
    elif payload.get("title"):
        parts.append(str(payload["title"]))
        explains = payload.get("explains")
        if explains:
            parts.append(f"[{str(explains).lower().replace('_', ' ')}]")
        if payload.get("angle"):
            parts.append(f"[angle: {payload['angle']}]")

    # Data quality.
    elif "check" in payload:
        parts.append(str(payload["check"]))
        failed, symbols = payload.get("failed"), payload.get("symbols") or []
        if failed:
            parts.append(f"{failed} symbol(s)")
        if symbols:
            parts.append(", ".join(str(s) for s in symbols[:4]))

    # Filings that carry no figure of their own.
    elif payload.get("form"):
        parts.append(str(payload["form"]))
        items = payload.get("items") or []
        if items:
            parts.append(f"items {', '.join(str(i) for i in items)}")

    if not parts and payload.get("label"):
        parts.append(str(payload["label"]))

    return "  ".join(parts)
