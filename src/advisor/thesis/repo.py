"""Loading a thesis: free-text row plus structured claims, as one object."""

from __future__ import annotations

import logging

from advisor.daemon.store import DaemonStore
from advisor.thesis.detect import completeness, is_substantive, original_prose
from advisor.thesis.models import StructuredThesis
from advisor.thesis.reachability import audit

logger = logging.getLogger(__name__)


class ThesisReadError(RuntimeError):
    """The thesis store exists but could not be read.

    Distinct from "no thesis written", which is a normal answer. Collapsing
    the two would let a malformed table read as an absent opinion — the
    advisor would quietly stop testing events against a thesis that is
    actually there.
    """


def load_thesis(store: DaemonStore, symbol: str) -> StructuredThesis | None:
    """The thesis for ``symbol``, or None when nothing has been written.

    A row whose content is still the blank template returns a thesis marked
    ``substantive=False`` rather than None: the record exists and the user
    should be told it is empty, not told nothing.
    """
    symbol = symbol.upper()
    claims = store.load_claims(symbol)

    row = None
    conn = store._conn  # noqa: SLF001 — the research module owns this table
    present = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='theses'"
    ).fetchone()
    if present:
        try:
            row = conn.execute(
                "SELECT title, content, conviction, status FROM theses "
                "WHERE upper(symbol) = ? ORDER BY updated_at DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.warning("thesis: %s exists but could not be read: %s", symbol, exc)
            raise ThesisReadError(str(exc)) from exc

    if row is None and not claims:
        return None

    content = row["content"] if row else ""
    written = is_substantive(content)
    note = ""
    if row is not None and not written:
        share = completeness(content)
        note = (
            f"the thesis document is still the blank template "
            f"({share:.0%} filled in) — nothing to test events against"
        )

    blocked = {
        r.claim_id: r.reason
        for r in audit(store, symbol, claims)
        if r.blocked and r.claim_id is not None
    }

    return StructuredThesis(
        symbol=symbol,
        title=(row["title"] if row else f"{symbol} claims"),
        conviction=(row["conviction"] if row else None),
        status=(row["status"] if row else None),
        # Claims are content in their own right: a thesis with testable claims
        # and no prose is more useful than prose with none.
        substantive=written or bool(claims),
        claims=claims,
        prose_note=note or (original_prose(content)[:400] if written else ""),
        blocked=blocked,
    )
