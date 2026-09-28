"""From a pick to a position decision: the entry engine, run for one name on demand.

A pick says two measured families agree; whether a position makes sense is
the entry engine's question, the same one the daemon asks every morning of
held and watchlist names: where the price sits against the name's own P/S
history, what the move was, which leg qualifies, how many shares the risk
budget allows and where the stop goes, and — because a decision is on the
table — the model's reading, with news pulled to explain the move if the
store has none (``entry.run.propose_all``). Nothing is reimplemented here.

The proposal is recorded in the ledger like any other (``entry_proposals``),
so an evaluated pick is tracked on the Tracking tab and judged by what the
price did next. It is a proposal: no order is staged.
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def evaluate_position(db_path, symbol: str, now: datetime, *, read: bool = True, runner=None):
    """Run the entry engine for ``symbol``; record and return its proposal.

    Returns ``{"proposal": dict | None, "errors": [...], "recorded": bool}``.
    """
    from pathlib import Path

    from advisor.daemon.store import DaemonStore
    from advisor.entry.run import propose_all
    from advisor.entry.store import EntryStore

    symbol = symbol.strip().upper()
    daemon = DaemonStore(Path(db_path))
    entry = EntryStore(Path(db_path))
    run = runner or propose_all
    before = {p.id for p in entry.list(since=now.date(), limit=5000) if p.symbol == symbol}
    proposals, errors = run(daemon, now, symbols=[symbol], entry_store=entry, read=read)
    if not proposals:
        return {"proposal": None, "errors": errors or [f"{symbol}: no proposal"], "recorded": False}
    p = proposals[0]
    return {
        "proposal": p.model_dump(mode="json"),
        "errors": errors,
        # Recorded unless the same session already held this exact call.
        "recorded": p.id not in before,
    }


def latest_evaluation(db_path, symbol: str) -> dict | None:
    """The newest proposal on file for ``symbol``, from any source."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT payload_json FROM entry_proposals WHERE symbol = ? "
            "ORDER BY session DESC, rowid DESC LIMIT 1",
            (symbol.strip().upper(),),
        ).fetchone()
    except sqlite3.OperationalError:
        return None  # no ledger yet
    finally:
        conn.close()
    if row is None:
        return None
    import json

    return json.loads(row[0])
