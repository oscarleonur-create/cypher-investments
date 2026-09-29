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
    _remember(db_path, p)
    return {
        "proposal": p.model_dump(mode="json"),
        "errors": errors,
        # Recorded unless the same session already held this exact call.
        "recorded": p.id not in before,
    }


# The last on-demand evaluation per name. The ledger keeps one proposal per
# session, name and action — the call as first made, which is what gets
# scored — so re-evaluating within a session (before the open, the session is
# still the previous day's) recorded nothing new and the screen reloaded the
# stored call, rationale and all, as if the click had done nothing.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS breadth_evaluations (
    symbol TEXT NOT NULL PRIMARY KEY,
    built_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
)
"""


def _remember(db_path, p) -> None:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(_SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO breadth_evaluations VALUES (?, ?, ?)",
            (p.symbol, p.built_at.isoformat(), p.model_dump_json()),
        )
        conn.commit()
    finally:
        conn.close()


def latest_evaluation(db_path, symbol: str) -> dict | None:
    """The newest proposal for ``symbol``: the last evaluation or the ledger's, if later."""
    import json
    import sqlite3
    from datetime import datetime

    sym = symbol.strip().upper()
    conn = sqlite3.connect(str(db_path))
    found = []
    try:
        for sql in (
            "SELECT payload_json FROM entry_proposals WHERE symbol = ? "
            "ORDER BY session DESC, rowid DESC LIMIT 1",
            "SELECT payload_json FROM breadth_evaluations WHERE symbol = ?",
        ):
            try:
                row = conn.execute(sql, (sym,)).fetchone()
            except sqlite3.OperationalError:
                continue  # that table does not exist yet
            if row is not None:
                found.append(json.loads(row[0]))
    finally:
        conn.close()
    if not found:
        return None
    return max(found, key=lambda p: datetime.fromisoformat(p["built_at"]))
