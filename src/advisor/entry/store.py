"""Entry proposals, kept for tracking. The entry module's own table.

Append-only, first wins per (session, symbol, action): a WAIT at 09:45 and an
ENTER at 11:45 are two rows, because the record of what the system said, and
when, is the point. What the user did about each is recorded against the
proposal id in the scanner's ``scan_decisions`` (the same journal), and what
the price did in ``outcomes``, which are merged and never overwritten.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from advisor.entry.proposal import Proposal
from advisor.learning import store as rule_versions

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS entry_proposals (
    id           TEXT NOT NULL PRIMARY KEY,   -- session:symbol:action
    session      TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    action       TEXT NOT NULL,
    payload_json TEXT NOT NULL,               -- Proposal
    created_at   TEXT DEFAULT (datetime('now')),
    updated_at   TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_entry_proposals_session ON entry_proposals(session DESC);
"""


class EntryStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        rule_versions.ensure_schema(self._conn)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def add(self, p: Proposal) -> bool:
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO entry_proposals (id, session, symbol, action, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (p.id, p.session.isoformat(), p.symbol, p.action.value, p.model_dump_json()),
        )
        new = cur.rowcount > 0
        if new:
            rule_versions.register(self._conn, p.rules)
        self._conn.commit()
        return new

    def list(
        self, *, session: date | None = None, since: date | None = None, limit: int = 5000
    ) -> list[Proposal]:
        clauses, args = [], []
        if session is not None:
            clauses.append("session = ?")
            args.append(session.isoformat())
        if since is not None:
            clauses.append("session >= ?")
            args.append(since.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT payload_json FROM entry_proposals {where} ORDER BY created_at DESC LIMIT ?",
            (*args, limit),
        ).fetchall()
        return [Proposal.model_validate_json(r["payload_json"]) for r in rows]

    def merge_outcomes(self, p: Proposal, outcomes: dict[str, float | None]) -> int:
        fresh = {k: v for k, v in outcomes.items() if k not in p.outcomes}
        if not fresh:
            return 0
        p.outcomes.update(fresh)
        self._conn.execute(
            "UPDATE entry_proposals SET payload_json = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (p.model_dump_json(), p.id),
        )
        self._conn.commit()
        return len(fresh)
