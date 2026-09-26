"""The scanner's own table in ``data/research.db``.

One row per (session, setup, symbol). The first detection of the day wins:
a stock that qualifies at 09:35 and again at 11:00 is one candidate priced at
09:35, because that is the entry the setup would actually have offered.
Outcomes are merged in later and never overwritten once written.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from advisor.learning import store as rule_versions
from advisor.scanner.models import Candidate, Phase, Setup, TradeDecision, candidate_id

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS scan_candidates (
    id           TEXT NOT NULL PRIMARY KEY,   -- session:setup:symbol
    session      TEXT NOT NULL,
    setup        TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    detected_at  TEXT NOT NULL,
    payload_json TEXT NOT NULL,               -- Candidate
    created_at   TEXT DEFAULT (datetime('now')),
    updated_at   TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_scan_candidates_session
    ON scan_candidates(session DESC, setup);

-- Append-only: a change of mind (a skip later corrected by a broker fill) is
-- a new row, so the record of what was believed when survives.
CREATE TABLE IF NOT EXISTS scan_decisions (
    candidate_id TEXT NOT NULL,
    taken        INTEGER NOT NULL,            -- 1 taken, 0 not taken
    source       TEXT NOT NULL,               -- broker | user
    payload_json TEXT NOT NULL,               -- TradeDecision
    decided_at   TEXT NOT NULL,
    created_at   TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_scan_decisions_candidate
    ON scan_decisions(candidate_id, decided_at DESC);
"""


class ScannerStore:
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

    def add(self, c: Candidate) -> bool:
        """Insert unless this (session, setup, symbol) exists. True when new.

        ``INSERT OR IGNORE`` rather than read-then-write, so two scans racing
        on the same mover cannot both record it.
        """
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO scan_candidates "
            "(id, session, setup, symbol, detected_at, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                c.id,
                c.session.isoformat(),
                c.setup.value,
                c.symbol,
                c.detected_at.isoformat(),
                c.model_dump_json(),
            ),
        )
        new = cur.rowcount > 0
        if new:
            rule_versions.register(self._conn, c.rules)
        self._conn.commit()
        return new

    def exists(
        self, session: date, setup: Setup, symbol: str, phase: Phase = Phase.SESSION
    ) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM scan_candidates WHERE id = ?",
            (candidate_id(session, setup, symbol, phase),),
        ).fetchone()
        return row is not None

    def get(self, cid: str) -> Candidate | None:
        row = self._conn.execute(
            "SELECT payload_json FROM scan_candidates WHERE id = ?", (cid,)
        ).fetchone()
        return Candidate.model_validate_json(row["payload_json"]) if row else None

    # ── Journal: what the user did about each candidate ───────────────────

    def record_decision(self, d: TradeDecision) -> None:
        """Append. The newest decision per candidate is the one that stands."""
        self._conn.execute(
            "INSERT INTO scan_decisions (candidate_id, taken, source, payload_json, decided_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                d.candidate_id,
                int(d.taken),
                d.source.value,
                d.model_dump_json(),
                d.decided_at.isoformat(),
            ),
        )
        self._conn.commit()

    def latest_decisions(self) -> dict[str, TradeDecision]:
        """Newest decision per candidate id."""
        rows = self._conn.execute(
            "SELECT payload_json FROM scan_decisions ORDER BY decided_at ASC, rowid ASC"
        ).fetchall()
        out: dict[str, TradeDecision] = {}
        for r in rows:
            d = TradeDecision.model_validate_json(r["payload_json"])
            out[d.candidate_id] = d
        return out

    def news_checked_count(self, session: date) -> int:
        """How many news lookups this session has spent (the credit budget)."""
        rows = self._conn.execute(
            "SELECT payload_json FROM scan_candidates WHERE session = ?",
            (session.isoformat(),),
        ).fetchall()
        return sum(1 for r in rows if json.loads(r["payload_json"]).get("news_checked"))

    def list(
        self,
        *,
        session: date | None = None,
        setup: Setup | None = None,
        since: date | None = None,
        limit: int = 500,
    ) -> list[Candidate]:
        clauses, args = [], []
        if session is not None:
            clauses.append("session = ?")
            args.append(session.isoformat())
        if since is not None:
            clauses.append("session >= ?")
            args.append(since.isoformat())
        if setup is not None:
            clauses.append("setup = ?")
            args.append(setup.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT payload_json FROM scan_candidates {where} "
            "ORDER BY detected_at DESC LIMIT ?",
            (*args, limit),
        ).fetchall()
        return [Candidate.model_validate_json(r["payload_json"]) for r in rows]

    def merge_outcomes(self, c: Candidate, outcomes: dict[str, float | None]) -> int:
        """Add outcome keys not already present. Returns how many were added.

        A key present with value None means "not applicable" (a +60m horizon
        past the close) and is final; a key absent means "not yet known".
        """
        fresh = {k: v for k, v in outcomes.items() if k not in c.outcomes}
        if not fresh:
            return 0
        c.outcomes.update(fresh)
        self._conn.execute(
            "UPDATE scan_candidates SET payload_json = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (c.model_dump_json(), c.id),
        )
        self._conn.commit()
        return len(fresh)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM scan_candidates").fetchone()[0]
