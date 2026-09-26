"""``rule_versions``: each version's parameters, written once, the first time it is seen.

The learning module owns the table. The stores that write records — the
scanner's candidates, the entry module's proposals — register the stamp on
their own connection as they insert, so a version is on file the moment a
record points at it and there is no second job that can fall behind.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime

from pydantic import BaseModel

from advisor.daemon.market_calendar import now_et
from advisor.learning.rules import PRE_REGISTRY, Kind, RuleStamp

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS rule_versions (
    version       TEXT NOT NULL PRIMARY KEY,   -- hash of ruleset + params
    ruleset       TEXT NOT NULL,
    params_json   TEXT NOT NULL,
    kinds_json    TEXT NOT NULL,
    first_code    TEXT NOT NULL,               -- git revision that first ran it
    first_seen_at TEXT NOT NULL                -- ISO, tz-aware ET
);
CREATE INDEX IF NOT EXISTS idx_rule_versions_ruleset
    ON rule_versions(ruleset, first_seen_at DESC);
"""

# The ledgers that carry a stamp: (table, label). Read here, never written.
LEDGERS: tuple[tuple[str, str], ...] = (
    ("scan_candidates", "candidates"),
    ("entry_proposals", "proposals"),
)


class RuleVersion(BaseModel):
    version: str
    ruleset: str
    params: dict
    kinds: dict[str, Kind]
    first_code: str
    first_seen_at: datetime


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def register(conn: sqlite3.Connection, s: RuleStamp | None) -> bool:
    """Record the stamp's version unless it is already on file. True when new.

    Does not commit: the caller commits with the record that carries the
    stamp, so the two land together. A stamp read back from a row has no
    parameters (they are not stored per row) and is skipped — its version was
    registered when that row was first written.
    """
    if s is None or not s.params:
        return False
    cur = conn.execute(
        "INSERT OR IGNORE INTO rule_versions "
        "(version, ruleset, params_json, kinds_json, first_code, first_seen_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            s.version,
            s.ruleset,
            json.dumps(s.params, sort_keys=True),
            json.dumps({k: v.value for k, v in s.kinds.items()}, sort_keys=True),
            s.code,
            now_et().isoformat(),
        ),
    )
    return cur.rowcount > 0


def _row(r) -> RuleVersion:
    return RuleVersion(
        version=r["version"],
        ruleset=r["ruleset"],
        params=json.loads(r["params_json"]),
        kinds=json.loads(r["kinds_json"]),
        first_code=r["first_code"],
        first_seen_at=datetime.fromisoformat(r["first_seen_at"]),
    )


class RuleStore:
    """Read side, for the CLI and, later, the evaluator."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        self._conn = conn
        ensure_schema(conn)

    def list(self, ruleset: str | None = None) -> list[RuleVersion]:
        where, args = ("WHERE ruleset = ?", (ruleset,)) if ruleset else ("", ())
        rows = self._conn.execute(
            f"SELECT * FROM rule_versions {where} ORDER BY ruleset, first_seen_at DESC, version",
            args,
        ).fetchall()
        return [_row(r) for r in rows]

    def get(self, version: str) -> RuleVersion | None:
        """By full version or an unambiguous prefix of it."""
        rows = self._conn.execute(
            "SELECT * FROM rule_versions WHERE version LIKE ? || '%'", (version,)
        ).fetchall()
        return _row(rows[0]) if len(rows) == 1 else None

    def record_counts(self) -> dict[str, dict[str, int]]:
        """``{label: {version or PRE_REGISTRY: rows}}`` over each ledger that exists."""
        tables = {
            r[0] for r in self._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        out: dict[str, dict[str, int]] = {}
        for table, label in LEDGERS:
            if table not in tables:
                continue
            rows = self._conn.execute(
                f"SELECT json_extract(payload_json, '$.rules.version') AS v, COUNT(*) AS n "
                f"FROM {table} GROUP BY v"
            ).fetchall()
            out[label] = {(r["v"] or PRE_REGISTRY): r["n"] for r in rows}
        return out


_TRADES_SCHEMA = """\
CREATE TABLE IF NOT EXISTS trades (
    id            TEXT NOT NULL PRIMARY KEY,   -- account:instrument:opened_at
    account       TEXT NOT NULL,
    underlying    TEXT NOT NULL,
    entry_session TEXT NOT NULL,
    book          TEXT NOT NULL,
    closed        INTEGER NOT NULL,
    payload_json  TEXT NOT NULL,               -- learning.trades.Trade
    updated_at    TEXT NOT NULL                -- ISO, tz-aware ET
);
CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(entry_session DESC);
"""


class TradeStore:
    """The user's round trips, rebuilt from the broker. The learning module's table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        self._conn = conn
        conn.executescript(_TRADES_SCHEMA)

    def upsert(self, t) -> int:
        """Write ``t`` if new or changed (an OPEN trade that closed). 1 if written."""
        payload = t.model_dump_json()
        row = self._conn.execute("SELECT payload_json FROM trades WHERE id = ?", (t.id,)).fetchone()
        if row is not None and row["payload_json"] == payload:
            return 0
        self._conn.execute(
            "INSERT OR REPLACE INTO trades "
            "(id, account, underlying, entry_session, book, closed, payload_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                t.id,
                t.account,
                t.underlying,
                t.entry_session.isoformat(),
                t.book.value,
                int(t.closed),
                payload,
                now_et().isoformat(),
            ),
        )
        self._conn.commit()
        return 1

    def list(self, *, book: str | None = None, since: date | None = None) -> list:
        from advisor.learning.trades import Trade

        clauses, args = [], []
        if book is not None:
            clauses.append("book = ?")
            args.append(book)
        if since is not None:
            clauses.append("entry_session >= ?")
            args.append(since.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT payload_json FROM trades {where} ORDER BY entry_session, id", args
        ).fetchall()
        return [Trade.model_validate_json(r["payload_json"]) for r in rows]
