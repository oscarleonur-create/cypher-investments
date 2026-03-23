"""SQLite persistence for research runs, sources, and search cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from research_agent.models import BatchResult, OpportunityCard

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    input_value TEXT NOT NULL,
    verdict TEXT,
    dip_type TEXT,
    card_json TEXT NOT NULL,
    batch_id TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES runs(id),
    source_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    publisher TEXT,
    tier INTEGER DEFAULT 3,
    UNIQUE(run_id, source_id)
);

CREATE TABLE IF NOT EXISTS search_cache (
    query_hash TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    symbols TEXT NOT NULL,
    total_requested INTEGER NOT NULL,
    total_completed INTEGER NOT NULL,
    total_failed INTEGER NOT NULL,
    failures_json TEXT DEFAULT '{}',
    elapsed_seconds REAL DEFAULT 0.0,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

_MIGRATIONS = [
    "ALTER TABLE runs ADD COLUMN batch_id TEXT",
]


class Store:
    """SQLite-backed persistence for research agent data."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self._run_migrations()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _run_migrations(self) -> None:
        """Apply migrations that may fail if columns already exist."""
        for migration in _MIGRATIONS:
            try:
                self._conn.execute(migration)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists

    def close(self) -> None:
        self._conn.close()

    # ── Runs ─────────────────────────────────────────────────────────────

    def save_run(self, card: OpportunityCard, batch_id: str | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs "
            "(id, mode, input_value, verdict, dip_type, card_json, batch_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                card.id,
                card.input.mode.value,
                card.input.value,
                card.verdict.value if card.verdict else None,
                card.dip_type.value if card.dip_type else None,
                card.model_dump_json(),
                batch_id,
            ),
        )
        # Save sources
        for idx, src in enumerate(card.sources):
            source_id = f"s{idx + 1}"
            self._conn.execute(
                "INSERT OR REPLACE INTO sources (run_id, source_id, url, title, publisher, tier) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (card.id, source_id, src.url, src.title, src.publisher, src.tier),
            )
        self._conn.commit()

    def load_run(self, run_id: str) -> OpportunityCard | None:
        row = self._conn.execute("SELECT card_json FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return OpportunityCard.model_validate_json(row["card_json"])

    def list_runs(
        self,
        ticker: str | None = None,
        mode: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        query = "SELECT id, mode, input_value, verdict, dip_type, created_at FROM runs"
        params: list = []
        conditions: list[str] = []
        if ticker:
            conditions.append("mode = 'ticker' AND input_value = ?")
            params.append(ticker.upper())
        if mode:
            conditions.append("mode = ?")
            params.append(mode)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ── Search cache ─────────────────────────────────────────────────────

    @staticmethod
    def _query_hash(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]

    def cache_search(self, query: str, response: dict, ttl_hours: int = 24) -> None:
        expires = datetime.now() + timedelta(hours=ttl_hours)
        self._conn.execute(
            "INSERT OR REPLACE INTO search_cache (query_hash, query, response, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (
                self._query_hash(query),
                query,
                json.dumps(response),
                expires.isoformat(),
            ),
        )
        self._conn.commit()

    def get_cached_search(self, query: str) -> dict | None:
        row = self._conn.execute(
            "SELECT response, expires_at FROM search_cache WHERE query_hash = ?",
            (self._query_hash(query),),
        ).fetchone()
        if row is None:
            return None
        if datetime.fromisoformat(row["expires_at"]) < datetime.now():
            self._conn.execute(
                "DELETE FROM search_cache WHERE query_hash = ?",
                (self._query_hash(query),),
            )
            self._conn.commit()
            return None
        return json.loads(row["response"])

    # ── Batches ──────────────────────────────────────────────────────────

    def save_batch(self, batch: BatchResult) -> None:
        """Persist a batch result and link its cards."""
        self._conn.execute(
            "INSERT OR REPLACE INTO batches "
            "(batch_id, symbols, total_requested, total_completed, total_failed, "
            "failures_json, elapsed_seconds) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                batch.batch_id,
                ",".join(
                    [c.input.value.upper() for c in batch.cards] + list(batch.failures.keys())
                ),
                batch.total_requested,
                batch.total_completed,
                batch.total_failed,
                json.dumps(batch.failures),
                batch.elapsed_seconds,
            ),
        )
        # Save individual cards linked to the batch
        for card in batch.cards:
            self.save_run(card, batch_id=batch.batch_id)
        self._conn.commit()

    def list_batches(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT batch_id, symbols, total_requested, total_completed, "
            "total_failed, elapsed_seconds, created_at "
            "FROM batches ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def load_batch_cards(self, batch_id: str) -> list[OpportunityCard]:
        rows = self._conn.execute(
            "SELECT card_json FROM runs WHERE batch_id = ?", (batch_id,)
        ).fetchall()
        return [OpportunityCard.model_validate_json(r["card_json"]) for r in rows]
