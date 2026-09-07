"""SQLite persistence for the daemon.

Shares `data/research.db` with `ResearchStore` and `research_agent.store.Store`,
following the same convention: each module owns its own tables and there are no
cross-module foreign keys.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from advisor.daemon.market_calendar import now_et
from advisor.daemon.models import (
    Event,
    EventSource,
    EventTier,
    Heartbeat,
    Watermark,
)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS events (
    id          TEXT NOT NULL PRIMARY KEY,
    ts          TEXT NOT NULL,
    source      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    tier        TEXT NOT NULL,           -- 'A' | 'B' | 'C'
    symbol      TEXT,                    -- NULL for book-level / macro-wide
    payload_json TEXT NOT NULL DEFAULT '{}',
    dedup_hash  TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now'))
);

-- The dedup guarantee: re-ingesting the same fact is a silent no-op.
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_dedup ON events(dedup_hash);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_symbol ON events(symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_tier ON events(tier, ts DESC);

CREATE TABLE IF NOT EXISTS watermarks (
    source           TEXT NOT NULL PRIMARY KEY,
    last_seen_ts     TEXT,
    last_seen_cursor TEXT NOT NULL DEFAULT '',
    updated_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS book_snapshots (
    as_of        TEXT NOT NULL PRIMARY KEY,
    snapshot_json TEXT NOT NULL,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_book_snapshots_as_of ON book_snapshots(as_of DESC);

CREATE TABLE IF NOT EXISTS macro_sensitivity (
    symbol      TEXT NOT NULL,
    asof        TEXT NOT NULL,
    payload_json TEXT NOT NULL,      -- SymbolSensitivity
    created_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, asof)
);

CREATE INDEX IF NOT EXISTS idx_macro_sensitivity_symbol
    ON macro_sensitivity(symbol, asof DESC);

CREATE TABLE IF NOT EXISTS book_exposure (
    asof        TEXT NOT NULL PRIMARY KEY,
    payload_json TEXT NOT NULL,      -- BookExposure
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS source_items (
    dedup_key   TEXT NOT NULL PRIMARY KEY,   -- SEC accession, or a hash of the URL
    symbol      TEXT NOT NULL,
    tier        TEXT NOT NULL,
    provider    TEXT NOT NULL,
    published_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,              -- SourceItem
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_source_items_symbol
    ON source_items(symbol, published_at DESC);

CREATE TABLE IF NOT EXISTS daemon_heartbeat (
    job         TEXT NOT NULL PRIMARY KEY,
    last_run_at TEXT,
    last_ok_at  TEXT,
    run_count   INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT NOT NULL DEFAULT ''
);
"""


class DaemonStore:
    """Event stream, ingest watermarks, and job heartbeats."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Events ────────────────────────────────────────────────────────────

    def emit(self, event: Event) -> bool:
        """Append ``event``; return False if it was already seen.

        Dedup is enforced by a unique index rather than a read-then-write, so
        two pollers racing on the same filing cannot both insert it.
        """
        try:
            self._conn.execute(
                "INSERT INTO events (id, ts, source, kind, tier, symbol, payload_json, "
                "dedup_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.id,
                    event.ts.isoformat(),
                    event.source.value,
                    event.kind,
                    event.tier.value,
                    event.symbol.upper() if event.symbol else None,
                    json.dumps(event.payload, default=str),
                    event.dedup_hash(),
                ),
            )
        except sqlite3.IntegrityError:
            return False
        self._conn.commit()
        return True

    def emit_many(self, events: list[Event]) -> int:
        """Append a batch; returns how many were new."""
        return sum(1 for e in events if self.emit(e))

    def recent_events(
        self,
        *,
        limit: int = 50,
        tier: EventTier | None = None,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> list[Event]:
        """Most recent events first, optionally filtered."""
        sql = "SELECT * FROM events WHERE 1=1"
        params: list = []
        if tier is not None:
            sql += " AND tier = ?"
            params.append(tier.value)
        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol.upper())
        if since is not None:
            sql += " AND ts >= ?"
            params.append(since.isoformat())
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return [self._row_to_event(r) for r in self._conn.execute(sql, params).fetchall()]

    def event_counts_by_tier(self, since: datetime | None = None) -> dict[str, int]:
        """Tier histogram, for `daemon status`."""
        sql = "SELECT tier, COUNT(*) AS n FROM events"
        params: list = []
        if since is not None:
            sql += " WHERE ts >= ?"
            params.append(since.isoformat())
        sql += " GROUP BY tier"
        return {r["tier"]: r["n"] for r in self._conn.execute(sql, params).fetchall()}

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"],
            ts=datetime.fromisoformat(row["ts"]),
            source=EventSource(row["source"]),
            kind=row["kind"],
            tier=EventTier(row["tier"]),
            symbol=row["symbol"],
            payload=json.loads(row["payload_json"]),
        )

    # ── Watermarks ────────────────────────────────────────────────────────

    def get_watermark(self, source: EventSource) -> Watermark:
        """Watermark for ``source``; a fresh one when the source is untouched."""
        row = self._conn.execute(
            "SELECT * FROM watermarks WHERE source = ?", (source.value,)
        ).fetchone()
        if row is None:
            return Watermark(source=source)
        return Watermark(
            source=source,
            last_seen_ts=(
                datetime.fromisoformat(row["last_seen_ts"]) if row["last_seen_ts"] else None
            ),
            last_seen_cursor=row["last_seen_cursor"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def set_watermark(
        self,
        source: EventSource,
        *,
        last_seen_ts: datetime | None = None,
        last_seen_cursor: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO watermarks (source, last_seen_ts, last_seen_cursor, "
            "updated_at) VALUES (?, ?, ?, ?)",
            (
                source.value,
                last_seen_ts.isoformat() if last_seen_ts else None,
                last_seen_cursor,
                now_et().isoformat(),
            ),
        )
        self._conn.commit()

    def all_watermarks(self) -> list[Watermark]:
        return [self.get_watermark(s) for s in EventSource]

    # ── Heartbeats ────────────────────────────────────────────────────────

    def get_heartbeat(self, job: str) -> Heartbeat:
        row = self._conn.execute("SELECT * FROM daemon_heartbeat WHERE job = ?", (job,)).fetchone()
        if row is None:
            return Heartbeat(job=job)
        return Heartbeat(
            job=job,
            last_run_at=(
                datetime.fromisoformat(row["last_run_at"]) if row["last_run_at"] else None
            ),
            last_ok_at=(datetime.fromisoformat(row["last_ok_at"]) if row["last_ok_at"] else None),
            run_count=row["run_count"],
            error_count=row["error_count"],
            last_error=row["last_error"],
        )

    def record_run(self, job: str, *, ok: bool, error: str = "") -> Heartbeat:
        """Stamp a job execution. Failures increment ``error_count`` but never
        clear ``last_ok_at``, so `daemon status` can show "failing since"."""
        hb = self.get_heartbeat(job)
        now = now_et()
        hb.last_run_at = now
        hb.run_count += 1
        if ok:
            hb.last_ok_at = now
            hb.last_error = ""
        else:
            hb.error_count += 1
            hb.last_error = error[:500]
        self._conn.execute(
            "INSERT OR REPLACE INTO daemon_heartbeat (job, last_run_at, last_ok_at, "
            "run_count, error_count, last_error) VALUES (?, ?, ?, ?, ?, ?)",
            (
                hb.job,
                hb.last_run_at.isoformat(),
                hb.last_ok_at.isoformat() if hb.last_ok_at else None,
                hb.run_count,
                hb.error_count,
                hb.last_error,
            ),
        )
        self._conn.commit()
        return hb

    # ── Book snapshots ────────────────────────────────────────────────────

    def save_book(self, snapshot, *, keep: int = 500) -> None:
        """Persist a book snapshot and prune the history.

        A partial snapshot (an account failed to load) is never saved: diffing
        against it would report every position in the missing account as
        closed, and fire a wave of false alerts.
        """
        if getattr(snapshot, "partial", False):
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO book_snapshots (as_of, snapshot_json) VALUES (?, ?)",
            (snapshot.as_of.isoformat(), snapshot.model_dump_json()),
        )
        self._conn.execute(
            "DELETE FROM book_snapshots WHERE as_of NOT IN "
            "(SELECT as_of FROM book_snapshots ORDER BY as_of DESC LIMIT ?)",
            (keep,),
        )
        self._conn.commit()

    def load_latest_book(self):
        """Most recent stored snapshot, or None when the book has never run."""
        from advisor.daemon.book import BookSnapshot

        row = self._conn.execute(
            "SELECT snapshot_json FROM book_snapshots ORDER BY as_of DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return BookSnapshot.model_validate_json(row["snapshot_json"])

    def book_snapshot_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM book_snapshots").fetchone()["n"]

    # ── Source items (filings, news) ──────────────────────────────────────

    def save_source_item(self, item) -> bool:
        """Archive one item. False when it was already stored.

        The primary key is the item's own dedup key — an SEC accession number
        when there is one — so two pollers racing on the same filing cannot
        both insert, and a re-run after a crash is a no-op rather than a
        duplicate.
        """
        try:
            self._conn.execute(
                "INSERT INTO source_items (dedup_key, symbol, tier, provider, published_at, "
                "retrieved_at, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    item.dedup_key(),
                    item.entity.symbol.upper(),
                    item.tier.value,
                    item.provider,
                    item.published_at.isoformat(),
                    item.retrieved_at.isoformat(),
                    item.model_dump_json(),
                ),
            )
        except sqlite3.IntegrityError:
            return False
        self._conn.commit()
        return True

    def recent_source_items(self, symbol: str | None = None, limit: int = 50) -> list:
        """Archived items, newest published first."""
        from advisor.news.models import SourceItem

        if symbol:
            rows = self._conn.execute(
                "SELECT payload_json FROM source_items WHERE symbol = ? "
                "ORDER BY published_at DESC LIMIT ?",
                (symbol.upper(), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload_json FROM source_items ORDER BY published_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [SourceItem.model_validate_json(r["payload_json"]) for r in rows]

    def source_items_between(self, symbol: str, start, end) -> list:
        """Items for ``symbol`` published in [start, end] — used by coverage scoring."""
        from advisor.news.models import SourceItem

        rows = self._conn.execute(
            "SELECT payload_json FROM source_items WHERE symbol = ? "
            "AND published_at >= ? AND published_at <= ? ORDER BY published_at",
            (symbol.upper(), start.isoformat(), end.isoformat()),
        ).fetchall()
        return [SourceItem.model_validate_json(r["payload_json"]) for r in rows]

    # ── Macro sensitivities and book exposure ────────────────────────────

    def save_sensitivity(self, sensitivity) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO macro_sensitivity (symbol, asof, payload_json) "
            "VALUES (?, ?, ?)",
            (
                sensitivity.symbol.upper(),
                sensitivity.asof.isoformat(),
                sensitivity.model_dump_json(),
            ),
        )
        self._conn.commit()

    def load_sensitivity(self, symbol: str):
        """Most recent estimate for ``symbol``, or None if never estimated."""
        from advisor.macro.sensitivity import SymbolSensitivity

        row = self._conn.execute(
            "SELECT payload_json FROM macro_sensitivity WHERE symbol = ? "
            "ORDER BY asof DESC LIMIT 1",
            (symbol.upper(),),
        ).fetchone()
        if row is None:
            return None
        return SymbolSensitivity.model_validate_json(row["payload_json"])

    def load_sensitivities(self, symbols: list[str] | None = None) -> dict:
        """Latest estimate per symbol, keyed by symbol."""
        out = {}
        rows = self._conn.execute("SELECT DISTINCT symbol FROM macro_sensitivity").fetchall()
        wanted = {s.upper() for s in symbols} if symbols else None
        for row in rows:
            symbol = row["symbol"]
            if wanted is not None and symbol not in wanted:
                continue
            estimate = self.load_sensitivity(symbol)
            if estimate is not None:
                out[symbol] = estimate
        return out

    def save_exposure(self, exposure) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO book_exposure (asof, payload_json) VALUES (?, ?)",
            (exposure.asof.isoformat(), exposure.model_dump_json()),
        )
        self._conn.commit()

    def load_latest_exposure(self):
        from advisor.macro.exposure import BookExposure

        row = self._conn.execute(
            "SELECT payload_json FROM book_exposure ORDER BY asof DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return BookExposure.model_validate_json(row["payload_json"])

    def all_heartbeats(self) -> list[Heartbeat]:
        rows = self._conn.execute("SELECT job FROM daemon_heartbeat ORDER BY job").fetchall()
        return [self.get_heartbeat(r["job"]) for r in rows]
