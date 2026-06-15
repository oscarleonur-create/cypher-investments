"""SQLite persistence for fundamental research reports.

Shares the file `data/research.db` with `research_agent.store.Store` (each
manages its own tables — no cross-table FKs). Splitting modules keeps the
dip-card pipeline cleanly separate from the deep-research engine.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from advisor.research.models import ResearchReport

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS research_reports (
    symbol TEXT NOT NULL,
    as_of TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, as_of)
);

CREATE INDEX IF NOT EXISTS idx_research_reports_symbol
    ON research_reports(symbol, created_at DESC);

CREATE TABLE IF NOT EXISTS research_artifacts (
    symbol TEXT NOT NULL,
    kind TEXT NOT NULL,         -- 'statements' | 'ratios' | 'red_flags' | 'filing'
    key TEXT NOT NULL,          -- e.g. accession number, or 'latest'
    payload_json TEXT NOT NULL,
    fetched_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, kind, key)
);

CREATE TABLE IF NOT EXISTS kpi_watchlist (
    symbol TEXT NOT NULL PRIMARY KEY,
    kpis_json TEXT NOT NULL,    -- JSON array of KpiDefinition
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kpi_history (
    symbol TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    result_json TEXT NOT NULL,  -- ThesisMonitorResult JSON
    PRIMARY KEY (symbol, checked_at)
);

CREATE INDEX IF NOT EXISTS idx_kpi_history_symbol
    ON kpi_history(symbol, checked_at DESC);

CREATE TABLE IF NOT EXISTS agent_conversations (
    id TEXT NOT NULL PRIMARY KEY,
    symbol TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    messages_json TEXT NOT NULL DEFAULT '[]',  -- JSON array of {role, content, tools?}
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_agent_conversations_symbol
    ON agent_conversations(symbol, updated_at DESC);

CREATE TABLE IF NOT EXISTS watchlist (
    symbol TEXT NOT NULL PRIMARY KEY,
    note TEXT NOT NULL DEFAULT '',
    added_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS theses (
    id TEXT NOT NULL PRIMARY KEY,             -- uuid4().hex[:12]
    symbol TEXT NOT NULL DEFAULT '',          -- '' = thematic (no ticker)
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',         -- markdown body
    tags_json TEXT NOT NULL DEFAULT '[]',     -- JSON array of strings
    conviction TEXT NOT NULL DEFAULT 'MEDIUM',-- HIGH | MEDIUM | LOW
    status TEXT NOT NULL DEFAULT 'DRAFT',     -- DRAFT | ACTIVE | ARCHIVED
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_theses_symbol
    ON theses(symbol, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_theses_updated
    ON theses(updated_at DESC);
"""


class ResearchStore:
    """SQLite-backed persistence for ResearchReport objects + cached artifacts."""

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

    # ── Reports ──────────────────────────────────────────────────────────

    def save_report(self, report: ResearchReport) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO research_reports (symbol, as_of, report_json) "
            "VALUES (?, ?, ?)",
            (report.symbol.upper(), report.as_of.isoformat(), report.model_dump_json()),
        )
        self._conn.commit()

    def load_latest_report(self, symbol: str) -> ResearchReport | None:
        row = self._conn.execute(
            "SELECT report_json FROM research_reports "
            "WHERE symbol = ? ORDER BY as_of DESC, created_at DESC LIMIT 1",
            (symbol.upper(),),
        ).fetchone()
        if row is None:
            return None
        return ResearchReport.model_validate_json(row["report_json"])

    def list_reports(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT symbol, as_of, created_at FROM research_reports "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Artifacts (per-layer cache) ──────────────────────────────────────

    def save_artifact(self, symbol: str, kind: str, key: str, payload_json: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO research_artifacts "
            "(symbol, kind, key, payload_json, fetched_at) VALUES (?, ?, ?, ?, ?)",
            (symbol.upper(), kind, key, payload_json, datetime.now().isoformat()),
        )
        self._conn.commit()

    def load_artifact(
        self, symbol: str, kind: str, key: str = "latest"
    ) -> tuple[str, datetime] | None:
        row = self._conn.execute(
            "SELECT payload_json, fetched_at FROM research_artifacts "
            "WHERE symbol = ? AND kind = ? AND key = ?",
            (symbol.upper(), kind, key),
        ).fetchone()
        if row is None:
            return None
        return row["payload_json"], datetime.fromisoformat(row["fetched_at"])

    # ── KPI watchlist ─────────────────────────────────────────────────────

    def save_kpi_watchlist(self, symbol: str, kpis: list) -> None:
        import json

        self._conn.execute(
            "INSERT OR REPLACE INTO kpi_watchlist (symbol, kpis_json, updated_at) "
            "VALUES (?, ?, ?)",
            (
                symbol.upper(),
                json.dumps([k.model_dump(mode="json") for k in kpis]),
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()

    def load_kpi_watchlist(self, symbol: str) -> list | None:
        import json

        from advisor.research.models import KpiDefinition

        row = self._conn.execute(
            "SELECT kpis_json FROM kpi_watchlist WHERE symbol = ?",
            (symbol.upper(),),
        ).fetchone()
        if row is None:
            return None
        return [KpiDefinition.model_validate(k) for k in json.loads(row["kpis_json"])]

    # ── KPI history ───────────────────────────────────────────────────────

    def save_kpi_check(self, symbol: str, result) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO kpi_history (symbol, checked_at, result_json) "
            "VALUES (?, ?, ?)",
            (symbol.upper(), result.checked_at.isoformat(), result.model_dump_json()),
        )
        self._conn.commit()

    def load_kpi_history(self, symbol: str, limit: int = 30) -> list:
        from advisor.research.models import ThesisMonitorResult

        rows = self._conn.execute(
            "SELECT result_json FROM kpi_history WHERE symbol = ? "
            "ORDER BY checked_at DESC LIMIT ?",
            (symbol.upper(), limit),
        ).fetchall()
        results = []
        for row in rows:
            try:
                results.append(ThesisMonitorResult.model_validate_json(row["result_json"]))
            except Exception:  # noqa: BLE001
                pass
        return results

    # ── Agent conversations ───────────────────────────────────────────────

    def create_conversation(self, symbol: str, title: str = "") -> str:
        import uuid

        conv_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO agent_conversations "
            "(id, symbol, title, messages_json, created_at, updated_at) "
            "VALUES (?, ?, ?, '[]', ?, ?)",
            (conv_id, symbol.upper(), title[:120], now, now),
        )
        self._conn.commit()
        return conv_id

    def append_messages(self, conversation_id: str, messages: list[dict]) -> None:
        """Append visible transcript turns to a conversation."""
        import json

        row = self._conn.execute(
            "SELECT messages_json FROM agent_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return
        existing = json.loads(row["messages_json"])
        existing.extend(messages)
        self._conn.execute(
            "UPDATE agent_conversations SET messages_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(existing), datetime.now().isoformat(), conversation_id),
        )
        self._conn.commit()

    def list_conversations(self, symbol: str, limit: int = 50) -> list[dict]:
        import json

        rows = self._conn.execute(
            "SELECT id, title, created_at, updated_at, messages_json "
            "FROM agent_conversations WHERE symbol = ? ORDER BY updated_at DESC LIMIT ?",
            (symbol.upper(), limit),
        ).fetchall()
        out = []
        for r in rows:
            try:
                n = len(json.loads(r["messages_json"]))
            except Exception:  # noqa: BLE001
                n = 0
            out.append(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "message_count": n,
                }
            )
        return out

    def load_conversation(self, conversation_id: str) -> dict | None:
        import json

        row = self._conn.execute(
            "SELECT id, symbol, title, messages_json, created_at, updated_at "
            "FROM agent_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "symbol": row["symbol"],
            "title": row["title"],
            "messages": json.loads(row["messages_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ── Watchlist ─────────────────────────────────────────────────────────

    def add_to_watchlist(self, symbol: str, note: str = "") -> None:
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO watchlist (symbol, note, added_at, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET note=excluded.note, updated_at=excluded.updated_at",
            (symbol.upper(), note, now, now),
        )
        self._conn.commit()

    def remove_from_watchlist(self, symbol: str) -> None:
        self._conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),))
        self._conn.commit()

    def load_watchlist(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT symbol, note, added_at, updated_at FROM watchlist ORDER BY added_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Theses ────────────────────────────────────────────────────────────

    def list_theses(self, symbol: str | None = None) -> list[dict]:
        """Metadata rows (no markdown body) for list views, newest first."""
        cols = "id, symbol, title, tags_json, conviction, status, created_at, updated_at"
        if symbol:
            rows = self._conn.execute(
                f"SELECT {cols} FROM theses WHERE symbol = ? ORDER BY updated_at DESC",
                (symbol.upper(),),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {cols} FROM theses ORDER BY updated_at DESC"
            ).fetchall()
        return [self._thesis_row(r, include_content=False) for r in rows]

    def get_thesis(self, thesis_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM theses WHERE id = ?", (thesis_id,)).fetchone()
        return self._thesis_row(row, include_content=True) if row else None

    def save_thesis(
        self,
        *,
        thesis_id: str | None = None,
        symbol: str = "",
        title: str = "",
        content: str = "",
        tags: list[str] | None = None,
        conviction: str = "MEDIUM",
        status: str = "DRAFT",
    ) -> dict:
        """Create (no id) or update (existing id) a thesis; returns the saved row.

        ``created_at`` is preserved on update; ``updated_at`` always advances.
        """
        import json
        import uuid

        now = datetime.now().isoformat()
        tags_json = json.dumps(tags or [])
        sym = (symbol or "").strip().upper()
        if thesis_id:
            self._conn.execute(
                "UPDATE theses SET symbol=?, title=?, content=?, tags_json=?, "
                "conviction=?, status=?, updated_at=? WHERE id=?",
                (sym, title, content, tags_json, conviction, status, now, thesis_id),
            )
        else:
            thesis_id = uuid.uuid4().hex[:12]
            self._conn.execute(
                "INSERT INTO theses (id, symbol, title, content, tags_json, "
                "conviction, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (thesis_id, sym, title, content, tags_json, conviction, status, now, now),
            )
        self._conn.commit()
        saved = self.get_thesis(thesis_id)
        assert saved is not None  # just written
        return saved

    def delete_thesis(self, thesis_id: str) -> None:
        self._conn.execute("DELETE FROM theses WHERE id = ?", (thesis_id,))
        self._conn.commit()

    @staticmethod
    def _thesis_row(row, *, include_content: bool) -> dict:
        import json

        out = {
            "id": row["id"],
            "symbol": row["symbol"],
            "title": row["title"],
            "tags": json.loads(row["tags_json"]),
            "conviction": row["conviction"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if include_content:
            out["content"] = row["content"]
        return out
