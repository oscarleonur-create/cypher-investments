"""Workstation persistence — saved scenarios (step 4).

Sibling to `advisor.research.store.ResearchStore` but lives in the dashboard
package because these tables capture user actions in the UI, not pipeline
output. Both stores write to the same SQLite file (`data/research.db`).

Step 4 ships the scenario store. Sessions + notes tables come in step 5.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from advisor.research.config import get_settings
from advisor.research.models import DcfAssumptions, DcfScenario

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS research_scenarios (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    assumptions_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    saved_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_research_scenarios_symbol
    ON research_scenarios(symbol, saved_at DESC);

CREATE TABLE IF NOT EXISTS research_sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    last_opened_at TEXT DEFAULT (datetime('now')),
    pinned INTEGER DEFAULT 0,
    current_symbol TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_research_sessions_last_opened
    ON research_sessions(last_opened_at DESC);

CREATE TABLE IF NOT EXISTS research_session_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts TEXT DEFAULT (datetime('now')),
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_session_events_session_ts
    ON research_session_events(session_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_session_events_symbol
    ON research_session_events(session_id, symbol, ts DESC);

CREATE TABLE IF NOT EXISTS research_notes (
    session_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    note_md TEXT DEFAULT '',
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, symbol)
);
"""


class WorkstationStore:
    """SQLite-backed persistence for workstation-level user state."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or get_settings().db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Scenarios ───────────────────────────────────────────────────────

    def save_scenario(
        self,
        symbol: str,
        name: str,
        assumptions: DcfAssumptions,
        result: DcfScenario,
    ) -> str:
        scenario_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO research_scenarios "
            "(id, symbol, name, assumptions_json, result_json, saved_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                scenario_id,
                symbol.upper(),
                name,
                assumptions.model_dump_json(),
                result.model_dump_json(),
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()
        return scenario_id

    def list_scenarios(self, symbol: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, name, assumptions_json, result_json, saved_at "
            "FROM research_scenarios WHERE symbol = ? "
            "ORDER BY saved_at DESC",
            (symbol.upper(),),
        ).fetchall()
        result: list[dict] = []
        for r in rows:
            result.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "assumptions": DcfAssumptions.model_validate_json(r["assumptions_json"]),
                    "result": DcfScenario.model_validate_json(r["result_json"]),
                    "saved_at": r["saved_at"],
                }
            )
        return result

    def delete_scenario(self, scenario_id: str) -> None:
        self._conn.execute("DELETE FROM research_scenarios WHERE id = ?", (scenario_id,))
        self._conn.commit()

    # ── Sessions ───────────────────────────────────────────────────────

    def new_session(self, name: str | None = None) -> str:
        session_id = str(uuid.uuid4())
        session_name = name or f"Session {datetime.now():%Y-%m-%d %H:%M}"
        self._conn.execute(
            "INSERT INTO research_sessions (id, name) VALUES (?, ?)",
            (session_id, session_name),
        )
        self._conn.commit()
        return session_id

    def list_sessions(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT s.id, s.name, s.created_at, s.last_opened_at, s.pinned, "
            "s.current_symbol, "
            "(SELECT COUNT(*) FROM research_session_events e "
            " WHERE e.session_id = s.id) AS event_count "
            "FROM research_sessions s "
            "ORDER BY s.pinned DESC, s.last_opened_at DESC "
            "LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def open_session(self, session_id: str, current_symbol: str | None = None) -> dict | None:
        if current_symbol is not None:
            self._conn.execute(
                "UPDATE research_sessions "
                "SET last_opened_at = datetime('now'), current_symbol = ? "
                "WHERE id = ?",
                (current_symbol.upper(), session_id),
            )
        else:
            self._conn.execute(
                "UPDATE research_sessions SET last_opened_at = datetime('now') WHERE id = ?",
                (session_id,),
            )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id, name, created_at, last_opened_at, pinned, current_symbol "
            "FROM research_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    def set_pinned(self, session_id: str, pinned: bool) -> None:
        self._conn.execute(
            "UPDATE research_sessions SET pinned = ? WHERE id = ?",
            (1 if pinned else 0, session_id),
        )
        self._conn.commit()

    def rename_session(self, session_id: str, name: str) -> None:
        self._conn.execute(
            "UPDATE research_sessions SET name = ? WHERE id = ?",
            (name, session_id),
        )
        self._conn.commit()

    def delete_session(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM research_session_events WHERE session_id = ?", (session_id,)
        )
        self._conn.execute("DELETE FROM research_notes WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM research_sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    # ── Session events ─────────────────────────────────────────────────

    def log_event(
        self,
        session_id: str,
        symbol: str,
        event_type: str,
        payload: dict | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO research_session_events "
            "(session_id, symbol, event_type, payload_json) "
            "VALUES (?, ?, ?, ?)",
            (
                session_id,
                symbol.upper(),
                event_type,
                json.dumps(payload or {}),
            ),
        )
        self._conn.commit()

    def list_events(
        self,
        session_id: str,
        symbol: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        if symbol:
            rows = self._conn.execute(
                "SELECT ts, symbol, event_type, payload_json FROM research_session_events "
                "WHERE session_id = ? AND symbol = ? ORDER BY ts DESC LIMIT ?",
                (session_id, symbol.upper(), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT ts, symbol, event_type, payload_json FROM research_session_events "
                "WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        result = []
        for r in rows:
            try:
                payload = json.loads(r["payload_json"]) if r["payload_json"] else {}
            except json.JSONDecodeError:
                payload = {}
            result.append(
                {
                    "ts": r["ts"],
                    "symbol": r["symbol"],
                    "event_type": r["event_type"],
                    "payload": payload,
                }
            )
        return result

    # ── Notes ──────────────────────────────────────────────────────────

    def save_note(self, session_id: str, symbol: str, note_md: str) -> None:
        self._conn.execute(
            "INSERT INTO research_notes (session_id, symbol, note_md, updated_at) "
            "VALUES (?, ?, ?, datetime('now')) "
            "ON CONFLICT(session_id, symbol) DO UPDATE SET "
            "note_md = excluded.note_md, updated_at = excluded.updated_at",
            (session_id, symbol.upper(), note_md),
        )
        self._conn.commit()

    def load_note(self, session_id: str, symbol: str) -> dict | None:
        row = self._conn.execute(
            "SELECT note_md, updated_at FROM research_notes " "WHERE session_id = ? AND symbol = ?",
            (session_id, symbol.upper()),
        ).fetchone()
        return dict(row) if row else None


# Module-level convenience accessor — opens a fresh connection per call so
# Streamlit's threading model doesn't bite us.


def get_store() -> WorkstationStore:
    return WorkstationStore()
