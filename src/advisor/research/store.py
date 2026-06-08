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
