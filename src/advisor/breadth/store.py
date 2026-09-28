"""The breadth store: its own SQLite file beside the live database.

Four years of daily bars for ~5,000 listings is a few million rows and well
over 100 MB. That does not belong in the shared ``research.db`` the daemon
and the UI read on every request; the replay store set the precedent
(``research-replay.db``). Everything here is rebuildable from free sources,
so losing the file costs a backfill, never a record.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS breadth_bars (
    symbol  TEXT NOT NULL,
    day     TEXT NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL NOT NULL,          -- split-adjusted, not dividend-adjusted
    volume  REAL,
    PRIMARY KEY (symbol, day)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS breadth_bar_state (
    symbol      TEXT NOT NULL PRIMARY KEY,
    first_day   TEXT,
    last_day    TEXT,
    synced_at   TEXT NOT NULL,
    rebased_at  TEXT,               -- last full refetch after a split changed history
    last_error  TEXT
);
CREATE TABLE IF NOT EXISTS breadth_facts (
    cik       INTEGER NOT NULL,
    concept   TEXT NOT NULL,
    frame     TEXT NOT NULL,        -- CY2026Q2 (a quarter) or CY2025 (a year)
    val       REAL NOT NULL,
    start     TEXT,
    end       TEXT NOT NULL,
    accn      TEXT,
    source    TEXT NOT NULL,        -- 'frames' or 'companyconcept'
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (cik, concept, frame)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS breadth_frame_state (
    concept    TEXT NOT NULL,
    frame      TEXT NOT NULL,
    status     INTEGER NOT NULL,    -- HTTP status: 200, or 404 for a frame nobody filed
    rows       INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (concept, frame)
);
CREATE TABLE IF NOT EXISTS breadth_concept_state (
    cik        INTEGER NOT NULL,
    concept    TEXT NOT NULL,
    rows       INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (cik, concept)
);
CREATE TABLE IF NOT EXISTS breadth_universe (
    day           TEXT NOT NULL,
    symbol        TEXT NOT NULL,     -- Yahoo spelling
    cik           INTEGER,
    exchange      TEXT,
    name          TEXT,
    eligible      INTEGER NOT NULL,
    reason        TEXT,              -- why not eligible; NULL when eligible
    price         REAL,
    dollar_volume REAL,              -- median of the window's close x volume
    sessions      INTEGER,
    rules         TEXT NOT NULL,     -- version of the E0 rules that decided it
    PRIMARY KEY (day, symbol)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS breadth_companies (
    cik        INTEGER NOT NULL PRIMARY KEY,
    sic        INTEGER,                -- NULL when the SEC lists none
    sic_desc   TEXT,
    name       TEXT,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS breadth_records (
    id            TEXT NOT NULL PRIMARY KEY,  -- origin:run:grp:symbol:day
    origin        TEXT NOT NULL,              -- 'live' or 'replay'
    run_id        TEXT NOT NULL,              -- 'live', or the replay run's id
    grp           TEXT NOT NULL,              -- 'P', 'F' or 'F+P'
    symbol        TEXT NOT NULL,
    cik           INTEGER,
    day           TEXT NOT NULL,              -- the session whose close decided it
    rules         TEXT NOT NULL,
    detail_json   TEXT NOT NULL,              -- which signals, and their numbers
    outcomes_json TEXT,                       -- forward returns vs matched controls
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_breadth_records_run ON breadth_records(run_id, grp, day);
CREATE TABLE IF NOT EXISTS breadth_replay_runs (
    id           TEXT NOT NULL PRIMARY KEY,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    code         TEXT NOT NULL,
    rules        TEXT NOT NULL,
    params_json  TEXT NOT NULL,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS breadth_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    ok           INTEGER,
    summary_json TEXT
);
"""


def breadth_path(db_path) -> Path:
    """The breadth store's own file, beside the live database."""
    db_path = Path(db_path)
    return db_path.with_name(f"{db_path.stem}-breadth{db_path.suffix}")


class BreadthStore:
    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30)
        self.conn.row_factory = sqlite3.Row
        # The daemon writes while the CLI and UI read.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> BreadthStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── runs ────────────────────────────────────────────────────────────

    def start_run(self, now: datetime) -> int:
        cur = self.conn.execute(
            "INSERT INTO breadth_runs (started_at) VALUES (?)", (now.isoformat(),)
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, now: datetime, ok: bool, summary: dict) -> None:
        self.conn.execute(
            "UPDATE breadth_runs SET finished_at = ?, ok = ?, summary_json = ? WHERE id = ?",
            (now.isoformat(), int(ok), json.dumps(summary, default=str), run_id),
        )
        self.conn.commit()

    def last_run(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM breadth_runs WHERE finished_at IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["summary"] = json.loads(out.pop("summary_json") or "{}")
        return out

    # ── universe ────────────────────────────────────────────────────────

    def write_universe(self, day: date, rows: list[dict]) -> None:
        """Replace ``day``'s snapshot. Re-running a day rewrites it, never duplicates."""
        with self.conn:
            self.conn.execute("DELETE FROM breadth_universe WHERE day = ?", (day.isoformat(),))
            self.conn.executemany(
                "INSERT INTO breadth_universe (day, symbol, cik, exchange, name, eligible, "
                "reason, price, dollar_volume, sessions, rules) VALUES "
                "(:day, :symbol, :cik, :exchange, :name, :eligible, :reason, :price, "
                ":dollar_volume, :sessions, :rules)",
                [{**r, "day": day.isoformat(), "eligible": int(r["eligible"])} for r in rows],
            )

    def latest_universe_day(self) -> date | None:
        row = self.conn.execute("SELECT MAX(day) FROM breadth_universe").fetchone()
        return date.fromisoformat(row[0]) if row and row[0] else None

    def universe(self, day: date, eligible_only: bool = False) -> list[dict]:
        sql = "SELECT * FROM breadth_universe WHERE day = ?"
        if eligible_only:
            sql += " AND eligible = 1"
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY symbol", (day.isoformat(),))]

    def counts(self) -> dict[str, int]:
        out = {}
        for table in (
            "breadth_bars",
            "breadth_bar_state",
            "breadth_facts",
            "breadth_frame_state",
            "breadth_universe",
        ):
            out[table] = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return out
