"""SQLite persistence for simulator chain snapshots, candidates, and results."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from advisor.simulator.models import CalibrationRecord, PCSCandidate, SimResult

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DB = _PROJECT_ROOT / "data" / "simulator.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS chain_snapshots (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    expiration TEXT NOT NULL,
    strike REAL NOT NULL,
    dte INTEGER,
    bid REAL,
    ask REAL,
    mid REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    iv REAL,
    underlying_price REAL,
    iv_rank REAL,
    iv_percentile REAL,
    snapshot_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, expiration, strike, snapshot_at)
);

CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    expiration TEXT NOT NULL,
    dte INTEGER,
    short_strike REAL NOT NULL,
    long_strike REAL NOT NULL,
    width REAL,
    short_bid REAL,
    short_ask REAL,
    long_bid REAL,
    long_ask REAL,
    net_credit REAL,
    mid_credit REAL,
    short_delta REAL,
    short_gamma REAL,
    short_theta REAL,
    short_vega REAL,
    short_iv REAL,
    long_delta REAL,
    long_iv REAL,
    underlying_price REAL,
    iv_percentile REAL,
    iv_rank REAL,
    pop_estimate REAL,
    sell_score REAL,
    buying_power REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sim_results (
    id TEXT PRIMARY KEY,
    candidate_id TEXT REFERENCES candidates(id),
    symbol TEXT NOT NULL,
    short_strike REAL,
    long_strike REAL,
    dte INTEGER,
    net_credit REAL,
    ev REAL,
    pop REAL,
    touch_prob REAL,
    cvar_95 REAL,
    stop_prob REAL,
    avg_hold_days REAL,
    ev_per_bp REAL,
    mc_std_err REAL DEFAULT 0,
    variance_reduction_factor REAL DEFAULT 1,
    cvar_95_is REAL DEFAULT 0,
    cvar_95_se REAL DEFAULT 0,
    pnl_p5 REAL,
    pnl_p25 REAL,
    pnl_p50 REAL,
    pnl_p75 REAL,
    pnl_p95 REAL,
    exit_profit_target REAL,
    exit_stop_loss REAL,
    exit_dte REAL,
    exit_expiration REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS calibration_tracking (
    id TEXT PRIMARY KEY,
    candidate_id TEXT REFERENCES candidates(id),
    symbol TEXT NOT NULL,
    predicted_pop REAL,
    predicted_touch REAL,
    predicted_stop REAL,
    predicted_ev REAL,
    actual_profit REAL,
    actual_touch REAL,
    actual_stop REAL,
    actual_pnl REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_snapshots_symbol ON chain_snapshots(symbol, snapshot_at);
CREATE INDEX IF NOT EXISTS idx_candidates_symbol ON candidates(symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_results_ev_bp ON sim_results(ev_per_bp DESC);
CREATE INDEX IF NOT EXISTS idx_calibration_symbol ON calibration_tracking(symbol, created_at);
"""


class SimulatorStore:
    """SQLite store for simulator data following research_agent/store.py pattern."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that may be missing from older databases."""
        migrations = [
            ("sim_results", "mc_std_err", "REAL DEFAULT 0"),
            ("sim_results", "variance_reduction_factor", "REAL DEFAULT 1"),
            ("sim_results", "cvar_95_is", "REAL DEFAULT 0"),
            ("sim_results", "cvar_95_se", "REAL DEFAULT 0"),
        ]
        for table, column, col_type in migrations:
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Chain snapshots ────────────────────────────────────────────────────────

    def save_chain_snapshot(self, records: list[dict], symbol: str) -> int:
        """Save enriched chain records as a snapshot. Returns count saved."""
        now = datetime.now().isoformat()
        rows = []
        for r in records:
            rows.append(
                (
                    str(uuid.uuid4())[:8],
                    symbol,
                    r.get("expiration", ""),
                    r.get("strike", 0),
                    r.get("dte", 0),
                    r.get("bid", 0),
                    r.get("ask", 0),
                    r.get("mid", 0),
                    r.get("delta", 0),
                    r.get("gamma", 0),
                    r.get("theta", 0),
                    r.get("vega", 0),
                    r.get("iv", 0),
                    r.get("underlying_price", 0),
                    r.get("iv_rank", 0),
                    r.get("iv_percentile", 0),
                    now,
                )
            )
        self._conn.executemany(
            "INSERT OR REPLACE INTO chain_snapshots "
            "(id, symbol, expiration, strike, dte, bid, ask, mid, "
            "delta, gamma, theta, vega, iv, underlying_price, iv_rank, iv_percentile, snapshot_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def get_chain_snapshots(self, symbol: str, limit: int = 500) -> list[dict]:
        """Get recent chain snapshots for a symbol."""
        cursor = self._conn.execute(
            "SELECT * FROM chain_snapshots WHERE symbol = ? ORDER BY snapshot_at DESC LIMIT ?",
            (symbol, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ── Candidates ─────────────────────────────────────────────────────────────

    def save_candidates_batch(self, candidates: list[PCSCandidate]) -> list[str]:
        """Save PCS candidates. Returns list of assigned IDs."""
        ids = []
        rows = []
        for c in candidates:
            cid = str(uuid.uuid4())[:8]
            ids.append(cid)
            rows.append(
                (
                    cid,
                    c.symbol,
                    c.expiration,
                    c.dte,
                    c.short_strike,
                    c.long_strike,
                    c.width,
                    c.short_bid,
                    c.short_ask,
                    c.long_bid,
                    c.long_ask,
                    c.net_credit,
                    c.mid_credit,
                    c.short_delta,
                    c.short_gamma,
                    c.short_theta,
                    c.short_vega,
                    c.short_iv,
                    c.long_delta,
                    c.long_iv,
                    c.underlying_price,
                    c.iv_percentile,
                    c.iv_rank,
                    c.pop_estimate,
                    c.sell_score,
                    c.buying_power,
                )
            )
        self._conn.executemany(
            "INSERT INTO candidates "
            "(id, symbol, expiration, dte, short_strike, long_strike, width, "
            "short_bid, short_ask, long_bid, long_ask, net_credit, mid_credit, "
            "short_delta, short_gamma, short_theta, short_vega, short_iv, "
            "long_delta, long_iv, underlying_price, iv_percentile, iv_rank, "
            "pop_estimate, sell_score, buying_power) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return ids

    # ── Sim results ────────────────────────────────────────────────────────────

    def save_sim_result(self, result: SimResult, candidate_id: str = "") -> str:
        """Save a simulation result. Returns assigned ID."""
        rid = str(uuid.uuid4())[:8]
        self._conn.execute(
            "INSERT INTO sim_results "
            "(id, candidate_id, symbol, short_strike, long_strike, dte, net_credit, "
            "ev, pop, touch_prob, cvar_95, stop_prob, avg_hold_days, ev_per_bp, "
            "mc_std_err, variance_reduction_factor, cvar_95_is, cvar_95_se, "
            "pnl_p5, pnl_p25, pnl_p50, pnl_p75, pnl_p95, "
            "exit_profit_target, exit_stop_loss, exit_dte, exit_expiration) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rid,
                candidate_id,
                result.symbol,
                result.short_strike,
                result.long_strike,
                result.dte,
                result.net_credit,
                result.ev,
                result.pop,
                result.touch_prob,
                result.cvar_95,
                result.stop_prob,
                result.avg_hold_days,
                result.ev_per_bp,
                result.mc_std_err,
                result.variance_reduction_factor,
                result.cvar_95_is,
                result.cvar_95_se,
                result.pnl_p5,
                result.pnl_p25,
                result.pnl_p50,
                result.pnl_p75,
                result.pnl_p95,
                result.exit_profit_target,
                result.exit_stop_loss,
                result.exit_dte,
                result.exit_expiration,
            ),
        )
        self._conn.commit()
        return rid

    def get_top_results(self, limit: int = 20) -> list[dict]:
        """Get top simulation results joined with candidate data, ranked by EV/BP."""
        cursor = self._conn.execute(
            """
            SELECT
                r.*, c.expiration, c.width, c.short_bid, c.short_ask,
                c.short_delta, c.short_iv, c.iv_percentile, c.sell_score, c.buying_power
            FROM sim_results r
            LEFT JOIN candidates c ON r.candidate_id = c.id
            ORDER BY r.ev_per_bp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_run_history(self, limit: int = 50) -> list[dict]:
        """Get recent sim results ordered by created_at DESC."""
        cursor = self._conn.execute(
            """
            SELECT
                r.*, c.expiration, c.width, c.short_delta, c.short_iv,
                c.iv_percentile, c.buying_power
            FROM sim_results r
            LEFT JOIN candidates c ON r.candidate_id = c.id
            ORDER BY r.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_results_by_date_range(self, start: str, end: str) -> list[dict]:
        """Get results within an ISO date range, joined with candidate data."""
        cursor = self._conn.execute(
            """
            SELECT
                r.*, c.expiration, c.width, c.short_delta, c.short_iv,
                c.iv_percentile, c.buying_power
            FROM sim_results r
            LEFT JOIN candidates c ON r.candidate_id = c.id
            WHERE r.created_at >= ? AND r.created_at <= ?
            ORDER BY r.created_at DESC
            """,
            (start, end),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ── Calibration tracking ──────────────────────────────────────────────────

    def save_calibration_record(self, record: CalibrationRecord) -> str:
        """Save a calibration record (predictions at sim time, actuals null)."""
        rid = str(uuid.uuid4())[:8]
        self._conn.execute(
            "INSERT INTO calibration_tracking "
            "(id, candidate_id, symbol, predicted_pop, predicted_touch, predicted_stop, "
            "predicted_ev, actual_profit, actual_touch, actual_stop, actual_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rid,
                record.candidate_id,
                record.symbol,
                record.predicted_pop,
                record.predicted_touch,
                record.predicted_stop,
                record.predicted_ev,
                record.actual_profit,
                record.actual_touch,
                record.actual_stop,
                record.actual_pnl,
            ),
        )
        self._conn.commit()
        return rid

    def update_calibration_outcome(
        self,
        candidate_id: str,
        actual_profit: float,
        actual_touch: float,
        actual_stop: float,
        actual_pnl: float,
    ) -> None:
        """Fill in actual outcomes for a calibration record."""
        self._conn.execute(
            "UPDATE calibration_tracking SET "
            "actual_profit = ?, actual_touch = ?, actual_stop = ?, actual_pnl = ? "
            "WHERE candidate_id = ?",
            (actual_profit, actual_touch, actual_stop, actual_pnl, candidate_id),
        )
        self._conn.commit()

    def get_pending_calibrations(self) -> list[dict]:
        """Get calibration records with no actuals whose candidates have expired.

        JOINs calibration_tracking with candidates to get strike/credit/IV data.
        """
        cursor = self._conn.execute(
            """
            SELECT
                ct.id AS calibration_id,
                ct.candidate_id,
                ct.symbol,
                ct.predicted_pop,
                ct.predicted_touch,
                ct.predicted_stop,
                ct.predicted_ev,
                ct.created_at,
                c.expiration,
                c.short_strike,
                c.long_strike,
                c.width,
                c.net_credit,
                c.short_iv,
                c.long_iv,
                c.underlying_price,
                c.dte
            FROM calibration_tracking ct
            JOIN candidates c ON ct.candidate_id = c.id
            WHERE ct.actual_profit IS NULL
              AND c.expiration < date('now')
            ORDER BY c.expiration
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_chain_snapshots_by_date_range(self, symbol: str, start: str, end: str) -> list[dict]:
        """Get chain snapshots for a symbol within a date range, ordered by snapshot_at + strike."""
        cursor = self._conn.execute(
            """
            SELECT * FROM chain_snapshots
            WHERE symbol = ?
              AND snapshot_at >= ?
              AND snapshot_at <= ?
            ORDER BY snapshot_at, strike
            """,
            (symbol, start, end + "T23:59:59"),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_resolved_calibrations(
        self, symbol: str | None = None, lookback_days: int = 90
    ) -> list[dict]:
        """Get calibration records where actuals have been filled in, joined with candidates."""
        where = "WHERE ct.actual_profit IS NOT NULL"
        params: list = []
        if symbol:
            where += " AND ct.symbol = ?"
            params.append(symbol)
        where += " AND ct.created_at >= datetime('now', ?)"
        params.append(f"-{lookback_days} days")

        cursor = self._conn.execute(
            f"""
            SELECT
                ct.*,
                c.expiration,
                c.short_strike,
                c.long_strike,
                c.width,
                c.net_credit,
                c.short_iv,
                c.long_iv,
                c.underlying_price,
                c.dte
            FROM calibration_tracking ct
            JOIN candidates c ON ct.candidate_id = c.id
            {where}
            ORDER BY ct.created_at
            """,
            params,
        )
        return [dict(row) for row in cursor.fetchall()]

    def compute_brier_scores(self, symbol: str | None = None, lookback_days: int = 90) -> dict:
        """Compute Brier scores for POP, touch, and stop predictions.

        Returns dict with pop_brier, touch_brier, stop_brier, n_samples.
        Scores below 0.10 are excellent, below 0.20 are good.
        Only includes records where actuals have been filled in.
        """
        where = "WHERE actual_profit IS NOT NULL"
        params: list = []
        if symbol:
            where += " AND symbol = ?"
            params.append(symbol)
        where += " AND created_at >= datetime('now', ?)"
        params.append(f"-{lookback_days} days")

        cursor = self._conn.execute(
            f"""
            SELECT
                AVG((predicted_pop - actual_profit)
                    * (predicted_pop - actual_profit)) as pop_brier,
                AVG((predicted_touch - actual_touch)
                    * (predicted_touch - actual_touch)) as touch_brier,
                AVG((predicted_stop - actual_stop) * (predicted_stop - actual_stop)) as stop_brier,
                COUNT(*) as n_samples
            FROM calibration_tracking
            {where}
            """,
            params,
        )
        row = cursor.fetchone()
        if row is None or row["n_samples"] == 0:
            return {"pop_brier": None, "touch_brier": None, "stop_brier": None, "n_samples": 0}
        return dict(row)
