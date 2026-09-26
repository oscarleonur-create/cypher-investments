"""Replay: the same rules, run day by day over history, with only what was known that day.

Live records accumulate at a few a day; a rule's edge, if it has one, needs
hundreds to show. Replay supplies the volume. It is only worth anything if it
runs **the code the daemon runs** — ``build_proposal``, ``relative_zone``,
``move_from_closes``, ``score`` — not a reimplementation, and if nothing it
reads on day *d* was unknown on day *d*:

- prices are sliced to *d* before any rule sees them;
- SEC revenue and shares are read ``as_of`` *d* (each value known 45 days after
  its quarter, 75 for a derived Q4 — ``valuation.history``);
- results dates are dates that were announced in advance;
- the decision is taken at *d*'s close, at *d*'s close price.

What cannot be replayed is left out and said so on every record: the event
stream and its tier-A blocker, news, the model's reading, holdings and the
user's thesis. A replayed proposal is therefore the entry rules *without*
their blockers — an upper bound on how often they would have fired.

The scanner's setups are replayed at **daily** resolution under their own
group names (``setup A~daily``, ``setup C~daily``) and never pooled with the
live five-minute setups: A is "gapped up at least ``gap_min`` at the open,
entered at the open" (the 09:35 hold, relative volume and news cannot be
known from daily bars); C is "closed down at least ``drop_min`` and
``sigma_min``σ, entered at the close" (no peer test, no news).

Records go to ``replay_records`` under a ``replay_runs`` row, never to the
live ledgers the daemon and the UI read.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from advisor.daemon import market_calendar as mc
from advisor.entry.proposal import Proposal, build_proposal
from advisor.entry.ruleset import entry_rules
from advisor.entry.sheet import Sheet, move_from_closes, next_earnings, sessions_until
from advisor.entry.track import Bar, score
from advisor.entry.zone import relative_zone
from advisor.learning.rules import Origin, code_rev
from advisor.scanner import detect
from advisor.scanner.ruleset import session_rules

logger = logging.getLogger(__name__)

NOMINAL_NET_LIQ = 10_000.0  # sizing does not change a return; legs still need a book
NOT_REPLAYED = (
    "replay: no event stream (tier-A blocker not applied)",
    "replay: no model reading",
    "replay: not held, no thesis",
)
DAILY_A = "setup A~daily"
DAILY_C = "setup C~daily"
DAILY_HORIZONS = {"next_close": 1, "d5": 5, "d10": 10, "d20": 20}


@dataclass(frozen=True)
class DayBar:
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SymbolData:
    symbol: str
    bars: list[DayBar]  # ascending
    series: object | None = None  # valuation.history.Series
    earnings: list[date] = field(default_factory=list)


# ── Scanner at daily resolution ──────────────────────────────────────────


def daily_setups(
    bars: list[DayBar], i: int, sigma: float | None, t: detect.Thresholds = detect.DEFAULT
) -> list[tuple[str, float]]:
    """(group, entry price) for the daily A and C on ``bars[i]``. Uses bars[:i+1] only."""
    if i < 1:
        return []
    today, prev = bars[i], bars[i - 1]
    if prev.close <= 0:
        return []
    out = []
    gap = today.open / prev.close - 1
    if gap >= t.gap_min and today.open >= t.a_min_price:
        out.append((DAILY_A, today.open))
    change = today.close / prev.close - 1
    if (
        change <= -t.drop_min
        and today.close >= t.c_min_price
        and sigma
        and abs(change) / sigma >= t.sigma_min
    ):
        out.append((DAILY_C, today.close))
    return out


def setup_outcomes(bars: list[DayBar], i: int, group: str, entry: float) -> dict[str, float]:
    """Returns from the daily entry: ``close`` for an open entry, then 1/5/10/20 sessions."""
    out: dict[str, float] = {}
    if group == DAILY_A:
        out["close"] = bars[i].close / entry - 1
    for key, n in DAILY_HORIZONS.items():
        if i + n < len(bars):
            out[key] = bars[i + n].close / entry - 1
    return out


def sigma_before(bars: list[DayBar], i: int, sessions: int = 60) -> float | None:
    """Daily-return σ over the ``sessions`` before bar i (today excluded)."""
    closes = [b.close for b in bars[max(0, i - sessions - 1) : i]]
    rets = [b / a - 1 for a, b in zip(closes, closes[1:]) if a > 0]
    return statistics.stdev(rets) if len(rets) >= 20 else None


# ── One symbol ────────────────────────────────────────────────────────────


def replay_symbol(
    data: SymbolData,
    start: date,
    end: date,
    *,
    thresholds: detect.Thresholds = detect.DEFAULT,
) -> tuple[list[Proposal], list[dict]]:
    """Proposals and daily setups for each session in [start, end]. Pure given ``data``."""
    bars = data.bars
    closes = [(b.day, b.close) for b in bars]
    score_bars = [Bar(b.day, b.high, b.low, b.close) for b in bars]
    horizon_end = bars[-1].day if bars else end
    scored_at = datetime.combine(horizon_end, mc.session_close(horizon_end), tzinfo=mc.MARKET_TZ)
    scored_at += timedelta(hours=1)
    proposals: list[Proposal] = []
    setups: list[dict] = []
    for i, b in enumerate(bars):
        d = b.day
        if d < start or d > end:
            continue
        hist = closes[: i + 1]
        assert hist[-1][0] == d  # nothing after d is visible below this line
        sigma = sigma_before(bars, i)
        ids = []
        for group, entry in daily_setups(bars, i, sigma, thresholds):
            sid = f"{d.isoformat()}:{group}:{data.symbol}"
            ids.append(sid)
            setups.append(
                {
                    "id": sid,
                    "group": group,
                    "session": d.isoformat(),
                    "symbol": data.symbol,
                    "entry": entry,
                    "sigma": sigma,
                    "outcomes": setup_outcomes(bars, i, group, entry),
                }
            )
        move = move_from_closes(hist, d)
        zone = relative_zone(hist, data.series, d, b.close) if data.series else None
        zone_prev = None
        if data.series and i >= 1:
            zone_prev = relative_zone(hist[:-1], data.series, hist[-2][0], hist[-2][1])
        upcoming = next_earnings(data.earnings, d)
        sheet = Sheet(
            symbol=data.symbol,
            built_at=datetime.combine(d, mc.session_close(d), tzinfo=mc.MARKET_TZ),
            move=move,
            zone=zone,
            zone_prev=zone_prev,
            candidates=ids,
            next_earnings=upcoming,
            earnings_in=sessions_until(d, upcoming) if upcoming else None,
            gaps=list(NOT_REPLAYED),
        )
        p = build_proposal(sheet, net_liq=NOMINAL_NET_LIQ)
        p.origin = Origin.REPLAY
        p.outcomes = score(p, score_bars, scored_at)
        proposals.append(p)
    return proposals, setups


# ── Data ──────────────────────────────────────────────────────────────────


def load_symbol(symbol: str, start: date, end: date) -> SymbolData | None:
    """Daily OHLCV from ``start`` to ``end``, the SEC series and results dates, or None."""
    try:
        import yfinance as yf

        df = yf.download(
            symbol,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("replay: no prices for %s: %s", symbol, exc)
        return None
    if df is None or df.empty:
        return None
    bars = []
    for ts, row in df.iterrows():
        vals = [float(row[k]) for k in ("Open", "High", "Low", "Close")]
        if all(v == v and v > 0 for v in vals):
            vol = float(row["Volume"]) if row["Volume"] == row["Volume"] else 0.0
            bars.append(DayBar(ts.date(), *vals, vol))
    if len(bars) < 300:
        return None
    series = None
    try:
        from advisor.valuation.history import load_series

        series = load_series(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.info("replay: no SEC series for %s: %s", symbol, exc)
    return SymbolData(symbol=symbol, bars=bars, series=series, earnings=_past_earnings(symbol))


def _past_earnings(symbol: str) -> list[date]:
    """Results dates, past and upcoming. Announced ahead, so known on the day they guard."""
    try:
        import yfinance as yf

        df = yf.Ticker(symbol).get_earnings_dates(limit=24)
    except Exception as exc:  # noqa: BLE001
        logger.info("replay: no results dates for %s: %s", symbol, exc)
        return []
    if df is None or len(df) == 0:
        return []
    return sorted({ts.date() for ts in df.index})


# ── Storage ───────────────────────────────────────────────────────────────

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS replay_runs (
    id          TEXT NOT NULL PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    code        TEXT NOT NULL,
    params_json TEXT NOT NULL,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS replay_records (
    run_id       TEXT NOT NULL,
    kind         TEXT NOT NULL,              -- proposal | setup
    id           TEXT NOT NULL,
    session      TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, kind, id)
);
CREATE INDEX IF NOT EXISTS idx_replay_records_run ON replay_records(run_id, kind);
"""


def compact(p: Proposal) -> dict:
    """What the judge and the report need from a replayed proposal, and nothing else.

    A full proposal is ~4 KB; a broad two-year replay is ~50,000 of them. The
    prose (reasons, notes) is dropped; the action, the inputs, the legs' stop
    distances, the outcomes and the rules version are kept.
    """
    return {
        "id": p.id,
        "session": p.session.isoformat(),
        "symbol": p.symbol,
        "action": p.action.value,
        "ruleset": p.rules.ruleset if p.rules else "entry",
        "version": p.rules.version if p.rules else None,
        "features": p.features,
        "legs": {g.horizon: (1 - g.stop / g.entry) if g.entry else None for g in p.legs},
        "blockers": len(p.blockers),
        "outcomes": p.outcomes,
    }


# CANNOT_SAY has no zone and no setup: nothing to learn, not stored.
KEEP_RUNS = 2  # finished runs whose records are kept
STORED_ACTIONS = frozenset({"ENTER", "ADD", "IN_ZONE", "WAIT", "NONE"})


class ReplayStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        self._conn = conn
        conn.executescript(_SCHEMA)

    def start(self, run_id: str, params: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO replay_runs (id, started_at, code, params_json) "
            "VALUES (?, ?, ?, ?)",
            (run_id, mc.now_et().isoformat(), code_rev(), json.dumps(params, sort_keys=True)),
        )
        self._conn.execute("DELETE FROM replay_records WHERE run_id = ?", (run_id,))
        self._conn.commit()

    def add(self, run_id: str, proposals: list[Proposal], setups: list[dict]) -> None:
        rows = [
            (run_id, "proposal", p.id, p.session.isoformat(), p.symbol, json.dumps(compact(p)))
            for p in proposals
            if p.action.value in STORED_ACTIONS
        ] + [(run_id, "setup", s["id"], s["session"], s["symbol"], json.dumps(s)) for s in setups]
        self._conn.executemany(
            "INSERT OR REPLACE INTO replay_records "
            "(run_id, kind, id, session, symbol, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def finish(self, run_id: str, summary: dict, *, keep: int = KEEP_RUNS) -> None:
        """Mark the run finished and drop the records of all but the ``keep`` latest runs.

        A broad two-year run is ~45 MB of records in the shared database. The
        run rows and their summaries are kept for every run; only the records
        of older runs go (sweeps re-run the replay in memory, not from here).
        """
        self._conn.execute(
            "UPDATE replay_runs SET finished_at = ?, summary_json = ? WHERE id = ?",
            (mc.now_et().isoformat(), json.dumps(summary, sort_keys=True), run_id),
        )
        latest = [
            r["id"]
            for r in self._conn.execute(
                "SELECT id FROM replay_runs WHERE finished_at IS NOT NULL "
                "ORDER BY finished_at DESC LIMIT ?",
                (keep,),
            ).fetchall()
        ]
        marks = ",".join("?" * len(latest))
        self._conn.execute(
            f"DELETE FROM replay_records WHERE run_id NOT IN ({marks}) "
            "AND run_id IN (SELECT id FROM replay_runs WHERE finished_at IS NOT NULL)",
            latest,
        )
        self._conn.commit()

    def runs(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM replay_runs ORDER BY started_at DESC").fetchall()
        return [
            {
                "id": r["id"],
                "started_at": r["started_at"],
                "finished_at": r["finished_at"],
                "code": r["code"],
                "params": json.loads(r["params_json"]),
                "summary": json.loads(r["summary_json"]) if r["summary_json"] else None,
            }
            for r in rows
        ]

    def latest_finished(self) -> str | None:
        row = self._conn.execute(
            "SELECT id FROM replay_runs WHERE finished_at IS NOT NULL "
            "ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None

    def proposals(self, run_id: str) -> list[dict]:
        """Compact proposals (see ``compact``)."""
        rows = self._conn.execute(
            "SELECT payload_json FROM replay_records WHERE run_id = ? AND kind = 'proposal'",
            (run_id,),
        ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def setups(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT payload_json FROM replay_records WHERE run_id = ? AND kind = 'setup'",
            (run_id,),
        ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]


# ── A run ─────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    run_id: str
    symbols: int = 0
    replayed: int = 0
    proposals: int = 0
    setups: int = 0
    no_data: list[str] = field(default_factory=list)
    no_series: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "symbols": self.symbols,
            "replayed": self.replayed,
            "proposals": self.proposals,
            "setups": self.setups,
            "no_data": self.no_data,
            "no_series": self.no_series,
        }


def run(
    store: ReplayStore,
    symbols: list[str],
    start: date,
    end: date,
    *,
    run_id: str | None = None,
    loader: Callable[[str, date, date], SymbolData | None] = load_symbol,
    history_days: int = 1100,
    progress: Callable[[str], None] | None = None,
) -> RunResult:
    """Replay ``symbols`` over [start, end]; ``history_days`` before start feed the zone and σ."""
    if start < mc.COVERED_FROM:
        # Horizons and the results guard count sessions; before the calendar's
        # tables every weekday is a session and every horizon would be shifted.
        raise ValueError(f"replay start {start} precedes the market calendar ({mc.COVERED_FROM})")
    if end < start:
        raise ValueError(f"replay end {end} is before its start {start}")
    run_id = run_id or f"replay-{mc.now_et().strftime('%Y%m%dT%H%M%S')}"
    store.start(
        run_id,
        {
            "symbols": symbols,
            "start": start.isoformat(),
            "end": end.isoformat(),
            # The versions of the rules this run replayed: its records are theirs.
            "entry_version": entry_rules().version,
            "scanner_version": session_rules().version,
        },
    )
    result = RunResult(run_id=run_id, symbols=len(symbols))
    for n, sym in enumerate(symbols, 1):
        data = loader(sym, start - timedelta(days=history_days), mc.now_et().date())
        if data is None:
            result.no_data.append(sym)
            continue
        if data.series is None:
            result.no_series.append(sym)
        proposals, setups = replay_symbol(data, start, end)
        store.add(run_id, proposals, setups)
        result.replayed += 1
        result.proposals += sum(p.action.value in STORED_ACTIONS for p in proposals)
        result.setups += len(setups)
        if progress:
            progress(f"{n}/{len(symbols)} {sym}: {len(proposals)} proposals, {len(setups)} setups")
    store.finish(run_id, result.summary())
    return result
