"""Shadow runs: a challenger decides on exactly what the live rules saw, and nobody acts on it.

The scanner's challengers reuse the champion's fetches — the same movers, the
same σ, the same peers, at the same minute — so the only difference between
their records is the one threshold that changed. News is not looked up for a
challenger (no credits spent on a rule nobody acts on).

The entry module's challengers decide on the same sheet, and the same model
reading, the live proposal was built from.

Each challenger writes to its own file beside the live database (see
``actuator.shadow_path``): record ids carry no version, so two challengers
sharing a ledger would overwrite each other's first-of-the-day rows.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from advisor.entry.store import EntryStore
from advisor.learning.actuator import (
    ENTRY,
    SESSION,
    ChangeStore,
    Status,
    active_entry_params,
    active_session_thresholds,
    challengers,
)
from advisor.learning.rules import Origin
from advisor.scanner.store import ScannerStore

logger = logging.getLogger(__name__)


def challenger_path(db_path: Path, change_id: str) -> Path:
    db_path = Path(db_path)
    return db_path.with_name(f"{db_path.stem}-shadow-{change_id}{db_path.suffix}")


class ShadowScannerStore(ScannerStore):
    def add(self, c) -> bool:
        c.origin = Origin.SHADOW
        return super().add(c)


class ShadowEntryStore(EntryStore):
    def add(self, p) -> bool:
        p.origin = Origin.SHADOW
        return super().add(p)


class Memo:
    """The champion's fetches, kept for its challengers in the same run."""

    def __init__(self, movers, sigma, catalysts, peers) -> None:
        self._movers, self._sigma, self._catalysts, self._peers = movers, sigma, catalysts, peers
        self._movers_result = None
        self._sigmas: dict[str, float] = {}
        self._asked: set[str] = set()
        self._peer_results: dict[str, tuple] = {}

    def movers(self):
        if self._movers_result is None:
            self._movers_result = self._movers()
        return self._movers_result

    def sigma(self, symbols: list[str]) -> dict[str, float]:
        missing = [s for s in symbols if s not in self._asked]
        if missing:
            self._sigmas.update(self._sigma(missing) or {})
            self._asked.update(missing)
        return {s: self._sigmas[s] for s in symbols if s in self._sigmas}

    def catalysts(self, symbol, name, since):
        return self._catalysts(symbol, name, since)

    def peers(self, symbol: str):
        if symbol not in self._peer_results:
            self._peer_results[symbol] = self._peers(symbol)
        return self._peer_results[symbol]


def scan_with_shadows(db_path: Path, now, *, memo: Memo | None = None) -> tuple[object, list[str]]:
    """The live scan under the active thresholds, then each SESSION challenger in shadow."""
    from advisor.scanner import sources
    from advisor.scanner.scan import run_scan

    memo = memo or Memo(
        sources.screen_movers, sources.daily_sigma, sources.find_catalysts, sources.peer_move
    )
    conn = sqlite3.connect(str(db_path))
    try:
        thresholds = active_session_thresholds(conn)
        shadows = challengers(conn, SESSION)
    finally:
        conn.close()
    store = ScannerStore(db_path)
    try:
        result = run_scan(
            store,
            now,
            fetch_movers=memo.movers,
            fetch_sigma=memo.sigma,
            fetch_catalysts=memo.catalysts,
            fetch_peers=memo.peers,
            thresholds=thresholds,
        )
    finally:
        store.close()
    notes = []
    for change_id, t in shadows:
        shadow = ShadowScannerStore(challenger_path(db_path, change_id))
        try:
            r = run_scan(
                shadow,
                now,
                fetch_movers=memo.movers,
                fetch_sigma=memo.sigma,
                fetch_catalysts=memo.catalysts,
                fetch_peers=memo.peers,
                thresholds=t,
                check_news=False,
            )
            notes.append(f"shadow {change_id}: {len(r.new)} new")
        except Exception as exc:  # noqa: BLE001
            logger.warning("shadow scan %s failed: %s", change_id, exc)
            notes.append(f"shadow {change_id}: failed ({exc})")
        finally:
            shadow.close()
    return result, notes


class EntryShadows:
    """The entry challengers for one run: open once, record each sheet, close."""

    def __init__(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        try:
            self.params = active_entry_params(conn)
            self._challengers = challengers(conn, ENTRY)
        finally:
            conn.close()
        self._stores = {
            cid: ShadowEntryStore(challenger_path(db_path, cid)) for cid, _ in self._challengers
        }
        self.recorded = 0

    def __call__(self, sheet, net_liq, reading) -> None:
        from advisor.entry.proposal import build_proposal

        for cid, params in self._challengers:
            p = build_proposal(sheet, net_liq=net_liq, reading=reading, params=params)
            self.recorded += int(self._stores[cid].add(p))

    def close(self) -> None:
        for s in self._stores.values():
            s.close()


def fill_shadow_outcomes(db_path: Path, now) -> str:
    """Outcomes for every challenger that has records, shadowing now or not."""
    from advisor.entry.track import fill_proposal_outcomes
    from advisor.scanner.outcomes import fill_outcomes

    conn = sqlite3.connect(str(db_path))
    try:
        changes = [
            c
            for c in ChangeStore(conn).list()
            if c.status in (Status.SHADOW, Status.ACTIVE, Status.RETIRED)
        ]
    finally:
        conn.close()
    notes = []
    for c in changes:
        path = challenger_path(db_path, c.id)
        if not path.exists():
            continue
        if c.ruleset == SESSION:
            s = ShadowScannerStore(path)
            try:
                notes.append(f"{c.id}: {fill_outcomes(s, now).summary()}")
            finally:
                s.close()
        elif c.ruleset == ENTRY:
            s = ShadowEntryStore(path)
            try:
                notes.append(f"{c.id}: {fill_proposal_outcomes(s, now).summary()}")
            finally:
                s.close()
    return "; ".join(notes) if notes else "no shadow records"


def load_shadow(db_path: Path) -> list:
    """Every challenger's records, for the judge."""
    from advisor.learning.evaluate import from_candidate, from_proposal

    conn = sqlite3.connect(str(db_path))
    try:
        changes = ChangeStore(conn).list()
    finally:
        conn.close()
    records = []
    for c in changes:
        path = challenger_path(db_path, c.id)
        if not path.exists():
            continue
        if c.ruleset == SESSION:
            s = ShadowScannerStore(path)
            try:
                records += [from_candidate(x) for x in s.list(limit=1_000_000)]
            finally:
                s.close()
        else:
            s = ShadowEntryStore(path)
            try:
                records += [from_proposal(x) for x in s.list(limit=1_000_000)]
            finally:
                s.close()
    return records
