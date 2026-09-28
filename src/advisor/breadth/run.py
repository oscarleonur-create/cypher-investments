"""One breadth sync: directory → bars → E0 snapshot → SEC facts. Safe to re-run.

Run nightly by the daemon and on demand by ``advisor breadth sync``. Each
stage reports what it did and what it could not; a refused bar pull leaves
the rest of the run to finish and the missing names to the next one.
"""

from __future__ import annotations

import logging
import sqlite3
import time as _time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

from advisor.breadth import bars as bars_mod
from advisor.breadth import facts as facts_mod
from advisor.breadth.listings import Directory, fetch_directory
from advisor.breadth.ruleset import universe_rules
from advisor.breadth.store import BreadthStore, breadth_path
from advisor.breadth.universe import funnel, snapshot

logger = logging.getLogger(__name__)


def _limit(directory: Directory, n: int) -> Directory:
    """The first ``n`` common-stock names and every excluded listing: a quick, honest sample."""
    keep = {x.yahoo.upper() for x in directory.common()[:n]}
    listings = [
        x
        for x in directory.listings
        if x.yahoo.upper() in keep or directory.reasons.get(x.yahoo.upper()) is not None
    ]
    return replace(directory, listings=listings)


def _register(db_path, stamp) -> None:
    from advisor.learning.store import ensure_schema, register

    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        register(conn, stamp)
        conn.commit()
    finally:
        conn.close()


def run_sync(
    db_path,
    now: datetime,
    *,
    bars: bool = True,
    facts: bool = True,
    limit: int | None = None,
    fetch_dir: Callable[[], Directory] = fetch_directory,
    fetch_bars: bars_mod.Fetch = bars_mod.yahoo_fetch,
    fetch_frame: facts_mod.FetchFrame = facts_mod.sec_frame,
    fetch_concept: facts_mod.FetchConcept = facts_mod.sec_concept,
    sleep: Callable[[float], None] = _time.sleep,
) -> dict:
    """Sync everything and return a summary. Never raises for a source failure."""
    started = _time.monotonic()
    summary: dict = {"ok": False}
    with BreadthStore(breadth_path(db_path)) as store:
        run_id = store.start_run(now)
        try:
            try:
                directory = fetch_dir()
            except Exception as exc:  # noqa: BLE001
                summary["error"] = f"symbol directory unavailable: {exc}"
                logger.warning("breadth: %s", summary["error"])
                return summary
            if limit:
                directory = _limit(directory, limit)
            common = [x.yahoo.upper() for x in directory.common()]
            summary["directory"] = {"listings": len(directory.listings), "common": len(common)}

            rate_limited = False
            if bars:
                report = bars_mod.sync_bars(store, common, now, fetch=fetch_bars, sleep=sleep)
                summary["bars"] = report.as_dict()
                rate_limited = report.rate_limited
                logger.info("breadth bars: %s", report.summary())

            day = bars_mod.last_closed_session(now)
            rules = universe_rules()
            _register(db_path, rules)
            rows = snapshot(store, directory, day, rules.version)
            summary["universe"] = {"day": day.isoformat(), "rules": rules.version, **funnel(rows)}

            if facts:
                fr = facts_mod.sync_frames(store, now, fetch=fetch_frame)
                eligible = sorted({r["cik"] for r in rows if r["eligible"] and r["cik"]})
                gaps = facts_mod.fill_gaps(store, eligible, now, fetch=fetch_concept)
                summary["facts"] = {
                    "frames": fr.as_dict(),
                    "gaps": gaps.as_dict(),
                    "coverage_of_eligible": facts_mod.coverage(store, eligible, now.date()),
                }
                logger.info("breadth facts: %s; %s", fr.summary(), gaps.summary())

            summary["ok"] = not rate_limited
            return summary
        finally:
            summary["seconds"] = round(_time.monotonic() - started, 1)
            store.finish_run(run_id, datetime.now(now.tzinfo), summary["ok"], summary)
