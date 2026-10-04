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


# Reasons a common stock is not eligible *today* that say nothing about its past.
_MARKET = {"no bars", "stale bars", "price below floor", "dollar volume below floor",
           "short history"}  # fmt: skip


def _sec_index():
    from advisor.breadth.filings import sec_index

    return sec_index


def _signals(db_path, store, rows, day, now, fetch_company, insider_fetch=None) -> dict:
    """Industries for the eligible, then the day's records and every live outcome due.

    Measurement only (B1): nothing here is shown to the user or reaches the
    digest. A failure is reported in the summary and never fails the sync.
    """
    from advisor.breadth.companies import sec_submission, sync_companies
    from advisor.breadth.insiders import sync_insiders
    from advisor.breadth.measure import record_live
    from advisor.breadth.ruleset import signal_rules

    try:
        eligible = sorted({r["cik"] for r in rows if r["eligible"] and r["cik"]})
        # Industries for every common stock, not only today's eligible: a
        # replay's records fall on names eligible on *their* day, and a record
        # with no industry is matched on size alone (55 of 551 insider records
        # were, in the first run).
        common = sorted(
            {r["cik"] for r in rows if r["cik"] and (r["eligible"] or r["reason"] in _MARKET)}
        )
        companies = sync_companies(store, common, now, fetch=fetch_company or sec_submission)
        insiders = sync_insiders(store, now, set(eligible), **(insider_fetch or {}))
        _register(db_path, signal_rules())
        live = record_live(store, day, now)
        from advisor.breadth.picks import build_picks

        built = build_picks(store, day, now, db_path)
        picks = {k: built.get(k) for k in ("ok", "day", "candidates", "error")}
        picks["symbols"] = [p["symbol"] for p in built.get("picks", [])]
        return {"companies": companies, "insiders": insiders, "live": live, "picks": picks}
    except Exception as exc:  # noqa: BLE001
        logger.exception("breadth signals failed")
        return {"error": str(exc)}


def run_sync(
    db_path,
    now: datetime,
    *,
    bars: bool = True,
    facts: bool = True,
    signals: bool = True,
    limit: int | None = None,
    fetch_dir: Callable[[], Directory] = fetch_directory,
    fetch_bars: bars_mod.Fetch = bars_mod.yahoo_fetch,
    fetch_frame: facts_mod.FetchFrame = facts_mod.sec_frame,
    fetch_concept: facts_mod.FetchConcept = facts_mod.sec_concept,
    fetch_company=None,
    fetch_index=None,
    insider_fetch: dict | None = None,  # {"get_text": ..., "get_bytes": ...} for tests
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
            if bars:
                on_day, before = bars_mod.day_coverage(store.conn, day)
                summary["coverage"] = {"day": day.isoformat(), "names": on_day, "before": before}
                if not bars_mod.complete(on_day, before):
                    # Nothing is built on a session the source has not finished
                    # serving: the universe, the records and the picks would all
                    # read a day without its closes. A later run fills it.
                    summary["error"] = (
                        f"bars for {day} incomplete: {on_day} names against {before} the "
                        f"session before; nothing built on it"
                    )
                    logger.warning("breadth: %s", summary["error"])
                    return summary
            rules = universe_rules()
            _register(db_path, rules)
            rows = snapshot(store, directory, day, rules.version)
            summary["universe"] = {"day": day.isoformat(), "rules": rules.version, **funnel(rows)}

            if facts:
                fr = facts_mod.sync_frames(store, now, fetch=fetch_frame)
                eligible = sorted({r["cik"] for r in rows if r["eligible"] and r["cik"]})
                gaps = facts_mod.fill_gaps(store, eligible, now, fetch=fetch_concept)
                from advisor.breadth.filings import sync_filings

                summary["filings"] = sync_filings(store, now, fetch=fetch_index or _sec_index())
                summary["facts"] = {
                    "frames": fr.as_dict(),
                    "gaps": gaps.as_dict(),
                    "coverage_of_eligible": facts_mod.coverage(store, eligible, now.date()),
                }
                logger.info("breadth facts: %s; %s", fr.summary(), gaps.summary())

            if signals:
                summary["signals"] = _signals(
                    db_path, store, rows, day, now, fetch_company, insider_fetch
                )

            # The sims you chose to follow: exits read from the bars just stored.
            from advisor.breadth import sim

            summary["sims"] = sim.update_open(store.conn)

            summary["ok"] = not rate_limited
            return summary
        finally:
            summary["seconds"] = round(_time.monotonic() - started, 1)
            store.finish_run(run_id, datetime.now(now.tzinfo), summary["ok"], summary)
