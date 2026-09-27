"""Is the system up to date? Code, jobs, rules and inputs, each with its age.

User decision (2026-09-27): old hypotheses must not make new decisions. The
guards that enforce it live where the decisions are made (``entry.freshness``,
``learning.actuator`` expiry); this module only reports, so the frontend can
show in one place what is current and what is not:

- **code**: the revision the daemon runs (its heartbeat carries it), the one
  this API process runs, and main as of the last fetch — how far behind each is.
- **jobs**: each scheduled job's last success against when its trigger says it
  should last have run (``Trigger.last_due``): ok, late, failing or never.
- **rules**: the entry rules version the latest proposals were made under
  against the code's current one; every live rule change with its expiry.
- **inputs**: the stale inputs the latest proposals named.

Read-only; the one subprocess is a local ``git`` read, no fetch.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from advisor.daemon import market_calendar as mc

# A rule change this close to expiry is flagged: the next sweep must renew it.
EXPIRY_WARN_DAYS = 7
_REPO = Path(__file__).resolve().parents[3]


class JobState(BaseModel):
    name: str
    description: str = ""
    schedule: str = ""
    state: str  # ok | late | failing | never | idle
    last_ok_at: str | None = None
    last_run_at: str | None = None
    due_since: str | None = None  # the latest moment it should have succeeded by
    last_error: str = ""


class CodeState(BaseModel):
    main: str | None = None  # origin/main as of the last fetch
    daemon: str | None = None
    daemon_behind: int | None = None  # commits the daemon's revision is behind main
    api: str | None = None
    api_behind: int | None = None
    dirty: list[str] = Field(default_factory=list)  # which runs uncommitted code


class RuleState(BaseModel):
    code_version: str | None = None  # the entry rules the code declares now
    proposals_version: str | None = None  # what the latest proposals were made under
    proposals_session: str | None = None
    changes: list[dict] = Field(default_factory=list)  # live rule changes, with expiry
    expired_recently: list[dict] = Field(default_factory=list)


class Status(BaseModel):
    now: str
    ok: bool  # nothing late, failing, behind or expired-in-force
    problems: list[str] = Field(default_factory=list)
    code: CodeState
    jobs: list[JobState]
    rules: RuleState
    stale_inputs: dict[str, list[str]] = Field(default_factory=dict)  # input -> symbols


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def behind(rev: str | None, main: str | None) -> int | None:
    """Commits in ``main`` that ``rev`` lacks. None when either is unknown."""
    if not rev or not main or rev == "unknown":
        return None
    sha = rev.removesuffix("+dirty")
    n = _git("rev-list", "--count", f"{sha}..{main}")
    return int(n) if n is not None and n.isdigit() else None


def code_state(daemon_rev: str | None) -> CodeState:
    from advisor.learning.rules import code_rev

    main = _git("rev-parse", "--short=12", "origin/main")
    api = code_rev()
    st = CodeState(main=main, daemon=daemon_rev or None, api=api)
    st.daemon_behind = behind(daemon_rev, main)
    st.api_behind = behind(api, main)
    runs = (("daemon", daemon_rev), ("api", api))
    st.dirty = [who for who, rev in runs if rev and "+dirty" in rev]
    return st


def job_state(job, hb, now: datetime) -> JobState:
    """One job's heartbeat against its schedule."""
    due = job.trigger.last_due(now)
    st = JobState(
        name=job.name,
        description=job.description,
        schedule=job.trigger.describe(),
        state="ok",
        last_ok_at=hb.last_ok_at.isoformat() if hb.last_ok_at else None,
        last_run_at=hb.last_run_at.isoformat() if hb.last_run_at else None,
        due_since=due.isoformat() if due else None,
        last_error=hb.last_error or "",
    )
    ok = mc.to_et(hb.last_ok_at) if hb.last_ok_at else None
    ran = mc.to_et(hb.last_run_at) if hb.last_run_at else None
    if ran is not None and (ok is None or ran > ok):
        st.state = "failing"
    elif ok is None:
        st.state = "never" if due is not None else "idle"
    elif due is not None and ok < due:
        st.state = "late"
    return st


def rule_state(conn, latest_rules: str | None, latest_session: str | None, now) -> RuleState:
    from advisor.entry.ruleset import entry_rules
    from advisor.learning.actuator import LIVE, ChangeStore, Status

    st = RuleState(
        code_version=entry_rules().version,
        proposals_version=latest_rules,
        proposals_session=latest_session,
    )
    store = ChangeStore(conn)
    for c in store.list():
        row = {
            "id": c.id,
            "ruleset": c.ruleset,
            "param": c.param,
            "value": c.value,
            "previous": c.previous,
            "status": c.status.value,
            "source": c.source,
            "note": c.note,
            "evidence_at": c.evidence_at,
            "expires_at": c.expires_at.isoformat(),
            "days_left": (c.expires_at - now).total_seconds() / 86400,
            "in_force": c.status is Status.ACTIVE and not c.expired(now),
        }
        if c.status in LIVE:
            st.changes.append(row)
        elif c.status is Status.EXPIRED and c.decided_at:
            if now - mc.to_et(datetime.fromisoformat(c.decided_at)) <= timedelta(days=30):
                st.expired_recently.append(row)
    return st


def system_status(db_path, now: datetime | None = None) -> Status:
    import sqlite3

    from advisor.daemon.models import EventSource
    from advisor.daemon.store import DaemonStore
    from advisor.daemon.supervisor import build_registry
    from advisor.entry.store import EntryStore

    now = mc.to_et(now or mc.now_et())
    daemon = DaemonStore(db_path)
    entries = EntryStore(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        jobs = [job_state(j, daemon.get_heartbeat(j.name), now) for j in build_registry()]
        daemon_rev = daemon.get_watermark(EventSource.DAEMON).last_seen_cursor
        recent = entries.list(since=now.date() - timedelta(days=10))
        latest_session = max((p.session for p in recent), default=None)
        latest = [p for p in recent if p.session == latest_session]
        newest = max(latest, key=lambda p: p.built_at) if latest else None
        rules = rule_state(
            conn,
            newest.rules.version if newest is not None and newest.rules else None,
            latest_session.isoformat() if latest_session else None,
            now,
        )
        stale: dict[str, set[str]] = {}
        newest_by_symbol: dict[str, object] = {}
        for p in latest:
            cur = newest_by_symbol.get(p.symbol)
            if cur is None or p.built_at > cur.built_at:
                newest_by_symbol[p.symbol] = p
        for p in newest_by_symbol.values():
            names = (p.features or {}).get("stale")
            for name in names.split(",") if isinstance(names, str) and names else []:
                stale.setdefault(name, set()).add(p.symbol)
    finally:
        daemon.close()
        entries.close()
        conn.close()

    code = code_state(daemon_rev)
    problems = []
    if code.daemon_behind:
        problems.append(
            f"the daemon runs {code.daemon}, {code.daemon_behind} commit(s) behind main "
            "(run-daemon.sh update)"
        )
    if code.api_behind:
        problems.append(f"this API runs {code.api}, {code.api_behind} commit(s) behind main")
    problems += [f"{who} runs uncommitted code" for who in code.dirty]
    for j in jobs:
        if j.state == "failing":
            problems.append(f"{j.name} is failing: {j.last_error[:120]}")
        elif j.state == "late":
            problems.append(f"{j.name} is late: last success {j.last_ok_at or 'never'}")
        elif j.state == "never":
            problems.append(f"{j.name} has never succeeded")
    if (
        rules.proposals_version
        and rules.code_version
        and rules.proposals_version != rules.code_version
        and not any(c["in_force"] for c in rules.changes)
    ):
        problems.append(
            f"the latest proposals ({rules.proposals_session}) were made under rules "
            f"{rules.proposals_version}; the code now declares {rules.code_version}"
        )
    for c in rules.changes:
        if c["status"] == "ACTIVE" and not c["in_force"]:
            problems.append(f"rule change {c['id']} is past its expiry and no longer applies")
        elif c["days_left"] < EXPIRY_WARN_DAYS:
            problems.append(
                f"rule change {c['id']} ({c['param']}) expires in {c['days_left']:.0f} day(s) "
                "unless a sweep renews it"
            )
    for name, symbols in sorted(stale.items()):
        problems.append(f"{name} stale on {len(symbols)} name(s) in the latest proposals")
    return Status(
        now=now.isoformat(),
        ok=not problems,
        problems=problems,
        code=code,
        jobs=jobs,
        rules=rules,
        stale_inputs={k: sorted(v) for k, v in stale.items()},
    )
