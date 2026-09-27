"""Rule changes: proposed, shadowed, approved, rolled back — never applied by the system itself.

Until now a rule could change only by editing code. That is the missing
actuator of a learning loop: an approved change has to take effect without a
code edit, and a proposed one has to be tried without taking effect.

A change is one threshold, one value, and moves through

    PENDING   proposed (by a sweep or by the user); does nothing
    SHADOW    runs live beside the active rules on the same data, recorded
              apart (``origin=shadow``) and never shown or acted on
    ACTIVE    the rule the daemon runs; activating one retires the previous
              ACTIVE change of the same parameter
    REJECTED  declined; the sweep will not propose the same value again soon
    RETIRED   was ACTIVE or SHADOW, stopped — the rollback
    EXPIRED   its evidence grew old, or a newer sweep no longer supports it:
              the system took it back and the code's value returned

Every transition past PENDING is the user's (CLI today, Telegram later), with
one exception the user chose (2026-09-27): **a change expires on its own**.
"We cannot have old hypotheses making new decisions" — so every change
carries the date of its evidence (``evidence_at``) and lives
``EVIDENCE_TTL_DAYS`` from it. Each monthly sweep renews what it still
supports: a PENDING or SHADOW value it finds again, and an ACTIVE value that
still beats the code's out of sample on the newest history. What it does not
renew expires; an ACTIVE change that fails the re-judgment expires at once.
Going back is the safe direction, so it needs no approval; re-approving is
the user's. An expired ACTIVE change stops applying the moment it is due,
whether or not anything has marked it yet (``in_force``).
Only parameters declared ``threshold`` can be changed, and only those the
code can actually take at run time: the entry rules' ``EntryParams`` and the
session scanner's ``Thresholds``. Risk budgets, the book limit and the
two-year window are ``decided`` and refused here, whatever asks.

A change is stamped like any rule: the effective parameters hash to a new
version, so records made under it are never pooled with the code's defaults.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import fields, replace
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel

from advisor.daemon.market_calendar import now_et, to_et
from advisor.learning.rules import Kind


class Status(StrEnum):
    PENDING = "PENDING"
    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"
    EXPIRED = "EXPIRED"


# A monthly sweep plus a week of slack: one sweep late is tolerated, a missed
# one is not. User decision (2026-09-27): evidence that no sweep renews expires.
EVIDENCE_TTL_DAYS = 35
LIVE = (Status.PENDING, Status.SHADOW, Status.ACTIVE)


ENTRY = "entry"
SESSION = "scanner.session"
ACTUATABLE = (ENTRY, SESSION)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS rule_changes (
    id            TEXT NOT NULL PRIMARY KEY,
    ruleset       TEXT NOT NULL,
    param         TEXT NOT NULL,              -- declared name, e.g. proposal.TRADE_STOP_SIGMAS
    value_json    TEXT NOT NULL,
    previous_json TEXT NOT NULL,              -- the code's value when proposed
    status        TEXT NOT NULL,
    source        TEXT NOT NULL,              -- sweep | user
    evidence_json TEXT NOT NULL DEFAULT '{}',
    note          TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,              -- ISO, tz-aware ET
    decided_at    TEXT,
    evidence_at   TEXT                        -- when its evidence was last current
);
CREATE INDEX IF NOT EXISTS idx_rule_changes_status ON rule_changes(ruleset, status);
"""


class RuleChange(BaseModel):
    id: str
    ruleset: str
    param: str
    value: float | int
    previous: float | int
    status: Status
    source: str
    evidence: dict = {}
    note: str = ""
    created_at: str
    decided_at: str | None = None
    evidence_at: str | None = None

    @property
    def expires_at(self) -> datetime:
        """When it stops applying unless a sweep renews it."""
        # In ET wall time, so 35 days is 35 days across a DST change.
        start = to_et(datetime.fromisoformat(self.evidence_at or self.created_at))
        return start + timedelta(days=EVIDENCE_TTL_DAYS)

    def expired(self, now: datetime) -> bool:
        return now >= self.expires_at


class ChangeError(ValueError):
    """A change that may not be made, with the reason."""


# ── What can change ───────────────────────────────────────────────────────


def _entry_field(param: str) -> str | None:
    from advisor.entry.proposal import PARAM_CONSTANTS

    by_const = {f"proposal.{c}": f for f, c in PARAM_CONSTANTS.items()}
    return by_const.get(param)


def _session_field(param: str) -> str | None:
    from advisor.scanner.detect import Thresholds

    name = param.removeprefix("thresholds.")
    return (
        name
        if param.startswith("thresholds.") and name in {f.name for f in fields(Thresholds)}
        else None
    )


def validate(ruleset: str, param: str, value) -> float | int:
    """The value to store, or ChangeError saying why this change may not be made."""
    if ruleset not in ACTUATABLE:
        raise ChangeError(f"{ruleset} has no run-time parameters yet; change it in code")
    if ruleset == ENTRY:
        from advisor.entry.ruleset import entry_rules

        stamp = entry_rules()
        field_ = _entry_field(param)
    else:
        from advisor.scanner.ruleset import session_rules

        stamp = session_rules()
        field_ = _session_field(param)
    kind = stamp.kinds.get(param)
    if kind is None:
        raise ChangeError(f"{param} is not a declared {ruleset} parameter")
    if kind is not Kind.THRESHOLD:
        raise ChangeError(f"{param} is {kind.value}: only the user changes it, in code")
    if field_ is None:
        raise ChangeError(f"{param} is a threshold the code cannot take at run time yet")
    current = stamp.params[param]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ChangeError(f"{param} needs a number, got {value!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        if float(value) != int(value):
            raise ChangeError(f"{param} is a count of sessions: {value} is not whole")
        value = int(value)
    else:
        value = float(value)
    if value <= 0:
        raise ChangeError(f"{param} must be positive")
    if value == current:
        raise ChangeError(f"{param} is already {current}")
    return value


# ── The table ─────────────────────────────────────────────────────────────


class ChangeStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        self._conn = conn
        conn.executescript(_SCHEMA)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(rule_changes)")}
        if "evidence_at" not in cols:  # a table from before expiry: evidence dates from filing
            conn.execute("ALTER TABLE rule_changes ADD COLUMN evidence_at TEXT")
            conn.execute("UPDATE rule_changes SET evidence_at = created_at")
            conn.commit()

    def propose(
        self,
        ruleset: str,
        param: str,
        value,
        *,
        source: str = "user",
        evidence: dict | None = None,
        note: str = "",
    ) -> RuleChange:
        value = validate(ruleset, param, value)
        previous = _code_value(ruleset, param)
        change = RuleChange(
            id=uuid.uuid4().hex[:8],
            ruleset=ruleset,
            param=param,
            value=value,
            previous=previous,
            status=Status.PENDING,
            source=source,
            evidence=evidence or {},
            note=note,
            created_at=now_et().isoformat(),
        )
        change.evidence_at = change.created_at
        self._conn.execute(
            "INSERT INTO rule_changes (id, ruleset, param, value_json, previous_json, status, "
            "source, evidence_json, note, created_at, evidence_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                change.id,
                ruleset,
                param,
                json.dumps(value),
                json.dumps(previous),
                change.status.value,
                source,
                json.dumps(change.evidence, sort_keys=True),
                note,
                change.created_at,
                change.evidence_at,
            ),
        )
        self._conn.commit()
        return change

    def get(self, change_id: str) -> RuleChange | None:
        row = self._conn.execute("SELECT * FROM rule_changes WHERE id = ?", (change_id,)).fetchone()
        return _row(row) if row else None

    def list(self, *, status: Status | None = None, ruleset: str | None = None) -> list[RuleChange]:
        clauses, args = [], []
        if status is not None:
            clauses.append("status = ?")
            args.append(status.value)
        if ruleset is not None:
            clauses.append("ruleset = ?")
            args.append(ruleset)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM rule_changes {where} ORDER BY created_at", args
        ).fetchall()
        return [_row(r) for r in rows]

    def _set(self, change_id: str, status: Status, note: str | None = None) -> RuleChange:
        change = self.get(change_id)
        if change is None:
            raise ChangeError(f"no change {change_id}")
        self._conn.execute(
            "UPDATE rule_changes SET status = ?, decided_at = ?, note = COALESCE(?, note) "
            "WHERE id = ?",
            (status.value, now_et().isoformat(), note, change_id),
        )
        return change

    def shadow(self, change_id: str) -> RuleChange:
        """PENDING → SHADOW: run beside the live rules, recorded apart, acted on by nobody."""
        change = self._require(change_id, {Status.PENDING})
        self._set(change_id, Status.SHADOW)
        self._conn.commit()
        return self.get(change.id)

    def activate(self, change_id: str) -> RuleChange:
        """PENDING/SHADOW → ACTIVE. The previous ACTIVE change of the parameter is retired.

        A sweep's change is activated on the sweep's evidence, so evidence past
        its date is refused. A change the user proposed by hand is backed by
        the user's decision: activating it dates its evidence now, and from
        then on the sweep must keep supporting it like any other.
        """
        change = self._require(change_id, {Status.PENDING, Status.SHADOW})
        now = now_et()
        if change.expired(now):
            self._expire(change, f"evidence from {change.evidence_at[:10]} expired before approval")
            self._conn.commit()
            raise ChangeError(
                f"{change_id}: its evidence ({change.evidence_at[:10]}) is older than "
                f"{EVIDENCE_TTL_DAYS} days; wait for the next sweep to find it again"
            )
        validate(change.ruleset, change.param, change.value)  # the code may have moved on
        if change.source == "user":
            self._conn.execute(
                "UPDATE rule_changes SET evidence_at = ? WHERE id = ?",
                (now.isoformat(), change_id),
            )
        for other in self.list(status=Status.ACTIVE, ruleset=change.ruleset):
            if other.param == change.param:
                self._set(other.id, Status.RETIRED, f"replaced by {change.id}")
        self._set(change_id, Status.ACTIVE)
        self._conn.commit()
        return self.get(change.id)

    def reject(self, change_id: str, note: str = "") -> RuleChange:
        self._require(change_id, {Status.PENDING, Status.SHADOW})
        self._set(change_id, Status.REJECTED, note or None)
        self._conn.commit()
        return self.get(change_id)

    def retire(self, change_id: str, note: str = "") -> RuleChange:
        """The rollback: an ACTIVE or SHADOW change stops; the code's value returns."""
        self._require(change_id, {Status.ACTIVE, Status.SHADOW})
        self._set(change_id, Status.RETIRED, note or None)
        self._conn.commit()
        return self.get(change_id)

    # ── Expiry ────────────────────────────────────────────────────────────

    def in_force(self, ruleset: str, now: datetime | None = None) -> list[RuleChange]:
        """The ACTIVE changes whose evidence is still current: what the daemon applies."""
        now = now or now_et()
        return [c for c in self.list(status=Status.ACTIVE, ruleset=ruleset) if not c.expired(now)]

    def renew(self, change_id: str, evidence: dict | None = None, note: str = "") -> RuleChange:
        """A sweep supports it again: its evidence is current from now."""
        change = self._require(change_id, set(LIVE))
        self._conn.execute(
            "UPDATE rule_changes SET evidence_at = ?, evidence_json = ?, note = ? WHERE id = ?",
            (
                now_et().isoformat(),
                json.dumps(evidence if evidence is not None else change.evidence, sort_keys=True),
                note or change.note,
                change_id,
            ),
        )
        self._conn.commit()
        return self.get(change_id)

    def expire(self, change_id: str, reason: str) -> RuleChange:
        """The system takes it back: the code's value returns."""
        change = self._require(change_id, set(LIVE))
        self._expire(change, reason)
        self._conn.commit()
        return self.get(change_id)

    def expire_due(self, now: datetime | None = None) -> list[RuleChange]:
        """Every live change past its evidence date, marked EXPIRED. Idempotent."""
        now = now or now_et()
        out = []
        for c in self.list():
            if c.status in LIVE and c.expired(now):
                since = (c.evidence_at or c.created_at)[:10]
                self._expire(c, f"no sweep renewed its evidence since {since}")
                out.append(self.get(c.id))
        self._conn.commit()
        return out

    def _expire(self, change: RuleChange, reason: str) -> None:
        self._set(change.id, Status.EXPIRED, reason)

    def _require(self, change_id: str, allowed: set[Status]) -> RuleChange:
        change = self.get(change_id)
        if change is None:
            raise ChangeError(f"no change {change_id}")
        if change.status not in allowed:
            names = "/".join(sorted(s.value for s in allowed))
            raise ChangeError(f"{change_id} is {change.status.value}; this needs {names}")
        return change


def _row(r) -> RuleChange:
    return RuleChange(
        id=r["id"],
        ruleset=r["ruleset"],
        param=r["param"],
        value=json.loads(r["value_json"]),
        previous=json.loads(r["previous_json"]),
        status=Status(r["status"]),
        source=r["source"],
        evidence=json.loads(r["evidence_json"] or "{}"),
        note=r["note"],
        created_at=r["created_at"],
        decided_at=r["decided_at"],
        evidence_at=r["evidence_at"],
    )


def _code_value(ruleset: str, param: str):
    if ruleset == ENTRY:
        from advisor.entry.ruleset import entry_rules

        return entry_rules().params[param]
    from advisor.scanner.ruleset import session_rules

    return session_rules().params[param]


# ── The rules the daemon runs ─────────────────────────────────────────────


def _apply_entry(base, changes: list[RuleChange]):
    updates = {_entry_field(c.param): c.value for c in changes if _entry_field(c.param)}
    return replace(base, **updates) if updates else base


def _apply_session(base, changes: list[RuleChange]):
    updates = {_session_field(c.param): c.value for c in changes if _session_field(c.param)}
    return replace(base, **updates) if updates else base


def active_entry_params(conn: sqlite3.Connection | None, now: datetime | None = None):
    """The entry thresholds in force: the code's, with every current ACTIVE change applied."""
    from advisor.entry.proposal import current_params

    base = current_params()
    if conn is None:
        return base
    return _apply_entry(base, ChangeStore(conn).in_force(ENTRY, now))


def active_session_thresholds(conn: sqlite3.Connection | None, now: datetime | None = None):
    from advisor.scanner.detect import DEFAULT

    if conn is None:
        return DEFAULT
    return _apply_session(DEFAULT, ChangeStore(conn).in_force(SESSION, now))


def code_values_of_active(store: ChangeStore) -> dict[str, tuple]:
    """{param: (the code's value,)} for each ACTIVE change: what a sweep must also measure."""
    return {c.param: (c.previous,) for c in store.list(status=Status.ACTIVE)}


def expired_event(change: RuleChange):
    """The notice for a change the system took back: tier B, so it reaches the digest."""
    from advisor.daemon.models import Event, EventSource, EventTier

    return Event(
        source=EventSource.DAEMON,
        kind="RULE_CHANGE_EXPIRED",
        tier=EventTier.B,
        payload={
            "change_id": change.id,
            "ruleset": change.ruleset,
            "param": change.param,
            "value": change.value,
            "restored": change.previous,
            "reason": change.note,
            "label": (
                f"rule change {change.id} expired: {change.param} {change.value} → "
                f"the code's {change.previous} ({change.note})"
            ),
        },
        dedup_key=change.id,
    )


def challengers(conn: sqlite3.Connection, ruleset: str) -> list[tuple[str, object]]:
    """(change id, parameters) for each SHADOW change: the active rules plus that one change."""
    store = ChangeStore(conn)
    now = now_et()
    active = store.in_force(ruleset, now)
    out = []
    for c in store.list(status=Status.SHADOW, ruleset=ruleset):
        if c.expired(now):
            continue
        others = [a for a in active if a.param != c.param]
        if ruleset == ENTRY:
            from advisor.entry.proposal import current_params

            out.append((c.id, _apply_entry(current_params(), [*others, c])))
        else:
            from advisor.scanner.detect import DEFAULT

            out.append((c.id, _apply_session(DEFAULT, [*others, c])))
    return out
