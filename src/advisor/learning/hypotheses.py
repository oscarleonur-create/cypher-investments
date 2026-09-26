"""Hypotheses: the model suggests where to look; the data, not the model, says what is true.

The last layer of the loop, and the most constrained. A language model is
good at noticing that "the ENTERs that lost were mostly just before results"
and bad at knowing whether that is so. So the model's only job is to propose
**testable** hypotheses in a fixed form — within one group, records whose
feature satisfies a condition do better (or worse) at a horizon — over
features that exist. The code then tests each one on the records, with the
judge's own machinery (own-drift excess, blocks of sessions), and the stored
status is the code's verdict, never the model's sentence.

A hypothesis changes nothing. SUPPORTED means "worth a human looking at a
rule change"; it does not become one by itself. The model sees aggregates
only — the judge's cells and calibration — and never a record it could quote.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import statistics
import uuid
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from advisor.daemon.market_calendar import now_et
from advisor.learning.evaluate import (
    HORIZON_SESSIONS,
    MIN_N,
    MIN_SESSIONS,
    Baselines,
    Record,
    blocks_of,
    cluster_ci,
)

logger = logging.getLogger(__name__)

MAX_HYPOTHESES = 5
# What a hypothesis may condition on: the inputs recorded with every proposal
# (entry.proposal.features_of) and every scanner candidate.
PROPOSAL_FEATURES = {
    "price", "day", "d5", "d20", "sigma", "move_z", "in_zone", "prev_in_zone", "ps",
    "ps_median", "ps_percentile", "zone_distance", "sessions_above", "zone_observations",
    "setups", "events_today", "tier_a_today", "events_week", "held", "weight", "thesis",
    "delivered_growth", "consensus_growth", "required_low", "required_high",
}  # fmt: skip
CANDIDATE_FEATURES = {
    "change", "gap", "rvol", "sigma", "market_cap", "peer_move", "news_checked", "has_catalyst",
}  # fmt: skip
FEATURES = PROPOSAL_FEATURES | CANDIDATE_FEATURES
OPS = ("<", "<=", ">", ">=", "==", "!=")


class Status(StrEnum):
    SUPPORTED = "SUPPORTED"  # the matching records did better (or worse), intervals apart
    REFUTED = "REFUTED"  # the opposite of the claim, intervals apart
    INCONCLUSIVE = "INCONCLUSIVE"  # intervals overlap, or too little data on a side
    INVALID = "INVALID"  # names a feature, group or horizon that does not exist


class Hypothesis(BaseModel):
    """The only form the model may propose in."""

    group: str = Field(description="an existing group, e.g. 'action ENTER' or 'setup C~daily'")
    feature: str = Field(description="one of the allowed feature names")
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    value: float | bool | str
    horizon: Literal["close", "next_close", "d5", "d10", "d20", "d60", "d120"]
    expect: Literal["better", "worse"] = Field(
        description="how records matching the condition do against the rest of the group"
    )
    why: str = Field(description="one sentence: what in the aggregates suggested this")


class Draft(BaseModel):
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=MAX_HYPOTHESES)


SYSTEM_PROMPT = f"""\
You review what a trading system's rules earned and propose where to look next.

You receive aggregates only: per rule group and horizon, the mean return in
excess of each name's own drift, how many records and independent windows it
rests on, and calibration findings. You propose at most {MAX_HYPOTHESES}
hypotheses, each of the form: within GROUP, records whose FEATURE OP VALUE do
BETTER or WORSE at HORIZON than the rest of GROUP.

Rules:
- Use only groups listed in the input and only the allowed feature names.
- Each hypothesis must be testable on those features; no outside knowledge,
  no tickers, no dates, no news.
- Prefer hypotheses that would change a threshold the system already has.
- Say in `why` which aggregate suggested it. Do not state it as a finding:
  it will be tested, and the test decides.
"""


def prompt_version() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]


def summary_for_model(cells: list, calibration: list, groups: dict[str, int]) -> str:
    """Numbers only, rounded, per group: what the model may reason from."""
    lines = ["GROUPS (records): " + ", ".join(f"{g} ({n})" for g, n in sorted(groups.items()))]
    lines.append("ALLOWED FEATURES: " + ", ".join(sorted(FEATURES)))
    lines.append("CELLS (group | origin | horizon | n | independent windows | excess | verdict):")
    for c in cells:
        ex = "n/a" if c.excess is None else f"{c.excess:+.2%}"
        lines.append(
            f"  {c.group} | {c.origin} | {c.horizon} | {c.n} | {c.windows} | {ex} | {c.verdict}"
        )
    for s in calibration:
        lines.append(
            f"CALIBRATION {s['leg']} stop: touched {s['observed']:.0%} vs {s['expected']:.0%} "
            f"implied ({s['n']}): {s['finding']}"
        )
    return "\n".join(lines)


# ── Testing ───────────────────────────────────────────────────────────────


def _matches(value, op: str, target) -> bool | None:
    if value is None:
        return None
    try:
        if op == "==":
            return value == target
        if op == "!=":
            return value != target
        v, t = float(value), float(target)
    except (TypeError, ValueError):
        return None
    return {"<": v < t, "<=": v <= t, ">": v > t, ">=": v >= t}[op]


def validate(h: Hypothesis, groups: set[str]) -> str | None:
    """Why the hypothesis cannot be tested, or None."""
    if h.feature not in FEATURES:
        return f"feature {h.feature!r} is not recorded"
    if h.group not in groups:
        return f"group {h.group!r} has no records"
    if h.op in ("<", "<=", ">", ">=") and isinstance(h.value, str | bool):
        return f"{h.op} needs a number, got {h.value!r}"
    return None


class Finding(BaseModel):
    status: Status
    reason: str
    matching: dict | None = None
    rest: dict | None = None


def examine(h: Hypothesis, records: list[Record], baselines: Baselines) -> Finding:
    """Matching vs the rest of the group, each as own-drift excess with a block interval."""
    groups = {r.group for r in records}
    invalid = validate(h, groups)
    if invalid:
        return Finding(status=Status.INVALID, reason=invalid)
    k = HORIZON_SESSIONS[h.horizon]
    match, rest = [], []
    for r in records:
        if r.group != h.group:
            continue
        ret = r.outcomes.get(h.horizon)
        base = baselines.get(r.symbol, k) if ret is not None else None
        if ret is None or base is None:
            continue
        m = _matches((r.extra.get("features") or {}).get(h.feature), h.op, h.value)
        if m is None:
            continue
        (match if m else rest).append((r.session, ret - base))

    def side(vals):
        windows = len(blocks_of(vals, k))
        ci = cluster_ci(vals, block=k) if windows >= MIN_SESSIONS else None
        return {
            "n": len(vals),
            "windows": windows,
            "excess": statistics.fmean(v for _, v in vals) if vals else None,
            "ci": list(ci) if ci else None,
        }

    a, b = side(match), side(rest)
    if min(a["n"], b["n"]) < MIN_N or a["ci"] is None or b["ci"] is None:
        return Finding(
            status=Status.INCONCLUSIVE, reason="too little data on a side", matching=a, rest=b
        )
    better = a["ci"][0] > b["ci"][1]
    worse = a["ci"][1] < b["ci"][0]
    if (better and h.expect == "better") or (worse and h.expect == "worse"):
        return Finding(status=Status.SUPPORTED, reason="intervals apart", matching=a, rest=b)
    if better or worse:
        return Finding(status=Status.REFUTED, reason="apart, the other way", matching=a, rest=b)
    return Finding(status=Status.INCONCLUSIVE, reason="intervals overlap", matching=a, rest=b)


# ── Storage ───────────────────────────────────────────────────────────────

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS learning_hypotheses (
    id              TEXT NOT NULL PRIMARY KEY,
    created_at      TEXT NOT NULL,              -- ISO, tz-aware ET
    model           TEXT,
    prompt_version  TEXT NOT NULL,
    hypothesis_json TEXT NOT NULL,
    status          TEXT NOT NULL,
    test_json       TEXT NOT NULL
);
"""


class HypothesisStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        self._conn = conn
        conn.executescript(_SCHEMA)

    def add(self, h: Hypothesis, result: Finding, *, model: str | None) -> str:
        hid = uuid.uuid4().hex[:8]
        self._conn.execute(
            "INSERT INTO learning_hypotheses (id, created_at, model, prompt_version, "
            "hypothesis_json, status, test_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                hid,
                now_et().isoformat(),
                model,
                prompt_version(),
                h.model_dump_json(),
                result.status.value,
                result.model_dump_json(),
            ),
        )
        self._conn.commit()
        return hid

    def list(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM learning_hypotheses ORDER BY created_at DESC"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "model": r["model"],
                "prompt_version": r["prompt_version"],
                "hypothesis": json.loads(r["hypothesis_json"]),
                "status": r["status"],
                "test": json.loads(r["test_json"]),
            }
            for r in rows
        ]


# ── A round ───────────────────────────────────────────────────────────────


def _default_model():
    from research_agent.config import ResearchConfig
    from research_agent.llm import OpenRouterLLM

    config = ResearchConfig()
    if not config.openrouter_api_key:
        return None
    llm = OpenRouterLLM(config)
    return (lambda system, user: llm.complete(system, user, response_model=Draft)), config.llm_model


def generate_and_test(
    records: list[Record],
    cells: list,
    calibration: list,
    baselines: Baselines,
    store: HypothesisStore,
    *,
    complete=None,
    model: str | None = None,
) -> list[dict]:
    """Ask for hypotheses on the aggregates, test each on the records, store both."""
    if complete is None:
        configured = _default_model()
        if configured is None:
            raise RuntimeError("no LLM configured (RESEARCH_AGENT_OPENROUTER_API_KEY)")
        complete, model = configured
    groups: dict[str, int] = {}
    for r in records:
        groups[r.group] = groups.get(r.group, 0) + 1
    draft = complete(SYSTEM_PROMPT, summary_for_model(cells, calibration, groups))
    out = []
    for h in draft.hypotheses[:MAX_HYPOTHESES]:
        result = examine(h, records, baselines)
        hid = store.add(h, result, model=model)
        out.append({"id": hid, "hypothesis": h.model_dump(), "result": result.model_dump()})
    return out
