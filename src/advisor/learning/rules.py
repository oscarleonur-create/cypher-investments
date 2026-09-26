"""Which rules produced a record, so an outcome can be pinned on them.

Every candidate and every proposal is judged later by what the price did. The
judgement is only worth anything if it can be pinned on the rules that
produced the record — and the rules change: the entry module's risk constants
changed three times on 2026-09-25 alone. A review that mixes proposals made
under different rules learns nothing about any of them.

A stamp has two parts, because a rule is both numbers and logic:

- ``version`` hashes the parameters: every threshold, limit and method
  constant the rule reads. Change one and the version changes.
- ``code`` is the git revision that ran. Numbers do not capture logic: "a
  sector move is not a news dip" changed which dips qualify without touching
  a single constant.

Each parameter has a kind, and the kind says who may change it:

    threshold   what qualifies, and where a stop sits: the learning loop may
                propose a new value, the user approves it
    decided     a limit the user set — risk budgets, the book limit, the
                two-year window. Never searched; only the user changes it
    model       how a quantity is measured — a volatility lookback, the
                intraday volume curve. Changed by reviewing the method

The kind is not hashed: re-labelling a constant changes nothing the rule does.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import subprocess
from datetime import date, datetime, time
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PRE_REGISTRY = "pre-registry"  # how a record written before stamps existed is grouped


class Kind(StrEnum):
    THRESHOLD = "threshold"
    DECIDED = "decided"
    MODEL = "model"


class Origin(StrEnum):
    """Where a record came from. Live and replayed records are never pooled blindly."""

    LIVE = "live"
    REPLAY = "replay"
    SHADOW = "shadow"  # a challenger run beside the live rules; acted on by nobody


class RuleStamp(BaseModel):
    ruleset: str  # e.g. "scanner.session", "entry"
    version: str  # hash of the ruleset name and its parameters
    code: str = "unknown"  # git revision, "+dirty" when src had uncommitted changes
    # In memory only. The record carries the version; the parameters are
    # written once, to ``rule_versions``, not repeated on every row.
    params: dict[str, Any] = Field(default_factory=dict, exclude=True)
    kinds: dict[str, Kind] = Field(default_factory=dict, exclude=True)


def _plain(value: Any) -> Any:
    """A parameter as JSON can hold it, the same way every time."""
    if isinstance(value, time | date | datetime):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    if isinstance(value, frozenset | set):
        return sorted(_plain(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def version_of(ruleset: str, params: dict[str, Any]) -> str:
    """Twelve hex characters of a hash over the ruleset's name and parameters.

    Canonical JSON (sorted keys, no whitespace), so the same parameters in any
    order give the same version. ``5`` and ``5.0`` differ on purpose: a
    constant retyped is a constant edited, and editing is what gets versioned.
    """
    body = json.dumps(
        {"ruleset": ruleset, "params": _plain(params)}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def stamp(ruleset: str, declared: dict[str, tuple[Any, Kind]]) -> RuleStamp:
    """A stamp for ``ruleset`` from ``{name: (value, kind)}``."""
    params = {name: _plain(value) for name, (value, _) in declared.items()}
    return RuleStamp(
        ruleset=ruleset,
        version=version_of(ruleset, params),
        code=code_rev(),
        params=params,
        kinds={name: kind for name, (_, kind) in declared.items()},
    )


_REPO = Path(__file__).resolve().parents[3]


@functools.cache
def code_rev() -> str:
    """The git revision this process runs, once per process. "unknown" if there is none.

    "+dirty" when anything under ``src/`` differs from that revision, untracked
    files included: an uncommitted edit is not the commit it sits on.
    """
    if not (_REPO / "src" / "advisor").is_dir():
        return "unknown"
    try:
        sha = subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(_REPO), "status", "--porcelain", "--", "src"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("rules: no git revision: %s", exc)
        return "unknown"
    if not sha:
        return "unknown"
    return f"{sha}+dirty" if dirty else sha


# Read now, when the rule modules that import this are themselves loaded, not
# at the first stamp: a checkout that moves under a running daemon would
# otherwise be reported as the code that daemon runs.
code_rev()
