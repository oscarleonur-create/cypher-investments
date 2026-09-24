"""Does either setup work? Aggregates only — candidates are never ranked.

Split by whether a catalyst was found, because that is the one distinction
the September review suggested matters, and it is the one a scanner can
observe. "News not checked" is its own group and never folded into "none".
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from advisor.scanner.models import Candidate
from advisor.scanner.outcomes import KEYS

_HORIZONS = ("r30", "r60", "r120", "close", "next_open", "next_close")


def _catalyst_label(c: Candidate) -> str:
    flag = c.has_catalyst
    return "unchecked" if flag is None else ("catalyst" if flag else "no catalyst")


def summarize(candidates: list[Candidate]) -> list[dict]:
    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in candidates:
        groups[(c.setup.value, _catalyst_label(c))].append(c)

    rows = []
    for (setup, label), members in sorted(groups.items()):
        row: dict = {"setup": setup, "catalyst": label, "n": len(members)}
        for key in _HORIZONS:
            values = [c.outcomes[key] for c in members if c.outcomes.get(key) is not None]
            row[key] = {
                "n": len(values),
                "mean": statistics.fmean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "positive": sum(v > 0 for v in values) / len(values) if values else None,
            }
        row["complete"] = sum(all(k in c.outcomes for k in KEYS) for c in members)
        rows.append(row)
    return rows
