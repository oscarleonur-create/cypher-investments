"""The proposals' track record, grouped by what the system said and what you did.

Aggregates only; no proposal is ranked. The groups answer the questions the
ledger exists for: does ENTER beat IN_ZONE? Do the WAITs dodge anything? When
you skip an ENTER, what does it cost?
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from advisor.entry.proposal import Proposal
from advisor.scanner.journal import status as decision_status

HORIZONS = (
    "next_close",
    "d5",
    "d10",
    "d20",
    "d60",
    "d120",
    "mae20",
    "mae120",
    "trade_stop",
    "pos_stop20",
)


def review(proposals: list[Proposal], decisions: dict) -> list[dict]:
    groups: dict[tuple[str, str], list[Proposal]] = defaultdict(list)
    for p in proposals:
        groups[(p.action.value, decision_status(p, decisions.get(p.id)))].append(p)
    rows = []
    for (action, st), members in sorted(groups.items()):
        row: dict = {"action": action, "status": st, "n": len(members)}
        for key in HORIZONS:
            vals = [p.outcomes[key] for p in members if p.outcomes.get(key) is not None]
            row[key] = {
                "n": len(vals),
                "mean": statistics.fmean(vals) if vals else None,
                "positive": sum(v > 0 for v in vals) / len(vals) if vals else None,
            }
        rows.append(row)
    return rows
