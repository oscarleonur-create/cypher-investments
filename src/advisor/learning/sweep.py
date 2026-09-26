"""Sweeps: which threshold values the replay says would have done better, out of sample.

The loop's generator. For each searchable threshold, a small grid of values
around the current one is replayed over the same history, and a value is
proposed only if it survives walk-forward: chosen on the past, judged on the
next stretch it never saw. It is only ever *proposed* (PENDING); the user
decides, ideally after a spell in shadow.

What makes a proposal earn its place:

- **Out of sample.** History is cut into ``FOLDS`` stretches. For each later
  stretch the best value is picked on everything before it and scored on it.
  Records whose outcome window crosses into the stretch being judged are
  purged from the choosing, so the future never leaks through a 20-session
  return that ends inside it.
- **Most of the time, not once, and not by noise.** The chosen value must
  beat the current one by ``MIN_EFFECT`` in at least ``MIN_WINS`` of the
  judged stretches, over all of them together, and with an interval clear of
  zero: a paired bootstrap over blocks of sessions (the same blocks for both
  values), so a 0.2% edge from noise across 35 variants does not pass.
- **Not a spike.** At least one neighbour on the grid must also beat the
  current value over the whole replay: a lone peak is usually noise fitted.
- **Not bought with the tail.** The worst decile may not be more than
  ``MAX_TAIL_WORSE`` deeper than the current value's.
- **Said as many times as it was tried.** The evidence records every value
  tested, so a proposal from 40 variants is read for what it is.

Each target has its own measure, because thresholds do different jobs: a
trade stop is judged by the trade it would have made (stopped or out at the
next close), a trigger by the 20-session return of the positions it would
have opened, a scanner threshold by the next-session return of the setups it
admits. All are excess over the name's own drift.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

from advisor.learning.evaluate import MIN_EFFECT, drift
from advisor.learning.replay import (
    DAILY_A,
    DAILY_C,
    MIN_HISTORY_BARS,
    SymbolData,
    daily_setups,
    day_sheets,
    decide,
    load_symbol,
)
from advisor.scanner import detect

logger = logging.getLogger(__name__)

FOLDS = 4
MIN_WINS = 2  # of FOLDS - 1 judged stretches
MIN_RECORDS = 30  # per value and stretch, to be judged at all
MAX_TAIL_WORSE = 0.01  # worst decile may be at most one point deeper
REJECTED_MEMORY_DAYS = 90


@dataclass(frozen=True)
class Target:
    name: str
    ruleset: str
    param: str  # declared name, as rule_changes stores it
    field: str  # EntryParams or Thresholds field
    grid: tuple  # candidate values, the current one included
    horizon: int  # sessions the measure spans (for purging)
    measure: str  # what is scored


def _grid(current, *, whole: bool = False) -> tuple:
    if whole:
        return tuple(sorted({max(1, current + d) for d in (-2, -1, 0, 1, 2)}))
    return tuple(sorted({round(current * m, 4) for m in (0.5, 0.75, 1.0, 1.25, 1.5)}))


def targets(base_params=None, base_thresholds=None) -> list[Target]:
    """Each searchable threshold, its grid centred on the value in force (not the code's)."""
    from advisor.entry.proposal import current_params

    p = base_params or current_params()
    t = base_thresholds or detect.DEFAULT
    return [
        Target("trade stop", "entry", "proposal.TRADE_STOP_SIGMAS", "trade_stop_sigmas",
               _grid(p.trade_stop_sigmas), 1, "trade leg: stop, else next close"),
        Target("position stop", "entry", "proposal.POSITION_STOP_SIGMAS", "position_stop_sigmas",
               _grid(p.position_stop_sigmas), 20, "position leg over 20 sessions, stop applied"),
        Target("dip trigger", "entry", "proposal.DIP_SIGMAS", "dip_sigmas",
               _grid(p.dip_sigmas), 20, "20-session return of positions opened"),
        Target("zone confirm", "entry", "proposal.ENTRY_CONFIRM_SESSIONS", "entry_confirm_sessions",
               _grid(p.entry_confirm_sessions, whole=True), 20,
               "20-session return of positions opened"),
        Target("A gap", "scanner.session", "thresholds.gap_min", "gap_min",
               _grid(t.gap_min), 1, "next close of daily A setups"),
        Target("C drop", "scanner.session", "thresholds.drop_min", "drop_min",
               _grid(t.drop_min), 1, "next close of daily C setups"),
        Target("C sigma", "scanner.session", "thresholds.sigma_min", "sigma_min",
               _grid(t.sigma_min), 1, "next close of daily C setups"),
    ]  # fmt: skip


# ── Measuring one value on one symbol ────────────────────────────────────


def _trade_result(bars, i: int, stop: float, entry: float) -> float | None:
    if i + 1 >= len(bars):
        return None
    nxt = bars[i + 1]
    return (stop / entry - 1) if nxt.low <= stop else (nxt.close / entry - 1)


def _position_result(bars, i: int, stop: float, entry: float, n: int = 20) -> float | None:
    if i + n >= len(bars):
        return None
    window = bars[i + 1 : i + n + 1]
    if min(b.low for b in window) <= stop:
        return stop / entry - 1
    return bars[i + n].close / entry - 1


def measure(
    target: Target,
    value,
    data: SymbolData,
    sheets,
    drifts: dict[int, float],
    base_params=None,
    base_thresholds=None,
) -> list:
    """(session, excess) for every record ``value`` produces on this symbol.

    Everything but the one threshold is the rules in force, so a value is
    compared with what the daemon actually runs.
    """
    from advisor.entry.proposal import current_params

    out = []
    bars = data.bars
    if target.ruleset == "scanner.session":
        t = replace(base_thresholds or detect.DEFAULT, **{target.field: value})
        group = DAILY_A if target.field == "gap_min" else DAILY_C
        base = drifts.get(1)
        if base is None:
            return out
        for ds in sheets:
            # The setups alone: a scanner threshold decides nothing a proposal adds.
            for g, entry in daily_setups(bars, ds.i, ds.sigma, t):
                if g != group or ds.i + 1 >= len(bars):
                    continue
                out.append((bars[ds.i].day, bars[ds.i + 1].close / entry - 1 - base))
        return out

    params = replace(base_params or current_params(), **{target.field: value})
    base = drifts.get(target.horizon)
    if base is None:
        return out
    for ds in sheets:
        p, _ = decide(data, ds, params=params)
        legs = {g.horizon: g for g in p.legs}
        if target.field == "trade_stop_sigmas":
            leg = legs.get("trade")
            r = _trade_result(bars, ds.i, leg.stop, leg.entry) if leg else None
        elif target.field == "position_stop_sigmas":
            leg = legs.get("position")
            r = _position_result(bars, ds.i, leg.stop, leg.entry) if leg else None
        else:
            leg = legs.get("position")
            r = None
            if leg and ds.i + 20 < len(bars):
                r = bars[ds.i + 20].close / leg.entry - 1
        if r is not None:
            out.append((bars[ds.i].day, r - base))
    return out


# ── Walk-forward ──────────────────────────────────────────────────────────


@dataclass
class Verdict:
    target: str
    param: str
    current: object
    proposed: object | None
    reason: str
    evidence: dict = field(default_factory=dict)


def _mean(vals):
    return statistics.fmean(vals) if vals else None


def paired_delta_ci(
    a: list[tuple[date, float]], b: list[tuple[date, float]], block: int, iters: int = 2000
) -> tuple[float, float] | None:
    """Interval of mean(a) - mean(b), resampling the same blocks of sessions for both."""
    import random

    days = sorted({d for d, _ in a} | {d for d, _ in b})
    block = max(1, block)
    groups = [set(days[i : i + block]) for i in range(0, len(days), block)]
    by_a = [[x for d, x in a if d in g] for g in groups]
    by_b = [[x for d, x in b if d in g] for g in groups]
    keep = [i for i in range(len(groups)) if by_a[i] or by_b[i]]
    if len(keep) < 2:
        return None
    rng = random.Random(0)
    deltas = []
    for _ in range(iters):
        pick = [rng.choice(keep) for _ in keep]
        xa = [x for i in pick for x in by_a[i]]
        xb = [x for i in pick for x in by_b[i]]
        if xa and xb:
            deltas.append(statistics.fmean(xa) - statistics.fmean(xb))
    if len(deltas) < iters // 2:
        return None
    deltas.sort()
    return deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas)) - 1]


def _p10(vals):
    if len(vals) < 10:
        return None
    return sorted(vals)[len(vals) // 10]


def walk_forward(target: Target, current, by_value: dict, sessions: list[date]) -> Verdict:
    """Choose on the past, judge on the next stretch; propose only what keeps winning."""
    if not sessions:
        return Verdict(target.name, target.param, current, None, "no sessions")
    days = sorted(set(sessions))
    size = max(1, len(days) // FOLDS)
    bounds = [days[min(k * size, len(days) - 1)] for k in range(FOLDS)] + [days[-1] + timedelta(1)]
    index = {d: n for n, d in enumerate(days)}

    def in_fold(k, dated: bool = False):
        lo, hi = bounds[k], bounds[k + 1]
        if dated:
            return {v: [(d, x) for d, x in recs if lo <= d < hi] for v, recs in by_value.items()}
        return {v: [x for d, x in recs if lo <= d < hi] for v, recs in by_value.items()}

    def before(k):
        # Purged: a record whose outcome window reaches into fold k is not used to choose.
        cut = index.get(bounds[k], len(days)) - target.horizon
        return {v: [x for d, x in recs if index[d] < cut] for v, recs in by_value.items()}

    folds = []
    chosen = []
    for k in range(1, FOLDS):
        past, test = before(k), in_fold(k)
        eligible = {v: _mean(x) for v, x in past.items() if len(x) >= MIN_RECORDS}
        if not eligible or current not in eligible:
            folds.append({"fold": k, "skipped": "too few records to choose on"})
            continue
        best = max(eligible, key=lambda v: eligible[v])
        cur_test, best_test = test.get(current, []), test.get(best, [])
        if len(cur_test) < MIN_RECORDS or len(best_test) < MIN_RECORDS:
            folds.append({"fold": k, "chosen": best, "skipped": "too few records to judge"})
            continue
        delta = _mean(best_test) - _mean(cur_test)
        chosen.append(best)
        folds.append(
            {
                "fold": k,
                "chosen": best,
                "in_sample": eligible,
                "oos_chosen": _mean(best_test),
                "oos_current": _mean(cur_test),
                "delta": delta,
                "n": (len(best_test), len(cur_test)),
            }
        )
    evidence = {
        "measure": target.measure,
        "grid": list(target.grid),
        "variants_tested": len(target.grid),
        "folds": folds,
    }
    judged = [f for f in folds if "delta" in f]
    if len(judged) < MIN_WINS:
        return Verdict(
            target.name, target.param, current, None, "too few judged stretches", evidence
        )
    mode, times = Counter(chosen).most_common(1)[0]
    if mode == current:
        return Verdict(
            target.name, target.param, current, None, "the current value kept winning", evidence
        )
    wins = sum(1 for f in judged if f["chosen"] == mode and f["delta"] > MIN_EFFECT)
    pooled_mode = [x for k in range(1, FOLDS) for x in in_fold(k).get(mode, [])]
    pooled_cur = [x for k in range(1, FOLDS) for x in in_fold(k).get(current, [])]
    pooled = _mean(pooled_mode) - _mean(pooled_cur) if pooled_mode and pooled_cur else None
    evidence.update({"mode": mode, "wins": wins, "pooled_oos_delta": pooled})
    if wins < MIN_WINS:
        return Verdict(
            target.name,
            target.param,
            current,
            None,
            f"won {wins} stretches, needs {MIN_WINS}",
            evidence,
        )
    if pooled is None or pooled <= MIN_EFFECT:
        return Verdict(
            target.name,
            target.param,
            current,
            None,
            "not better over all judged stretches",
            evidence,
        )
    dated_mode = [p for k in range(1, FOLDS) for p in in_fold(k, dated=True).get(mode, [])]
    dated_cur = [p for k in range(1, FOLDS) for p in in_fold(k, dated=True).get(current, [])]
    ci = paired_delta_ci(dated_mode, dated_cur, target.horizon)
    evidence["pooled_oos_ci"] = list(ci) if ci else None
    if ci is None or ci[0] <= 0:
        return Verdict(
            target.name,
            target.param,
            current,
            None,
            "the out-of-sample gain is within noise (interval includes zero)",
            evidence,
        )
    tail_mode, tail_cur = _p10(pooled_mode), _p10(pooled_cur)
    evidence.update({"p10_proposed": tail_mode, "p10_current": tail_cur})
    if tail_mode is not None and tail_cur is not None and tail_mode < tail_cur - MAX_TAIL_WORSE:
        return Verdict(
            target.name, target.param, current, None, "better on average, deeper tail", evidence
        )
    grid = list(target.grid)
    pos = grid.index(mode)
    neighbours = [grid[j] for j in (pos - 1, pos + 1) if 0 <= j < len(grid) and grid[j] != current]
    all_past = {v: _mean([x for _, x in recs]) for v, recs in by_value.items() if recs}
    cur_all = all_past.get(current)
    plateau = any(
        all_past.get(n) is not None and cur_all is not None and all_past[n] > cur_all
        for n in neighbours
    )
    evidence["plateau"] = plateau
    if neighbours and not plateau:
        return Verdict(
            target.name, target.param, current, None, "a lone peak: no neighbour agrees", evidence
        )
    return Verdict(target.name, target.param, current, mode, "survived walk-forward", evidence)


# ── A sweep ───────────────────────────────────────────────────────────────


@dataclass
class SweepResult:
    verdicts: list[Verdict] = field(default_factory=list)
    symbols: int = 0
    used: int = 0
    proposed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def sweep(
    symbols: list[str],
    start: date,
    end: date,
    *,
    loader: Callable = load_symbol,
    history_days: int = 1100,
    only: set[str] | None = None,
    progress: Callable[[str], None] | None = None,
    base_params=None,
    base_thresholds=None,
) -> SweepResult:
    """Measure every target's grid over the symbols, then walk each forward.

    ``base_params``/``base_thresholds``: the rules in force (approved changes
    applied); default the code's. The comparison is always against them.
    """
    chosen = [t for t in targets(base_params, base_thresholds) if not only or t.name in only]
    records: dict[str, dict] = {t.name: defaultdict(list) for t in chosen}
    sessions: dict[str, list] = {t.name: [] for t in chosen}
    result = SweepResult(symbols=len(symbols))
    today = end
    for n, sym in enumerate(symbols, 1):
        data = loader(sym, start - timedelta(days=history_days), today)
        if data is None or len(data.bars) < MIN_HISTORY_BARS:
            continue
        result.used += 1
        closes = {b.day: b.close for b in data.bars}
        drifts = {h: drift(closes, h) for h in {1, 20}}
        sheets = day_sheets(data, start, end)
        for t in chosen:
            for v in t.grid:
                recs = measure(t, v, data, sheets, drifts, base_params, base_thresholds)
                records[t.name][v].extend(recs)
                sessions[t.name].extend(d for d, _ in recs)
        if progress:
            progress(f"{n}/{len(symbols)} {sym}")
    for t in chosen:
        current = _current(t, base_params, base_thresholds)
        result.verdicts.append(walk_forward(t, current, records[t.name], sessions[t.name]))
    return result


def _current(t: Target, base_params=None, base_thresholds=None):
    from advisor.entry.proposal import current_params

    if t.ruleset == "entry":
        return getattr(base_params or current_params(), t.field)
    return getattr(base_thresholds or detect.DEFAULT, t.field)


def file_proposals(result: SweepResult, store) -> SweepResult:
    """PENDING changes for the verdicts that survived, unless on file or lately rejected."""
    from advisor.daemon.market_calendar import now_et
    from advisor.learning.actuator import ChangeError, Status

    existing = store.list()
    now = now_et()
    for v in result.verdicts:
        if v.proposed is None:
            continue
        same = [c for c in existing if c.param == v.param and c.value == v.proposed]
        if any(c.status in (Status.PENDING, Status.SHADOW, Status.ACTIVE) for c in same):
            result.skipped.append(f"{v.param}={v.proposed}: already on file")
            continue
        recent_no = [
            c
            for c in same
            if c.status is Status.REJECTED
            and c.decided_at
            and (now - _parse(c.decided_at)).days < REJECTED_MEMORY_DAYS
        ]
        if recent_no:
            result.skipped.append(
                f"{v.param}={v.proposed}: rejected {recent_no[-1].decided_at[:10]}"
            )
            continue
        try:
            c = store.propose(
                "entry" if v.param.startswith("proposal.") else "scanner.session",
                v.param,
                v.proposed,
                source="sweep",
                evidence=v.evidence,
                note=f"{v.target}: {v.reason}",
            )
            result.proposed.append(c.id)
        except ChangeError as exc:
            result.skipped.append(f"{v.param}={v.proposed}: {exc}")
    return result


def _parse(iso: str):
    from datetime import datetime

    return datetime.fromisoformat(iso)
