"""The judge: what each rule version has earned, said no more confidently than the data allows.

A loop that learns from noise is worse than no loop: it will change a rule
because of three lucky sessions and call it improvement. So every figure here
is built to resist that:

- **Clustered by session.** Twenty candidates on one Tuesday are one market
  day, not twenty observations. Confidence intervals bootstrap *sessions*,
  and every cell reports how many distinct sessions it rests on.
- **Against the name's own drift.** "ENTER averaged +3% in 20 sessions" means
  nothing in a market where the same names averaged +3% on any day. Each
  record is scored as its return minus its symbol's average return over the
  same horizon (the null: being in this name on an ordinary day). This is not
  a benchmark index — the user declined one — it is the floor a rule must
  clear to be doing anything at all.
- **UNDETERMINED by default.** A cell is EDGE only when its interval clears
  zero, NEGATIVE only when it sits below, and neither with fewer than
  ``MIN_N`` records or ``MIN_SESSIONS`` sessions. The report also says how
  many cells were tested, because at 95% one in forty will clear zero by
  chance.
- **Tails beside means.** Maximum adverse excursion is reported with every
  mean: a threshold that raises the average by taking deeper drawdowns is
  not an improvement.

Calibration asks whether the system's own models are right, separately from
whether its rules pay: do stops set at kσ get touched about as often as a
volatility of σ implies, and do the model's stances order the outcomes they
claim to (CONSTRUCTIVE above AT_RISK)?
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from advisor.learning.rules import PRE_REGISTRY

MIN_N = 20
MIN_SESSIONS = 10
BOOTSTRAP = 2000
CONFIDENCE = 0.95
# An interval must clear zero by this much to count: a "certain" 0.01% edge is
# rounding, not a rule doing something, and costs more than it earns to trade.
MIN_EFFECT = 0.001

# Horizon -> sessions after entry. "close" is the entry session itself.
HORIZON_SESSIONS: dict[str, int] = {
    "close": 0,
    "next_close": 1,
    "d5": 5,
    "d10": 10,
    "d20": 20,
    "d60": 60,
    "d120": 120,
}
# The worst excursion that goes with each return horizon, where recorded.
TAIL_FOR = {"close": "mae", "d20": "mae20", "d120": "mae120"}


class Verdict(StrEnum):
    EDGE = "EDGE"  # beats its own names' drift, interval clear of zero
    NEGATIVE = "NEGATIVE"  # does worse than its names' drift, interval clear of zero
    UNDETERMINED = "UNDETERMINED"  # the interval straddles zero, or too little data


@dataclass
class Record:
    """One candidate or proposal, reduced to what the judge needs."""

    ruleset: str
    version: str
    group: str  # "setup C", "setup A@pre", "action ENTER"
    session: date
    symbol: str
    outcomes: dict[str, float | None]
    origin: str = "live"
    extra: dict = field(default_factory=dict)


def from_candidate(c) -> Record:
    pre = getattr(c.phase, "value", c.phase) == "premarket"
    default_rs = "scanner.premarket" if pre else "scanner.session"
    return Record(
        ruleset=c.rules.ruleset if c.rules else default_rs,
        version=c.rules.version if c.rules else PRE_REGISTRY,
        group=f"setup {c.setup.value}{'@pre' if pre else ''}",
        session=c.session,
        symbol=c.symbol.upper(),
        outcomes=dict(c.outcomes),
        origin=getattr(getattr(c, "origin", None), "value", "live"),
    )


def from_proposal(p) -> Record:
    legs = {g.horizon: g for g in p.legs}
    extra = {
        "stance": p.stance,
        "reading_prompt": getattr(p, "reading_prompt", None),
        "sigma": (p.features or {}).get("sigma"),
    }
    for horizon, g in legs.items():
        if g.entry:
            extra[f"{horizon}_stop_pct"] = 1 - g.stop / g.entry
    return Record(
        ruleset=p.rules.ruleset if p.rules else "entry",
        version=p.rules.version if p.rules else PRE_REGISTRY,
        group=f"action {p.action.value}",
        session=p.session,
        symbol=p.symbol.upper(),
        outcomes=dict(p.outcomes),
        origin=getattr(getattr(p, "origin", None), "value", "live"),
        extra=extra,
    )


# ── Baseline: the name's own drift ──────────────────────────────────────


def drift(closes: dict[date, float], sessions: int) -> float | None:
    """Mean ``sessions``-ahead close-to-close return over every day available."""
    if sessions <= 0:
        return 0.0  # within one session drift is noise-sized; taken as zero
    days = sorted(d for d, px in closes.items() if px and px > 0)
    rets = [closes[days[i + sessions]] / closes[days[i]] - 1 for i in range(len(days) - sessions)]
    return statistics.fmean(rets) if len(rets) >= 20 else None


class Baselines:
    """Per (symbol, horizon) drift, closes fetched once per symbol."""

    def __init__(self, closes_fn: Callable[[str], dict[date, float] | None]) -> None:
        self._fn = closes_fn
        self._closes: dict[str, dict[date, float] | None] = {}
        self._drift: dict[tuple[str, int], float | None] = {}

    def get(self, symbol: str, sessions: int) -> float | None:
        key = (symbol, sessions)
        if key not in self._drift:
            if symbol not in self._closes:
                self._closes[symbol] = self._fn(symbol)
            closes = self._closes[symbol]
            self._drift[key] = drift(closes, sessions) if closes else None
        return self._drift[key]


# ── Statistics ───────────────────────────────────────────────────────────


def cluster_ci(
    values: list[tuple[date, float]], *, iters: int = BOOTSTRAP, seed: int = 0
) -> tuple[float, float] | None:
    """Percentile interval of the mean, resampling whole sessions. Deterministic."""
    by_session: dict[date, list[float]] = defaultdict(list)
    for day, v in values:
        by_session[day].append(v)
    sessions = list(by_session)
    if len(sessions) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        pool: list[float] = []
        for _ in sessions:
            pool.extend(by_session[rng.choice(sessions)])
        means.append(statistics.fmean(pool))
    means.sort()
    lo = means[int((1 - CONFIDENCE) / 2 * iters)]
    hi = means[int((1 + CONFIDENCE) / 2 * iters) - 1]
    return lo, hi


def verdict(n: int, sessions: int, ci: tuple[float, float] | None) -> tuple[Verdict, str]:
    if n < MIN_N or sessions < MIN_SESSIONS:
        return Verdict.UNDETERMINED, f"too little data ({n} records, {sessions} sessions)"
    if ci is None:
        return Verdict.UNDETERMINED, "no interval"
    if ci[0] > MIN_EFFECT:
        return Verdict.EDGE, "interval above its names' own drift"
    if ci[1] < -MIN_EFFECT:
        return Verdict.NEGATIVE, "interval below its names' own drift"
    return Verdict.UNDETERMINED, "interval does not clear zero by the minimum effect"


@dataclass
class Cell:
    ruleset: str
    version: str
    group: str
    horizon: str
    origin: str
    n: int
    sessions: int
    mean: float | None
    median: float | None
    hit: float | None
    baseline: float | None  # mean drift of the same names over the horizon
    excess: float | None  # mean of (return - own drift)
    ci: tuple[float, float] | None  # of the excess
    tail: float | None  # mean worst excursion, where recorded
    unbased: int  # records with no baseline (too little price history), left out
    verdict: Verdict
    reason: str

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["verdict"] = self.verdict.value
        d["ci"] = list(self.ci) if self.ci else None
        return d


def evaluate(
    records: list[Record], baselines: Baselines, horizons: tuple[str, ...] | None = None
) -> list[Cell]:
    """One cell per (ruleset, version, group, origin, horizon) with any outcome."""
    groups: dict[tuple[str, str, str, str], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r.ruleset, r.version, r.group, r.origin)].append(r)
    cells = []
    for (ruleset, version, group, origin), members in sorted(groups.items()):
        for horizon, k in HORIZON_SESSIONS.items():
            if horizons and horizon not in horizons:
                continue
            have = [r for r in members if r.outcomes.get(horizon) is not None]
            if not have:
                continue
            rets = [r.outcomes[horizon] for r in have]
            excess, unbased = [], 0
            base_vals = []
            for r in have:
                b = baselines.get(r.symbol, k)
                if b is None:
                    unbased += 1
                    continue
                base_vals.append(b)
                excess.append((r.session, r.outcomes[horizon] - b))
            ex_sessions = len({d for d, _ in excess})
            # Resampling three sessions gives an interval that looks precise and
            # is not: below MIN_SESSIONS no interval is shown at all.
            ci = cluster_ci(excess) if ex_sessions >= MIN_SESSIONS else None
            v, why = verdict(len(excess), ex_sessions, ci)
            tail_key = TAIL_FOR.get(horizon)
            tails = [
                r.outcomes[tail_key]
                for r in have
                if tail_key and r.outcomes.get(tail_key) is not None
            ]
            cells.append(
                Cell(
                    ruleset=ruleset,
                    version=version,
                    group=group,
                    horizon=horizon,
                    origin=origin,
                    n=len(have),
                    sessions=len({r.session for r in have}),
                    mean=statistics.fmean(rets),
                    median=statistics.median(rets),
                    hit=sum(x > 0 for x in rets) / len(rets),
                    baseline=statistics.fmean(base_vals) if base_vals else None,
                    excess=statistics.fmean(v for _, v in excess) if excess else None,
                    ci=ci,
                    tail=statistics.fmean(tails) if tails else None,
                    unbased=unbased,
                    verdict=v,
                    reason=why,
                )
            )
    return cells


# ── Calibration ──────────────────────────────────────────────────────────


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def touch_probability(k_sigmas: float, sessions: float) -> float:
    """P(a driftless walk with daily σ touches -k·σ within ``sessions``): 2Φ(-k/√T)."""
    if sessions <= 0:
        return 0.0
    return min(1.0, 2 * _phi(-k_sigmas / math.sqrt(sessions)))


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# The trade leg runs from entry (intraday) to the next session's close: about
# a session and a half. The position leg's stop is checked over 20 sessions.
STOP_SPANS = {"trade": ("trade_stop", 1.5), "position": ("pos_stop20", 20.0)}


def stop_calibration(records: list[Record]) -> list[dict]:
    """Observed stop-touch rate vs what each record's own σ implies."""
    out = []
    for leg, (key, span) in STOP_SPANS.items():
        observed, expected = [], []
        for r in records:
            sigma = r.extra.get("sigma")
            pct = r.extra.get(f"{leg}_stop_pct")
            hit = r.outcomes.get(key)
            if hit is None or not sigma or not pct or sigma <= 0:
                continue
            observed.append(hit)
            expected.append(touch_probability(pct / sigma, span))
        if not observed:
            continue
        n, hits = len(observed), int(sum(observed))
        exp = statistics.fmean(expected)
        ci = wilson(hits, n)
        if n < MIN_N or ci is None:
            finding = "too little data"
        elif ci[0] > exp:
            finding = (
                "stops touched more often than σ implies: vol under-estimated or stop too tight"
            )
        elif ci[1] < exp:
            finding = "stops touched less often than σ implies: vol over-estimated or stop too wide"
        else:
            finding = "consistent with σ"
        out.append(
            {
                "leg": leg,
                "n": n,
                "observed": hits / n,
                "observed_ci": list(ci) if ci else None,
                "expected": exp,
                "finding": finding,
            }
        )
    return out


STANCE_ORDER = ("CONSTRUCTIVE", "NEUTRAL", "CAUTIOUS", "AT_RISK")


def stance_calibration(records: list[Record], baselines: Baselines, horizon: str = "d20") -> dict:
    """Do stances order outcomes the way they claim? Per prompt version, per stance."""
    k = HORIZON_SESSIONS[horizon]
    by: dict[tuple[str | None, str], list[tuple[date, float]]] = defaultdict(list)
    for r in records:
        stance, ret = r.extra.get("stance"), r.outcomes.get(horizon)
        if stance is None or ret is None:
            continue
        b = baselines.get(r.symbol, k)
        if b is None:
            continue
        by[(r.extra.get("reading_prompt"), stance)].append((r.session, ret - b))
    prompts: dict[str | None, dict] = defaultdict(dict)
    for (prompt, stance), vals in by.items():
        n_sessions = len({d for d, _ in vals})
        ci = cluster_ci(vals) if n_sessions >= MIN_SESSIONS else None
        prompts[prompt][stance] = {
            "n": len(vals),
            "sessions": n_sessions,
            "excess": statistics.fmean(v for _, v in vals),
            "ci": list(ci) if ci else None,
        }
    result = {}
    for prompt, stances in prompts.items():
        top, bottom = stances.get("CONSTRUCTIVE"), stances.get("AT_RISK")
        finding = "UNDETERMINED: needs both CONSTRUCTIVE and AT_RISK with data"
        if top and bottom and top["ci"] and bottom["ci"]:
            if min(top["n"], bottom["n"]) < MIN_N:
                finding = "UNDETERMINED: too little data"
            elif top["ci"][0] > bottom["ci"][1]:
                finding = "ORDERED: CONSTRUCTIVE beat AT_RISK"
            elif top["ci"][1] < bottom["ci"][0]:
                finding = "INVERTED: AT_RISK beat CONSTRUCTIVE"
            else:
                finding = "UNDETERMINED: intervals overlap"
        result[prompt or "unstamped"] = {"horizon": horizon, "stances": stances, "finding": finding}
    return result


# ── Report ───────────────────────────────────────────────────────────────


def chance_edges(cells: list[Cell]) -> float:
    """How many EDGE verdicts chance alone would produce across the cells judged."""
    judged = sum(1 for c in cells if not c.reason.startswith("too little"))
    return judged * (1 - CONFIDENCE) / 2


def load_live(db_path) -> list[Record]:
    """Every live candidate and proposal on file, as records."""
    from advisor.entry.store import EntryStore
    from advisor.scanner.store import ScannerStore

    scanner, entries = ScannerStore(db_path), EntryStore(db_path)
    try:
        return [from_candidate(c) for c in scanner.list(limit=1_000_000)] + [
            from_proposal(p) for p in entries.list(limit=1_000_000)
        ]
    finally:
        scanner.close()
        entries.close()


def yahoo_closes(years: int = 3) -> Callable[[str], dict[date, float] | None]:
    """Daily closes over the last ``years`` years, for the drift baseline."""
    from datetime import timedelta

    from advisor.daemon.market_calendar import now_et
    from advisor.scanner.sources import daily_closes

    end = now_et().date() + timedelta(days=1)
    start = end - timedelta(days=365 * years)
    return lambda symbol: daily_closes(symbol, start, end)
