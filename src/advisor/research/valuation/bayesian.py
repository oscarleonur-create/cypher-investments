"""Bayesian fair-value engine.

Instead of three point DCF scenarios, this treats each DCF driver as a **prior
distribution** (mean + uncertainty), lets research signals **update** those
priors (precision-weighted Normal conjugate update), optionally perturbs them
with **ecosystem** toggles, then Monte-Carlos the implied price to produce a
**posterior distribution over fair value**.

The Monte-Carlo projection is a vectorised (numpy) mirror of
`dcf.compute_dcf_scenario` — same FCF fade, Gordon / exit-multiple terminal
value, net-debt bridge and per-share floor — so thousands of draws recompute in
milliseconds (fast enough for live what-if sliders). A unit test pins the
vectorised path to the canonical scalar function at the prior mean so the two
cannot silently diverge.

Public API:
- `build_bayesian_pricing(report)` → baseline `BayesianPriceResult` (stored).
- `recompute_bayesian(report, overrides)` → recomputed result for slider moves.
"""

from __future__ import annotations

import logging

import numpy as np

from advisor.research.models import (
    BayesianOverrides,
    BayesianPriceResult,
    DcfAssumptions,
    DcfResult,
    EcosystemFactor,
    EvidenceSignal,
    HistogramBin,
    PriorDriver,
    ResearchReport,
)
from advisor.research.valuation.dcf import compute_dcf_scenario, dcf_inputs_from_report

logger = logging.getLogger(__name__)

_PROJECTION_YEARS = 10
_DEFAULT_DRAWS = 8000
_MAX_DRAWS = 40000

# Slider / clamp ranges per driver — kept in lock-step with the clamps in
# dcf.py so a sampled assumption never escapes the range the DCF defends.
_DRIVER_BOUNDS: dict[str, tuple[float, float, str, str]] = {
    # key: (min, max, unit, label)
    "revenue_growth_yr1_3": (-0.30, 0.60, "pct", "Revenue growth (yr 1-3)"),
    "revenue_growth_yr4_10": (0.01, 0.30, "pct", "Revenue growth (yr 4-10)"),
    "target_fcf_margin": (0.0, 0.45, "pct", "Terminal FCF margin"),
    "wacc": (0.06, 0.30, "pct", "Discount rate (WACC)"),
    "terminal_growth_rate": (0.0, 0.04, "pct", "Terminal growth"),
    "terminal_exit_multiple": (2.0, 40.0, "x", "Exit multiple (EV/EBITDA)"),
}

# A currently unprofitable company whose `target_fcf_margin` was seeded from its
# (floored) trailing margin gets valued as if it never turns a profit — the DCF
# then collapses to ~zero and the margin/growth sliders do nothing. For names
# with revenue but a sub-threshold margin we instead anchor the *terminal* FCF
# margin prior on a normalized at-scale margin (wide uncertainty), so the
# posterior reflects "if they reach profitability" and the sliders come alive.
_UNPROFITABLE_MARGIN = 0.05
_NORMALIZED_TERMINAL_MARGIN = 0.15
_NORMALIZED_TERMINAL_STD = 0.08

# Solver bounds for the reverse-DCF / implied-growth bisection. A result pinned
# to a bound means the DCF can't bracket the target at any sane growth, so the
# "implied growth" is a saturated artefact and must not be fed back as evidence.
_GROWTH_SOLVER_LO = -0.10
_GROWTH_SOLVER_HI = 0.50

# Minimum prior uncertainty when the bull/bear spread is thin or absent, as a
# fraction of |mean| with an absolute floor — so every slider stays meaningful.
_STD_REL_FLOOR = 0.15
_STD_ABS_FLOOR: dict[str, float] = {
    "revenue_growth_yr1_3": 0.03,
    "revenue_growth_yr4_10": 0.02,
    "target_fcf_margin": 0.02,
    "wacc": 0.01,
    "terminal_growth_rate": 0.004,
    "terminal_exit_multiple": 2.0,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ── Priors ───────────────────────────────────────────────────────────────────


def build_priors(report: ResearchReport) -> list[PriorDriver]:
    """Seed driver priors from the base DCF scenario; uncertainty from bull/bear."""
    dcf = report.dcf
    if dcf is None or dcf.base is None:
        return []

    base = dcf.base.assumptions
    bull = dcf.bull.assumptions if dcf.bull else None
    bear = dcf.bear.assumptions if dcf.bear else None

    # Trailing FCF margin (un-floored) tells us whether the company is currently
    # profitable; if not, we re-anchor the terminal-margin prior (see below).
    base_rev = float(dcf.base_revenue or 0.0)
    seed_margin = (float(dcf.seed_fcf or 0.0) / base_rev) if base_rev > 0 else None
    unprofitable = seed_margin is not None and seed_margin < _UNPROFITABLE_MARGIN

    drivers: list[PriorDriver] = []
    for key, (lo, hi, unit, label) in _DRIVER_BOUNDS.items():
        mean = getattr(base, key, None)
        if mean is None:
            continue  # e.g. terminal_exit_multiple absent → skip that slider
        mean = float(mean)

        # 1σ from the bull/bear spread (covers ~95% of [bear, bull]); fall back
        # to a relative/absolute floor when the scenarios don't spread the knob.
        spread = 0.0
        if bull is not None and bear is not None:
            bv, rv = getattr(bull, key, None), getattr(bear, key, None)
            if bv is not None and rv is not None:
                spread = abs(float(bv) - float(rv)) / 4.0
        std = max(spread, _STD_REL_FLOOR * abs(mean), _STD_ABS_FLOOR.get(key, 0.01))

        # Re-anchor the terminal FCF-margin prior for currently-unprofitable but
        # revenue-generating names onto a normalized at-scale margin, with wide
        # uncertainty — otherwise the DCF assumes perpetual break-even and the
        # valuation collapses to zero regardless of the sliders.
        if key == "target_fcf_margin" and unprofitable:
            mean = _NORMALIZED_TERMINAL_MARGIN
            std = max(std, _NORMALIZED_TERMINAL_STD)

        drivers.append(
            PriorDriver(
                key=key,
                label=label,
                mean=_clamp(mean, lo, hi),
                std=std,
                min=lo,
                max=hi,
                unit=unit,
            )
        )
    return drivers


# ── Evidence ─────────────────────────────────────────────────────────────────


def build_evidence(report: ResearchReport) -> list[EvidenceSignal]:
    """Derive the research signals that update the priors toward a posterior."""
    dcf = report.dcf
    if dcf is None or dcf.base is None:
        return []

    signals: list[EvidenceSignal] = []
    base_wacc = float(dcf.base.assumptions.wacc)

    # Sell-side consensus target → the revenue growth it would imply in our DCF.
    consensus = report.consensus
    target = consensus.target_price_mean if consensus else None
    if target:
        implied = _growth_implied_by_price(dcf, float(target))
        if implied is not None:
            signals.append(
                EvidenceSignal(
                    key="analyst_target",
                    label=f"Analyst target ${target:,.0f}",
                    target_driver="revenue_growth_yr1_3",
                    observed=implied,
                    precision=1.0,
                    weight=0.6,
                    note=f"{consensus.n_analysts} analysts · implies "
                    f"{implied * 100:.1f}% growth",
                )
            )

    # Reverse-DCF: the growth the *market* is currently pricing in (an anchor).
    # Skip it when the solver pinned to a bound — that's a saturated artefact
    # (the DCF couldn't bracket the price), not a real implied-growth estimate.
    if dcf.implied_growth_rate is not None and not _is_saturated(dcf.implied_growth_rate):
        signals.append(
            EvidenceSignal(
                key="reverse_dcf",
                label="Market-implied growth",
                target_driver="revenue_growth_yr1_3",
                observed=float(dcf.implied_growth_rate),
                precision=0.8,
                weight=0.5,
                note=f"reverse-DCF: {dcf.implied_growth_rate * 100:.1f}% priced in",
            )
        )

    # Options ATM IV → market's expected dispersion. Widens the growth prior
    # (uncertainty-only signal: no mean shift).
    of = report.options_flow
    if of is not None and of.atm_iv:
        signals.append(
            EvidenceSignal(
                key="options_iv",
                label=f"Option-implied vol {of.atm_iv * 100:.0f}%",
                target_driver="revenue_growth_yr1_3",
                observed=None,
                precision=min(float(of.atm_iv), 1.0),
                weight=0.5,
                note="higher IV → wider fair-value distribution",
            )
        )

    # Management quality → discount-rate (WACC) nudge: a 0–100 composite mapped
    # to ±1.5pp around the base WACC (better stewardship → lower discount).
    mq = report.management_quality
    if mq is not None:
        implied_wacc = _clamp(base_wacc - (mq.composite_score - 50.0) / 50.0 * 0.015, 0.06, 0.30)
        signals.append(
            EvidenceSignal(
                key="mgmt_quality",
                label=f"Management quality {mq.composite_score:.0f}/100 ({mq.quality_tier})",
                target_driver="wacc",
                observed=implied_wacc,
                precision=0.6,
                weight=0.4,
                note="capital-allocation quality adjusts the discount rate",
            )
        )

    # Red flags → raise the discount rate (risk premium), capped at +3pp.
    rf = report.red_flags
    if rf is not None and rf.flags:
        high = rf.high_severity_count
        bump = min(len(rf.flags) * 0.005 + high * 0.005, 0.03)
        signals.append(
            EvidenceSignal(
                key="red_flags",
                label=f"{len(rf.flags)} red flag(s) · {high} high",
                target_driver="wacc",
                observed=_clamp(base_wacc + bump, 0.06, 0.30),
                precision=0.7,
                weight=0.5,
                note="accounting / quality risks widen the discount",
            )
        )

    # Earnings-call events → near-term growth nudge. The most recently reported
    # quarter's management tone (and the cross-quarter trend) is a forward signal
    # the priors (seeded from trailing fundamentals) don't otherwise see:
    # bullish guidance nudges yr1-3 growth up, bearish nudges it down.
    tr = report.transcripts
    if tr is not None and tr.summaries:
        latest = tr.summaries[0]
        tone_val = _TONE_SCORE.get(str(latest.tone), 0)
        trend = (tr.tone_trend or "").lower()
        if any(w in trend for w in ("improv", "bullish", "accelerat", "strength")):
            tone_val += 1
        elif any(w in trend for w in ("deterior", "bearish", "weaken", "slow")):
            tone_val -= 1
        tone_val = int(_clamp(tone_val, -2, 2))
        if tone_val != 0:
            base_g = float(dcf.base.assumptions.revenue_growth_yr1_3)
            lo, hi = _DRIVER_BOUNDS["revenue_growth_yr1_3"][:2]
            observed = _clamp(base_g + tone_val * 0.03, lo, hi)
            direction = "bullish" if tone_val > 0 else "bearish"
            signals.append(
                EvidenceSignal(
                    key="transcript_tone",
                    label=f"Earnings call {latest.quarter} · {latest.tone}",
                    target_driver="revenue_growth_yr1_3",
                    observed=observed,
                    precision=0.5,
                    weight=0.4,
                    note=f"management guidance {direction}"
                    + (f" · trend: {tr.tone_trend}" if tr.tone_trend else ""),
                )
            )

    return signals


# Earnings-call tone → integer score used to nudge near-term growth evidence.
_TONE_SCORE: dict[str, int] = {"bullish": 1, "neutral": 0, "bearish": -1}


def _is_saturated(g: float, eps: float = 1e-3) -> bool:
    """True if a reverse-DCF growth result is pinned to a solver bound."""
    return abs(g - _GROWTH_SOLVER_LO) < eps or abs(g - _GROWTH_SOLVER_HI) < eps


def _growth_implied_by_price(dcf: DcfResult, target_price: float) -> float | None:
    """Bisection for the constant growth (yr 1-10) that makes the DCF == target_price.

    Mirrors `reverse_dcf.solve_implied_growth` but for an arbitrary target price
    (here, the sell-side consensus target rather than the market price).
    """
    if dcf.base is None or target_price <= 0:
        return None
    a0 = dcf.base.assumptions
    margin = max(a0.target_fcf_margin, 0.001)
    base_revenue = dcf.base.projected_fcf[0] / margin if dcf.base.projected_fcf else None
    seed_fcf = dcf.base.projected_fcf[0] if dcf.base.projected_fcf else None
    if base_revenue is None or seed_fcf is None:
        return None

    def price_at(g: float) -> float:
        a = DcfAssumptions(
            scenario="implied",
            revenue_growth_yr1_3=g,
            revenue_growth_yr4_10=g,
            target_fcf_margin=a0.target_fcf_margin,
            capex_intensity=a0.capex_intensity,
            terminal_growth_rate=a0.terminal_growth_rate,
            terminal_exit_multiple=a0.terminal_exit_multiple,
            wacc=a0.wacc,
        )
        return compute_dcf_scenario(
            a, base_revenue, seed_fcf, dcf.net_debt, dcf.shares_outstanding, target_price
        ).implied_price

    lo, hi = _GROWTH_SOLVER_LO, _GROWTH_SOLVER_HI
    # If the DCF can't bracket the target at either bound the "implied growth" is
    # a saturated artefact — return None so the caller drops the signal rather
    # than feeding a pinned bound (e.g. 0.50) back in as high-weight evidence.
    if price_at(lo) > target_price or price_at(hi) < target_price:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        p = price_at(mid)
        if abs(p - target_price) / max(target_price, 1) < 0.001:
            return mid
        if p < target_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ── Ecosystem factors ────────────────────────────────────────────────────────


def build_ecosystem_factors(report: ResearchReport) -> list[EcosystemFactor]:
    """Surface ecosystem nodes that, when toggled active, perturb a driver.

    These default to *inactive* — they represent "what if this relationship
    turns" scenarios the user can switch on. Effects are deliberately modest and
    bounded by the driver clamps downstream.
    """
    eco = report.ecosystem
    if eco is None:
        return []

    factors: list[EcosystemFactor] = []

    # Customer concentration: a key customer cutting back dents growth + widens it.
    for c in (eco.top_customers or [])[:3]:
        conc = c.concentration_pct
        weight = (conc / 100.0) if conc else 0.10
        factors.append(
            EcosystemFactor(
                key=f"customer:{c.name}",
                label=f"{c.name} pulls back" + (f" ({conc:.0f}% of rev)" if conc else ""),
                kind="customer",
                driver="revenue_growth_yr1_3",
                mean_delta=-(0.02 + 0.04 * min(weight, 1.0)),
                std_delta=0.02,
                note="customer concentration risk",
            )
        )

    # Supplier stress: a key/sole-source supplier squeezes margin.
    for s in (eco.top_suppliers or [])[:2]:
        factors.append(
            EcosystemFactor(
                key=f"supplier:{s.name}",
                label=f"{s.name} input cost shock",
                kind="supplier",
                driver="target_fcf_margin",
                mean_delta=-0.03,
                std_delta=0.01,
                note=s.category or "key supplier",
            )
        )

    # Institutional holder exit: large holder unwinding lifts the discount rate.
    holders = eco.holders.top_holders if eco.holders else []
    for h in (holders or [])[:2]:
        factors.append(
            EcosystemFactor(
                key=f"holder:{h.name}",
                label=f"{h.name} unwinds stake",
                kind="holder",
                driver="wacc",
                mean_delta=0.01,
                std_delta=0.005,
                note="ownership / liquidity risk",
            )
        )

    return factors


# ── Posterior update ─────────────────────────────────────────────────────────


def posterior_update(
    priors: list[PriorDriver], evidence: list[EvidenceSignal]
) -> list[PriorDriver]:
    """Precision-weighted Normal conjugate update of each driver's prior.

    For a driver with prior (μ₀, σ₀) and observations xᵢ (precision τᵢ scaled by
    the signal's user weight), the posterior mean/precision are the standard
    conjugate combination. Uncertainty-only signals (observed is None) leave the
    mean untouched and inflate σ.
    """
    by_driver: dict[str, list[EvidenceSignal]] = {}
    for s in evidence:
        by_driver.setdefault(s.target_driver, []).append(s)

    out: list[PriorDriver] = []
    for d in priors:
        mu0, sigma0 = d.mean, max(d.std, 1e-6)
        tau0 = 1.0 / (sigma0 * sigma0)

        precision = tau0
        mean_num = tau0 * mu0
        var_inflation = 1.0
        for s in by_driver.get(d.key, []):
            w = _clamp(s.weight, 0.0, 1.0)
            if w <= 0:
                continue
            if s.observed is None:
                # Uncertainty-only: widen σ proportional to weight·precision.
                var_inflation *= 1.0 + w * float(s.precision)
                continue
            tau_i = w * float(s.precision) * tau0
            precision += tau_i
            mean_num += tau_i * float(s.observed)

        post_mean = mean_num / precision
        post_sigma = (1.0 / precision) ** 0.5 * var_inflation

        out.append(
            d.model_copy(
                update={
                    "mean": _clamp(post_mean, d.min, d.max),
                    "std": max(post_sigma, 1e-4),
                }
            )
        )
    return out


def apply_ecosystem(
    drivers: list[PriorDriver], factors: list[EcosystemFactor]
) -> list[PriorDriver]:
    """Apply active ecosystem factors' deltas to the matching driver mean/std."""
    deltas: dict[str, tuple[float, float]] = {}
    for f in factors:
        if not f.active:
            continue
        dm, ds = deltas.get(f.driver, (0.0, 0.0))
        deltas[f.driver] = (dm + f.mean_delta, ds + f.std_delta)

    if not deltas:
        return drivers

    out: list[PriorDriver] = []
    for d in drivers:
        dm, ds = deltas.get(d.key, (0.0, 0.0))
        out.append(
            d.model_copy(
                update={
                    "mean": _clamp(d.mean + dm, d.min, d.max),
                    "std": max(d.std + ds, 1e-4),
                }
            )
        )
    return out


# ── Monte-Carlo simulation ───────────────────────────────────────────────────


def _simulate_prices(
    drivers: list[PriorDriver],
    base_revenue: float,
    seed_fcf: float,
    net_debt: float,
    shares: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorised mirror of dcf.compute_dcf_scenario over `n` sampled draws.

    Returns an array of implied per-share prices. Keep this in lock-step with
    `compute_dcf_scenario` (tests/test_bayesian.py pins them at the prior mean).
    """
    by_key = {d.key: d for d in drivers}

    def sample(key: str, default: float) -> np.ndarray:
        d = by_key.get(key)
        if d is None:
            return np.full(n, default)
        draws = rng.normal(d.mean, max(d.std, 1e-9), n)
        return np.clip(draws, d.min, d.max)

    g13 = sample("revenue_growth_yr1_3", 0.05)
    g410 = sample("revenue_growth_yr4_10", 0.03)
    margin = sample("target_fcf_margin", 0.10)
    wacc = sample("wacc", 0.09)
    g_term = sample("terminal_growth_rate", 0.025)
    has_exit = "terminal_exit_multiple" in by_key
    exit_mult = sample("terminal_exit_multiple", 0.0) if has_exit else None

    shares = max(shares, 1.0)

    # Starting FCF margin, clamped exactly as compute_dcf_scenario does.
    if base_revenue:
        start_margin = float(np.clip(seed_fcf / max(base_revenue, 1.0), -0.30, 0.40))
    else:
        start_margin = float(margin.mean())

    revenue = np.full(n, float(base_revenue))
    pv_fcf = np.zeros(n)
    last_fcf = np.zeros(n)
    for year in range(1, _PROJECTION_YEARS + 1):
        g = g13 if year <= 3 else g410
        mp = year / _PROJECTION_YEARS
        fcf_margin = (1 - mp) * start_margin + mp * margin
        revenue = revenue * (1 + g)
        fcf = revenue * fcf_margin
        pv_fcf += fcf / (1 + wacc) ** year
        if year == _PROJECTION_YEARS:
            last_fcf = fcf

    # Gordon terminal value (only meaningful with positive terminal FCF).
    terminal_fcf = last_fcf * (1 + g_term)
    tv_gordon = np.where(terminal_fcf > 0, terminal_fcf / np.maximum(wacc - g_term, 0.001), 0.0)
    pv_gordon = tv_gordon / (1 + wacc) ** _PROJECTION_YEARS

    pv_terminal = pv_gordon
    if exit_mult is not None:
        safe_margin = np.where(margin > 0, margin, np.nan)
        ebitda_proxy = last_fcf / safe_margin * 0.25
        pv_exit = (ebitda_proxy * exit_mult) / (1 + wacc) ** _PROJECTION_YEARS
        use_exit = (last_fcf > 0) & (margin > 0)
        pv_terminal = np.where(use_exit, pv_exit, pv_gordon)

    enterprise_value = pv_fcf + pv_terminal
    equity_value = np.maximum(enterprise_value - net_debt, 0.0)
    implied = equity_value / shares
    return implied[np.isfinite(implied)]


def simulate(
    drivers: list[PriorDriver],
    inputs,  # dcf.DcfInputs
    *,
    n: int = _DEFAULT_DRAWS,
    seed: int | None = 7,
) -> dict:
    """Run the Monte-Carlo and return posterior summary stats + a histogram."""
    n = max(200, min(int(n), _MAX_DRAWS))
    rng = np.random.default_rng(seed)
    prices = _simulate_prices(
        drivers,
        inputs.base_revenue,
        inputs.seed_fcf,
        inputs.net_debt,
        inputs.shares,
        n,
        rng,
    )
    current = float(inputs.current_price)

    if prices.size == 0:
        return {
            "n_draws": n,
            "mean_price": 0.0,
            "median_price": 0.0,
            "p5": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "prob_undervalued": 0.0,
            "expected_upside_pct": 0.0,
            "histogram": [],
        }

    p5, p25, median, p75, p95 = (float(x) for x in np.percentile(prices, [5, 25, 50, 75, 95]))
    prob_under = float(np.mean(prices > current)) if current > 0 else 0.0

    # Histogram over the central mass (p1..p99) so a few tail draws don't squash
    # the chart; counts include the clipped tails in the edge bins.
    lo, hi = np.percentile(prices, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(np.clip(prices, lo, hi), bins=30, range=(float(lo), float(hi)))
    centres = (edges[:-1] + edges[1:]) / 2
    histogram = [
        HistogramBin(x=round(float(cx), 4), count=int(ct)) for cx, ct in zip(centres, counts)
    ]

    return {
        "n_draws": n,
        "mean_price": float(np.mean(prices)),
        "median_price": median,
        "p5": p5,
        "p25": p25,
        "p75": p75,
        "p95": p95,
        "prob_undervalued": prob_under,
        "expected_upside_pct": (median / current - 1.0) if current > 0 else 0.0,
        "histogram": histogram,
    }


# ── Orchestration ────────────────────────────────────────────────────────────


def recompute_bayesian(
    report: ResearchReport, overrides: BayesianOverrides | None = None
) -> BayesianPriceResult:
    """Build priors/evidence/ecosystem, apply user overrides, and simulate."""
    overrides = overrides or BayesianOverrides()
    symbol = report.symbol.upper()

    inputs = dcf_inputs_from_report(report)
    priors = build_priors(report)
    if inputs is None or not priors:
        # No usable DCF → an empty result the UI renders as "no Bayesian model".
        current = float(report.dcf.current_price) if report.dcf else 0.0
        return BayesianPriceResult(
            symbol=symbol,
            current_price=current,
            meaningful=False,
            note="No usable DCF for this name — can't build a fair-value posterior.",
        )

    # Apply driver overrides (mean / uncertainty sliders).
    priors = [
        d.model_copy(
            update={
                "mean": _clamp(overrides.driver_mean.get(d.key, d.mean), d.min, d.max),
                "std": max(overrides.driver_std.get(d.key, d.std), 1e-4),
            }
        )
        for d in priors
    ]

    # Apply evidence weight overrides.
    evidence = [
        s.model_copy(
            update={"weight": _clamp(overrides.evidence_weight.get(s.key, s.weight), 0.0, 1.0)}
        )
        for s in build_evidence(report)
    ]

    # Apply ecosystem toggle overrides.
    ecosystem = [
        f.model_copy(update={"active": overrides.ecosystem_active.get(f.key, f.active)})
        for f in build_ecosystem_factors(report)
    ]

    posterior = apply_ecosystem(posterior_update(priors, evidence), ecosystem)
    stats = simulate(posterior, inputs, n=overrides.n_draws or _DEFAULT_DRAWS)

    meaningful, note = _assess_meaningfulness(inputs, stats["median_price"])

    return BayesianPriceResult(
        symbol=symbol,
        current_price=float(inputs.current_price),
        drivers=priors,
        evidence=evidence,
        ecosystem=ecosystem,
        meaningful=meaningful,
        note=note,
        **stats,
    )


def _assess_meaningfulness(inputs, median: float) -> tuple[bool, str]:
    """Flag posteriors a revenue-growth DCF can't legitimately produce.

    Pre-revenue names and deeply cash-burning ones collapse to ~net-cash or
    zero, so a precise "fair value" there is misleading. We surface an honest
    note instead of fake precision; a fair value far below price (but non-zero)
    stays meaningful but is annotated as carrying a large speculative premium.
    """
    current = float(inputs.current_price)
    if inputs.base_revenue <= 0:
        return False, (
            "Pre-revenue company — a revenue-growth DCF doesn't apply. Fair value "
            "here is essentially net cash per share; the market is pricing "
            "optionality this model can't capture."
        )
    if median <= max(0.02 * current, 0.01):
        return False, (
            "Projected equity value collapses toward zero (cash burn / leverage). "
            "The market is pricing optionality a discounted-cash-flow model can't "
            "represent — treat this as speculative, not a fair-value estimate."
        )
    if current > 0 and median < 0.30 * current:
        return True, (
            "Fair value sits well below the current price — the market is pricing "
            "growth or optionality beyond what this DCF models. Read the gap as the "
            "speculative premium baked into the quote."
        )
    return True, ""


def build_bayesian_pricing(report: ResearchReport) -> BayesianPriceResult:
    """Baseline posterior (default weights/toggles) — persisted during review."""
    return recompute_bayesian(report, BayesianOverrides())
