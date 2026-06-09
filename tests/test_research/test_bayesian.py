"""Tests for the Bayesian fair-value engine."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest
from advisor.research.models import (
    AnalystConsensus,
    BayesianOverrides,
    DcfAssumptions,
    EvidenceSignal,
    PriorDriver,
    ResearchReport,
)


def _build_report() -> ResearchReport:
    """A report with a realistic DCF (mocked yfinance) + a consensus signal."""
    from advisor.research.valuation.dcf import build_dcf
    from advisor.research.valuation.reverse_dcf import solve_implied_growth

    fake_info = {
        "currentPrice": 50.0,
        "sharesOutstanding": 100e6,
        "totalDebt": 500e6,
        "totalCash": 200e6,
        "beta": 1.2,
        "totalRevenue": 1000e6,
        "freeCashflow": 150e6,
        "revenueGrowth": 0.10,
        "enterpriseToEbitda": 15.0,
    }
    with (
        patch("yfinance.Ticker") as mock_ticker,
        patch("advisor.research.valuation.dcf._risk_free_rate", return_value=0.043),
    ):
        mock_ticker.return_value.info = fake_info
        dcf = build_dcf("TEST", statements=None)
    dcf.implied_growth_rate = solve_implied_growth(dcf)

    return ResearchReport(
        symbol="TEST",
        as_of=date.today(),
        dcf=dcf,
        consensus=AnalystConsensus(symbol="TEST", n_analysts=12, target_price_mean=70.0),
    )


def test_build_priors_seeds_from_base_scenario():
    from advisor.research.valuation.bayesian import build_priors

    priors = build_priors(_build_report())
    keys = {d.key for d in priors}
    assert "revenue_growth_yr1_3" in keys
    assert "wacc" in keys
    for d in priors:
        assert d.std > 0
        assert d.min <= d.mean <= d.max


def test_vectorized_matches_scalar_at_mean():
    """With σ=0 every draw equals the canonical compute_dcf_scenario price."""
    import numpy as np
    from advisor.research.valuation.bayesian import _simulate_prices, build_priors
    from advisor.research.valuation.dcf import compute_dcf_scenario, dcf_inputs_from_report

    report = _build_report()
    inputs = dcf_inputs_from_report(report)
    priors = [d.model_copy(update={"std": 0.0}) for d in build_priors(report)]

    prices = _simulate_prices(
        priors,
        inputs.base_revenue,
        inputs.seed_fcf,
        inputs.net_debt,
        inputs.shares,
        n=64,
        rng=np.random.default_rng(0),
    )

    by_key = {d.key: d.mean for d in priors}
    base = report.dcf.base.assumptions
    assump = DcfAssumptions(
        scenario="base",
        revenue_growth_yr1_3=by_key["revenue_growth_yr1_3"],
        revenue_growth_yr4_10=by_key["revenue_growth_yr4_10"],
        target_fcf_margin=by_key["target_fcf_margin"],
        capex_intensity=base.capex_intensity,
        terminal_growth_rate=by_key["terminal_growth_rate"],
        terminal_exit_multiple=by_key.get("terminal_exit_multiple"),
        wacc=by_key["wacc"],
    )
    scalar = compute_dcf_scenario(
        assump,
        inputs.base_revenue,
        inputs.seed_fcf,
        inputs.net_debt,
        inputs.shares,
        inputs.current_price,
    )
    assert prices.std() == pytest.approx(0.0, abs=1e-6)
    assert float(prices.mean()) == pytest.approx(scalar.implied_price, rel=1e-9)


def test_build_bayesian_pricing_is_sane():
    from advisor.research.valuation.bayesian import build_bayesian_pricing

    res = build_bayesian_pricing(_build_report())
    assert res.symbol == "TEST"
    assert res.current_price == pytest.approx(50.0)
    assert res.p5 <= res.p25 <= res.median_price <= res.p75 <= res.p95
    assert all(map(lambda v: v == v and abs(v) < 1e9, [res.p5, res.p95, res.mean_price]))
    assert 0.0 <= res.prob_undervalued <= 1.0
    assert len(res.histogram) > 0
    assert res.evidence  # analyst target + reverse-DCF signals present


def test_posterior_update_monotonic_in_weight():
    """A signal observed above the prior mean pulls the posterior up with weight."""
    from advisor.research.valuation.bayesian import posterior_update

    prior = PriorDriver(
        key="revenue_growth_yr1_3",
        label="g",
        mean=0.10,
        std=0.05,
        min=-0.30,
        max=0.60,
        unit="pct",
    )

    def post_mean(w: float) -> float:
        sig = EvidenceSignal(
            key="analyst_target",
            label="t",
            target_driver="revenue_growth_yr1_3",
            observed=0.25,
            precision=1.0,
            weight=w,
        )
        return posterior_update([prior], [sig])[0].mean

    m0, m_half, m_full = post_mean(0.0), post_mean(0.5), post_mean(1.0)
    assert m0 == pytest.approx(0.10)  # no evidence → prior unchanged
    assert m0 < m_half < m_full  # more weight → closer to the 0.25 observation


def test_uncertainty_only_signal_widens_without_shifting():
    from advisor.research.valuation.bayesian import posterior_update

    prior = PriorDriver(
        key="revenue_growth_yr1_3",
        label="g",
        mean=0.10,
        std=0.05,
        min=-0.30,
        max=0.60,
        unit="pct",
    )
    iv = EvidenceSignal(
        key="options_iv",
        label="iv",
        target_driver="revenue_growth_yr1_3",
        observed=None,
        precision=0.5,
        weight=1.0,
    )
    out = posterior_update([prior], [iv])[0]
    assert out.mean == pytest.approx(0.10)  # mean untouched
    assert out.std > prior.std  # uncertainty inflated


def test_overrides_change_posterior():
    from advisor.research.valuation.bayesian import recompute_bayesian

    report = _build_report()
    base = recompute_bayesian(report, BayesianOverrides())
    bumped = recompute_bayesian(
        report, BayesianOverrides(driver_mean={"revenue_growth_yr1_3": 0.30})
    )
    assert bumped.median_price > base.median_price
