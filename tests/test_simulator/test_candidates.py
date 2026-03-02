"""Tests for candidate generation — delta filtering, credit calculation, scoring."""

import pytest
from advisor.simulator.candidates import (
    compute_sell_score,
    generate_pcs_candidates,
    get_adaptive_delta,
    pre_score_candidates,
    score_liquidity,
)
from advisor.simulator.models import SimConfig


@pytest.fixture
def mock_chain():
    """Enriched chain data mimicking get_enriched_chain() output."""
    return [
        {
            "symbol": "TEST",
            "expiration": "2026-04-03",
            "dte": 35,
            "strike": 40.0,
            "bid": 0.10,
            "ask": 0.15,
            "mid": 0.125,
            "delta": -0.05,
            "gamma": 0.005,
            "theta": -0.01,
            "vega": 0.02,
            "iv": 0.35,
            "underlying_price": 50.0,
        },
        {
            "symbol": "TEST",
            "expiration": "2026-04-03",
            "dte": 35,
            "strike": 42.0,
            "bid": 0.20,
            "ask": 0.30,
            "mid": 0.25,
            "delta": -0.10,
            "gamma": 0.01,
            "theta": -0.02,
            "vega": 0.04,
            "iv": 0.34,
            "underlying_price": 50.0,
        },
        {
            "symbol": "TEST",
            "expiration": "2026-04-03",
            "dte": 35,
            "strike": 44.0,
            "bid": 0.40,
            "ask": 0.55,
            "mid": 0.475,
            "delta": -0.18,
            "gamma": 0.02,
            "theta": -0.04,
            "vega": 0.08,
            "iv": 0.33,
            "underlying_price": 50.0,
        },
        {
            "symbol": "TEST",
            "expiration": "2026-04-03",
            "dte": 35,
            "strike": 46.0,
            "bid": 0.80,
            "ask": 1.00,
            "mid": 0.90,
            "delta": -0.28,
            "gamma": 0.03,
            "theta": -0.06,
            "vega": 0.12,
            "iv": 0.32,
            "underlying_price": 50.0,
        },
        {
            "symbol": "TEST",
            "expiration": "2026-04-03",
            "dte": 35,
            "strike": 48.0,
            "bid": 1.50,
            "ask": 1.80,
            "mid": 1.65,
            "delta": -0.40,
            "gamma": 0.04,
            "theta": -0.08,
            "vega": 0.15,
            "iv": 0.31,
            "underlying_price": 50.0,
        },
    ]


@pytest.fixture
def config():
    return SimConfig(min_credit=0.10, min_width=2.0, max_width=10.0, max_buying_power=5000.0)


class TestAdaptiveDelta:
    def test_high_iv(self):
        assert get_adaptive_delta(80) == 0.35

    def test_medium_iv(self):
        assert get_adaptive_delta(50) == 0.28

    def test_low_iv(self):
        assert get_adaptive_delta(10) == 0.16

    def test_boundary_75(self):
        assert get_adaptive_delta(75) == 0.35

    def test_boundary_25(self):
        assert get_adaptive_delta(25) == 0.28


class TestScoreLiquidity:
    def test_tight_spread(self):
        score = score_liquidity(1.00, 1.02)
        assert score > 80

    def test_wide_spread(self):
        score = score_liquidity(1.00, 2.00)
        assert score < 20

    def test_zero_bid(self):
        assert score_liquidity(0, 1.00) == 0.0


class TestGenerateCandidates:
    def test_generates_candidates(self, mock_chain, config):
        candidates = generate_pcs_candidates(mock_chain, "TEST", config, iv_percentile=50.0)
        assert len(candidates) > 0

    def test_net_credit_conservative(self, mock_chain, config):
        """Net credit should use short_bid - long_ask."""
        candidates = generate_pcs_candidates(mock_chain, "TEST", config, iv_percentile=50.0)
        for c in candidates:
            expected = round(c.short_bid - c.long_ask, 2)
            assert c.net_credit == expected, f"Credit mismatch: {c.net_credit} vs {expected}"

    def test_width_constraints(self, mock_chain, config):
        candidates = generate_pcs_candidates(mock_chain, "TEST", config, iv_percentile=50.0)
        for c in candidates:
            assert c.width >= config.min_width
            assert c.width <= config.max_width

    def test_buying_power_constraint(self, mock_chain, config):
        config.max_buying_power = 300.0
        candidates = generate_pcs_candidates(mock_chain, "TEST", config, iv_percentile=50.0)
        for c in candidates:
            assert c.buying_power <= 300.0

    def test_min_credit_filter(self, mock_chain, config):
        config.min_credit = 0.50
        candidates = generate_pcs_candidates(mock_chain, "TEST", config, iv_percentile=50.0)
        for c in candidates:
            assert c.net_credit >= 0.50

    def test_pop_estimate(self, mock_chain, config):
        candidates = generate_pcs_candidates(mock_chain, "TEST", config, iv_percentile=50.0)
        for c in candidates:
            expected_pop = round(1 - abs(c.short_delta), 4)
            assert c.pop_estimate == expected_pop

    def test_empty_chain(self, config):
        candidates = generate_pcs_candidates([], "TEST", config, iv_percentile=50.0)
        assert candidates == []


class TestComputeSellScore:
    def test_score_range(self, mock_chain, config):
        candidates = generate_pcs_candidates(mock_chain, "TEST", config, iv_percentile=60.0)
        for c in candidates:
            score = compute_sell_score(c, 60.0)
            assert 0 <= score <= 100

    def test_high_iv_scores_higher(self, mock_chain, config):
        low_iv = generate_pcs_candidates(mock_chain, "TEST", config, iv_percentile=20.0)
        high_iv = generate_pcs_candidates(mock_chain, "TEST", config, iv_percentile=80.0)
        if low_iv and high_iv:
            low_scores = [compute_sell_score(c, 20.0) for c in low_iv]
            high_scores = [compute_sell_score(c, 80.0) for c in high_iv]
            assert max(high_scores) > max(low_scores)


class TestPreScore:
    def test_sorted_descending(self, mock_chain, config):
        candidates = generate_pcs_candidates(mock_chain, "TEST", config, iv_percentile=50.0)
        scored = pre_score_candidates(candidates, 50.0)
        scores = [c.sell_score for c in scored]
        assert scores == sorted(scores, reverse=True)


class TestNaNFiltering:
    def test_nan_bid_ask_skipped(self, config):
        """Chain records with NaN bid/ask should be excluded."""
        chain = [
            {
                "symbol": "TEST",
                "expiration": "2026-04-03",
                "dte": 35,
                "strike": 44.0,
                "bid": float("nan"),
                "ask": 0.55,
                "mid": 0.475,
                "delta": -0.18,
                "gamma": 0.02,
                "theta": -0.04,
                "vega": 0.08,
                "iv": 0.33,
                "underlying_price": 50.0,
            },
            {
                "symbol": "TEST",
                "expiration": "2026-04-03",
                "dte": 35,
                "strike": 46.0,
                "bid": 0.80,
                "ask": 1.00,
                "mid": 0.90,
                "delta": -0.28,
                "gamma": 0.03,
                "theta": -0.06,
                "vega": 0.12,
                "iv": 0.32,
                "underlying_price": 50.0,
            },
            {
                "symbol": "TEST",
                "expiration": "2026-04-03",
                "dte": 35,
                "strike": 42.0,
                "bid": 0.20,
                "ask": float("nan"),
                "mid": 0.25,
                "delta": -0.10,
                "gamma": 0.01,
                "theta": -0.02,
                "vega": 0.04,
                "iv": 0.34,
                "underlying_price": 50.0,
            },
        ]
        candidates = generate_pcs_candidates(chain, "TEST", config, iv_percentile=50.0)
        # NaN strikes should not appear as short legs
        for c in candidates:
            assert c.short_strike != 44.0, "NaN bid short should be excluded"

    def test_zero_bid_skipped(self, config):
        """Chain records with bid=0 should be excluded as short legs."""
        chain = [
            {
                "symbol": "TEST",
                "expiration": "2026-04-03",
                "dte": 35,
                "strike": 44.0,
                "bid": 0.0,
                "ask": 0.55,
                "mid": 0.275,
                "delta": -0.18,
                "gamma": 0.02,
                "theta": -0.04,
                "vega": 0.08,
                "iv": 0.33,
                "underlying_price": 50.0,
            },
            {
                "symbol": "TEST",
                "expiration": "2026-04-03",
                "dte": 35,
                "strike": 42.0,
                "bid": 0.20,
                "ask": 0.30,
                "mid": 0.25,
                "delta": -0.10,
                "gamma": 0.01,
                "theta": -0.02,
                "vega": 0.04,
                "iv": 0.34,
                "underlying_price": 50.0,
            },
        ]
        candidates = generate_pcs_candidates(chain, "TEST", config, iv_percentile=50.0)
        for c in candidates:
            assert c.short_strike != 44.0, "Zero bid short should be excluded"
