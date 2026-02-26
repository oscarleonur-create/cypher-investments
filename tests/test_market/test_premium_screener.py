"""Tests for premium screener — POP, liquidity, sell score, adaptive delta."""

from __future__ import annotations

from datetime import date, timedelta

from advisor.market.premium_screener import (
    LiquidityScore,
    PremiumOpportunity,
    compute_sell_score,
    get_adaptive_delta,
    score_liquidity,
)

# ── Adaptive Delta ────────────────────────────────────────────────────────────


def test_adaptive_delta_high_iv():
    assert get_adaptive_delta(80) == 0.35
    assert get_adaptive_delta(75) == 0.35
    assert get_adaptive_delta(99) == 0.35


def test_adaptive_delta_normal_iv():
    assert get_adaptive_delta(50) == 0.28
    assert get_adaptive_delta(25) == 0.28
    assert get_adaptive_delta(74) == 0.28


def test_adaptive_delta_low_iv():
    assert get_adaptive_delta(10) == 0.16
    assert get_adaptive_delta(0) == 0.16
    assert get_adaptive_delta(24) == 0.16


# ── Liquidity Score ───────────────────────────────────────────────────────────


def test_liquidity_score_perfect():
    """High volume, high OI, tight spread, good vol/OI = ~100."""
    liq = score_liquidity(volume=1000, oi=5000, bid=1.00, ask=1.05)
    assert liq.total >= 80
    assert liq.volume_pts > 0
    assert liq.oi_pts > 0
    assert liq.spread_pts > 0
    assert liq.vol_oi_pts > 0


def test_liquidity_score_zero():
    """No volume, no OI = 0."""
    liq = score_liquidity(volume=0, oi=0, bid=0, ask=0)
    assert liq.total == 0


def test_liquidity_score_wide_spread():
    """Wide spread should reduce score."""
    tight = score_liquidity(volume=100, oi=500, bid=1.00, ask=1.05)
    wide = score_liquidity(volume=100, oi=500, bid=1.00, ask=2.00)
    assert tight.spread_pts > wide.spread_pts


def test_liquidity_score_components():
    """Verify individual component ranges."""
    liq = score_liquidity(volume=250, oi=1000, bid=0.50, ask=0.60)
    assert 0 <= liq.volume_pts <= 30
    assert 0 <= liq.oi_pts <= 30
    assert 0 <= liq.spread_pts <= 25
    assert 0 <= liq.vol_oi_pts <= 15
    assert liq.total == liq.volume_pts + liq.oi_pts + liq.spread_pts + liq.vol_oi_pts


def test_liquidity_capped_at_100():
    """Score should never exceed 100."""
    liq = score_liquidity(volume=10000, oi=50000, bid=5.00, ask=5.05)
    assert liq.total <= 100


# ── Sell Score ────────────────────────────────────────────────────────────────


def _make_opp(**kwargs) -> PremiumOpportunity:
    """Helper to create a PremiumOpportunity with sensible defaults."""
    defaults = {
        "symbol": "TEST",
        "strategy": "naked_put",
        "strike": 90.0,
        "expiry": date.today() + timedelta(days=35),
        "dte": 35,
        "credit": 0.85,
        "bid": 0.80,
        "ask": 0.90,
        "delta": 0.25,
        "iv": 0.45,
        "pop": 0.75,
        "iv_percentile": 65.0,
        "annualized_yield": 1.20,
        "expected_move": 8.0,
        "strike_vs_em": 1.2,
        "liquidity": LiquidityScore(
            total=70, volume_pts=20, oi_pts=20, spread_pts=20, vol_oi_pts=10
        ),
        "term_structure": "contango",
        "earnings_days": None,
        "margin_req": 500.0,
        "max_loss": 500.0,
    }
    defaults.update(kwargs)
    return PremiumOpportunity(**defaults)


def test_sell_score_range():
    """Score should be between 0 and 100."""
    opp = _make_opp()
    score = compute_sell_score(opp)
    assert 0 <= score <= 100


def test_sell_score_high_iv_percentile_boosts():
    """Higher IV percentile should increase score."""
    low_iv = _make_opp(iv_percentile=20)
    high_iv = _make_opp(iv_percentile=90)
    assert compute_sell_score(high_iv) > compute_sell_score(low_iv)


def test_sell_score_pop_boost():
    """Higher POP should increase score."""
    low_pop = _make_opp(pop=0.62)
    high_pop = _make_opp(pop=0.88)
    assert compute_sell_score(high_pop) > compute_sell_score(low_pop)


def test_sell_score_contango_vs_backwardation():
    """Contango should score higher than backwardation."""
    contango = _make_opp(term_structure="contango")
    backwardation = _make_opp(term_structure="backwardation")
    assert compute_sell_score(contango) > compute_sell_score(backwardation)


def test_sell_score_earnings_safety():
    """No earnings nearby should score higher than earnings within window."""
    safe = _make_opp(earnings_days=None)
    risky = _make_opp(earnings_days=3)
    assert compute_sell_score(safe) > compute_sell_score(risky)


def test_sell_score_liquidity_impact():
    """Better liquidity should increase score."""
    low_liq = _make_opp(
        liquidity=LiquidityScore(total=20, volume_pts=5, oi_pts=5, spread_pts=5, vol_oi_pts=5)
    )
    high_liq = _make_opp(
        liquidity=LiquidityScore(total=90, volume_pts=25, oi_pts=25, spread_pts=25, vol_oi_pts=15)
    )
    assert compute_sell_score(high_liq) > compute_sell_score(low_liq)


def test_sell_score_strike_vs_em():
    """Strikes further from expected move should score higher."""
    close = _make_opp(strike_vs_em=0.3)
    far = _make_opp(strike_vs_em=1.8)
    assert compute_sell_score(far) > compute_sell_score(close)


def test_sell_score_ideal_opportunity():
    """A 'perfect' opportunity should score close to 100."""
    opp = _make_opp(
        iv_percentile=95,
        pop=0.92,
        annualized_yield=1.50,
        liquidity=LiquidityScore(total=95, volume_pts=28, oi_pts=28, spread_pts=24, vol_oi_pts=15),
        term_structure="contango",
        strike_vs_em=2.0,
        earnings_days=None,
    )
    score = compute_sell_score(opp)
    assert score >= 85


def test_sell_score_poor_opportunity():
    """A poor opportunity should score low."""
    opp = _make_opp(
        iv_percentile=10,
        pop=0.55,
        annualized_yield=0.10,
        liquidity=LiquidityScore(total=10, volume_pts=2, oi_pts=3, spread_pts=3, vol_oi_pts=2),
        term_structure="backwardation",
        strike_vs_em=0.2,
        earnings_days=2,
    )
    score = compute_sell_score(opp)
    assert score < 25
