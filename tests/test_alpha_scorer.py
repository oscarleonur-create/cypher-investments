"""Unit tests for the alpha scorer."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from advisor.confluence.alpha_scorer import (
    _normalize_dip,
    _normalize_fundamental,
    _normalize_mispricing,
    _normalize_ml,
    _normalize_pead,
    _normalize_sentiment,
    _normalize_smart_money,
    _normalize_technical,
    classify_signal,
    compute_alpha,
)
from advisor.confluence.models import AlphaSignal

# ── Normalizer tests ────────────────────────────────────────────────────


class TestNormalizeTechnical:
    def test_bullish_no_volume_bonus(self):
        r = SimpleNamespace(is_bullish=True, volume_ratio=1.0, price=100, sma_20=95)
        assert _normalize_technical(r) == 70.0

    def test_bullish_with_volume_bonus(self):
        r = SimpleNamespace(is_bullish=True, volume_ratio=2.0, price=100, sma_20=95)
        assert _normalize_technical(r) == 100.0

    def test_bullish_partial_volume_bonus(self):
        r = SimpleNamespace(is_bullish=True, volume_ratio=1.5, price=100, sma_20=95)
        assert _normalize_technical(r) == 85.0

    def test_no_breakout_at_sma(self):
        r = SimpleNamespace(is_bullish=False, volume_ratio=0.8, price=100, sma_20=100)
        assert _normalize_technical(r) == pytest.approx(55.0)

    def test_no_breakout_far_below(self):
        r = SimpleNamespace(is_bullish=False, volume_ratio=0.8, price=85, sma_20=100)
        assert _normalize_technical(r) == 0.0

    def test_no_breakout_zero_sma(self):
        r = SimpleNamespace(is_bullish=False, volume_ratio=0.5, price=50, sma_20=0)
        assert _normalize_technical(r) == 0.0


class TestNormalizeSentiment:
    def test_passthrough(self):
        r = SimpleNamespace(score=72.5)
        assert _normalize_sentiment(r) == 72.5

    def test_clamp_high(self):
        r = SimpleNamespace(score=120)
        assert _normalize_sentiment(r) == 100.0

    def test_clamp_low(self):
        r = SimpleNamespace(score=-5)
        assert _normalize_sentiment(r) == 0.0


class TestNormalizeFundamental:
    def test_clear_and_insider(self):
        r = SimpleNamespace(is_clear=True, insider_buying_detected=True)
        assert _normalize_fundamental(r) == 100.0

    def test_clear_only(self):
        r = SimpleNamespace(is_clear=True, insider_buying_detected=False)
        assert _normalize_fundamental(r) == 60.0

    def test_insider_only(self):
        r = SimpleNamespace(is_clear=False, insider_buying_detected=True)
        assert _normalize_fundamental(r) == 40.0

    def test_neither(self):
        r = SimpleNamespace(is_clear=False, insider_buying_detected=False)
        assert _normalize_fundamental(r) == 0.0


class TestNormalizeSmartMoney:
    def test_max_score(self):
        r = SimpleNamespace(total_score=100)
        assert _normalize_smart_money(r) == 100.0

    def test_min_score(self):
        r = SimpleNamespace(total_score=-35)
        assert _normalize_smart_money(r) == 0.0

    def test_mid_score(self):
        r = SimpleNamespace(total_score=32.5)
        assert round(_normalize_smart_money(r), 1) == 50.0

    def test_clamp_below(self):
        r = SimpleNamespace(total_score=-50)
        assert _normalize_smart_money(r) == 0.0


class TestNormalizeMispricing:
    def test_passthrough(self):
        r = SimpleNamespace(total_score=65)
        assert _normalize_mispricing(r) == 65.0

    def test_clamp(self):
        r = SimpleNamespace(total_score=110)
        assert _normalize_mispricing(r) == 100.0


class TestNormalizeML:
    def test_high_prob(self):
        r = SimpleNamespace(win_probability=0.85)
        assert _normalize_ml(r) == 85.0

    def test_zero(self):
        r = SimpleNamespace(win_probability=0.0)
        assert _normalize_ml(r) == 0.0


class TestNormalizeDip:
    @pytest.mark.parametrize(
        "score,expected",
        [
            ("FAIL", 0),
            ("WEAK", 20),
            ("WATCH", 40),
            ("LEAN_BUY", 60),
            ("BUY", 80),
            ("STRONG_BUY", 100),
        ],
    )
    def test_all_levels(self, score, expected):
        r = SimpleNamespace(overall_score=score)
        assert _normalize_dip(r) == expected

    def test_unknown(self):
        r = SimpleNamespace(overall_score="UNKNOWN")
        assert _normalize_dip(r) == 0


class TestNormalizePead:
    @pytest.mark.parametrize(
        "score,expected",
        [
            ("FAIL", 0),
            ("WATCH", 30),
            ("LEAN_BUY", 55),
            ("BUY", 80),
            ("STRONG_BUY", 100),
        ],
    )
    def test_all_levels(self, score, expected):
        r = SimpleNamespace(overall_score=score)
        assert _normalize_pead(r) == expected


# ── Signal classification ───────────────────────────────────────────────


class TestClassifySignal:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (100, AlphaSignal.STRONG_BUY),
            (80, AlphaSignal.STRONG_BUY),
            (79, AlphaSignal.BUY),
            (65, AlphaSignal.BUY),
            (64, AlphaSignal.LEAN_BUY),
            (55, AlphaSignal.LEAN_BUY),
            (54, AlphaSignal.NEUTRAL),
            (40, AlphaSignal.NEUTRAL),
            (39, AlphaSignal.LEAN_SELL),
            (25, AlphaSignal.LEAN_SELL),
            (24, AlphaSignal.AVOID),
            (0, AlphaSignal.AVOID),
        ],
    )
    def test_thresholds(self, score, expected):
        assert classify_signal(score) == expected


# ── Weight redistribution ───────────────────────────────────────────────


def _make_runner(normalized: float):
    """Return a (runner, normalizer) pair that always succeeds with *normalized*."""
    return (
        lambda sym: SimpleNamespace(total_score=normalized, score=normalized),
        lambda r: normalized,
    )


class TestWeightRedistribution:
    @patch("advisor.confluence.alpha_scorer._LAYERS")
    def test_all_layers_available(self, mock_layers):
        """When all layers available, weights should sum to 1.0."""
        runner, normalizer = _make_runner(50.0)
        mock_layers.__iter__ = lambda self: iter(
            [("a", runner, normalizer), ("b", runner, normalizer)]
        )
        mock_layers.__len__ = lambda self: 2

        result = compute_alpha("TEST", weights={"a": 0.6, "b": 0.4})
        active = [ls for ls in result.layers if ls.available]
        total_weight = sum(ls.weight for ls in active)
        assert abs(total_weight - 1.0) < 1e-6
        assert result.alpha_score == 50.0

    @patch("advisor.confluence.alpha_scorer._LAYERS")
    def test_one_layer_fails(self, mock_layers):
        """When one layer fails, remaining weight is redistributed."""

        def fail_runner(sym):
            return None

        runner, normalizer = _make_runner(80.0)
        mock_layers.__iter__ = lambda self: iter(
            [
                ("a", runner, normalizer),
                ("b", fail_runner, normalizer),
                ("c", runner, normalizer),
            ]
        )
        mock_layers.__len__ = lambda self: 3

        result = compute_alpha("TEST", weights={"a": 0.5, "b": 0.3, "c": 0.2})
        active = [ls for ls in result.layers if ls.available]
        total_weight = sum(ls.weight for ls in active)
        assert abs(total_weight - 1.0) < 1e-6
        # All active layers normalized to 80 → composite should be 80
        assert result.alpha_score == 80.0

    @patch("advisor.confluence.alpha_scorer._LAYERS")
    def test_all_layers_fail(self, mock_layers):
        """When all layers fail, score is 0 and signal is AVOID."""

        def fail_runner(sym):
            return None

        mock_layers.__iter__ = lambda self: iter([("a", fail_runner, lambda r: 0)])
        mock_layers.__len__ = lambda self: 1

        result = compute_alpha("TEST", weights={"a": 1.0})
        assert result.alpha_score == 0.0
        assert result.signal == AlphaSignal.AVOID
        assert result.active_layers == 0


# ── Score bounds ────────────────────────────────────────────────────────


class TestScoreBounds:
    @patch("advisor.confluence.alpha_scorer._LAYERS")
    def test_max_100(self, mock_layers):
        runner, normalizer = _make_runner(100.0)
        mock_layers.__iter__ = lambda self: iter([("a", runner, normalizer)])
        mock_layers.__len__ = lambda self: 1

        result = compute_alpha("TEST", weights={"a": 1.0})
        assert result.alpha_score <= 100.0

    @patch("advisor.confluence.alpha_scorer._LAYERS")
    def test_min_0(self, mock_layers):
        runner, normalizer = _make_runner(0.0)
        mock_layers.__iter__ = lambda self: iter([("a", runner, normalizer)])
        mock_layers.__len__ = lambda self: 1

        result = compute_alpha("TEST", weights={"a": 1.0})
        assert result.alpha_score >= 0.0


# ── skip_layers parameter ──────────────────────────────────────────────


class TestSkipLayers:
    @patch("advisor.confluence.alpha_scorer._LAYERS")
    def test_skip_marks_unavailable(self, mock_layers):
        runner, normalizer = _make_runner(50.0)
        mock_layers.__iter__ = lambda self: iter(
            [("a", runner, normalizer), ("b", runner, normalizer)]
        )
        mock_layers.__len__ = lambda self: 2

        result = compute_alpha("TEST", weights={"a": 0.5, "b": 0.5}, skip_layers={"b"})
        layer_b = next(ls for ls in result.layers if ls.name == "b")
        assert not layer_b.available
        assert layer_b.error == "skipped"
        # Only layer a active → full weight on a
        assert result.alpha_score == 50.0

    @patch("advisor.confluence.alpha_scorer._LAYERS")
    def test_skip_all_returns_zero(self, mock_layers):
        runner, normalizer = _make_runner(50.0)
        mock_layers.__iter__ = lambda self: iter([("a", runner, normalizer)])
        mock_layers.__len__ = lambda self: 1

        result = compute_alpha("TEST", weights={"a": 1.0}, skip_layers={"a"})
        assert result.alpha_score == 0.0
        assert result.signal == AlphaSignal.AVOID
