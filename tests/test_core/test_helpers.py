"""Tests for advisor.core.helpers — safe conversion utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

from advisor.core.helpers import safe_float, safe_info, safe_int

# ── safe_float ───────────────────────────────────────────────────────────────


class TestSafeFloat:
    def test_normal_float(self):
        assert safe_float(3.14) == 3.14

    def test_int_input(self):
        assert safe_float(42) == 42.0

    def test_string_number(self):
        assert safe_float("2.5") == 2.5

    def test_none_returns_default(self):
        assert safe_float(None) is None
        assert safe_float(None, default=0.0) == 0.0

    def test_nan_returns_default(self):
        assert safe_float(float("nan")) is None
        assert safe_float(float("nan"), default=-1.0) == -1.0

    def test_inf_returns_default(self):
        assert safe_float(float("inf")) is None
        assert safe_float(float("-inf"), default=0.0) == 0.0

    def test_non_numeric_string(self):
        assert safe_float("abc") is None
        assert safe_float("abc", default=0.0) == 0.0

    def test_zero(self):
        assert safe_float(0) == 0.0

    def test_negative(self):
        assert safe_float(-5.5) == -5.5

    def test_bool_converts(self):
        # bool is a subclass of int in Python
        assert safe_float(True) == 1.0
        assert safe_float(False) == 0.0


# ── safe_int ─────────────────────────────────────────────────────────────────


class TestSafeInt:
    def test_normal_int(self):
        assert safe_int(7) == 7

    def test_float_truncates(self):
        assert safe_int(3.9) == 3

    def test_string_number(self):
        assert safe_int("5") == 5

    def test_string_float(self):
        assert safe_int("3.0") == 3

    def test_none_returns_default(self):
        assert safe_int(None) == 0
        assert safe_int(None, default=99) == 99

    def test_nan_returns_default(self):
        assert safe_int(float("nan")) == 0
        assert safe_int(float("nan"), default=-1) == -1

    def test_inf_returns_default(self):
        assert safe_int(float("inf")) == 0

    def test_non_numeric_string(self):
        assert safe_int("abc") == 0
        assert safe_int("abc", default=42) == 42

    def test_zero(self):
        assert safe_int(0) == 0

    def test_negative(self):
        assert safe_int(-3.7) == -3


# ── safe_info ────────────────────────────────────────────────────────────────


class TestSafeInfo:
    def test_normal_info(self):
        ticker = MagicMock()
        ticker.info = {"symbol": "AAPL", "shortName": "Apple"}
        assert safe_info(ticker) == {"symbol": "AAPL", "shortName": "Apple"}

    def test_none_info(self):
        ticker = MagicMock()
        ticker.info = None
        assert safe_info(ticker) == {}

    def test_exception_returns_empty(self):
        ticker = MagicMock()
        type(ticker).info = property(lambda self: (_ for _ in ()).throw(RuntimeError("network")))
        assert safe_info(ticker) == {}

    def test_empty_dict(self):
        ticker = MagicMock()
        ticker.info = {}
        assert safe_info(ticker) == {}
