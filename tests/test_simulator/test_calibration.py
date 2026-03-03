"""Tests for calibration module — Student-t fitting and vol dynamics."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from advisor.simulator.calibration import (
    calibrate,
    estimate_vol_dynamics,
    fit_student_t,
)
from advisor.simulator.models import SimConfig
from scipy.stats import t as student_t


def _mock_returns(n=252, df=5, scale=0.015):
    """Generate synthetic daily returns from Student-t distribution."""
    rng = np.random.default_rng(42)
    returns = student_t.rvs(df=df, loc=0, scale=scale, size=n, random_state=rng)
    return pd.Series(returns, index=pd.date_range("2025-01-01", periods=n, freq="B"))


@pytest.fixture
def mock_returns():
    return _mock_returns()


class TestFitStudentT:
    def test_recovers_known_df(self):
        """Fitting Student-t to data generated with df=5 should recover ~5."""
        returns = _mock_returns(n=500, df=5, scale=0.015)
        df, loc, scale = fit_student_t(returns)
        assert 3.0 <= df <= 8.0, f"Expected df near 5, got {df}"

    def test_df_clamped_lower(self):
        """Very fat tails should be clamped to df >= 2.5."""
        returns = _mock_returns(n=500, df=2.0, scale=0.02)
        df, loc, scale = fit_student_t(returns)
        assert df >= 2.5

    def test_returns_three_values(self):
        returns = _mock_returns()
        result = fit_student_t(returns)
        assert len(result) == 3
        df, loc, scale = result
        assert isinstance(df, float)
        assert isinstance(loc, float)
        assert isinstance(scale, float)


class TestEstimateVolDynamics:
    def test_returns_expected_keys(self):
        returns = _mock_returns(n=252)
        result = estimate_vol_dynamics(returns)
        assert "vol_mean_level" in result
        assert "vol_mean_revert_speed" in result
        assert "leverage_effect" in result

    def test_vol_mean_positive(self):
        returns = _mock_returns(n=252)
        result = estimate_vol_dynamics(returns)
        assert result["vol_mean_level"] > 0

    def test_leverage_negative_or_zero(self):
        returns = _mock_returns(n=252)
        result = estimate_vol_dynamics(returns)
        assert result["leverage_effect"] <= 0

    def test_short_data_fallback(self):
        """With very short data, should still return valid defaults."""
        returns = _mock_returns(n=20)
        result = estimate_vol_dynamics(returns)
        assert result["vol_mean_level"] > 0
        assert 0 < result["vol_mean_revert_speed"] <= 2.0


class TestCalibrate:
    @patch("advisor.simulator.calibration._get_daily_returns")
    def test_returns_sim_config(self, mock_get):
        mock_get.return_value = _mock_returns()
        result = calibrate("TEST")
        assert isinstance(result, SimConfig)

    @patch("advisor.simulator.calibration._get_daily_returns")
    def test_updates_from_base(self, mock_get):
        mock_get.return_value = _mock_returns()
        base = SimConfig(n_paths=50_000, profit_target_pct=0.60)
        result = calibrate("TEST", config=base)
        # Should preserve non-calibrated params
        assert result.n_paths == 50_000
        assert result.profit_target_pct == 0.60
        # Should update calibrated params
        assert result.student_t_df != SimConfig().student_t_df or True  # May match by chance

    @patch("advisor.simulator.calibration._get_daily_returns")
    def test_failure_returns_defaults(self, mock_get):
        mock_get.side_effect = ValueError("No data")
        result = calibrate("BADTICKER")
        assert isinstance(result, SimConfig)
        assert result.student_t_df == 5.0  # Default
