"""Tests for validation module — resolve predictions against historical prices."""

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from advisor.simulator.models import SimConfig
from advisor.simulator.validation import (
    ResolvedOutcome,
    _compute_calibration_buckets,
    resolve_outcomes,
    resolve_single_outcome,
)


def _make_ohlcv(
    entry_date: date,
    n_trade_days: int,
    base_price: float = 50.0,
    daily_returns: list[float] | None = None,
    low_overrides: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame for testing.

    Generates a buffer of 60 trading days before entry_date at base_price
    (for HV computation), then applies daily_returns starting from entry.

    Args:
        entry_date: First trading day of the trade.
        n_trade_days: Number of trading days from entry through expiration.
        base_price: Starting price at entry.
        daily_returns: Optional list of daily returns applied starting day 1 of trade.
        low_overrides: Dict of {trade_day_index: low_price} overrides (0-indexed from entry).
    """
    buffer_days = 60
    total_days = buffer_days + n_trade_days
    dates = pd.bdate_range(start=entry_date - timedelta(days=90), periods=total_days)

    # Flat price in the buffer, then apply returns during trade period
    prices = np.full(total_days, base_price)

    if daily_returns is not None:
        for i in range(buffer_days + 1, total_days):
            ret_idx = i - buffer_days - 1
            if ret_idx < len(daily_returns):
                prices[i] = prices[i - 1] * (1 + daily_returns[ret_idx])
            else:
                prices[i] = prices[i - 1]

    lows = prices * 0.998

    if low_overrides:
        for trade_day, low_val in low_overrides.items():
            abs_idx = buffer_days + trade_day
            if abs_idx < total_days:
                lows[abs_idx] = low_val

    df = pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.002,
            "Low": lows,
            "Close": prices,
            "Volume": np.full(total_days, 1_000_000),
        },
        index=dates,
    )

    return df


class _MockDataProvider:
    """Mock data provider that returns pre-built OHLCV data."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_stock_history(self, symbol, start, end, interval="1d"):
        mask = (self._df.index >= pd.Timestamp(start)) & (self._df.index <= pd.Timestamp(end))
        result = self._df.loc[mask]
        if result.empty:
            raise ValueError(f"No data for {symbol}")
        return result


@pytest.fixture
def config():
    return SimConfig(
        profit_target_pct=0.50,
        stop_loss_multiplier=2.0,
        close_at_dte=7,
        slippage_pct=0.02,
        risk_free_rate=0.05,
    )


class TestResolveSingleOutcome:
    def test_profitable_expiration(self, config):
        """Price stays well above short strike -> profitable at expiration."""
        entry = date(2025, 1, 2)
        exp = "2025-02-07"

        # Generate flat price at 50 (short strike at 45, long at 42)
        df = _make_ohlcv(entry, n_trade_days=60, base_price=50.0)
        provider = _MockDataProvider(df)

        outcome = resolve_single_outcome(
            symbol="TEST",
            entry_date=entry,
            expiration=exp,
            short_strike=45.0,
            long_strike=42.0,
            net_credit=0.40,
            width=3.0,
            short_iv=0.35,
            long_iv=0.38,
            config=config,
            data_provider=provider,
        )

        assert outcome.actual_profit == 1.0, "Should be profitable"
        assert outcome.actual_touch == 0.0, "Should not touch short strike"
        assert outcome.actual_pnl > 0, f"P&L should be positive, got {outcome.actual_pnl}"

    def test_touch_detection(self, config):
        """Force Low below short strike on one day -> touch detected."""
        entry = date(2025, 1, 2)
        exp = "2025-02-07"

        # Force Low below short strike on trade day 10
        df = _make_ohlcv(entry, n_trade_days=60, base_price=50.0, low_overrides={10: 44.5})
        provider = _MockDataProvider(df)

        outcome = resolve_single_outcome(
            symbol="TEST",
            entry_date=entry,
            expiration=exp,
            short_strike=45.0,
            long_strike=42.0,
            net_credit=0.40,
            width=3.0,
            short_iv=0.35,
            long_iv=0.38,
            config=config,
            data_provider=provider,
        )

        assert outcome.actual_touch == 1.0, "Should detect touch"

    def test_stop_loss_triggered(self):
        """Price crashes hard below long strike -> stop loss triggered."""
        entry = date(2025, 1, 2)
        exp = "2025-04-03"  # Long DTE so DTE-close doesn't trigger first

        # Aggressive stop-loss config: stop at 1x credit
        stop_config = SimConfig(
            profit_target_pct=0.99,  # Won't hit
            stop_loss_multiplier=1.0,  # Stop at 1x credit loss
            close_at_dte=3,  # Low threshold
            slippage_pct=0.02,
        )

        # Create a severe crash: -4% per day for 10 days
        n_trade_days = 100
        returns = [(-0.04 if i < 10 else 0.0) for i in range(n_trade_days)]
        df = _make_ohlcv(entry, n_trade_days=n_trade_days, base_price=50.0, daily_returns=returns)
        provider = _MockDataProvider(df)

        outcome = resolve_single_outcome(
            symbol="TEST",
            entry_date=entry,
            expiration=exp,
            short_strike=45.0,
            long_strike=42.0,
            net_credit=0.40,
            width=3.0,
            short_iv=0.35,
            long_iv=0.38,
            config=stop_config,
            data_provider=provider,
        )

        assert outcome.actual_stop == 1.0, (
            f"Should trigger stop loss, got exit_reason={outcome.exit_reason}, "
            f"pnl={outcome.actual_pnl}"
        )
        assert outcome.exit_reason == "stop_loss"
        assert outcome.actual_pnl < 0, f"P&L should be negative, got {outcome.actual_pnl}"

    def test_dte_close_exit(self):
        """Verify exit when remaining DTE <= close_at_dte."""
        # Use a very high close_at_dte to force early exit
        config = SimConfig(
            profit_target_pct=0.99,  # Very high -> won't hit
            stop_loss_multiplier=100.0,  # Very high -> won't hit
            close_at_dte=30,  # Close when <= 30 DTE remaining
            slippage_pct=0.02,
        )

        entry = date(2025, 1, 2)
        exp = "2025-02-28"  # ~57 days out

        df = _make_ohlcv(entry, n_trade_days=80, base_price=50.0)
        provider = _MockDataProvider(df)

        outcome = resolve_single_outcome(
            symbol="TEST",
            entry_date=entry,
            expiration=exp,
            short_strike=45.0,
            long_strike=42.0,
            net_credit=0.40,
            width=3.0,
            short_iv=0.35,
            long_iv=0.38,
            config=config,
            data_provider=provider,
        )

        assert (
            outcome.exit_reason == "dte_close"
        ), f"Expected dte_close exit, got {outcome.exit_reason}"
        # Should exit well before expiration
        total_dte = (date.fromisoformat(exp) - entry).days
        assert outcome.exit_day < total_dte, "Should exit before expiration"

    def test_profit_target_exit(self):
        """Spread value decays enough -> profit target hit.

        Uses ATM-ish strikes with high IV so the entry spread has significant
        value, then price rallies hard to decay the spread and hit profit target.
        """
        entry = date(2025, 1, 2)
        exp = "2025-04-03"  # ~90 DTE

        # Noisy buffer for HV, then strong rally from entry
        buffer_days = 60
        n_trade_days = 100
        total_days = buffer_days + n_trade_days
        dates = pd.bdate_range(start=entry - timedelta(days=90), periods=total_days)

        rng = np.random.default_rng(42)
        prices = np.full(total_days, 50.0)
        for i in range(1, buffer_days):
            prices[i] = prices[i - 1] * np.exp(rng.normal(0, 0.012))
        prices[buffer_days] = 50.0
        # Strong rally: +2%/day for 20 days then flat
        for i in range(buffer_days + 1, total_days):
            day = i - buffer_days
            if day <= 20:
                prices[i] = prices[i - 1] * 1.02
            else:
                prices[i] = prices[i - 1]

        df = pd.DataFrame(
            {
                "Open": prices * 0.999,
                "High": prices * 1.002,
                "Low": prices * 0.998,
                "Close": prices,
                "Volume": np.full(total_days, 1e6),
            },
            index=dates,
        )
        provider = _MockDataProvider(df)

        # Spread: 48/45 (closer to money) with high IV -> significant entry value
        # With entry_spread ~20.50 and credit=80, threshold must be achievable
        profit_config = SimConfig(
            profit_target_pct=0.15,  # Take profit at 15% of credit = $12
            stop_loss_multiplier=10.0,
            close_at_dte=0,  # Disable DTE close so profit target wins
            slippage_pct=0.005,
        )

        outcome = resolve_single_outcome(
            symbol="TEST",
            entry_date=entry,
            expiration=exp,
            short_strike=48.0,  # Near ATM
            long_strike=45.0,
            net_credit=0.80,
            width=3.0,
            short_iv=0.35,
            long_iv=0.38,
            config=profit_config,
            data_provider=provider,
        )

        assert (
            outcome.exit_reason == "profit_target"
        ), f"Expected profit_target, got {outcome.exit_reason}, pnl={outcome.actual_pnl}"
        assert outcome.actual_profit == 1.0
        assert outcome.actual_pnl > 0

    def test_insufficient_data_handled(self, config):
        """Should handle case with insufficient price data gracefully."""
        entry = date(2025, 1, 2)
        exp = "2025-01-03"  # Only 1 day

        # Very short DataFrame — provider will cover the fetch range
        dates = pd.bdate_range(start=entry - timedelta(days=90), periods=65)
        prices = np.full(len(dates), 50.0)
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 1,
                "Low": prices - 1,
                "Close": prices,
                "Volume": np.full(len(dates), 1e6),
            },
            index=dates,
        )
        provider = _MockDataProvider(df)

        outcome = resolve_single_outcome(
            symbol="TEST",
            entry_date=entry,
            expiration=exp,
            short_strike=45.0,
            long_strike=42.0,
            net_credit=0.40,
            width=3.0,
            short_iv=0.35,
            long_iv=0.38,
            config=config,
            data_provider=provider,
        )

        # Should not crash — returns some outcome
        assert isinstance(outcome, ResolvedOutcome)


class TestResolveOutcomes:
    def test_no_pending_records(self):
        """No pending records -> empty list."""
        store = MagicMock()
        store.get_pending_calibrations.return_value = []

        result = resolve_outcomes(store)
        assert result == []

    def test_future_expiration_skipped(self):
        """Records with future expiration should be skipped."""
        future_date = (date.today() + timedelta(days=30)).isoformat()
        store = MagicMock()
        store.get_pending_calibrations.return_value = [
            {
                "candidate_id": "c1",
                "symbol": "TEST",
                "expiration": future_date,
                "short_strike": 45.0,
                "long_strike": 42.0,
                "width": 3.0,
                "net_credit": 0.40,
                "short_iv": 0.35,
                "long_iv": 0.38,
                "underlying_price": 50.0,
                "dte": 30,
                "created_at": date.today().isoformat(),
                "predicted_pop": 0.80,
                "predicted_touch": 0.25,
                "predicted_stop": 0.10,
                "predicted_ev": 10.0,
            }
        ]

        result = resolve_outcomes(store)
        assert result == []
        store.update_calibration_outcome.assert_not_called()


class TestComputeCalibrationBuckets:
    def test_empty_data(self):
        """Empty data should return 10 buckets all with count=0."""
        buckets = _compute_calibration_buckets([], [])
        assert len(buckets) == 10
        assert all(b["count"] == 0 for b in buckets)

    def test_concentrated_data(self):
        """Predictions clustered in one bucket -> correct counts."""
        # All predictions between 0.7-0.8
        predicted = [0.75, 0.72, 0.78, 0.71]
        actual = [1.0, 1.0, 0.0, 1.0]

        buckets = _compute_calibration_buckets(predicted, actual)
        assert len(buckets) == 10

        # Find the 70-80% bucket
        bucket_70 = next(b for b in buckets if b["bucket"] == "70-80%")
        assert bucket_70["count"] == 4
        assert bucket_70["actual_mean"] == pytest.approx(0.75, abs=0.01)

        # Other buckets should be empty
        other_counts = sum(b["count"] for b in buckets if b["bucket"] != "70-80%")
        assert other_counts == 0

    def test_spread_across_buckets(self):
        """Predictions spread across range -> multiple buckets populated."""
        predicted = [0.15, 0.25, 0.55, 0.85]
        actual = [0.0, 0.0, 1.0, 1.0]

        buckets = _compute_calibration_buckets(predicted, actual)
        populated = [b for b in buckets if b["count"] > 0]
        assert len(populated) == 4

    def test_edge_case_probability_1(self):
        """Probability of exactly 1.0 should go in last bucket."""
        predicted = [1.0]
        actual = [1.0]

        buckets = _compute_calibration_buckets(predicted, actual)
        last_bucket = buckets[-1]
        assert last_bucket["count"] == 1

    def test_bucket_labels(self):
        """Check that bucket labels are formatted correctly."""
        buckets = _compute_calibration_buckets([], [])
        expected_labels = [
            "0-10%",
            "10-20%",
            "20-30%",
            "30-40%",
            "40-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-90%",
            "90-100%",
        ]
        actual_labels = [b["bucket"] for b in buckets]
        assert actual_labels == expected_labels
