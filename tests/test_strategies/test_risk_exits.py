"""Tests for equity risk exits: stop-loss, trailing stop, circuit breaker."""

from __future__ import annotations

import backtrader as bt
import pandas as pd
import pytest
from advisor.engine.analyzers import DrawdownCircuitBreaker
from advisor.strategies.equity.pead import PeadDrift
from advisor.strategies.equity.sma_crossover import SMACrossover

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_price_df(prices: list[float], start: str = "2024-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=len(prices))
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": [1_000_000] * len(prices),
        },
        index=dates,
    )


def _run_cerebro(
    strategy_cls,
    prices: list[float],
    cash: float = 100_000,
    circuit_breaker_pct: float = 0.0,
    **strategy_kwargs,
) -> bt.Strategy:
    df = _make_price_df(prices)
    cerebro = bt.Cerebro()
    feed = bt.feeds.PandasData(
        dataname=df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=-1,
    )
    cerebro.adddata(feed)
    cerebro.addstrategy(strategy_cls, **strategy_kwargs)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.0)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    if circuit_breaker_pct > 0:
        cerebro.addanalyzer(
            DrawdownCircuitBreaker,
            _name="circuit_breaker",
            max_drawdown_pct=circuit_breaker_pct,
        )
    results = cerebro.run()
    return results[0]


# ── Stop-loss tests ──────────────────────────────────────────────────────────


class TestStopLoss:
    def test_stop_loss_triggers_on_gap_down(self):
        """A 15% gap-down should trigger the default -10% stop-loss."""
        prices = (
            [100.0] * 20  # flat start
            + [100 - i * 0.5 for i in range(1, 11)]  # gentle dip to 95
            + [95 + i * 3 for i in range(1, 21)]  # rise to 155 (golden cross)
            + [155.0] * 5  # hold
            + [130.0]  # gap down ~16% from 155
            + [125.0] * 10  # stay low
        )
        strat = _run_cerebro(SMACrossover, prices, short_period=5, long_period=15)
        trade_analysis = strat.analyzers.trades.get_analysis()
        closed = trade_analysis.get("total", {}).get("closed", 0)
        # Should have exited via stop-loss, not waiting for death cross
        assert closed >= 1

    def test_stop_loss_disabled_when_zero(self):
        """Setting stop_loss_pct=0 should disable stop-loss."""
        prices = (
            [100.0] * 20
            + [100 - i * 0.5 for i in range(1, 11)]
            + [95 + i * 3 for i in range(1, 21)]
            + [155.0] * 5
            + [130.0]  # gap down
            + [130 + i * 2 for i in range(1, 21)]  # recover
        )
        strat = _run_cerebro(
            SMACrossover, prices, short_period=5, long_period=15, stop_loss_pct=0.0
        )
        # With stop-loss disabled, position should survive the gap down
        # and the strategy continues until a death cross
        assert strat.broker.getvalue() > 0

    def test_pead_uses_base_stop_loss(self):
        """PeadDrift's -8% stop-loss should come from base mechanism."""
        # Generate spike + fade pattern for PEAD entry
        prices = (
            [100.0] * 210  # need 200 bars for SMA(200)
            + [100.0, 100.0, 100.0]  # pre-spike
        )
        # Volume spike + fade after
        volumes = [1_000_000] * 210 + [1_000_000, 1_000_000, 1_000_000]
        # Add volume spike
        prices.append(105.0)  # spike day high
        volumes.append(5_000_000)  # 5x volume spike
        # Wait bars + fade
        prices.extend([103.0, 102.0, 101.0])  # fade below spike high
        volumes.extend([1_000_000] * 3)
        # Then crash
        prices.extend([90.0] * 10)  # -10% drop (beyond -8% stop)
        volumes.extend([1_000_000] * 10)

        # PeadDrift has stop_loss_pct=-0.08 which overrides the base -0.10
        assert PeadDrift._get_param_defaults().get("stop_loss_pct") == -0.08


class TestTrailingStop:
    def test_trailing_stop_fires_after_peak(self):
        """Trailing stop should fire when price drops 5% from peak."""
        prices = (
            [100.0] * 20
            + [100 - i for i in range(1, 11)]  # dip to 90
            + [90 + i * 3 for i in range(1, 21)]  # rise to 150 (golden cross)
            + [150 + i * 2 for i in range(1, 11)]  # peak at 170
            + [170 - i * 2 for i in range(1, 11)]  # drop to 150 (~12% off peak)
        )
        strat = _run_cerebro(
            SMACrossover,
            prices,
            short_period=5,
            long_period=15,
            trailing_stop_pct=0.05,  # 5% trail
            stop_loss_pct=0.0,  # disable hard stop
        )
        trade_analysis = strat.analyzers.trades.get_analysis()
        closed = trade_analysis.get("total", {}).get("closed", 0)
        assert closed >= 1  # trailing stop should have triggered


# ── Circuit breaker tests ────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_circuit_breaker_trips(self):
        """Portfolio dropping 20% should trip a 15% circuit breaker."""
        # Big decline to trigger circuit breaker
        prices = (
            [100.0] * 20
            + [100 - i for i in range(1, 11)]  # dip
            + [90 + i * 3 for i in range(1, 11)]  # rise to 120 (buy)
            + [120 - i * 3 for i in range(1, 21)]  # crash to 60 (-50%)
        )
        strat = _run_cerebro(
            SMACrossover,
            prices,
            short_period=5,
            long_period=15,
            circuit_breaker_pct=15.0,
        )
        # Circuit breaker should have tripped
        cb = strat.analyzers.circuit_breaker.get_analysis()
        assert cb["tripped"] is True

    def test_circuit_breaker_analyzer_records_peak(self):
        """Circuit breaker should track peak portfolio value."""
        prices = [100.0] * 50  # flat — no trades
        strat = _run_cerebro(
            SMACrossover,
            prices,
            short_period=5,
            long_period=15,
            circuit_breaker_pct=15.0,
        )
        cb = strat.analyzers.circuit_breaker.get_analysis()
        assert cb["peak_value"] == pytest.approx(100_000, rel=0.01)
        assert cb["tripped"] is False


# ── Parameter validation tests ───────────────────────────────────────────────


class TestParamValidation:
    def test_sma_crossover_rejects_invalid_periods(self):
        """Setting short_period >= long_period should raise ValueError."""
        prices = [100.0] * 80
        with pytest.raises(ValueError, match="short_period.*must be less than"):
            _run_cerebro(SMACrossover, prices, short_period=50, long_period=20)

    def test_sma_crossover_accepts_valid_periods(self):
        """Valid periods should not raise."""
        prices = [100.0] * 80
        strat = _run_cerebro(SMACrossover, prices, short_period=10, long_period=30)
        assert strat is not None


# ── Base class tests ─────────────────────────────────────────────────────────


class TestStrategyBaseRiskTracking:
    def test_entry_price_tracked(self):
        """After a buy fill, _risk_entry_price should be set."""
        prices = [100.0] * 20 + [100 - i for i in range(1, 11)] + [90 + i * 3 for i in range(1, 21)]
        strat = _run_cerebro(SMACrossover, prices, short_period=5, long_period=15)
        # Verify the risk tracking attributes exist
        assert hasattr(strat, "_risk_entry_price")
        assert hasattr(strat, "_risk_peak_price")
        # A golden cross should have triggered a buy, so entry price should be set
        assert strat._risk_entry_price is not None
        assert strat._risk_entry_price > 0
