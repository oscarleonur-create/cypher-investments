"""Tests for PEAD (Post-Earnings Announcement Drift) strategy."""

from __future__ import annotations

import sys

import backtrader as bt
import pandas as pd
from advisor.strategies.equity.pead import PeadDrift
from advisor.strategies.registry import StrategyRegistry

_PEAD_MOD = "advisor.strategies.equity.pead"


def _fresh_discover() -> StrategyRegistry:
    """discover() that works even when pead was already imported."""
    sys.modules.pop(_PEAD_MOD, None)
    registry = StrategyRegistry()
    registry.discover()
    return registry


def _make_price_df(
    prices: list[float],
    volumes: list[int] | None = None,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    """Build an OHLCV DataFrame from close prices and optional volumes."""
    dates = pd.bdate_range(start=start, periods=len(prices))
    if volumes is None:
        volumes = [1_000_000] * len(prices)
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": volumes,
        },
        index=dates,
    )


def _run_cerebro(
    prices: list[float],
    volumes: list[int] | None = None,
    **strategy_kwargs,
) -> bt.Strategy:
    """Run a minimal Cerebro with PeadDrift on synthetic data."""
    df = _make_price_df(prices, volumes)
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
    cerebro.addstrategy(PeadDrift, **strategy_kwargs)
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.0)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    results = cerebro.run()
    return results[0]


# ===========================================================================
# Registration tests
# ===========================================================================


class TestPeadRegistration:
    def test_discovered_by_registry(self):
        registry = _fresh_discover()
        assert "pead" in registry.names

    def test_correct_class_from_registry(self):
        registry = _fresh_discover()
        cls = registry.get("pead")
        assert cls is not None
        assert cls.strategy_name == "pead"

    def test_metadata(self):
        meta = PeadDrift.get_metadata()
        assert meta["name"] == "pead"
        assert meta["type"] == "equity"
        assert meta["version"] == "1.1.0"
        assert "hold_days" in meta["params"]
        assert "volume_spike_factor" in meta["params"]
        assert "fade_pct" in meta["params"]
        assert "sma_long" in meta["params"]
        assert "stop_loss_pct" in meta["params"]

    def test_default_params(self):
        meta = PeadDrift.get_metadata()
        assert meta["params"]["hold_days"] == 45
        assert meta["params"]["volume_spike_factor"] == 2.0
        assert meta["params"]["fade_pct"] == -0.02
        assert meta["params"]["sma_long"] == 200
        assert meta["params"]["stop_loss_pct"] == -0.08

    def test_description_is_nonempty(self):
        assert len(PeadDrift.description) > 0

    def test_force_all_confluence(self):
        assert PeadDrift.force_all_confluence is True


# ===========================================================================
# Logic tests (synthetic data, no network)
# ===========================================================================


class TestPeadLogic:
    def test_flat_prices_no_volume_spike_no_trades(self):
        """Constant prices and uniform volume should never trigger entry."""
        prices = [100.0] * 250
        volumes = [1_000_000] * 250
        strat = _run_cerebro(prices, volumes, sma_long=50, volume_avg_period=10)
        trade_analysis = strat.analyzers.trades.get_analysis()
        total = trade_analysis.get("total", {}).get("total", 0)
        assert total == 0

    def test_volume_spike_with_fade_triggers_buy(self):
        """Volume spike followed by a price fade should trigger a buy."""
        n_warmup = 210
        base_price = 100.0
        base_vol = 1_000_000

        prices = [base_price] * n_warmup
        volumes = [base_vol] * n_warmup

        # Spike day: volume 3x, price jumps
        prices.append(108.0)
        volumes.append(base_vol * 3)

        # Day after spike: still high
        prices.append(106.0)
        volumes.append(base_vol)

        # Wait bars (2-3): price fades below spike high
        prices.append(105.0)
        volumes.append(base_vol)

        # Hold for enough bars to exit
        for _ in range(50):
            prices.append(105.0)
            volumes.append(base_vol)

        strat = _run_cerebro(
            prices,
            volumes,
            sma_long=200,
            volume_avg_period=20,
            volume_spike_factor=2.0,
            fade_pct=-0.02,
            hold_days=10,
            wait_bars_min=2,
            wait_bars_max=3,
        )
        trade_analysis = strat.analyzers.trades.get_analysis()
        total = trade_analysis.get("total", {}).get("total", 0)
        assert total >= 1

    def test_hold_days_exit(self):
        """Position should be closed after hold_days bars."""
        n_warmup = 210
        base_price = 100.0
        base_vol = 1_000_000

        prices = [base_price] * n_warmup
        volumes = [base_vol] * n_warmup

        # Spike
        prices.append(108.0)
        volumes.append(base_vol * 3)

        # Post-spike
        prices.append(106.0)
        volumes.append(base_vol)

        # Fade entry
        prices.append(105.0)
        volumes.append(base_vol)

        # Hold for hold_days + buffer
        for _ in range(20):
            prices.append(105.0)
            volumes.append(base_vol)

        strat = _run_cerebro(
            prices,
            volumes,
            sma_long=200,
            volume_avg_period=20,
            hold_days=5,
            wait_bars_min=2,
            wait_bars_max=3,
        )
        trade_analysis = strat.analyzers.trades.get_analysis()
        closed = trade_analysis.get("total", {}).get("closed", 0)
        assert closed >= 1

    def test_no_buy_below_sma200(self):
        """Should not buy if price is below SMA(200)."""
        prices = [100.0] * 200
        prices.extend([60.0] * 10)
        volumes = [1_000_000] * len(prices)

        # Spike while below SMA
        prices.append(65.0)
        volumes.append(3_000_000)

        prices.append(63.0)
        volumes.append(1_000_000)

        # Fade
        prices.append(60.0)
        volumes.append(1_000_000)

        for _ in range(10):
            prices.append(60.0)
            volumes.append(1_000_000)

        strat = _run_cerebro(
            prices,
            volumes,
            sma_long=200,
            volume_avg_period=20,
            wait_bars_min=2,
            wait_bars_max=3,
        )
        trade_analysis = strat.analyzers.trades.get_analysis()
        total = trade_analysis.get("total", {}).get("total", 0)
        assert total == 0

    def test_stop_loss_triggers_early_exit(self):
        """Stop-loss should close position before hold_days if price drops."""
        n_warmup = 210
        base_price = 100.0
        base_vol = 1_000_000

        prices = [base_price] * n_warmup
        volumes = [base_vol] * n_warmup

        # Spike
        prices.append(108.0)
        volumes.append(base_vol * 3)

        # Post-spike
        prices.append(106.0)
        volumes.append(base_vol)

        # Fade entry at ~105
        prices.append(105.0)
        volumes.append(base_vol)

        # Price crashes: 105 → 90 (>8% drop from 105)
        # Entry price is 105, stop at -8% = 96.6
        prices.append(96.0)  # below stop-loss threshold
        volumes.append(base_vol)

        # Continue at low price (well within hold_days=45)
        for _ in range(10):
            prices.append(90.0)
            volumes.append(base_vol)

        strat = _run_cerebro(
            prices,
            volumes,
            sma_long=200,
            volume_avg_period=20,
            hold_days=45,
            stop_loss_pct=-0.08,
            wait_bars_min=2,
            wait_bars_max=3,
        )
        trade_analysis = strat.analyzers.trades.get_analysis()
        closed = trade_analysis.get("total", {}).get("closed", 0)
        # Should have closed early due to stop-loss, not waited 45 days
        assert closed >= 1

    def test_no_stop_loss_when_price_holds(self):
        """Price staying above stop-loss should not trigger early exit."""
        n_warmup = 210
        base_price = 100.0
        base_vol = 1_000_000

        prices = [base_price] * n_warmup
        volumes = [base_vol] * n_warmup

        # Spike
        prices.append(108.0)
        volumes.append(base_vol * 3)

        # Post-spike
        prices.append(106.0)
        volumes.append(base_vol)

        # Fade entry at ~105
        prices.append(105.0)
        volumes.append(base_vol)

        # Price dips but stays above stop-loss (105 * 0.92 = 96.6)
        for _ in range(8):
            prices.append(100.0)  # -4.8%, above -8% stop
            volumes.append(base_vol)

        strat = _run_cerebro(
            prices,
            volumes,
            sma_long=200,
            volume_avg_period=20,
            hold_days=45,
            stop_loss_pct=-0.08,
            wait_bars_min=2,
            wait_bars_max=3,
        )
        trade_analysis = strat.analyzers.trades.get_analysis()
        # Should still be holding (only 8 bars, hold_days=45, no stop-loss hit)
        open_trades = trade_analysis.get("total", {}).get("open", 0)
        assert open_trades == 1
