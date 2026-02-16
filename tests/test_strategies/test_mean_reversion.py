"""Tests for Mean Reversion (Short-Term Bounce) strategy."""

from __future__ import annotations

import sys

import backtrader as bt
import pandas as pd
from advisor.strategies.equity.mean_reversion import MeanReversion
from advisor.strategies.registry import StrategyRegistry

_MR_MOD = "advisor.strategies.equity.mean_reversion"


def _fresh_discover() -> StrategyRegistry:
    """discover() that works even when mean_reversion was already imported."""
    sys.modules.pop(_MR_MOD, None)
    registry = StrategyRegistry()
    registry.discover()
    return registry


def _make_price_df(
    prices: list[float],
    volumes: list[int] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    """Build an OHLCV DataFrame from close prices and optional volumes/highs/lows."""
    dates = pd.bdate_range(start=start, periods=len(prices))
    if volumes is None:
        volumes = [1_000_000] * len(prices)
    if highs is None:
        highs = [p * 1.02 for p in prices]
    if lows is None:
        lows = [p * 0.98 for p in prices]
    return pd.DataFrame(
        {
            "Open": prices,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
        },
        index=dates,
    )


def _run_cerebro(
    prices: list[float],
    volumes: list[int] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    **strategy_kwargs,
) -> bt.Strategy:
    """Run a minimal Cerebro with MeanReversion on synthetic data."""
    df = _make_price_df(prices, volumes, highs, lows)
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
    cerebro.addstrategy(MeanReversion, **strategy_kwargs)
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.0)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    results = cerebro.run()
    return results[0]


# ===========================================================================
# Registration tests
# ===========================================================================


class TestMeanReversionRegistration:
    def test_discovered_by_registry(self):
        registry = _fresh_discover()
        assert "mean_reversion" in registry.names

    def test_correct_class_from_registry(self):
        registry = _fresh_discover()
        cls = registry.get("mean_reversion")
        assert cls is not None
        assert cls.strategy_name == "mean_reversion"

    def test_metadata_keys(self):
        meta = MeanReversion.get_metadata()
        assert meta["name"] == "mean_reversion"
        assert meta["type"] == "equity"
        assert meta["version"] == "1.0.0"
        assert "rsi_period" in meta["params"]
        assert "rsi_threshold" in meta["params"]
        assert "ema_period" in meta["params"]
        assert "atr_period" in meta["params"]
        assert "atr_multiplier" in meta["params"]
        assert "volume_avg_period" in meta["params"]
        assert "volume_spike_factor" in meta["params"]

    def test_default_params(self):
        meta = MeanReversion.get_metadata()
        assert meta["params"]["rsi_period"] == 14
        assert meta["params"]["rsi_threshold"] == 25
        assert meta["params"]["ema_period"] == 20
        assert meta["params"]["atr_period"] == 14
        assert meta["params"]["atr_multiplier"] == 2.0
        assert meta["params"]["volume_avg_period"] == 20
        assert meta["params"]["volume_spike_factor"] == 1.5

    def test_description_is_nonempty(self):
        assert len(MeanReversion.description) > 0

    def test_force_all_confluence(self):
        assert MeanReversion.force_all_confluence is True


# ===========================================================================
# Logic tests (synthetic data, no network)
# ===========================================================================


class TestMeanReversionLogic:
    def test_flat_prices_no_trades(self):
        """Constant prices and uniform volume → RSI ~50, no trades."""
        prices = [100.0] * 250
        volumes = [1_000_000] * 250
        strat = _run_cerebro(prices, volumes)
        trade_analysis = strat.analyzers.trades.get_analysis()
        total = trade_analysis.get("total", {}).get("total", 0)
        assert total == 0

    def test_oversold_atr_drop_volume_spike_triggers_buy(self):
        """Steep decline with volume spike should trigger at least 1 trade."""
        base_vol = 1_000_000

        # 100-bar warmup: stable at 100
        n_warmup = 100
        prices = [100.0] * n_warmup
        highs = [102.0] * n_warmup
        lows = [98.0] * n_warmup
        volumes = [base_vol] * n_warmup

        # 10-bar steep decline: price drops ~3% per bar with volume spike
        # This creates: low RSI, price far below EMA, high ATR
        price = 100.0
        for i in range(10):
            price *= 0.97
            prices.append(price)
            highs.append(price * 1.01)
            lows.append(price * 0.99)
            volumes.append(int(base_vol * 2.5))  # volume spike

        # Recovery: price bounces back toward EMA over 15 bars
        for i in range(15):
            price *= 1.02
            prices.append(price)
            highs.append(price * 1.01)
            lows.append(price * 0.99)
            volumes.append(base_vol)

        # Padding for indicators to settle
        for _ in range(30):
            prices.append(price)
            highs.append(price * 1.01)
            lows.append(price * 0.99)
            volumes.append(base_vol)

        strat = _run_cerebro(
            prices,
            volumes,
            highs=highs,
            lows=lows,
            rsi_period=14,
            rsi_threshold=30,  # slightly relaxed for synthetic data
            ema_period=20,
            atr_period=14,
            atr_multiplier=1.5,
            volume_avg_period=20,
            volume_spike_factor=1.5,
        )
        trade_analysis = strat.analyzers.trades.get_analysis()
        total = trade_analysis.get("total", {}).get("total", 0)
        assert total >= 1

    def test_exit_at_ema_reversion(self):
        """Position should close when price recovers to EMA."""
        base_vol = 1_000_000

        # 100-bar warmup
        n_warmup = 100
        prices = [100.0] * n_warmup
        highs = [102.0] * n_warmup
        lows = [98.0] * n_warmup
        volumes = [base_vol] * n_warmup

        # Steep decline with volume spike
        price = 100.0
        for i in range(10):
            price *= 0.97
            prices.append(price)
            highs.append(price * 1.01)
            lows.append(price * 0.99)
            volumes.append(int(base_vol * 2.5))

        # Recovery back above EMA
        for i in range(20):
            price *= 1.02
            prices.append(price)
            highs.append(price * 1.01)
            lows.append(price * 0.99)
            volumes.append(base_vol)

        # Padding
        for _ in range(30):
            prices.append(price)
            highs.append(price * 1.01)
            lows.append(price * 0.99)
            volumes.append(base_vol)

        strat = _run_cerebro(
            prices,
            volumes,
            highs=highs,
            lows=lows,
            rsi_period=14,
            rsi_threshold=30,
            ema_period=20,
            atr_period=14,
            atr_multiplier=1.5,
            volume_avg_period=20,
            volume_spike_factor=1.5,
        )
        trade_analysis = strat.analyzers.trades.get_analysis()
        closed = trade_analysis.get("total", {}).get("closed", 0)
        assert closed >= 1

    def test_no_volume_spike_no_trade(self):
        """Deep oversold + far below EMA but normal volume → no trade."""
        base_vol = 1_000_000

        # 100-bar warmup
        n_warmup = 100
        prices = [100.0] * n_warmup
        highs = [102.0] * n_warmup
        lows = [98.0] * n_warmup
        volumes = [base_vol] * n_warmup

        # Steep decline but NO volume spike (volume stays at base)
        price = 100.0
        for i in range(10):
            price *= 0.97
            prices.append(price)
            highs.append(price * 1.01)
            lows.append(price * 0.99)
            volumes.append(base_vol)  # no spike

        # Padding
        for _ in range(40):
            prices.append(price)
            highs.append(price * 1.01)
            lows.append(price * 0.99)
            volumes.append(base_vol)

        strat = _run_cerebro(
            prices,
            volumes,
            highs=highs,
            lows=lows,
            rsi_period=14,
            rsi_threshold=30,
            ema_period=20,
            atr_period=14,
            atr_multiplier=1.5,
            volume_avg_period=20,
            volume_spike_factor=1.5,
        )
        trade_analysis = strat.analyzers.trades.get_analysis()
        total = trade_analysis.get("total", {}).get("total", 0)
        assert total == 0

    def test_rsi_above_threshold_no_trade(self):
        """Below EMA with volume spike but RSI not oversold → no trade."""
        base_vol = 1_000_000

        # 100-bar warmup
        n_warmup = 100
        prices = [100.0] * n_warmup
        highs = [102.0] * n_warmup
        lows = [98.0] * n_warmup
        volumes = [base_vol] * n_warmup

        # Gentle decline (not enough to push RSI below threshold)
        # ~1% per bar for 5 bars — RSI will stay around 35-40
        price = 100.0
        for i in range(5):
            price *= 0.99
            prices.append(price)
            highs.append(price * 1.01)
            lows.append(price * 0.99)
            volumes.append(int(base_vol * 2.5))  # volume spike present

        # Padding
        for _ in range(40):
            prices.append(price)
            highs.append(price * 1.01)
            lows.append(price * 0.99)
            volumes.append(base_vol)

        strat = _run_cerebro(
            prices,
            volumes,
            highs=highs,
            lows=lows,
            rsi_period=14,
            rsi_threshold=25,  # strict threshold
            ema_period=20,
            atr_period=14,
            atr_multiplier=2.0,
            volume_avg_period=20,
            volume_spike_factor=1.5,
        )
        trade_analysis = strat.analyzers.trades.get_analysis()
        total = trade_analysis.get("total", {}).get("total", 0)
        assert total == 0
