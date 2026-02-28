#!/usr/bin/env python3
"""
Options Backtester — Black-Scholes based historical simulation.
Strategies: naked_put, wheel, put_credit_spread.
Uses yfinance for price data, core/pricing.py for BSM pricing with full Greeks.
"""

import logging
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from advisor.core.enums import OptionType
from advisor.core.pricing import bsm_price
from advisor.market.premium_screener import get_adaptive_delta

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class BacktestConfig:
    """Tunable parameters for options backtesting."""

    # Entry signals
    use_adaptive_delta: bool = True  # use IV-based delta vs fixed 0.25
    base_rsi_threshold: float = 40.0  # default RSI entry threshold
    rsi_relax_in_high_iv: bool = True  # relax to 50 when IV pctile > 75
    use_regime_filter: bool = True  # skip entries in high-vol HMM regime
    use_iv_term_filter: bool = False  # require near-term HV > long-term HV

    # Position management
    profit_target_pct: float = 0.50  # close at 50% of credit
    stop_loss_multiplier: float = 3.0  # close at 3x credit
    close_at_dte: int = 21  # close when DTE <= 21
    use_gamma_exit: bool = True  # close on gamma spike
    gamma_threshold: float = 0.03  # gamma * S threshold
    use_iv_crush_exit: bool = True  # close early on IV drop
    iv_crush_pnl_pct: float = 0.30  # min unrealized P&L for crush exit
    iv_crush_drop_pct: float = 0.20  # min IV drop from entry


# ── IV estimation ────────────────────────────────────────────────────────────


def estimate_iv(prices: pd.Series, window: int = 30) -> pd.Series:
    """Rolling historical volatility as IV proxy (annualized)."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ── Strike selection ─────────────────────────────────────────────────────────


def find_strike_for_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float,
    option_type: str = "put",
    step: float = 0.50,
) -> float:
    """Find strike closest to target delta via grid search."""
    if option_type == "put":
        strikes = np.arange(S * 0.70, S * 1.01, step)
        deltas = [abs(bsm_price(S, k, T, r, sigma, OptionType.PUT).delta) for k in strikes]
    else:
        strikes = np.arange(S * 1.0, S * 1.30, step)
        deltas = [bsm_price(S, k, T, r, sigma, OptionType.CALL).delta for k in strikes]

    if not len(deltas):
        return round(S * 0.90 if option_type == "put" else S * 1.10, 2)

    idx = np.argmin(np.abs(np.array(deltas) - target_delta))
    return round(float(strikes[idx]) * 2) / 2  # round to $0.50


# ── Trade tracking ───────────────────────────────────────────────────────────


@dataclass
class Trade:
    strategy: str
    entry_date: str
    expiry_date: str
    strike: float
    option_type: str  # put / call / spread
    premium: float  # per share, credit received
    contracts: int = 1
    exit_date: Optional[str] = None
    exit_premium: Optional[float] = None
    pnl: Optional[float] = None
    reason: Optional[str] = None
    # spread fields
    long_strike: Optional[float] = None
    long_premium: Optional[float] = None
    # entry context
    entry_iv: Optional[float] = None
    entry_delta: Optional[float] = None

    def net_credit(self) -> float:
        """Total credit received (per contract = 100 shares)."""
        if self.long_premium is not None:
            return (self.premium - self.long_premium) * 100 * self.contracts
        return self.premium * 100 * self.contracts

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


# ── Strategies ───────────────────────────────────────────────────────────────


class Backtester:
    def __init__(
        self,
        symbol: str,
        start: str,
        end: str,
        cash: float,
        risk_free: float = 0.045,
        config: BacktestConfig | None = None,
    ):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_cash = cash
        self.cash = cash
        self.r = risk_free
        self.config = config or BacktestConfig()
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []
        self._load_data()

    def _load_data(self):
        # Fetch with extra buffer for IV calc + IV percentile
        buf_start = (pd.Timestamp(self.start) - timedelta(days=365)).strftime("%Y-%m-%d")
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=buf_start, end=self.end, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data for {self.symbol}")
        df.index = df.index.tz_localize(None)
        df["IV"] = estimate_iv(df["Close"], window=30)
        df["RSI"] = compute_rsi(df["Close"], period=14)
        df["RedDay"] = df["Close"] < df["Open"]

        # Rolling IV percentile: rank current IV against trailing 252-day distribution
        df["IV_Pctile"] = (
            df["IV"]
            .rolling(252, min_periods=60)
            .apply(lambda w: (w.iloc[:-1] < w.iloc[-1]).mean() * 100)
        )

        # HMM regime labels
        if self.config.use_regime_filter:
            try:
                from advisor.ml.regime import RegimeDetector

                if RegimeDetector.model_exists():
                    detector = RegimeDetector.load()
                    regime_df = detector.compute_regime_features(df.index)
                    df["regime"] = regime_df["hmm_regime"]
                else:
                    logger.info("No HMM model found, regime filter disabled")
                    df["regime"] = 1  # default to normal
            except Exception as e:
                logger.warning("Regime detection failed, defaulting to normal: %s", e)
                df["regime"] = 1

        # IV term structure proxy
        if self.config.use_iv_term_filter:
            df["IV_60"] = estimate_iv(df["Close"], window=60)
            df["iv_contango"] = df["IV"] > df["IV_60"]  # near-term elevated

        self.df = df.loc[self.start : self.end].copy()
        self.all_df = df  # keep full for lookback
        print(f"Loaded {len(self.df)} trading days for {self.symbol}", file=sys.stderr)

    # ── Shared position management ────────────────────────────────────────

    def _close_trade(
        self, trade: Trade, date_str: str, exit_premium: float, reason: str
    ) -> tuple[bool, Trade]:
        """Close a trade with given exit premium and reason."""
        initial_credit = trade.premium - (trade.long_premium or 0)
        trade.exit_date = date_str
        trade.exit_premium = round(exit_premium, 2)
        trade.pnl = round((initial_credit - exit_premium) * 100 * trade.contracts, 2)
        trade.reason = reason
        return (True, trade)

    def _manage_position(
        self, trade: Trade, S: float, iv: float, date, date_str: str
    ) -> tuple[bool, Trade | None]:
        """Check exits for an open position. Returns (closed, trade_if_closed)."""
        expiry = pd.Timestamp(trade.expiry_date)
        dte = (expiry - date).days
        T = max(dte / 365.0, 0.001)

        # Get full Greeks via bsm_price
        if trade.option_type == "spread":
            short_result = bsm_price(S, trade.strike, T, self.r, iv, OptionType.PUT)
            long_result = bsm_price(S, trade.long_strike, T, self.r, iv, OptionType.PUT)
            current_premium = short_result.price - long_result.price
            gamma = short_result.gamma - long_result.gamma  # net gamma
        elif trade.option_type == "put":
            result = bsm_price(S, trade.strike, T, self.r, iv, OptionType.PUT)
            current_premium = result.price
            gamma = result.gamma
        else:  # call
            result = bsm_price(S, trade.strike, T, self.r, iv, OptionType.CALL)
            current_premium = result.price
            gamma = result.gamma

        initial_credit = trade.premium - (trade.long_premium or 0)

        # --- Profit target ---
        if current_premium <= initial_credit * self.config.profit_target_pct:
            return self._close_trade(trade, date_str, current_premium, "profit_target")

        # --- Stop loss ---
        if current_premium >= initial_credit * self.config.stop_loss_multiplier:
            return self._close_trade(trade, date_str, current_premium, "stop_loss")

        # --- DTE exit ---
        if self.config.close_at_dte and dte <= self.config.close_at_dte and dte > 0:
            return self._close_trade(trade, date_str, current_premium, "dte_exit")

        # --- Gamma spike exit ---
        if self.config.use_gamma_exit and abs(gamma) * S > self.config.gamma_threshold:
            return self._close_trade(trade, date_str, current_premium, "gamma_exit")

        # --- IV crush exit ---
        if self.config.use_iv_crush_exit and trade.entry_iv:
            iv_drop = (trade.entry_iv - iv) / trade.entry_iv
            unrealized_pnl_pct = (
                (initial_credit - current_premium) / initial_credit if initial_credit > 0 else 0
            )
            if (
                iv_drop > self.config.iv_crush_drop_pct
                and unrealized_pnl_pct > self.config.iv_crush_pnl_pct
            ):
                return self._close_trade(trade, date_str, current_premium, "iv_crush_exit")

        # --- Expiry ---
        if dte <= 0:
            return self._close_at_expiry(trade, S, date_str)

        return (False, None)  # still open

    def _close_at_expiry(self, trade: Trade, S: float, date_str: str) -> tuple[bool, Trade]:
        """Handle trade at expiration with intrinsic value settlement."""
        if trade.option_type == "spread":
            short_itm = max(trade.strike - S, 0)
            long_itm = max(trade.long_strike - S, 0)
            intrinsic = short_itm - long_itm
        elif trade.option_type == "put":
            intrinsic = max(trade.strike - S, 0)
        else:  # call
            intrinsic = max(S - trade.strike, 0)

        initial_credit = trade.premium - (trade.long_premium or 0)
        trade.exit_date = date_str
        trade.exit_premium = round(intrinsic, 2)
        trade.pnl = round((initial_credit - intrinsic) * 100 * trade.contracts, 2)
        if trade.option_type == "spread":
            trade.reason = "expired"
        else:
            trade.reason = "expired_otm" if intrinsic == 0 else "expired_itm"
        return (True, trade)

    def _should_enter(self, row, rsi: float) -> bool:
        """Check entry signal filters (regime, IV term structure, RSI threshold)."""
        iv_pctile = row.get("IV_Pctile", 50)
        if np.isnan(iv_pctile):
            iv_pctile = 50

        rsi_thresh = self.config.base_rsi_threshold
        if self.config.rsi_relax_in_high_iv and iv_pctile >= 75:
            rsi_thresh = 50.0

        if self.config.use_regime_filter and row.get("regime") == 2:
            return False

        if self.config.use_iv_term_filter and not row.get("iv_contango", True):
            return False

        if not row["RedDay"] or rsi >= rsi_thresh:
            return False

        return True

    def _get_target_delta(self, row, default: float = 0.25) -> float:
        """Get target delta, adaptive or fixed."""
        if self.config.use_adaptive_delta:
            iv_pctile = row.get("IV_Pctile", 50)
            if np.isnan(iv_pctile):
                iv_pctile = 50
            return get_adaptive_delta(iv_pctile)
        return default

    # ── Naked Put ────────────────────────────────────────────────────────

    def run_naked_put(self):
        df = self.df
        open_trades: list[Trade] = []
        max_positions = int(self.initial_cash / (df["Close"].median() * 100)) or 1
        max_positions = min(max_positions, 2)  # conservative with $5K

        for date, row in df.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            S = row["Close"]
            iv = row["IV"]
            rsi = row["RSI"]

            if np.isnan(iv) or np.isnan(rsi):
                self.equity_curve.append(self.cash)
                continue

            # Check/close existing trades
            still_open = []
            for t in open_trades:
                closed, closed_trade = self._manage_position(t, S, iv, date, date_str)
                if closed:
                    self.cash += closed_trade.pnl
                    self.trades.append(closed_trade)
                else:
                    still_open.append(t)
            open_trades = still_open

            # Entry: adaptive signals
            if self._should_enter(row, rsi) and len(open_trades) < max_positions:
                target_dte = 35
                expiry_date = (date + timedelta(days=target_dte)).strftime("%Y-%m-%d")
                T = target_dte / 365.0
                target_delta = self._get_target_delta(row, default=0.25)
                strike = find_strike_for_delta(S, T, self.r, iv, target_delta, "put")

                # Margin: max(20% underlying - OTM, 10% strike) * 100 + premium
                otm_amount = max(S - strike, 0)
                margin_req = max(0.20 * S - otm_amount, 0.10 * strike) * 100
                if margin_req > self.cash * 0.80:
                    self.equity_curve.append(self.cash)
                    continue

                bsm_result = bsm_price(S, strike, T, self.r, iv, OptionType.PUT)
                premium = bsm_result.price
                if premium < 0.10:  # skip tiny premiums
                    self.equity_curve.append(self.cash)
                    continue

                trade = Trade(
                    strategy="naked_put",
                    entry_date=date_str,
                    expiry_date=expiry_date,
                    strike=strike,
                    option_type="put",
                    premium=round(premium, 2),
                    contracts=1,
                    entry_iv=round(iv, 4),
                    entry_delta=round(abs(bsm_result.delta), 4),
                )
                self.cash += trade.net_credit()  # receive premium
                open_trades.append(trade)

            self.equity_curve.append(self.cash)

        # Close remaining at end
        for t in open_trades:
            S = df.iloc[-1]["Close"]
            iv = df.iloc[-1]["IV"]
            if np.isnan(iv):
                iv = self.all_df["IV"].dropna().iloc[-1]
            dte = max((pd.Timestamp(t.expiry_date) - df.index[-1]).days, 0)
            T = max(dte / 365.0, 0.001)
            current_premium = bsm_price(S, t.strike, T, self.r, iv, OptionType.PUT).price
            t.exit_date = df.index[-1].strftime("%Y-%m-%d")
            t.exit_premium = round(current_premium, 2)
            t.pnl = round((t.premium - current_premium) * 100 * t.contracts, 2)
            t.reason = "end_of_backtest"
            self.cash += t.pnl
            self.trades.append(t)

    # ── Wheel ────────────────────────────────────────────────────────────

    def run_wheel(self):
        df = self.df
        state = "selling_puts"  # or "selling_calls"
        shares = 0
        cost_basis = 0.0
        open_trade: Optional[Trade] = None

        for date, row in df.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            S = row["Close"]
            iv = row["IV"]
            if np.isnan(iv):
                self.equity_curve.append(self.cash + shares * S)
                continue

            # ── Manage open trade ──
            if open_trade:
                expiry = pd.Timestamp(open_trade.expiry_date)
                dte = (expiry - date).days
                T = max(dte / 365.0, 0.001)

                # Wheel has special assignment/called-away logic at expiry
                # Use _manage_position for standard exits (profit target, stop loss,
                # DTE exit, gamma exit, IV crush), but handle expiry ourselves
                if dte > 0:
                    # Check standard exits (non-expiry)
                    closed, closed_trade = self._manage_position(open_trade, S, iv, date, date_str)
                    if closed:
                        self.cash += closed_trade.pnl
                        self.trades.append(closed_trade)
                        open_trade = None
                        self.equity_curve.append(self.cash + shares * S)
                        continue
                else:
                    # Expiry — wheel-specific assignment/called-away logic
                    if open_trade.option_type == "put":
                        intrinsic = max(open_trade.strike - S, 0)
                        if intrinsic > 0:
                            # Assigned — buy 100 shares
                            open_trade.exit_date = date_str
                            open_trade.pnl = round((open_trade.premium - intrinsic) * 100, 2)
                            open_trade.reason = "assigned"
                            self.cash += open_trade.pnl
                            cost_basis = open_trade.strike - open_trade.premium
                            shares = 100
                            self.cash -= open_trade.strike * 100
                            self.trades.append(open_trade)
                            open_trade = None
                            state = "selling_calls"
                        else:
                            open_trade.exit_date = date_str
                            open_trade.exit_premium = 0.0
                            open_trade.pnl = round(open_trade.premium * 100, 2)
                            open_trade.reason = "expired_otm"
                            self.cash += open_trade.pnl
                            self.trades.append(open_trade)
                            open_trade = None

                    elif open_trade.option_type == "call":
                        if S >= open_trade.strike:
                            # Called away
                            open_trade.exit_date = date_str
                            open_trade.pnl = round(
                                (open_trade.strike - cost_basis + open_trade.premium) * 100,
                                2,
                            )
                            open_trade.reason = "called_away"
                            self.cash += open_trade.strike * 100 + open_trade.premium * 100
                            shares = 0
                            self.trades.append(open_trade)
                            open_trade = None
                            state = "selling_puts"
                        else:
                            open_trade.exit_date = date_str
                            open_trade.exit_premium = 0.0
                            open_trade.pnl = round(open_trade.premium * 100, 2)
                            open_trade.reason = "expired_otm"
                            self.cash += open_trade.pnl
                            self.trades.append(open_trade)
                            open_trade = None

                    self.equity_curve.append(self.cash + shares * S)
                    continue

                # Not expired yet, continue
                self.equity_curve.append(self.cash + shares * S)
                continue

            # ── Open new trade ──
            if state == "selling_puts":
                # Need enough cash for assignment
                affordable_strike = self.cash / 100.0
                if affordable_strike < S * 0.70:
                    self.equity_curve.append(self.cash + shares * S)
                    continue

                target_dte = 35
                T = target_dte / 365.0
                expiry_date = (date + timedelta(days=target_dte)).strftime("%Y-%m-%d")
                target_delta = self._get_target_delta(row, default=0.25)
                strike = find_strike_for_delta(S, T, self.r, iv, target_delta, "put")
                strike = min(strike, affordable_strike)
                bsm_result = bsm_price(S, strike, T, self.r, iv, OptionType.PUT)
                premium = bsm_result.price
                if premium < 0.10:
                    self.equity_curve.append(self.cash + shares * S)
                    continue

                open_trade = Trade(
                    strategy="wheel",
                    entry_date=date_str,
                    expiry_date=expiry_date,
                    strike=round(strike, 2),
                    option_type="put",
                    premium=round(premium, 2),
                    entry_iv=round(iv, 4),
                    entry_delta=round(abs(bsm_result.delta), 4),
                )
                self.cash += open_trade.net_credit()

            elif state == "selling_calls" and shares >= 100:
                target_dte = 35
                T = target_dte / 365.0
                expiry_date = (date + timedelta(days=target_dte)).strftime("%Y-%m-%d")
                # Sell call above cost basis
                target_delta = self._get_target_delta(row, default=0.30)
                strike = find_strike_for_delta(S, T, self.r, iv, target_delta, "call")
                strike = max(strike, cost_basis * 1.02)  # at least break even
                bsm_result = bsm_price(S, strike, T, self.r, iv, OptionType.CALL)
                premium = bsm_result.price
                if premium < 0.05:
                    self.equity_curve.append(self.cash + shares * S)
                    continue

                open_trade = Trade(
                    strategy="wheel",
                    entry_date=date_str,
                    expiry_date=expiry_date,
                    strike=round(strike, 2),
                    option_type="call",
                    premium=round(premium, 2),
                    entry_iv=round(iv, 4),
                    entry_delta=round(bsm_result.delta, 4),
                )
                self.cash += open_trade.net_credit()

            self.equity_curve.append(self.cash + shares * S)

        # End cleanup — mark-to-market the last open trade
        if open_trade:
            S = df.iloc[-1]["Close"]
            iv = df.iloc[-1]["IV"]
            if np.isnan(iv):
                iv = self.all_df["IV"].dropna().iloc[-1]
            dte = max((pd.Timestamp(open_trade.expiry_date) - df.index[-1]).days, 0)
            T = max(dte / 365.0, 0.001)
            if open_trade.option_type == "put":
                current = bsm_price(S, open_trade.strike, T, self.r, iv, OptionType.PUT).price
            else:
                current = bsm_price(S, open_trade.strike, T, self.r, iv, OptionType.CALL).price
            open_trade.exit_date = df.index[-1].strftime("%Y-%m-%d")
            open_trade.exit_premium = round(current, 2)
            open_trade.pnl = round((open_trade.premium - current) * 100, 2)
            open_trade.reason = "end_of_backtest"
            self.cash += open_trade.pnl
            self.trades.append(open_trade)
        if shares > 0:
            self.cash += shares * df.iloc[-1]["Close"]

    # ── Put Credit Spread ────────────────────────────────────────────────

    def run_put_credit_spread(self):
        df = self.df
        open_trades: list[Trade] = []
        spread_width = 5.0
        max_positions = 3

        for date, row in df.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            S = row["Close"]
            iv = row["IV"]
            rsi = row["RSI"]
            if np.isnan(iv) or np.isnan(rsi):
                self.equity_curve.append(self.cash)
                continue

            # Manage existing
            still_open = []
            for t in open_trades:
                closed, closed_trade = self._manage_position(t, S, iv, date, date_str)
                if closed:
                    self.cash += closed_trade.pnl
                    self.trades.append(closed_trade)
                else:
                    still_open.append(t)
            open_trades = still_open

            # Entry: adaptive signals (use base threshold of 45 for spreads)
            iv_pctile = row.get("IV_Pctile", 50)
            if np.isnan(iv_pctile):
                iv_pctile = 50
            rsi_thresh = 45.0
            if self.config.rsi_relax_in_high_iv and iv_pctile >= 75:
                rsi_thresh = 55.0

            can_enter = row["RedDay"] and rsi < rsi_thresh
            if self.config.use_regime_filter and row.get("regime") == 2:
                can_enter = False
            if self.config.use_iv_term_filter and not row.get("iv_contango", True):
                can_enter = False

            if can_enter and len(open_trades) < max_positions:
                target_dte = 35
                T = target_dte / 365.0
                expiry_date = (date + timedelta(days=target_dte)).strftime("%Y-%m-%d")

                target_delta = self._get_target_delta(row, default=0.25)
                short_strike = find_strike_for_delta(S, T, self.r, iv, target_delta, "put")
                long_strike = short_strike - spread_width

                short_result = bsm_price(S, short_strike, T, self.r, iv, OptionType.PUT)
                long_result = bsm_price(S, long_strike, T, self.r, iv, OptionType.PUT)
                short_prem = short_result.price
                long_prem = long_result.price
                net_credit = short_prem - long_prem

                if net_credit < 0.15:
                    self.equity_curve.append(self.cash)
                    continue

                # Max risk per spread
                max_risk = (spread_width - net_credit) * 100
                if max_risk > self.cash * 0.15:
                    self.equity_curve.append(self.cash)
                    continue

                trade = Trade(
                    strategy="put_credit_spread",
                    entry_date=date_str,
                    expiry_date=expiry_date,
                    strike=round(short_strike, 2),
                    option_type="spread",
                    premium=round(short_prem, 2),
                    long_strike=round(long_strike, 2),
                    long_premium=round(long_prem, 2),
                    entry_iv=round(iv, 4),
                    entry_delta=round(abs(short_result.delta), 4),
                )
                self.cash += trade.net_credit()
                open_trades.append(trade)

            self.equity_curve.append(self.cash)

        # Close remaining — mark-to-market
        for t in open_trades:
            S = df.iloc[-1]["Close"]
            iv = df.iloc[-1]["IV"]
            if np.isnan(iv):
                iv = self.all_df["IV"].dropna().iloc[-1]
            initial_credit = t.premium - (t.long_premium or 0)
            dte = max((pd.Timestamp(t.expiry_date) - df.index[-1]).days, 0)
            T = max(dte / 365.0, 0.001)
            short_val = bsm_price(S, t.strike, T, self.r, iv, OptionType.PUT).price
            long_val = bsm_price(S, t.long_strike, T, self.r, iv, OptionType.PUT).price
            spread_val = short_val - long_val
            t.exit_date = df.index[-1].strftime("%Y-%m-%d")
            t.exit_premium = round(spread_val, 2)
            t.pnl = round((initial_credit - spread_val) * 100 * t.contracts, 2)
            t.reason = "end_of_backtest"
            self.cash += t.pnl
            self.trades.append(t)

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self, strategy: str):
        dispatch = {
            "naked_put": self.run_naked_put,
            "wheel": self.run_wheel,
            "put_credit_spread": self.run_put_credit_spread,
        }
        if strategy not in dispatch:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(dispatch)}")
        print(f"Running {strategy} on {self.symbol}...", file=sys.stderr)
        dispatch[strategy]()
        return self.summary(strategy)

    def summary(self, strategy: str) -> dict:
        trades = self.trades
        if not trades:
            return {"error": "No trades generated"}

        pnls = [t.pnl for t in trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        premiums = [t.premium for t in trades]

        # Days in trade
        days_in_trade = []
        for t in trades:
            if t.exit_date and t.entry_date:
                d = (pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days
                days_in_trade.append(d)

        # Max drawdown from equity curve
        eq = np.array(self.equity_curve) if self.equity_curve else np.array([self.initial_cash])
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = float(np.min(dd)) if len(dd) else 0.0

        # Sharpe (daily returns from equity curve)
        if len(eq) > 1:
            daily_ret = np.diff(eq) / eq[:-1]
            sharpe = float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-9) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Exit reason breakdown
        exit_reasons: dict[str, int] = {}
        for t in trades:
            r = t.reason or "unknown"
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        result = {
            "strategy": strategy,
            "symbol": self.symbol,
            "period": f"{self.start} → {self.end}",
            "initial_cash": self.initial_cash,
            "final_cash": round(self.cash, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl / self.initial_cash * 100, 2),
            "num_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
            "avg_premium": round(np.mean(premiums), 2),
            "avg_pnl_per_trade": round(np.mean(pnls), 2) if pnls else 0,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_days_in_trade": round(np.mean(days_in_trade), 1) if days_in_trade else 0,
            "exit_reasons": exit_reasons,
            "trades": [t.to_dict() for t in trades],
        }
        return result


def print_summary(result: dict):
    from rich.console import Console
    from rich.table import Table

    c = Console()
    c.print(f"\n[bold cyan]═══ {result['strategy'].upper()} Backtest: {result['symbol']} ═══[/]")
    c.print(f"Period: {result['period']}")
    c.print(f"Initial: ${result['initial_cash']:,.2f}  →  Final: ${result['final_cash']:,.2f}")
    c.print(
        f"[{'green' if result['total_pnl'] >= 0 else 'red'}]"
        f"Total P&L: ${result['total_pnl']:,.2f} ({result['total_return_pct']:+.1f}%)[/]"
    )
    c.print()

    t = Table(show_header=True, header_style="bold")
    for col in ["Trades", "Win Rate", "Avg Premium", "Avg P&L", "Max DD", "Sharpe", "Avg Days"]:
        t.add_column(col, justify="right")
    t.add_row(
        str(result["num_trades"]),
        f"{result['win_rate_pct']}%",
        f"${result['avg_premium']:.2f}",
        f"${result['avg_pnl_per_trade']:.2f}",
        f"{result['max_drawdown_pct']:.1f}%",
        f"{result['sharpe_ratio']:.2f}",
        f"{result['avg_days_in_trade']:.0f}",
    )
    c.print(t)

    # Exit reasons breakdown
    exit_reasons = result.get("exit_reasons", {})
    if exit_reasons:
        c.print("\n[bold]Exit Reasons:[/]")
        er_table = Table(show_header=True, header_style="dim")
        er_table.add_column("Reason")
        er_table.add_column("Count", justify="right")
        er_table.add_column("Pct", justify="right")
        total = sum(exit_reasons.values())
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total else 0
            er_table.add_row(reason, str(count), f"{pct:.0f}%")
        c.print(er_table)

    # Trade log
    c.print(f"\n[bold]Trade Log ({len(result['trades'])} trades):[/]")
    log = Table(show_header=True, header_style="dim")
    for col in ["Entry", "Expiry", "Type", "Strike", "Premium", "Exit", "P&L", "Reason"]:
        log.add_column(col)
    for tr in result["trades"]:
        pnl = tr.get("pnl", 0) or 0
        pnl_style = "green" if pnl >= 0 else "red"
        strike_str = f"${tr['strike']}"
        if tr.get("long_strike"):
            strike_str += f"/${tr['long_strike']}"
        log.add_row(
            tr["entry_date"],
            tr["expiry_date"],
            tr["option_type"],
            strike_str,
            f"${tr['premium']:.2f}",
            tr.get("exit_date", "—"),
            f"[{pnl_style}]${pnl:,.2f}[/]",
            tr.get("reason", "—"),
        )
    c.print(log)
