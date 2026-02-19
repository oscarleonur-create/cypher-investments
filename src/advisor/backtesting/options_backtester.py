#!/usr/bin/env python3
"""
Options Backtester — Black-Scholes based historical simulation.
Strategies: naked_put, wheel, put_credit_spread.
Uses yfinance for price data, scipy for BS pricing.
"""

import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# ── Black-Scholes ────────────────────────────────────────────────────────────


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price. T in years."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


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
        deltas = [abs(bs_put_delta(S, k, T, r, sigma)) for k in strikes]
    else:
        strikes = np.arange(S * 1.0, S * 1.30, step)
        deltas = [bs_call_delta(S, k, T, r, sigma) for k in strikes]

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

    def net_credit(self) -> float:
        """Total credit received (per contract = 100 shares)."""
        if self.long_premium is not None:
            return (self.premium - self.long_premium) * 100 * self.contracts
        return self.premium * 100 * self.contracts

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


# ── Strategies ───────────────────────────────────────────────────────────────


class Backtester:
    def __init__(self, symbol: str, start: str, end: str, cash: float, risk_free: float = 0.045):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_cash = cash
        self.cash = cash
        self.r = risk_free
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []
        self._load_data()

    def _load_data(self):
        # Fetch with extra buffer for IV calc
        buf_start = (pd.Timestamp(self.start) - timedelta(days=60)).strftime("%Y-%m-%d")
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=buf_start, end=self.end, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data for {self.symbol}")
        df.index = df.index.tz_localize(None)
        df["IV"] = estimate_iv(df["Close"], window=30)
        df["RSI"] = compute_rsi(df["Close"], period=14)
        df["RedDay"] = df["Close"] < df["Open"]
        self.df = df.loc[self.start : self.end].copy()
        self.all_df = df  # keep full for lookback
        print(f"Loaded {len(self.df)} trading days for {self.symbol}", file=sys.stderr)

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
                expiry = pd.Timestamp(t.expiry_date)
                dte = (expiry - date).days
                T = max(dte / 365.0, 0.001)
                current_premium = bs_put_price(S, t.strike, T, self.r, iv)

                # Profit target: bought back at 50% of credit
                if current_premium <= t.premium * 0.50:
                    t.exit_date = date_str
                    t.exit_premium = current_premium
                    t.pnl = (t.premium - current_premium) * 100 * t.contracts
                    t.reason = "profit_target_50pct"
                    self.cash -= current_premium * 100 * t.contracts  # buy back
                    self.trades.append(t)
                    continue

                # Stop loss: 200% of credit
                if current_premium >= t.premium * 3.0:
                    t.exit_date = date_str
                    t.exit_premium = current_premium
                    t.pnl = (t.premium - current_premium) * 100 * t.contracts
                    t.reason = "stop_loss_200pct"
                    self.cash -= current_premium * 100 * t.contracts  # buy back
                    self.trades.append(t)
                    continue

                # Expiry
                if dte <= 0:
                    intrinsic = max(t.strike - S, 0)
                    t.exit_date = date_str
                    t.exit_premium = intrinsic
                    t.pnl = (t.premium - intrinsic) * 100 * t.contracts
                    t.reason = "expired_otm" if intrinsic == 0 else "expired_itm"
                    self.cash -= intrinsic * 100 * t.contracts  # assignment cost
                    self.trades.append(t)
                    continue

                still_open.append(t)
            open_trades = still_open

            # Entry: red day, RSI < 40, room for position
            if row["RedDay"] and rsi < 40 and len(open_trades) < max_positions:
                target_dte = 35
                expiry_date = (date + timedelta(days=target_dte)).strftime("%Y-%m-%d")
                T = target_dte / 365.0
                strike = find_strike_for_delta(S, T, self.r, iv, 0.25, "put")

                # Margin: max(20% underlying - OTM, 10% strike) * 100 + premium
                otm_amount = max(S - strike, 0)
                margin_req = max(0.20 * S - otm_amount, 0.10 * strike) * 100
                if margin_req > self.cash * 0.80:
                    continue

                premium = bs_put_price(S, strike, T, self.r, iv)
                if premium < 0.10:  # skip tiny premiums
                    continue

                trade = Trade(
                    strategy="naked_put",
                    entry_date=date_str,
                    expiry_date=expiry_date,
                    strike=strike,
                    option_type="put",
                    premium=round(premium, 2),
                    contracts=1,
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
            current_premium = bs_put_price(S, t.strike, T, self.r, iv)
            t.exit_date = df.index[-1].strftime("%Y-%m-%d")
            t.exit_premium = round(current_premium, 2)
            t.pnl = round((t.premium - current_premium) * 100 * t.contracts, 2)
            t.reason = "end_of_backtest"
            self.cash -= current_premium * 100 * t.contracts
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

                if open_trade.option_type == "put":
                    current = bs_put_price(S, open_trade.strike, T, self.r, iv)

                    # 50% profit
                    if current <= open_trade.premium * 0.50:
                        open_trade.exit_date = date_str
                        open_trade.exit_premium = round(current, 2)
                        open_trade.pnl = round((open_trade.premium - current) * 100, 2)
                        open_trade.reason = "profit_target"
                        self.cash += open_trade.pnl
                        self.trades.append(open_trade)
                        open_trade = None
                        self.equity_curve.append(self.cash + shares * S)
                        continue

                    # Expiry
                    if dte <= 0:
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
                        self.equity_curve.append(self.cash + shares * S)
                        continue

                elif open_trade.option_type == "call":
                    current = bs_call_price(S, open_trade.strike, T, self.r, iv)

                    if current <= open_trade.premium * 0.50:
                        open_trade.exit_date = date_str
                        open_trade.exit_premium = round(current, 2)
                        open_trade.pnl = round((open_trade.premium - current) * 100, 2)
                        open_trade.reason = "profit_target"
                        self.cash += open_trade.pnl
                        self.trades.append(open_trade)
                        open_trade = None
                        self.equity_curve.append(self.cash + shares * S)
                        continue

                    if dte <= 0:
                        if S >= open_trade.strike:
                            # Called away
                            open_trade.exit_date = date_str
                            open_trade.pnl = round(
                                (open_trade.strike - cost_basis + open_trade.premium) * 100, 2
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
                strike = find_strike_for_delta(S, T, self.r, iv, 0.25, "put")
                strike = min(strike, affordable_strike)
                premium = bs_put_price(S, strike, T, self.r, iv)
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
                )
                self.cash += open_trade.net_credit()

            elif state == "selling_calls" and shares >= 100:
                target_dte = 35
                T = target_dte / 365.0
                expiry_date = (date + timedelta(days=target_dte)).strftime("%Y-%m-%d")
                # Sell call above cost basis
                strike = find_strike_for_delta(S, T, self.r, iv, 0.30, "call")
                strike = max(strike, cost_basis * 1.02)  # at least break even
                premium = bs_call_price(S, strike, T, self.r, iv)
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
                current = bs_put_price(S, open_trade.strike, T, self.r, iv)
            else:
                current = bs_call_price(S, open_trade.strike, T, self.r, iv)
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
                expiry = pd.Timestamp(t.expiry_date)
                dte = (expiry - date).days
                T = max(dte / 365.0, 0.001)

                short_val = bs_put_price(S, t.strike, T, self.r, iv)
                long_val = bs_put_price(S, t.long_strike, T, self.r, iv)
                spread_val = short_val - long_val
                initial_credit = t.premium - (t.long_premium or 0)

                if spread_val <= initial_credit * 0.50:
                    t.exit_date = date_str
                    t.exit_premium = round(spread_val, 2)
                    t.pnl = round((initial_credit - spread_val) * 100 * t.contracts, 2)
                    t.reason = "profit_target_50pct"
                    self.cash += t.pnl
                    self.trades.append(t)
                    continue

                # Max loss = spread width - credit
                max_loss = spread_width - initial_credit
                if spread_val >= initial_credit + max_loss * 0.75:
                    t.exit_date = date_str
                    t.exit_premium = round(spread_val, 2)
                    t.pnl = round((initial_credit - spread_val) * 100 * t.contracts, 2)
                    t.reason = "stop_loss"
                    self.cash += t.pnl
                    self.trades.append(t)
                    continue

                if dte <= 0:
                    short_itm = max(t.strike - S, 0)
                    long_itm = max(t.long_strike - S, 0)
                    net = short_itm - long_itm
                    t.exit_date = date_str
                    t.pnl = round((initial_credit - net) * 100 * t.contracts, 2)
                    t.reason = "expired"
                    self.cash += t.pnl
                    self.trades.append(t)
                    continue

                still_open.append(t)
            open_trades = still_open

            # Entry
            if row["RedDay"] and rsi < 45 and len(open_trades) < max_positions:
                target_dte = 35
                T = target_dte / 365.0
                expiry_date = (date + timedelta(days=target_dte)).strftime("%Y-%m-%d")

                short_strike = find_strike_for_delta(S, T, self.r, iv, 0.25, "put")
                long_strike = short_strike - spread_width

                short_prem = bs_put_price(S, short_strike, T, self.r, iv)
                long_prem = bs_put_price(S, long_strike, T, self.r, iv)
                net_credit = short_prem - long_prem

                if net_credit < 0.15:
                    continue

                # Max risk per spread
                max_risk = (spread_width - net_credit) * 100
                if max_risk > self.cash * 0.15:
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
            short_val = bs_put_price(S, t.strike, T, self.r, iv)
            long_val = bs_put_price(S, t.long_strike, T, self.r, iv)
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
