"""Premium screener — find the best options to sell based on composite scoring.

Ranks opportunities by a 0-100 "sell score" combining:
  IV percentile (25 pts), POP (20 pts), yield (15 pts), liquidity (15 pts),
  term structure (10 pts), strike vs expected move (10 pts), earnings safety (5 pts).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import numpy as np
import yfinance as yf
from pydantic import BaseModel, Field

from advisor.core.enums import OptionType
from advisor.core.pricing import bsm_price
from advisor.market.iv_analysis import (
    IVPercentileResult,
    TermStructureResult,
    classify_term_structure,
    compute_expected_move,
    compute_iv_percentile,
    get_next_earnings_date,
)
from advisor.market.options_scanner import _filter_expirations, _get_price

logger = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────


class LiquidityScore(BaseModel):
    total: int = 0  # 0-100
    volume_pts: int = 0  # 0-30
    oi_pts: int = 0  # 0-30
    spread_pts: int = 0  # 0-25
    vol_oi_pts: int = 0  # 0-15


class PremiumOpportunity(BaseModel):
    symbol: str
    strategy: str  # "naked_put" or "put_credit_spread"
    strike: float
    long_strike: float | None = None  # spreads only
    expiry: date
    dte: int
    credit: float  # mid price (per share)
    bid: float
    ask: float
    delta: float = 0.0
    iv: float = 0.0
    pop: float = 0.0  # probability of profit, 0-1
    iv_percentile: float = 0.0  # 0-100
    annualized_yield: float = 0.0
    expected_move: float = 0.0
    strike_vs_em: float = 0.0  # how many expected moves OTM
    liquidity: LiquidityScore = Field(default_factory=LiquidityScore)
    term_structure: str = "flat"
    earnings_date: date | None = None
    earnings_days: int | None = None  # days until earnings
    margin_req: float = 0.0
    max_loss: float = 0.0  # spreads: width - credit; naked: margin
    sell_score: float = 0.0  # 0-100 composite
    flags: list[str] = Field(default_factory=list)


class PremiumScanResult(BaseModel):
    scanned_at: datetime = Field(default_factory=datetime.now)
    regime: str = "normal"
    target_delta: float = 0.28
    account_size: float = 5000.0
    tickers_scanned: int = 0
    opportunities: list[PremiumOpportunity] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ── Liquidity scoring ─────────────────────────────────────────────────────────


def score_liquidity(volume: int, oi: int, bid: float, ask: float) -> LiquidityScore:
    """Score option liquidity on a 0-100 scale.

    Components:
      - Volume (0-30): 0 at 0, 30 at 500+
      - Open Interest (0-30): 0 at 0, 30 at 2000+
      - Spread tightness (0-25): based on (ask-bid)/mid
      - Vol/OI ratio (0-15): 0 at 0, 15 at 0.5+
    """
    # Volume: 0-30
    volume_pts = min(30, int(volume / 500 * 30)) if volume > 0 else 0

    # Open Interest: 0-30
    oi_pts = min(30, int(oi / 2000 * 30)) if oi > 0 else 0

    # Spread tightness: 0-25
    spread_pts = 0
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2
        if mid > 0:
            spread_pct = (ask - bid) / mid
            if spread_pct <= 0.05:
                spread_pts = 25
            elif spread_pct <= 0.10:
                spread_pts = 20
            elif spread_pct <= 0.15:
                spread_pts = 15
            elif spread_pct <= 0.25:
                spread_pts = 10
            elif spread_pct <= 0.40:
                spread_pts = 5

    # Vol/OI ratio: 0-15
    vol_oi_pts = 0
    if oi > 0 and volume > 0:
        ratio = volume / oi
        vol_oi_pts = min(15, int(ratio / 0.5 * 15))

    total = volume_pts + oi_pts + spread_pts + vol_oi_pts
    return LiquidityScore(
        total=min(100, total),
        volume_pts=volume_pts,
        oi_pts=oi_pts,
        spread_pts=spread_pts,
        vol_oi_pts=vol_oi_pts,
    )


# ── Adaptive delta ────────────────────────────────────────────────────────────


def get_adaptive_delta(iv_percentile: float) -> float:
    """Choose target delta based on IV environment.

    High IV (>75th pctile) -> wider strikes (0.35 delta) to capture more premium.
    Normal (25-75) -> standard 0.28 delta.
    Low IV (<25th) -> tighter strikes (0.16 delta) since premium is thin.
    """
    if iv_percentile >= 75:
        return 0.35
    elif iv_percentile >= 25:
        return 0.28
    else:
        return 0.16


def get_regime_name() -> str:
    """Try to detect market regime via fitted HMM, fallback to VIX heuristic."""
    try:
        from advisor.ml.regime import RegimeDetector

        if RegimeDetector.model_exists():
            detector = RegimeDetector.load()
            result = detector.detect_regime()
            return result["regime_name"]
    except Exception:
        pass

    # VIX fallback
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if not hist.empty:
            vix_level = float(hist["Close"].iloc[-1])
            if vix_level >= 25:
                return "high_vol"
            elif vix_level <= 15:
                return "low_vol"
    except Exception:
        pass

    return "normal"


# ── Sell score ────────────────────────────────────────────────────────────────


def compute_sell_score(opp: PremiumOpportunity) -> float:
    """Compute the composite sell score (0-100) for a premium opportunity.

    Weights:
      IV Percentile:           25 pts
      Probability of Profit:   20 pts
      Annualized Yield:        15 pts
      Liquidity:               15 pts
      Term Structure:          10 pts
      Strike vs Expected Move: 10 pts
      Earnings Safety:          5 pts
    """
    score = 0.0

    # IV Percentile (25 pts): linear scale 0-100 -> 0-25
    score += min(25.0, opp.iv_percentile / 100 * 25)

    # POP (20 pts): 60% POP = 0, 90% POP = 20 (linear in [0.60, 0.90])
    if opp.pop >= 0.90:
        score += 20.0
    elif opp.pop >= 0.60:
        score += (opp.pop - 0.60) / 0.30 * 20.0

    # Annualized Yield (15 pts): 0% = 0, 200%+ = 15
    ann_yield_pct = (
        opp.annualized_yield * 100 if opp.annualized_yield <= 10 else opp.annualized_yield
    )
    score += min(15.0, ann_yield_pct / 200 * 15)

    # Liquidity (15 pts): direct scale from 0-100 -> 0-15
    score += opp.liquidity.total / 100 * 15

    # Term Structure (10 pts): contango = 10, flat = 5, backwardation = 0
    if opp.term_structure == "contango":
        score += 10.0
    elif opp.term_structure == "flat":
        score += 5.0

    # Strike vs Expected Move (10 pts): >1.5 EM = 10, 1.0 EM = 5, <0.5 EM = 0
    if opp.strike_vs_em >= 1.5:
        score += 10.0
    elif opp.strike_vs_em >= 0.5:
        score += (opp.strike_vs_em - 0.5) / 1.0 * 10.0

    # Earnings Safety (5 pts): no earnings in window = 5, earnings close = 0
    if opp.earnings_days is None or opp.earnings_days > 14:
        score += 5.0
    elif opp.earnings_days > 7:
        score += 2.5

    return round(score, 1)


# ── Main screener ─────────────────────────────────────────────────────────────


class PremiumScreener:
    """Scan tickers for premium-selling opportunities with composite scoring."""

    def __init__(
        self,
        account_size: float = 5000.0,
        min_iv_pctile: float = 30.0,
        strategies: list[str] | None = None,
        min_dte: int = 25,
        max_dte: int = 45,
        earnings_buffer: int = 7,
        top_n: int = 15,
        tt_data: dict | None = None,
    ):
        self.account_size = account_size
        self.min_iv_pctile = min_iv_pctile
        self.strategies = strategies or ["naked_put", "put_credit_spread"]
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.earnings_buffer = earnings_buffer
        self.top_n = top_n
        self.tt_data = tt_data

    def scan(self, tickers: list[str]) -> PremiumScanResult:
        """Run the full premium screening pipeline."""
        regime = get_regime_name()
        today = date.today()
        target_min = today + timedelta(days=self.min_dte)
        target_max = today + timedelta(days=self.max_dte)

        result = PremiumScanResult(
            regime=regime,
            account_size=self.account_size,
        )

        # Phase 1 & 2: Per-ticker IV analysis + filtering
        qualified: list[dict] = []
        for symbol in tickers:
            try:
                ticker = yf.Ticker(symbol.upper())
                price = _get_price(ticker)
                if price is None or price <= 0:
                    result.errors.append(f"{symbol}: no price data")
                    continue

                # IV percentile
                iv_result = compute_iv_percentile(symbol, tt_data=self.tt_data, ticker=ticker)

                # Filter: minimum IV percentile
                if iv_result.iv_percentile < self.min_iv_pctile:
                    logger.debug(
                        f"{symbol}: IV percentile {iv_result.iv_percentile:.0f} "
                        f"< threshold {self.min_iv_pctile}"
                    )
                    continue

                # Earnings date
                earnings_date = get_next_earnings_date(symbol, ticker=ticker)

                # Filter: no earnings within DTE window + buffer
                if earnings_date is not None:
                    days_to_earnings = (earnings_date - today).days
                    if 0 < days_to_earnings <= self.max_dte + self.earnings_buffer:
                        logger.debug(
                            f"{symbol}: earnings in {days_to_earnings}d, "
                            f"within DTE window + buffer"
                        )
                        # Don't skip entirely — flag it instead
                        pass

                # Term structure
                ts_result = classify_term_structure(symbol, ticker=ticker, price=price)

                result.tickers_scanned += 1
                qualified.append(
                    {
                        "symbol": symbol.upper(),
                        "ticker": ticker,
                        "price": price,
                        "iv_result": iv_result,
                        "earnings_date": earnings_date,
                        "ts_result": ts_result,
                    }
                )

            except Exception as e:
                result.errors.append(f"{symbol}: {e}")
                logger.warning(f"Error in phase 1 for {symbol}: {e}")

        # Phase 3 & 4: Chain scan + enrichment
        for q in qualified:
            try:
                self._scan_ticker(q, target_min, target_max, today, result)
            except Exception as e:
                result.errors.append(f"{q['symbol']}: chain scan failed: {e}")
                logger.warning(f"Chain scan error for {q['symbol']}: {e}")

        # Phase 5: Score and rank
        for opp in result.opportunities:
            opp.sell_score = compute_sell_score(opp)

        result.opportunities.sort(key=lambda o: o.sell_score, reverse=True)
        result.opportunities = result.opportunities[: self.top_n]

        # Set target delta from first qualifying ticker's IV percentile
        if qualified:
            avg_pctile = np.mean([q["iv_result"].iv_percentile for q in qualified])
            result.target_delta = get_adaptive_delta(avg_pctile)

        return result

    def _scan_ticker(
        self,
        q: dict,
        target_min: date,
        target_max: date,
        today: date,
        result: PremiumScanResult,
    ) -> None:
        """Scan a single ticker's option chains for opportunities."""
        symbol = q["symbol"]
        ticker = q["ticker"]
        price = q["price"]
        iv_result: IVPercentileResult = q["iv_result"]
        ts_result: TermStructureResult = q["ts_result"]
        earnings_date = q["earnings_date"]

        expirations = ticker.options
        if not expirations:
            return

        valid_expiries = _filter_expirations(expirations, target_min, target_max)
        if not valid_expiries:
            return

        target_delta = get_adaptive_delta(iv_result.iv_percentile)
        current_iv = iv_result.current_iv if iv_result.current_iv > 0 else 0.30

        for exp_str in valid_expiries:
            exp_date = date.fromisoformat(exp_str)
            dte = (exp_date - today).days
            if dte <= 0:
                continue

            try:
                chain = ticker.option_chain(exp_str)
            except Exception:
                continue

            puts = chain.puts
            if puts.empty:
                continue

            em = compute_expected_move(price, current_iv, dte)

            # Scan naked puts
            if "naked_put" in self.strategies:
                self._scan_naked_puts(
                    puts,
                    symbol,
                    price,
                    exp_date,
                    dte,
                    target_delta,
                    current_iv,
                    em,
                    iv_result,
                    ts_result,
                    earnings_date,
                    today,
                    result,
                )

            # Scan put credit spreads
            if "put_credit_spread" in self.strategies:
                self._scan_put_spreads(
                    puts,
                    symbol,
                    price,
                    exp_date,
                    dte,
                    target_delta,
                    current_iv,
                    em,
                    iv_result,
                    ts_result,
                    earnings_date,
                    today,
                    result,
                )

    def _scan_naked_puts(
        self,
        puts,
        symbol: str,
        price: float,
        exp_date: date,
        dte: int,
        target_delta: float,
        current_iv: float,
        expected_move: float,
        iv_result: IVPercentileResult,
        ts_result: TermStructureResult,
        earnings_date: date | None,
        today: date,
        result: PremiumScanResult,
    ) -> None:
        """Find qualifying naked put candidates with enrichment."""
        T = dte / 365.0
        r = 0.05  # risk-free rate assumption

        for _, row in puts.iterrows():
            strike = float(row["strike"])
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)

            if bid < 0.10 or strike >= price:
                continue

            otm_pct = (price - strike) / price
            if not (0.03 <= otm_pct <= 0.25):
                continue

            # Spread filter
            if ask > 0 and (ask - bid) / ask > 0.40:
                continue

            mid = (bid + ask) / 2
            iv = float(row.get("impliedVolatility", 0) or 0) or current_iv
            volume = int(row.get("volume", 0) or 0)
            oi = int(row.get("openInterest", 0) or 0)

            # Delta via BSM
            bsm = bsm_price(price, strike, T, r, iv, OptionType.PUT)
            delta = abs(bsm.delta)

            # Filter: only consider strikes near target delta
            if delta > target_delta + 0.15 or delta < max(0.05, target_delta - 0.15):
                continue

            # POP = 1 - |delta|
            pop = 1.0 - delta

            # Margin: max(20% underlying - OTM amount, 10% strike) * 100 + premium
            otm_amount = price - strike
            premium_100 = mid * 100
            margin_req = max(0.20 * price - otm_amount, 0.10 * strike) * 100 + premium_100

            ann_yield = (
                (premium_100 / margin_req) * (365 / dte) if margin_req > 0 and dte > 0 else 0
            )

            # Strike vs expected move
            strike_vs_em = otm_amount / expected_move if expected_move > 0 else 0

            # Liquidity
            liq = score_liquidity(volume, oi, bid, ask)

            # Earnings proximity
            earnings_days = None
            if earnings_date is not None:
                earnings_days = (earnings_date - today).days

            # Flags
            flags = []
            if liq.total < 50:
                flags.append("low-liq")
            if earnings_days is not None and 0 < earnings_days <= dte:
                flags.append("earnings")
            if margin_req > self.account_size * 0.20:
                flags.append("margin")

            result.opportunities.append(
                PremiumOpportunity(
                    symbol=symbol,
                    strategy="naked_put",
                    strike=strike,
                    expiry=exp_date,
                    dte=dte,
                    credit=mid,
                    bid=bid,
                    ask=ask,
                    delta=round(delta, 3),
                    iv=round(iv, 4),
                    pop=round(pop, 3),
                    iv_percentile=iv_result.iv_percentile,
                    annualized_yield=round(ann_yield, 4),
                    expected_move=round(expected_move, 2),
                    strike_vs_em=round(strike_vs_em, 2),
                    liquidity=liq,
                    term_structure=ts_result.classification,
                    earnings_date=earnings_date,
                    earnings_days=earnings_days,
                    margin_req=round(margin_req, 2),
                    max_loss=round(margin_req, 2),
                    flags=flags,
                )
            )

    def _scan_put_spreads(
        self,
        puts,
        symbol: str,
        price: float,
        exp_date: date,
        dte: int,
        target_delta: float,
        current_iv: float,
        expected_move: float,
        iv_result: IVPercentileResult,
        ts_result: TermStructureResult,
        earnings_date: date | None,
        today: date,
        result: PremiumScanResult,
    ) -> None:
        """Find qualifying put credit spread candidates with enrichment."""
        T = dte / 365.0
        r = 0.05

        # Collect OTM put candidates
        candidates = []
        for _, row in puts.iterrows():
            strike = float(row["strike"])
            if strike >= price:
                continue
            otm_pct = (price - strike) / price
            if not (0.03 <= otm_pct <= 0.30):
                continue
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            iv = float(row.get("impliedVolatility", 0) or 0) or current_iv
            volume = int(row.get("volume", 0) or 0)
            oi = int(row.get("openInterest", 0) or 0)
            candidates.append((strike, bid, ask, iv, volume, oi))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for i in range(len(candidates) - 1):
            short_strike, short_bid, _, short_iv, short_vol, short_oi = candidates[i]
            long_strike, _, long_ask, _, long_vol, long_oi = candidates[i + 1]

            width = short_strike - long_strike
            if width <= 0:
                continue

            net_credit = short_bid - long_ask
            if net_credit <= 0.05:
                continue

            # Short delta
            bsm_short = bsm_price(price, short_strike, T, r, short_iv, OptionType.PUT)
            short_delta = abs(bsm_short.delta)

            # Filter by target delta
            if short_delta > target_delta + 0.15 or short_delta < max(0.05, target_delta - 0.15):
                continue

            # POP = 1 - |short delta|
            pop = 1.0 - short_delta

            max_loss = (width - net_credit) * 100
            credit_100 = net_credit * 100
            ror = credit_100 / max_loss if max_loss > 0 else 0
            ann_yield = ror * (365 / dte) if dte > 0 else 0

            mid_credit = net_credit  # already using bid/ask
            otm_amount = price - short_strike
            strike_vs_em = otm_amount / expected_move if expected_move > 0 else 0

            # Liquidity: average of both legs
            avg_vol = (short_vol + long_vol) // 2
            avg_oi = (short_oi + long_oi) // 2
            liq = score_liquidity(avg_vol, avg_oi, short_bid, long_ask + width)

            earnings_days = None
            if earnings_date is not None:
                earnings_days = (earnings_date - today).days

            flags = []
            if liq.total < 50:
                flags.append("low-liq")
            if earnings_days is not None and 0 < earnings_days <= dte:
                flags.append("earnings")
            if max_loss > self.account_size * 0.20:
                flags.append("margin")

            result.opportunities.append(
                PremiumOpportunity(
                    symbol=symbol,
                    strategy="put_credit_spread",
                    strike=short_strike,
                    long_strike=long_strike,
                    expiry=exp_date,
                    dte=dte,
                    credit=round(mid_credit, 2),
                    bid=short_bid,
                    ask=long_ask,
                    delta=round(short_delta, 3),
                    iv=round(short_iv, 4),
                    pop=round(pop, 3),
                    iv_percentile=iv_result.iv_percentile,
                    annualized_yield=round(ann_yield, 4),
                    expected_move=round(expected_move, 2),
                    strike_vs_em=round(strike_vs_em, 2),
                    liquidity=liq,
                    term_structure=ts_result.classification,
                    earnings_date=earnings_date,
                    earnings_days=earnings_days,
                    margin_req=round(max_loss, 2),
                    max_loss=round(max_loss, 2),
                    flags=flags,
                )
            )
