"""Mispricing screener — detects stocks where market price diverges from fair value.

Three signal layers scored 0-100:
  1. Fundamental mispricing (Piotroski F-Score + relative valuation) — 0-40 pts
  2. Options market signal (IV vs realized vol) — 0-30 pts
  3. Earnings estimate revisions (pre-PEAD) — 0-30 pts
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

import numpy as np
import yfinance as yf
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CACHE_DIR = Path("/tmp/advisor_mispricing_cache")

# TTLs per layer
_TTL_FUNDAMENTAL = 24 * 3600  # 24h
_TTL_OPTIONS = 4 * 3600  # 4h
_TTL_ESTIMATES = 12 * 3600  # 12h
_TTL_SECTOR = 24 * 3600  # 24h


# ── Models ───────────────────────────────────────────────────────────────────


class MispricingSignal(StrEnum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WATCH = "WATCH"
    HOLD = "HOLD"


class FundamentalMispricingScore(BaseModel):
    score: float = 0.0
    f_score: int = 0
    f_score_details: dict = Field(default_factory=dict)
    pe_ratio: float | None = None
    sector_pe: float | None = None
    pb_ratio: float | None = None
    sector_pb: float | None = None
    ev_ebitda: float | None = None
    sector_ev_ebitda: float | None = None
    discount_pct: float | None = None


class OptionsMarketScore(BaseModel):
    score: float = 0.0
    implied_vol: float | None = None
    realized_vol_20d: float | None = None
    iv_rv_ratio: float | None = None
    iv_rank: float | None = None  # 0-100, IV rank across multiple expirations
    put_call_oi_ratio: float | None = None
    skew_pct: float | None = None
    notable_strikes: list[str] = Field(default_factory=list)


class EstimateRevisionScore(BaseModel):
    score: float = 0.0
    current_price: float | None = None
    target_price: float | None = None
    upside_pct: float | None = None
    recent_upgrades: int = 0
    recent_downgrades: int = 0
    recommendation_mean: float | None = None
    earnings_growth_est: float | None = None


class MispricingResult(BaseModel):
    symbol: str
    sector: str = ""
    total_score: float = 0.0
    signal: MispricingSignal = MispricingSignal.HOLD
    fundamental: FundamentalMispricingScore = Field(default_factory=FundamentalMispricingScore)
    options_market: OptionsMarketScore = Field(default_factory=OptionsMarketScore)
    estimate_revisions: EstimateRevisionScore = Field(default_factory=EstimateRevisionScore)
    scanned_at: datetime = Field(default_factory=datetime.now)


# ── Cache helpers ────────────────────────────────────────────────────────────


def _cache_key(prefix: str, key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:24]
    return CACHE_DIR / f"{prefix}_{h}.json"


def _cache_get(prefix: str, key: str, ttl: int | None = None) -> dict | None:
    p = _cache_key(prefix, key)
    effective_ttl = ttl or _TTL_FUNDAMENTAL
    if p.exists() and (time.time() - p.stat().st_mtime) < effective_ttl:
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _cache_set(prefix: str, key: str, data: dict) -> None:
    """Atomic cache write — write to temp file then rename."""
    try:
        p = _cache_key(prefix, key)
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        closed = False
        try:
            os.write(fd, json.dumps(data, default=str).encode())
            os.close(fd)
            closed = True
            os.replace(tmp, p)
        except Exception:
            if not closed:
                os.close(fd)
            Path(tmp).unlink(missing_ok=True)
            raise
    except Exception as e:
        logger.debug(f"Cache write failed: {e}")


# ── Shared yfinance helper ──────────────────────────────────────────────────

# Token-bucket rate limiter: allows up to _YF_BURST concurrent calls while
# keeping average rate at ~1/_YF_MIN_INTERVAL calls/sec.  Unlike the old
# lock-and-sleep approach, threads only hold the lock long enough to grab a
# slot, then sleep *outside* the lock so other threads can do non-API work.
_YF_MIN_INTERVAL = 0.25  # seconds between calls (avg ~4 req/s)
_yf_lock = threading.Lock()
_yf_next_slot = 0.0


def _yf_throttle() -> None:
    """Reserve a time-slot for the next yfinance call, sleep only if needed."""
    global _yf_next_slot
    with _yf_lock:
        now = time.time()
        if _yf_next_slot <= now:
            _yf_next_slot = now + _YF_MIN_INTERVAL
            return  # no wait needed
        wait_until = _yf_next_slot
        _yf_next_slot += _YF_MIN_INTERVAL

    # Sleep outside the lock so other threads can claim later slots
    delay = wait_until - time.time()
    if delay > 0:
        time.sleep(delay)


# ── Helper: safe float from info dict ────────────────────────────────────────


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


# ── 1. Fundamental Mispricing ────────────────────────────────────────────────


def _compute_piotroski(ticker: yf.Ticker) -> tuple[int, dict]:
    """Compute Piotroski F-Score (0-9) from yfinance data."""
    details: dict[str, bool] = {}
    score = 0

    try:
        financials = ticker.financials
        balance = ticker.balance_sheet
        cashflow = ticker.cashflow

        # Need at least 2 years of data for YoY comparisons
        if financials is None or financials.empty or balance is None or balance.empty:
            return 0, details

        # Helper to get a row value for a given column index
        def _get(df, row_name, col_idx=0):
            for name in [row_name] if isinstance(row_name, str) else row_name:
                if name in df.index:
                    try:
                        return float(df.iloc[df.index.get_loc(name), col_idx])
                    except (ValueError, TypeError, IndexError):
                        pass
            return None

        # 1. Positive net income
        ni = _get(financials, ["Net Income", "Net Income From Continuing Operations"])
        if ni is not None and ni > 0:
            details["positive_net_income"] = True
            score += 1
        else:
            details["positive_net_income"] = False

        # 2. Positive operating cash flow
        ocf = None
        if cashflow is not None and not cashflow.empty:
            ocf = _get(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        if ocf is not None and ocf > 0:
            details["positive_ocf"] = True
            score += 1
        else:
            details["positive_ocf"] = False

        # 3. ROA increasing YoY
        total_assets_0 = _get(balance, "Total Assets", 0)
        total_assets_1 = _get(balance, "Total Assets", 1) if balance.shape[1] > 1 else None
        ni_1 = (
            _get(financials, ["Net Income", "Net Income From Continuing Operations"], 1)
            if financials.shape[1] > 1
            else None
        )
        roa_0 = (ni / total_assets_0) if (ni is not None and total_assets_0) else None
        roa_1 = (ni_1 / total_assets_1) if (ni_1 is not None and total_assets_1) else None
        if roa_0 is not None and roa_1 is not None and roa_0 > roa_1:
            details["roa_increasing"] = True
            score += 1
        else:
            details["roa_increasing"] = False

        # 4. OCF > Net Income (accrual quality)
        if ocf is not None and ni is not None and ocf > ni:
            details["accrual_quality"] = True
            score += 1
        else:
            details["accrual_quality"] = False

        # 5. Long-term debt ratio decreasing
        ltd_0 = _get(balance, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0)
        ltd_1 = (
            _get(balance, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 1)
            if balance.shape[1] > 1
            else None
        )
        if (
            ltd_0 is not None
            and ltd_1 is not None
            and total_assets_0 is not None
            and total_assets_0 != 0
            and total_assets_1 is not None
            and total_assets_1 != 0
        ):
            ratio_0 = ltd_0 / total_assets_0
            ratio_1 = ltd_1 / total_assets_1
            details["debt_decreasing"] = ratio_0 < ratio_1
            if details["debt_decreasing"]:
                score += 1
        elif ltd_0 is not None and ltd_0 == 0:
            details["debt_decreasing"] = True
            score += 1
        else:
            details["debt_decreasing"] = False

        # 6. Current ratio increasing
        ca_0 = _get(balance, ["Current Assets", "Total Current Assets"], 0)
        cl_0 = _get(balance, ["Current Liabilities", "Total Current Liabilities"], 0)
        ca_1 = (
            _get(balance, ["Current Assets", "Total Current Assets"], 1)
            if balance.shape[1] > 1
            else None
        )
        cl_1 = (
            _get(balance, ["Current Liabilities", "Total Current Liabilities"], 1)
            if balance.shape[1] > 1
            else None
        )
        cr_0 = (ca_0 / cl_0) if (ca_0 is not None and cl_0) else None
        cr_1 = (ca_1 / cl_1) if (ca_1 is not None and cl_1) else None
        if cr_0 is not None and cr_1 is not None and cr_0 > cr_1:
            details["current_ratio_up"] = True
            score += 1
        else:
            details["current_ratio_up"] = False

        # 7. No new shares issued
        shares_0 = _get(
            balance,
            ["Ordinary Shares Number", "Common Stock Shares Outstanding", "Share Issued"],
            0,
        )
        shares_1 = (
            _get(
                balance,
                ["Ordinary Shares Number", "Common Stock Shares Outstanding", "Share Issued"],
                1,
            )
            if balance.shape[1] > 1
            else None
        )
        if shares_0 is not None and shares_1 is not None and shares_0 <= shares_1:
            details["no_dilution"] = True
            score += 1
        else:
            details["no_dilution"] = shares_0 is not None and shares_1 is None

        # 8. Gross margin increasing
        gp_0 = _get(financials, "Gross Profit", 0)
        rev_0 = _get(financials, ["Total Revenue", "Revenue"], 0)
        gp_1 = _get(financials, "Gross Profit", 1) if financials.shape[1] > 1 else None
        rev_1 = (
            _get(financials, ["Total Revenue", "Revenue"], 1) if financials.shape[1] > 1 else None
        )
        gm_0 = (gp_0 / rev_0) if (gp_0 is not None and rev_0) else None
        gm_1 = (gp_1 / rev_1) if (gp_1 is not None and rev_1) else None
        if gm_0 is not None and gm_1 is not None and gm_0 > gm_1:
            details["gross_margin_up"] = True
            score += 1
        else:
            details["gross_margin_up"] = False

        # 9. Asset turnover increasing
        at_0 = (rev_0 / total_assets_0) if (rev_0 is not None and total_assets_0) else None
        rev_1_val = (
            _get(financials, ["Total Revenue", "Revenue"], 1) if financials.shape[1] > 1 else None
        )
        at_1 = (rev_1_val / total_assets_1) if (rev_1_val is not None and total_assets_1) else None
        if at_0 is not None and at_1 is not None and at_0 > at_1:
            details["asset_turnover_up"] = True
            score += 1
        else:
            details["asset_turnover_up"] = False

    except Exception as e:
        logger.warning(f"Piotroski computation failed: {e}")

    return score, details


def _fscore_to_points(f_score: int) -> float:
    """Convert Piotroski F-Score to point allocation (0-20)."""
    if f_score >= 8:
        return 20.0
    elif f_score == 7:
        return 15.0
    elif f_score == 6:
        return 10.0
    elif f_score == 5:
        return 5.0
    return 0.0


def _get_sector_medians(sector: str) -> dict:
    """Get sector median P/E, P/B, EV/EBITDA from S&P 500 sector peers.

    Uses Wikipedia GICS sector data to target only tickers in the same sector,
    then samples up to 20 of those for valuation metrics.
    """
    cached = _cache_get("sector_medians", sector, ttl=_TTL_SECTOR)
    if cached:
        return cached

    from advisor.confluence.smart_money_screener import get_sp500_by_sector

    try:
        import random

        by_sector = get_sp500_by_sector()
        # Find the matching sector (case-insensitive partial match)
        sector_tickers: list[str] = []
        sector_lower = sector.lower()
        for s, tickers in by_sector.items():
            if s.lower() == sector_lower or sector_lower in s.lower():
                sector_tickers = tickers
                break

        if not sector_tickers:
            logger.debug(f"No S&P 500 tickers found for sector '{sector}'")
            result = {"pe": None, "pb": None, "ev_ebitda": None, "sample_size": 0}
            _cache_set("sector_medians", sector, result)
            return result

        # Sample up to 20 known sector peers (not random from full universe)
        sample = random.sample(sector_tickers, min(20, len(sector_tickers)))

        pe_vals, pb_vals, ev_vals = [], [], []
        for sym in sample:
            try:
                _yf_throttle()
                info = yf.Ticker(sym).info or {}
                pe = _safe_float(info.get("trailingPE"))
                pb = _safe_float(info.get("priceToBook"))
                ev = _safe_float(info.get("enterpriseToEbitda"))
                if pe is not None and 0 < pe < 200:
                    pe_vals.append(pe)
                if pb is not None and 0 < pb < 50:
                    pb_vals.append(pb)
                if ev is not None and 0 < ev < 100:
                    ev_vals.append(ev)
            except Exception:
                continue

        result = {
            "pe": float(np.median(pe_vals)) if pe_vals else None,
            "pb": float(np.median(pb_vals)) if pb_vals else None,
            "ev_ebitda": float(np.median(ev_vals)) if ev_vals else None,
            "sample_size": len(pe_vals),
        }
    except Exception as e:
        logger.warning(f"Sector median computation failed: {e}")
        result = {"pe": None, "pb": None, "ev_ebitda": None, "sample_size": 0}

    _cache_set("sector_medians", sector, result)
    return result


def _score_relative_valuation(info: dict, sector: str) -> tuple[float, dict]:
    """Score relative valuation vs sector medians (0-20 pts)."""
    medians = _get_sector_medians(sector)
    score = 0.0
    details = {
        "pe_ratio": _safe_float(info.get("trailingPE")),
        "sector_pe": medians.get("pe"),
        "pb_ratio": _safe_float(info.get("priceToBook")),
        "sector_pb": medians.get("pb"),
        "ev_ebitda": _safe_float(info.get("enterpriseToEbitda")),
        "sector_ev_ebitda": medians.get("ev_ebitda"),
    }

    pe = details["pe_ratio"]
    s_pe = details["sector_pe"]
    pb = details["pb_ratio"]
    s_pb = details["sector_pb"]
    ev = details["ev_ebitda"]
    s_ev = details["sector_ev_ebitda"]

    overvalued = False

    # P/E discount
    if pe and s_pe and pe > 0 and s_pe > 0:
        discount = (s_pe - pe) / s_pe
        if discount > 0.30:
            score += 10
        if pe > s_pe * 1.5:
            overvalued = True

    # P/B discount
    if pb and s_pb and pb > 0 and s_pb > 0:
        discount = (s_pb - pb) / s_pb
        if discount > 0.20:
            score += 5
        if pb > s_pb * 1.5:
            overvalued = True

    # EV/EBITDA discount
    if ev and s_ev and ev > 0 and s_ev > 0:
        discount = (s_ev - ev) / s_ev
        if discount > 0.25:
            score += 5
        if ev > s_ev * 1.5:
            overvalued = True

    if overvalued:
        score = min(score, 5.0)

    return min(score, 20.0), details


def _fetch_fundamental_score(
    symbol: str, ticker: yf.Ticker | None = None
) -> FundamentalMispricingScore:
    """Layer 1: Fundamental mispricing score (0-40 pts)."""
    cached = _cache_get("fundamental", symbol, ttl=_TTL_FUNDAMENTAL)
    if cached:
        return FundamentalMispricingScore(**cached)

    try:
        if ticker is None:
            _yf_throttle()
            ticker = yf.Ticker(symbol)

        info = ticker.info or {}
        sector = info.get("sector", "")

        # Piotroski F-Score (0-20 pts)
        f_score, f_details = _compute_piotroski(ticker)
        f_points = _fscore_to_points(f_score)

        # Relative valuation (0-20 pts)
        val_points, val_details = _score_relative_valuation(info, sector)

        total = f_points + val_points

        # Compute overall discount %
        pe = val_details.get("pe_ratio")
        s_pe = val_details.get("sector_pe")
        discount_pct = None
        if pe and s_pe and s_pe > 0:
            discount_pct = round(((s_pe - pe) / s_pe) * 100, 1)

        result = FundamentalMispricingScore(
            score=total,
            f_score=f_score,
            f_score_details=f_details,
            pe_ratio=val_details.get("pe_ratio"),
            sector_pe=val_details.get("sector_pe"),
            pb_ratio=val_details.get("pb_ratio"),
            sector_pb=val_details.get("sector_pb"),
            ev_ebitda=val_details.get("ev_ebitda"),
            sector_ev_ebitda=val_details.get("sector_ev_ebitda"),
            discount_pct=discount_pct,
        )
        _cache_set("fundamental", symbol, result.model_dump())
        return result

    except Exception as e:
        logger.warning(f"Fundamental score failed for {symbol}: {e}")
        return FundamentalMispricingScore()


# ── 2. Options Market Signal ─────────────────────────────────────────────────


def _fetch_options_market_score(symbol: str, ticker: yf.Ticker | None = None) -> OptionsMarketScore:
    """Layer 2: Options market signal — IV vs realized vol (0-30 pts)."""
    cached = _cache_get("options_mkt", symbol, ttl=_TTL_OPTIONS)
    if cached:
        return OptionsMarketScore(**cached)

    try:
        if ticker is None:
            _yf_throttle()
            ticker = yf.Ticker(symbol)

        # Realized vol (20-day)
        _yf_throttle()
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 30:
            return OptionsMarketScore()

        daily_returns = hist["Close"].pct_change().dropna()
        realized_vol_20d = float(daily_returns.tail(20).std() * np.sqrt(252))

        # Get options chain
        expirations = ticker.options
        if not expirations:
            return OptionsMarketScore()

        current_price = float(hist["Close"].iloc[-1])
        atm_range = current_price * 0.05

        # Use nearest expiry for primary IV and scoring
        _yf_throttle()
        chain = ticker.option_chain(expirations[0])
        calls = chain.calls
        puts = chain.puts

        # Near-ATM options for IV
        atm_calls = calls[
            (calls["strike"] >= current_price - atm_range)
            & (calls["strike"] <= current_price + atm_range)
        ]
        atm_puts = puts[
            (puts["strike"] >= current_price - atm_range)
            & (puts["strike"] <= current_price + atm_range)
        ]

        all_atm_iv = []
        if not atm_calls.empty:
            all_atm_iv.extend(atm_calls["impliedVolatility"].dropna().tolist())
        if not atm_puts.empty:
            all_atm_iv.extend(atm_puts["impliedVolatility"].dropna().tolist())

        if not all_atm_iv:
            return OptionsMarketScore()

        implied_vol = float(np.mean(all_atm_iv))
        iv_rv_ratio = implied_vol / realized_vol_20d if realized_vol_20d > 0 else None

        # IV/RV scoring (0-15 pts)
        score = 0.0
        if iv_rv_ratio is not None:
            if iv_rv_ratio < 0.7:
                score += 15
            elif iv_rv_ratio < 0.85:
                score += 10
            elif iv_rv_ratio <= 1.15:
                score += 5
            # iv_rv > 1.3 = 0 pts

        # Put/Call OI ratio
        total_call_oi = int(calls["openInterest"].fillna(0).sum())
        total_put_oi = int(puts["openInterest"].fillna(0).sum())
        pc_oi_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else None

        if pc_oi_ratio is not None and pc_oi_ratio < 0.5:
            score += 5  # bullish positioning

        # IV Rank: sample ATM IV across multiple expirations to build a range,
        # then rank the front-month IV within that range.
        # IV Rank = (front_IV - min_IV) / (max_IV - min_IV) * 100
        iv_rank = None
        term_ivs = [implied_vol]  # front-month already computed
        # Fetch up to 4 additional expirations for term-structure range
        extra_expiries = expirations[1:5]
        for exp in extra_expiries:
            try:
                _yf_throttle()
                exp_chain = ticker.option_chain(exp)
                exp_calls = exp_chain.calls[
                    (exp_chain.calls["strike"] >= current_price - atm_range)
                    & (exp_chain.calls["strike"] <= current_price + atm_range)
                ]
                exp_puts = exp_chain.puts[
                    (exp_chain.puts["strike"] >= current_price - atm_range)
                    & (exp_chain.puts["strike"] <= current_price + atm_range)
                ]
                exp_iv = []
                if not exp_calls.empty:
                    exp_iv.extend(exp_calls["impliedVolatility"].dropna().tolist())
                if not exp_puts.empty:
                    exp_iv.extend(exp_puts["impliedVolatility"].dropna().tolist())
                if exp_iv:
                    term_ivs.append(float(np.mean(exp_iv)))
            except Exception:
                continue

        if len(term_ivs) >= 3:
            iv_min = min(term_ivs)
            iv_max = max(term_ivs)
            if iv_max > iv_min:
                iv_rank = round((implied_vol - iv_min) / (iv_max - iv_min) * 100, 1)
            else:
                iv_rank = 50.0  # flat term structure
            if iv_rank < 20:
                score += 5  # front-month IV near bottom of term structure

        # Skew: OTM put IV vs OTM call IV
        skew_pct = None
        otm_puts = puts[puts["strike"] < current_price * 0.95]
        otm_calls = calls[calls["strike"] > current_price * 1.05]
        if not otm_puts.empty and not otm_calls.empty:
            avg_put_iv = float(otm_puts["impliedVolatility"].dropna().mean())
            avg_call_iv = float(otm_calls["impliedVolatility"].dropna().mean())
            if avg_call_iv > 0:
                skew_pct = round(((avg_put_iv - avg_call_iv) / avg_call_iv) * 100, 1)
                if skew_pct > 10:
                    score += 5  # fear premium

        # Notable strikes
        notable_strikes: list[str] = []
        if not atm_calls.empty:
            top = atm_calls.nlargest(3, "openInterest")
            for _, row in top.iterrows():
                notable_strikes.append(f"${row['strike']:.0f}C OI:{int(row['openInterest'])}")

        result = OptionsMarketScore(
            score=min(score, 30.0),
            implied_vol=round(implied_vol, 4),
            realized_vol_20d=round(realized_vol_20d, 4),
            iv_rv_ratio=round(iv_rv_ratio, 2) if iv_rv_ratio else None,
            iv_rank=iv_rank,
            put_call_oi_ratio=round(pc_oi_ratio, 2) if pc_oi_ratio else None,
            skew_pct=skew_pct,
            notable_strikes=notable_strikes,
        )
        _cache_set("options_mkt", symbol, result.model_dump())
        return result

    except Exception as e:
        logger.debug(f"Options market score failed for {symbol}: {e}")
        return OptionsMarketScore()


# ── 3. Earnings Estimate Revisions ───────────────────────────────────────────


def _fetch_estimate_revisions(
    symbol: str, ticker: yf.Ticker | None = None
) -> EstimateRevisionScore:
    """Layer 3: Earnings estimate revisions / analyst consensus (0-30 pts)."""
    cached = _cache_get("estimates", symbol, ttl=_TTL_ESTIMATES)
    if cached:
        return EstimateRevisionScore(**cached)

    try:
        if ticker is None:
            _yf_throttle()
            ticker = yf.Ticker(symbol)

        info = ticker.info or {}
        current_price = _safe_float(info.get("currentPrice")) or _safe_float(
            info.get("regularMarketPrice")
        )
        target_price = _safe_float(info.get("targetMeanPrice"))
        rec_mean = _safe_float(info.get("recommendationMean"))

        score = 0.0

        # Analyst target upside
        upside_pct = None
        if current_price and target_price and current_price > 0:
            upside_pct = round(((target_price - current_price) / current_price) * 100, 1)
            if upside_pct > 30:
                score += 10
            elif upside_pct > 20:
                score += 7
            elif upside_pct > 10:
                score += 3

        # Recent upgrades/downgrades (last 30 days)
        recent_upgrades = 0
        recent_downgrades = 0
        try:
            recs = ticker.recommendations
            if recs is not None and not recs.empty:
                # Reset index to get date as column if it's the index
                if hasattr(recs.index, "name") and recs.index.name:
                    recs = recs.reset_index()

                # Find date column
                date_col = None
                for col in recs.columns:
                    if "date" in str(col).lower():
                        date_col = col
                        break

                cutoff = datetime.now() - timedelta(days=30)

                if date_col:
                    import pandas as pd

                    recs[date_col] = pd.to_datetime(recs[date_col], errors="coerce")
                    recent = recs[recs[date_col] >= cutoff]
                else:
                    # Use last 10 entries as proxy
                    recent = recs.tail(10)

                for _, row in recent.iterrows():
                    to_grade = str(row.get("toGrade", row.get("To Grade", ""))).lower()
                    from_grade = str(row.get("fromGrade", row.get("From Grade", ""))).lower()

                    buy_terms = ("buy", "outperform", "overweight", "strong buy")
                    sell_terms = ("sell", "underperform", "underweight", "reduce")

                    if any(t in to_grade for t in buy_terms):
                        if not any(t in from_grade for t in buy_terms):
                            recent_upgrades += 1
                    elif any(t in to_grade for t in sell_terms):
                        recent_downgrades += 1
        except Exception as e:
            logger.debug(f"Recommendations fetch failed for {symbol}: {e}")

        net_upgrades = recent_upgrades - recent_downgrades
        if recent_upgrades >= 3 and recent_downgrades == 0:
            score += 10
        elif net_upgrades >= 2:
            score += 7
        elif net_upgrades >= 1:
            score += 3

        # Recommendation mean
        if rec_mean is not None:
            if rec_mean < 2.0:
                score += 5
            elif rec_mean < 2.5:
                score += 3

        # Earnings growth estimate
        earnings_growth = _safe_float(info.get("earningsGrowth"))
        earnings_quarterly_growth = _safe_float(info.get("earningsQuarterlyGrowth"))
        eg_val = earnings_growth or earnings_quarterly_growth
        if eg_val is not None:
            if eg_val > 0.20:
                score += 5
            elif eg_val > 0.10:
                score += 3

        result = EstimateRevisionScore(
            score=min(score, 30.0),
            current_price=current_price,
            target_price=target_price,
            upside_pct=upside_pct,
            recent_upgrades=recent_upgrades,
            recent_downgrades=recent_downgrades,
            recommendation_mean=rec_mean,
            earnings_growth_est=round(eg_val * 100, 1) if eg_val is not None else None,
        )
        _cache_set("estimates", symbol, result.model_dump())
        return result

    except Exception as e:
        logger.warning(f"Estimate revisions failed for {symbol}: {e}")
        return EstimateRevisionScore()


# ── Main screener ────────────────────────────────────────────────────────────


def screen_mispricing(symbol: str) -> MispricingResult:
    """Run all 3 layers and return combined score."""
    sector = ""
    try:
        _yf_throttle()
        ticker = yf.Ticker(symbol)
        sector = (ticker.info or {}).get("sector", "")
    except Exception:
        ticker = None

    fundamental = _fetch_fundamental_score(symbol, ticker=ticker)
    options_market = _fetch_options_market_score(symbol, ticker=ticker)
    estimate_revisions = _fetch_estimate_revisions(symbol, ticker=ticker)

    total = fundamental.score + options_market.score + estimate_revisions.score

    if total >= 75:
        signal = MispricingSignal.STRONG_BUY
    elif total >= 60:
        signal = MispricingSignal.BUY
    elif total >= 40:
        signal = MispricingSignal.WATCH
    else:
        signal = MispricingSignal.HOLD

    return MispricingResult(
        symbol=symbol.upper(),
        sector=sector,
        total_score=total,
        signal=signal,
        fundamental=fundamental,
        options_market=options_market,
        estimate_revisions=estimate_revisions,
    )
