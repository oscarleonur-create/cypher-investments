"""Options flow fetcher — pulls live chain data from yfinance.

Computes:
  - Put/call ratio (volume and OI)
  - Max pain price for the nearest expiration
  - ATM implied volatility (avg across strikes within ±5% of spot)
  - Unusual activity (volume/OI ratio > threshold)
  - Per-expiration volume/OI summary (nearest 8 expirations)
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from advisor.research.models import (
    OptionsExpirationSummary,
    OptionsFlowResult,
    UnusualOptionTrade,
)

logger = logging.getLogger(__name__)

_UNUSUAL_RATIO_MIN = 3.0  # volume must be 3× open interest to flag
_UNUSUAL_OI_MIN = 100  # ignore micro-OI contracts
_UNUSUAL_VOL_MIN = 200  # minimum volume to appear in unusual list
_MAX_UNUSUAL = 20  # cap the list length
_MAX_EXPIRATIONS = 8  # per-expiry breakdown depth


def build_options_flow(symbol: str) -> OptionsFlowResult | None:
    """Return an OptionsFlowResult for `symbol`, or None on failure."""
    sym = symbol.upper()
    try:
        ticker = yf.Ticker(sym)
        current_price: float | None = None
        info = ticker.fast_info
        try:
            current_price = float(info.last_price)
        except Exception:
            pass

        expirations = ticker.options
        if not expirations:
            logger.info("No options chain for %s", sym)
            return OptionsFlowResult(symbol=sym, current_price=current_price)

        all_calls: list[pd.DataFrame] = []
        all_puts: list[pd.DataFrame] = []
        expiry_summaries: list[OptionsExpirationSummary] = []

        for exp in expirations[:_MAX_EXPIRATIONS]:
            try:
                chain = ticker.option_chain(exp)
            except Exception as exc:
                logger.debug("Chain fetch failed for %s exp %s: %s", sym, exp, exc)
                continue

            calls = _clean(chain.calls, "call")
            puts = _clean(chain.puts, "put")

            c_vol = int(calls["volume"].sum())
            p_vol = int(puts["volume"].sum())
            c_oi = int(calls["openInterest"].sum())
            p_oi = int(puts["openInterest"].sum())

            expiry_summaries.append(
                OptionsExpirationSummary(
                    expiration=exp,
                    call_volume=c_vol,
                    put_volume=p_vol,
                    call_oi=c_oi,
                    put_oi=p_oi,
                    pcr_volume=(p_vol / c_vol) if c_vol > 0 else None,
                )
            )

            all_calls.append(calls)
            all_puts.append(puts)

        if not all_calls and not all_puts:
            return OptionsFlowResult(symbol=sym, current_price=current_price)

        calls_all = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        puts_all = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

        total_call_vol = int(calls_all["volume"].sum()) if not calls_all.empty else 0
        total_put_vol = int(puts_all["volume"].sum()) if not puts_all.empty else 0
        total_call_oi = int(calls_all["openInterest"].sum()) if not calls_all.empty else 0
        total_put_oi = int(puts_all["openInterest"].sum()) if not puts_all.empty else 0

        pcr = (total_put_vol / total_call_vol) if total_call_vol > 0 else None

        # ATM IV — average IV of contracts within ±5% of spot
        atm_iv: float | None = None
        if current_price:
            lo, hi = current_price * 0.95, current_price * 1.05
            atm_mask_c = (
                calls_all["strike"].between(lo, hi)
                if not calls_all.empty
                else pd.Series(dtype=bool)
            )
            atm_mask_p = (
                puts_all["strike"].between(lo, hi) if not puts_all.empty else pd.Series(dtype=bool)
            )
            atm_ivs = pd.concat(
                [
                    calls_all.loc[atm_mask_c, "impliedVolatility"],
                    puts_all.loc[atm_mask_p, "impliedVolatility"],
                ]
            )
            atm_ivs = atm_ivs.replace(0, float("nan")).dropna()
            if not atm_ivs.empty:
                atm_iv = float(atm_ivs.mean())

        # Max pain for nearest expiry
        max_pain: float | None = None
        if all_calls and all_puts:
            max_pain = _compute_max_pain(all_calls[0], all_puts[0])

        # Unusual activity across all expirations
        unusual = _find_unusual(calls_all, puts_all, expirations[:_MAX_EXPIRATIONS])

        nearest = expirations[0] if expirations else ""

        return OptionsFlowResult(
            symbol=sym,
            current_price=current_price,
            put_call_ratio=pcr,
            total_call_volume=total_call_vol,
            total_put_volume=total_put_vol,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            max_pain_price=max_pain,
            atm_iv=atm_iv,
            unusual_activity=unusual,
            expirations=expiry_summaries,
            nearest_expiry=nearest,
        )

    except Exception as exc:
        logger.warning("options_flow failed for %s: %s", sym, exc)
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _clean(df: pd.DataFrame, opt_type: str) -> pd.DataFrame:
    """Normalise a raw yfinance chain DataFrame."""
    df = df.copy()
    df["option_type"] = opt_type
    for col in ("volume", "openInterest"):
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)
    df["impliedVolatility"] = pd.to_numeric(df.get("impliedVolatility", 0), errors="coerce").fillna(
        0.0
    )
    df["strike"] = pd.to_numeric(df.get("strike", 0), errors="coerce").fillna(0.0)
    df["lastPrice"] = pd.to_numeric(df.get("lastPrice", 0), errors="coerce").fillna(0.0)
    return df


def _compute_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    """Return the max-pain strike for a single expiration.

    For every candidate strike S:
      pain(S) = Σ max(0, S - K) * call_OI  +  Σ max(0, K - S) * put_OI
    The strike that minimises pain(S) is max pain.
    """
    strikes = sorted(set(calls["strike"].tolist()) | set(puts["strike"].tolist()))
    if not strikes:
        return None

    call_map = dict(zip(calls["strike"], calls["openInterest"]))
    put_map = dict(zip(puts["strike"], puts["openInterest"]))

    best_strike, best_pain = None, float("inf")
    for s in strikes:
        pain = sum(max(0.0, s - k) * call_map.get(k, 0) for k in strikes)
        pain += sum(max(0.0, k - s) * put_map.get(k, 0) for k in strikes)
        if pain < best_pain:
            best_pain, best_strike = pain, s

    return best_strike


def _find_unusual(
    calls: pd.DataFrame, puts: pd.DataFrame, expirations: list[str]
) -> list[UnusualOptionTrade]:
    """Return contracts where volume is ≥ _UNUSUAL_RATIO_MIN × open interest."""
    rows: list[UnusualOptionTrade] = []

    for df in (calls, puts):
        if df.empty:
            continue
        mask = (
            (df["volume"] >= _UNUSUAL_VOL_MIN)
            & (df["openInterest"] >= _UNUSUAL_OI_MIN)
            & (df["volume"] / df["openInterest"].replace(0, float("nan")) >= _UNUSUAL_RATIO_MIN)
        )
        flagged = df[mask].copy()
        flagged["vol_oi_ratio"] = flagged["volume"] / flagged["openInterest"].replace(0, 1)
        flagged = flagged.sort_values("vol_oi_ratio", ascending=False)

        for _, row in flagged.iterrows():
            # Attempt to extract expiry from contractSymbol (AAPL241220C00150000)
            exp = _exp_from_symbol(str(row.get("contractSymbol", "")), expirations)
            rows.append(
                UnusualOptionTrade(
                    contract_symbol=str(row.get("contractSymbol", "")),
                    expiration=exp,
                    strike=float(row["strike"]),
                    option_type=str(row.get("option_type", "")),
                    volume=int(row["volume"]),
                    open_interest=int(row["openInterest"]),
                    volume_oi_ratio=round(float(row["vol_oi_ratio"]), 1),
                    implied_volatility=round(float(row["impliedVolatility"]), 4),
                    last_price=float(row["lastPrice"]),
                    estimated_premium=round(
                        float(row["volume"]) * float(row["lastPrice"]) * 100, 0
                    ),
                )
            )

    rows.sort(key=lambda r: r.volume_oi_ratio, reverse=True)
    return rows[:_MAX_UNUSUAL]


def _exp_from_symbol(contract: str, expirations: list[str]) -> str:
    """Try to match a contract symbol to one of the known expiration date strings."""
    for exp in expirations:
        short = exp.replace("-", "")[2:]  # "241220" from "2024-12-20"
        if short in contract:
            return exp
    return ""
