"""Candidate generation — build and score PCS candidates from enriched chain data."""

from __future__ import annotations

import logging
import math
from typing import Callable

from advisor.simulator.db import SimulatorStore
from advisor.simulator.models import PCSCandidate, SimConfig

logger = logging.getLogger(__name__)


def get_adaptive_delta(iv_percentile: float) -> float:
    """Adaptive delta target based on IV percentile.

    Mirrors premium_screener.get_adaptive_delta().
    """
    if iv_percentile >= 75:
        return 0.35
    elif iv_percentile >= 25:
        return 0.28
    else:
        return 0.16


def score_liquidity(bid: float, ask: float) -> float:
    """Simple liquidity score based on bid-ask spread tightness (0-100)."""
    if bid <= 0 or ask <= 0:
        return 0.0
    mid = (bid + ask) / 2
    if mid <= 0:
        return 0.0
    spread_pct = (ask - bid) / mid
    # Tight spread (< 5%) = 100, wide spread (> 50%) = 0
    return max(0.0, min(100.0, (0.5 - spread_pct) / 0.45 * 100))


def compute_sell_score(candidate: PCSCandidate, iv_percentile: float) -> float:
    """Compute composite sell score (0-100) for a PCS candidate.

    Weighting mirrors premium_screener.compute_sell_score():
    - IV Percentile: 25 pts
    - POP: 20 pts
    - Annualized Yield: 15 pts
    - Liquidity: 15 pts
    - Credit/Width ratio: 15 pts
    - Delta quality: 10 pts
    """
    score = 0.0

    # IV percentile (25 pts): linear 0-100 -> 0-25
    score += min(iv_percentile / 100.0, 1.0) * 25

    # POP (20 pts): linear 60-95%
    pop = candidate.pop_estimate
    score += max(0.0, min(1.0, (pop - 0.60) / 0.35)) * 20

    # Annualized yield (15 pts): credit / buying_power * 365/dte
    if candidate.buying_power > 0 and candidate.dte > 0:
        ann_yield = (candidate.net_credit * 100 / candidate.buying_power) * (365 / candidate.dte)
        score += min(ann_yield / 2.0, 1.0) * 15  # Cap at 200% ann yield

    # Liquidity (15 pts): based on bid-ask spread tightness
    liq = score_liquidity(candidate.short_bid, candidate.short_ask)
    score += (liq / 100.0) * 15

    # Credit/width ratio (15 pts): higher is better
    if candidate.width > 0:
        ratio = candidate.net_credit / candidate.width
        score += min(ratio / 0.40, 1.0) * 15  # Cap at 40% of width

    # Delta quality (10 pts): closer to adaptive target is better
    target_delta = get_adaptive_delta(iv_percentile)
    delta_diff = abs(abs(candidate.short_delta) - target_delta)
    score += max(0.0, 1.0 - delta_diff / 0.15) * 10

    return round(score, 1)


def generate_pcs_candidates(
    chain: list[dict],
    symbol: str,
    config: SimConfig,
    iv_percentile: float = 50.0,
    iv_rank: float = 0.0,
) -> list[PCSCandidate]:
    """Generate PCS candidates from enriched chain data.

    Finds short puts near the adaptive delta target, pairs with long puts
    at $2-$10 width. Net credit = short_bid - long_ask (conservative).
    """
    delta_target = config.delta_target or get_adaptive_delta(iv_percentile)

    # Group by expiration
    by_expiry: dict[str, list[dict]] = {}
    for rec in chain:
        exp = rec.get("expiration", "")
        if exp:
            by_expiry.setdefault(exp, []).append(rec)

    candidates = []

    for expiry, strikes in by_expiry.items():
        # Sort by strike ascending
        strikes.sort(key=lambda r: r["strike"])

        # Find short put candidates near delta target
        for i, short in enumerate(strikes):
            short_bid = short.get("bid", 0)
            short_ask = short.get("ask", 0)
            # Skip records with NaN or missing bid/ask
            if not short_bid or not short_ask or math.isnan(short_bid) or math.isnan(short_ask):
                continue

            short_delta = abs(short.get("delta", 0))
            if short_delta < 0.05 or short_delta > 0.50:
                continue
            if abs(short_delta - delta_target) > 0.12:
                continue
            if short_bid <= 0:
                continue

            # Find matching long puts (lower strikes)
            for j in range(i - 1, -1, -1):
                long = strikes[j]
                width = short["strike"] - long["strike"]

                if width < config.min_width:
                    continue
                if width > config.max_width:
                    break

                long_ask = long.get("ask", 0)
                long_bid = long.get("bid", 0)
                # Skip NaN bid/ask on long leg
                if not long_ask or math.isnan(long_ask):
                    continue
                if long_bid and math.isnan(long_bid):
                    continue
                if long_ask <= 0:
                    continue

                net_credit = short["bid"] - long_ask
                if net_credit < config.min_credit:
                    continue

                short_mid = (short.get("bid", 0) + short.get("ask", 0)) / 2
                long_mid = (long.get("bid", 0) + long.get("ask", 0)) / 2
                mid_credit = short_mid - long_mid

                buying_power = (width - net_credit) * 100
                if buying_power > config.max_buying_power:
                    continue

                cand = PCSCandidate(
                    symbol=symbol,
                    expiration=expiry,
                    dte=short.get("dte", 0),
                    short_strike=short["strike"],
                    long_strike=long["strike"],
                    width=width,
                    short_bid=short.get("bid", 0),
                    short_ask=short.get("ask", 0),
                    long_bid=long.get("bid", 0),
                    long_ask=long_ask,
                    net_credit=round(net_credit, 2),
                    mid_credit=round(mid_credit, 2),
                    short_delta=short.get("delta", 0),
                    short_gamma=short.get("gamma", 0),
                    short_theta=short.get("theta", 0),
                    short_vega=short.get("vega", 0),
                    short_iv=short.get("iv", 0.30),
                    long_delta=long.get("delta", 0),
                    long_iv=long.get("iv", 0.30),
                    underlying_price=short.get("underlying_price", 0),
                    iv_percentile=iv_percentile,
                    iv_rank=iv_rank,
                    pop_estimate=round(1 - short_delta, 4),
                    buying_power=round(buying_power, 2),
                )
                cand.sell_score = compute_sell_score(cand, iv_percentile)
                candidates.append(cand)

    return candidates


def pre_score_candidates(
    candidates: list[PCSCandidate], iv_percentile: float
) -> list[PCSCandidate]:
    """Score and sort candidates by sell_score descending."""
    for c in candidates:
        c.sell_score = compute_sell_score(c, iv_percentile)
    candidates.sort(key=lambda c: c.sell_score, reverse=True)
    return candidates


async def scan_and_generate(
    symbols: list[str],
    config: SimConfig,
    store: SimulatorStore | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[PCSCandidate]:
    """Async: fetch enriched chains, save snapshots, generate + score candidates."""
    from advisor.market.tastytrade_client import (
        get_enriched_chain,
        get_market_metrics,
        get_session,
    )

    session = await get_session()

    # Fetch IV metrics for all symbols
    metrics = await get_market_metrics(session, symbols)

    all_candidates = []

    for symbol in symbols:
        if progress_callback:
            progress_callback(f"Scanning {symbol}...")

        try:
            chain = await get_enriched_chain(session, symbol)
        except Exception as e:
            logger.warning("Failed to fetch chain for %s: %s", symbol, e)
            continue

        if not chain:
            continue

        # Get IV data
        sym_metrics = metrics.get(symbol, {})
        iv_percentile = sym_metrics.get("iv_percentile", 50.0)
        iv_rank = sym_metrics.get("iv_rank", 0.0)

        # Inject IV data into chain records
        for rec in chain:
            rec["iv_percentile"] = iv_percentile
            rec["iv_rank"] = iv_rank

        # Save snapshot
        if store:
            store.save_chain_snapshot(chain, symbol)

        # Generate candidates
        candidates = generate_pcs_candidates(chain, symbol, config, iv_percentile, iv_rank)
        candidates = pre_score_candidates(candidates, iv_percentile)
        all_candidates.extend(candidates)

    # Final sort across all symbols
    all_candidates.sort(key=lambda c: c.sell_score, reverse=True)
    return all_candidates
