"""Macro ingest: factor sensitivities, book exposure, and the events they imply.

Two cadences, because the two things move at different speeds:

- **Sensitivities refresh weekly.** Factor loadings are slow-moving by
  construction (a 250-day window barely shifts in five sessions), and the
  estimate costs a batch download plus a regression per symbol. Recomputing
  daily would burn bandwidth to produce the same numbers.
- **Factor moves are checked daily**, against the stored exposure. That is the
  cheap half, and it is where the events come from.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from advisor.daemon.book import BookSnapshot
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.macro.exposure import BookExposure, build_book_exposure
from advisor.macro.factors import build_factor_returns, fetch_prices, log_returns
from advisor.macro.sensitivity import (
    SymbolSensitivity,
    estimate_sensitivity,
    residual_z,
)

logger = logging.getLogger(__name__)

# A factor move this many standard deviations from its own trailing history is
# a shock worth checking against the book.
#
# 2.0 was the first choice and it was wrong. Checking nine factors daily at
# 2 sigma fires on ~34% of sessions (1 - 0.954^9) — a multiple-comparisons
# problem, measured at 168 of 503 real sessions. That is ~1.7 interrupt-days a
# week against a budget of 0-3 pings.
FACTOR_SHOCK_Z = 2.5

# The book must actually be exposed for a shock to matter.
MATERIAL_BOOK_LOADING = 0.30

# ...and the exposure must translate into a move worth waking someone for.
# This is the filter that does the real work: the question is never "did a
# factor twitch" but "does my book move enough that I would act".
MIN_EXPECTED_BOOK_MOVE = 0.010  # 1.0% of net liq

# How far off model a single name must move to be worth surfacing.
RESIDUAL_DIVERGENCE_Z = 2.0


def estimate_universe(
    symbols: list[str], *, period: str = "2y"
) -> tuple[dict[str, SymbolSensitivity], pd.DataFrame, list[str]]:
    """Estimate sensitivities for ``symbols``.

    Returns ``(estimates, factor_returns, skipped)``. A symbol is skipped when
    it has no price history or too little of it — a recent listing cannot
    support a nine-factor regression, and saying so beats inventing a loading.
    """
    factors = build_factor_returns(period=period)
    if factors.empty:
        logger.warning("macro: factor panel unavailable")
        return {}, factors, list(symbols)

    prices = fetch_prices(symbols, period=period)
    if prices.empty:
        return {}, factors, list(symbols)
    returns = log_returns(prices)

    estimates: dict[str, SymbolSensitivity] = {}
    skipped: list[str] = []
    for symbol in symbols:
        if symbol not in returns.columns:
            skipped.append(symbol)
            continue
        estimate = estimate_sensitivity(symbol, returns[symbol], factors)
        if estimate is None:
            skipped.append(symbol)
            continue
        estimates[symbol] = estimate

    return estimates, factors, skipped


def refresh_sensitivities(
    store: DaemonStore, book: BookSnapshot, symbols: list[str] | None = None
) -> tuple[BookExposure | None, list[str]]:
    """Re-estimate loadings, rebuild book exposure, and persist both."""
    watched = symbols or book.symbols
    if not watched:
        return None, []

    estimates, _factors, skipped = estimate_universe(watched)
    for estimate in estimates.values():
        store.save_sensitivity(estimate)

    exposure = build_book_exposure(book, estimates)
    store.save_exposure(exposure)
    store.set_watermark(EventSource.MACRO, last_seen_ts=None, last_seen_cursor=str(date.today()))
    return exposure, skipped


def factor_shock_events(
    factors: pd.DataFrame, exposure: BookExposure, *, lookback: int = 250
) -> list[Event]:
    """Tier A events where a large factor move meets a real book exposure.

    Both halves are required. A 3-sigma move in a factor the book has no
    exposure to is not news, and a large exposure on a quiet day is not either.
    """
    if factors.empty or exposure.net_liq <= 0:
        return []

    events: list[Event] = []
    recent = factors.iloc[-lookback:]
    latest = factors.iloc[-1]
    session = factors.index[-1].date().isoformat()

    for factor in factors.columns:
        history = recent[factor].dropna()
        if len(history) < 60:
            continue
        sigma = float(history.std())
        if sigma <= 0:
            continue
        move = float(latest[factor])
        z = move / sigma
        loading = exposure.loading(factor)
        expected_move = loading * move
        if (
            abs(z) < FACTOR_SHOCK_Z
            or abs(loading) < MATERIAL_BOOK_LOADING
            or abs(expected_move) < MIN_EXPECTED_BOOK_MOVE
        ):
            continue

        entry = exposure.factors.get(factor)
        events.append(
            Event(
                source=EventSource.MACRO,
                kind="FACTOR_SHOCK_HITTING_BOOK",
                tier=EventTier.A,
                symbol=None,  # book-level
                dedup_key=f"{factor}:{session}",
                payload={
                    "factor": factor,
                    "move": round(move, 5),
                    "z": round(z, 2),
                    "book_loading": round(loading, 3),
                    "expected_book_move": round(expected_move, 5),
                    "top_contributors": entry.top_contributors if entry else [],
                },
            )
        )
    return events


def residual_divergence_events(
    book: BookSnapshot,
    sensitivities: dict[str, SymbolSensitivity],
    symbol_returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> list[Event]:
    """Tier B events where a holding moved for a reason macro cannot explain.

    The most genuinely alpha-adjacent signal here and it costs nothing: if the
    factors say a name should have fallen 2% and it rose instead, something
    company-specific is happening, often before the news that explains it.
    """
    if factors.empty or symbol_returns.empty:
        return []

    session = symbol_returns.index[-1].date().isoformat()
    moves = {f: float(factors[f].iloc[-1]) for f in factors.columns}
    events: list[Event] = []

    for symbol in book.symbols:
        estimate = sensitivities.get(symbol)
        if estimate is None or symbol not in symbol_returns.columns:
            continue
        actual = symbol_returns[symbol].iloc[-1]
        if pd.isna(actual):
            continue
        z = residual_z(estimate, float(actual), moves)
        if abs(z) < RESIDUAL_DIVERGENCE_Z:
            continue
        expected = float(actual) - z * estimate.resid_vol
        events.append(
            Event(
                source=EventSource.MACRO,
                kind="RESIDUAL_DIVERGENCE",
                tier=EventTier.B,
                symbol=symbol,
                dedup_key=f"{symbol}:{session}:residual",
                payload={
                    "actual_return": round(float(actual), 5),
                    "expected_return": round(expected, 5),
                    "residual_z": round(z, 2),
                    "direction": "outperformed" if z > 0 else "underperformed",
                    "r2": estimate.r2,
                },
            )
        )
    return events


async def daily_macro_events(store: DaemonStore, book: BookSnapshot) -> list[Event]:
    """Check today's factor moves against the stored exposure.

    Cheap half of the macro layer: no regression, just today's factor returns
    measured against loadings estimated last week. Returns an empty list rather
    than raising if the exposure has never been built or the panel is
    unreachable — a macro outage must not stop the position mechanics.
    """
    exposure = store.load_latest_exposure()
    if exposure is None:
        logger.debug("macro: no exposure yet, run macro_refresh first")
        return []

    try:
        factors = build_factor_returns()
        prices = fetch_prices(book.symbols)
    except Exception as exc:  # noqa: BLE001
        logger.warning("macro: panel unavailable: %s", exc)
        return []

    if factors.empty:
        return []

    events = factor_shock_events(factors, exposure)
    if not prices.empty:
        sensitivities = store.load_sensitivities(book.symbols)
        events.extend(residual_divergence_events(book, sensitivities, log_returns(prices), factors))
    return events
