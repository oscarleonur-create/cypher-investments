"""Catalyst layer for the scalping scanner.

Scalp signals are pure price triggers; on their own they can't tell a real,
catalyst-driven move from noise. This layer adds *why it's moving* and uses it to
**gate on volume** then **rank on news**, mirroring the confluence philosophy
(cheap signals first, expensive checks only on what already triggered):

- Tier 1 (free, from candles): relative volume (RVOL) + overnight gap %.
- Tier 2 (cheap, yfinance): earnings-session proximity.
- Tier 3 (gated, only triggered names): yfinance news headlines + optional
  LLM-scored sentiment (`confluence.sentiment.check_sentiment`).

`enrich_signals` is the entry point. The Tier 2/3 functions hit the network and
are designed to be monkeypatched in tests; Tier 1 is pure.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pandas as pd

from advisor.scalping.models import ScalpAction, ScalpSignal

logger = logging.getLogger(__name__)

_FADE_STRATEGIES = {"vwap_reversion", "rsi2_mean_reversion"}


def _clamp(v: float) -> float:
    return round(max(0.0, min(100.0, v)), 1)


# ── Tier 1: volume context (pure, from candles) ─────────────────────────────


def volume_context(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return ``(rvol, gap_pct)`` from a candle frame.

    RVOL compares the most recent bars' volume to the session baseline (a proxy
    for true time-of-day RVOL, robust to the limited intraday history we hold).
    gap_pct is the latest session's open vs. the prior session's close.
    """
    rvol = _rvol(df)
    gap = _gap_pct(df)
    return rvol, gap


def _rvol(df: pd.DataFrame, recent_bars: int = 3) -> float | None:
    vol = df["volume"].dropna()
    if len(vol) < recent_bars + 3:
        return None
    recent = float(vol.iloc[-recent_bars:].mean())
    baseline = float(vol.iloc[:-recent_bars].median())
    if baseline <= 0:
        return None
    return round(recent / baseline, 2)


def _gap_pct(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    days = pd.Index(df.index).normalize()
    distinct = days.unique()
    if len(distinct) < 2:
        return None
    last_day = distinct[-1]
    prior = df[days == distinct[-2]]
    today = df[days == last_day]
    if prior.empty or today.empty:
        return None
    prev_close = float(prior["close"].iloc[-1])
    today_open = float(today["open"].iloc[0])
    if prev_close <= 0:
        return None
    return round((today_open - prev_close) / prev_close, 4)


# ── Tier 2: earnings proximity (cheap, yfinance) ────────────────────────────


def earnings_context(symbol: str) -> tuple[bool, int | None]:
    """Return ``(earnings_today, days_to_earnings)`` using the yfinance calendar."""
    try:
        import yfinance as yf

        cal = yf.Ticker(symbol.upper()).calendar
        dates = cal.get("Earnings Date", []) if isinstance(cal, dict) else []
        if not isinstance(dates, list):
            dates = [dates]
        today = date.today()
        deltas = []
        for raw in dates:
            try:
                d = pd.Timestamp(raw).date()
                deltas.append((d - today).days)
            except Exception:  # noqa: BLE001
                continue
        if not deltas:
            return False, None
        # nearest absolute, but prefer the soonest upcoming
        upcoming = [x for x in deltas if x >= 0]
        nearest = min(upcoming) if upcoming else max(deltas)
        return (nearest == 0), nearest
    except Exception as exc:  # noqa: BLE001
        logger.debug("earnings_context failed for %s: %s", symbol, exc)
        return False, None


# ── Tier 3a: recent news headlines (gated, yfinance) ────────────────────────


def news_headlines(symbol: str, max_age_hours: float = 48.0, limit: int = 5) -> list[str]:
    """Recent yfinance news titles for a symbol (free, no LLM)."""
    try:
        import yfinance as yf

        items = yf.Ticker(symbol.upper()).news or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("news_headlines failed for %s: %s", symbol, exc)
        return []

    now = datetime.now(timezone.utc).timestamp()
    out: list[str] = []
    for it in items:
        # yfinance shapes vary by version: flat dict or {"content": {...}}.
        content = it.get("content", it)
        title = content.get("title") or it.get("title")
        ts = it.get("providerPublishTime")
        if ts is not None and (now - float(ts)) > max_age_hours * 3600:
            continue
        if title:
            out.append(str(title))
        if len(out) >= limit:
            break
    return out


# ── Tier 3b: LLM-scored sentiment (gated, Tavily + LLM) ──────────────────────


def llm_sentiment(symbol: str) -> tuple[float | None, bool | None, list[str]]:
    """Return ``(score 0–100, is_bullish, key_headlines)`` via check_sentiment."""
    try:
        from advisor.confluence.sentiment import check_sentiment

        res = check_sentiment(symbol)
        return res.score, res.is_bullish, list(res.key_headlines or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("llm_sentiment failed for %s: %s", symbol, exc)
        return None, None, []


# ── Orchestration: gate on volume, rank on news ─────────────────────────────


def enrich_signals(
    signals: list[ScalpSignal],
    candles: dict[str, pd.DataFrame],
    min_rvol: float = 1.5,
    use_llm: bool = False,
) -> tuple[list[ScalpSignal], int]:
    """Apply Tier 1 to all signals, gate on RVOL, then enrich + rescore survivors.

    Returns ``(kept_signals, gated_out_count)``. Earnings/news lookups are cached
    per symbol so multiple strategies on the same ticker don't refetch.
    """
    # Tier 1 on everything (pure, free).
    for sig in signals:
        df = candles.get(sig.symbol)
        if df is not None and not df.empty:
            sig.rvol, sig.gap_pct = volume_context(df)

    # Gate on volume — keep signals whose RVOL clears the bar (unknown RVOL passes).
    kept = [s for s in signals if s.rvol is None or s.rvol >= min_rvol]
    gated_out = len(signals) - len(kept)

    earn_cache: dict[str, tuple[bool, int | None]] = {}
    news_cache: dict[str, list[str]] = {}
    sent_cache: dict[str, tuple[float | None, bool | None, list[str]]] = {}

    for sig in kept:
        sym = sig.symbol
        sig.earnings_today, sig.days_to_earnings = earn_cache.setdefault(sym, earnings_context(sym))
        sig.headlines = news_cache.setdefault(sym, news_headlines(sym))
        if use_llm:
            score, bull, heads = sent_cache.setdefault(sym, llm_sentiment(sym))
            sig.sentiment_score = score
            if heads and not sig.headlines:
                sig.headlines = heads
        _rescore(sig)

    kept.sort(key=lambda s: s.score, reverse=True)
    return kept, gated_out


def _rescore(sig: ScalpSignal) -> None:
    """Fold catalyst context into the final score and write a 'why' note."""
    sig.technical_score = sig.score
    final = sig.score
    is_fade = sig.strategy in _FADE_STRATEGIES
    parts: list[str] = []

    if sig.rvol is not None:
        final += min(20.0, max(0.0, (sig.rvol - 1.0)) * 10.0)
        parts.append(f"RVOL {sig.rvol:.1f}×")
    if sig.gap_pct is not None and abs(sig.gap_pct) >= 0.01:
        parts.append(f"gap {sig.gap_pct * 100:+.1f}%")

    if sig.earnings_today:
        # A fresh earnings move is momentum, not something to fade.
        final += -15.0 if is_fade else 10.0
        parts.append("earnings today")
    elif sig.days_to_earnings is not None and 0 < sig.days_to_earnings <= 3:
        parts.append(f"earnings in {sig.days_to_earnings}d")

    if sig.sentiment_score is not None:
        bull = sig.sentiment_score >= 55
        bear = sig.sentiment_score <= 45
        aligned = (sig.action == ScalpAction.LONG and bull) or (
            sig.action == ScalpAction.SHORT and bear
        )
        against = (sig.action == ScalpAction.LONG and bear) or (
            sig.action == ScalpAction.SHORT and bull
        )
        if aligned:
            final += 15.0
        elif against:
            final -= 20.0 if is_fade else 10.0
        tone = "bullish" if bull else "bearish" if bear else "neutral"
        parts.append(f"{tone} news ({sig.sentiment_score:.0f})")
    elif sig.headlines:
        parts.append(f"{len(sig.headlines)} headline{'s' if len(sig.headlines) > 1 else ''}")

    sig.score = _clamp(final)
    sig.catalyst_note = " · ".join(parts)
