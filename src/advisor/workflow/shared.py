"""Shared workflow phases used by both stocks and options morning/afternoon routines."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging

from advisor.workflow.models import (
    AccountSnapshot,
    RegimeContext,
    ScannerHit,
)

logger = logging.getLogger(__name__)


# ── Phase: Regime Detection ──────────────────────────────────────────────────


def run_regime_check() -> RegimeContext:
    """Detect market regime via HMM. Returns RegimeContext."""
    try:
        from advisor.ml.regime import RegimeDetector

        detector = RegimeDetector()
        result = detector.detect_regime()

        regime_name = result["regime_name"]
        vix = result["vix"]
        warning = None
        if regime_name == "high_vol" and vix > 30:
            warning = "HIGH VOLATILITY REGIME"

        return RegimeContext(
            regime_name=regime_name,
            vix=vix,
            spy_vol=result["spy_vol"],
            regime_prob=result["regime_prob"],
            warning=warning,
        )
    except Exception as e:
        logger.warning("Regime detection failed: %s", e)
        return RegimeContext(warning=f"Regime detection failed: {e}")


# ── Phase: Account Snapshot ──────────────────────────────────────────────────


def run_account_snapshot(mode: str, account_number: str | None = None) -> AccountSnapshot:
    """Fetch TastyTrade balances and positions, filtered by mode."""
    from advisor.market.tastytrade_client import get_balances, get_positions

    try:
        loop = _get_or_create_loop()

        balances = loop.run_until_complete(get_balances(account_number=account_number))
        positions = loop.run_until_complete(get_positions(account_number=account_number))

        # Filter positions by instrument type
        if mode == "stocks":
            filtered = [p for p in positions if p.get("instrument_type") == "Equity"]
        else:
            filtered = [p for p in positions if p.get("instrument_type") != "Equity"]

        net_liq = balances.get("net_liq", 0)
        buying_power = balances.get("buying_power", 0)
        warning = None
        if net_liq > 0 and buying_power < net_liq * 0.20:
            warning = f"LOW BUYING POWER — {buying_power/net_liq:.0%} of net liq"

        return AccountSnapshot(
            net_liq=net_liq,
            cash=balances.get("cash", 0),
            buying_power=buying_power,
            positions=filtered,
            warning=warning,
        )
    except Exception as e:
        logger.warning("Account snapshot failed: %s", e)
        return AccountSnapshot(warning=f"Account fetch failed: {e}")


# ── Phase: Opportunity Discovery ─────────────────────────────────────────────


def run_discovery(
    universes: list[str],
    watchlist: list[str],
    errors: list[str],
    top_n: int = 10,
) -> list[ScannerHit]:
    """Run 4 scanners in parallel, merge hits, apply overlap gate."""
    symbols = _resolve_symbols(universes, watchlist)
    if not symbols:
        errors.append("No symbols resolved for discovery")
        return []

    # Run scanners in parallel
    def _run_smart_money() -> dict[str, float]:
        from advisor.confluence.smart_money_screener import screen_smart_money

        scores = {}
        for sym in symbols[:top_n]:
            try:
                result = screen_smart_money(sym)
                raw = result.total_score
                normalized = max(0.0, min(100.0, (raw + 35) / 135 * 100))
                if normalized > 20:
                    scores[sym] = round(normalized, 2)
            except Exception as e:
                logger.debug("Smart money failed for %s: %s", sym, e)
        return scores

    def _run_mispricing() -> dict[str, float]:
        from advisor.confluence.mispricing_screener import screen_mispricing

        scores = {}
        for sym in symbols[:top_n]:
            try:
                result = screen_mispricing(sym)
                if result.total_score > 20:
                    scores[sym] = round(result.total_score, 2)
            except Exception as e:
                logger.debug("Mispricing failed for %s: %s", sym, e)
        return scores

    def _run_alpha() -> dict[str, float]:
        from advisor.confluence.alpha_scorer import compute_alpha

        scores = {}
        for sym in symbols:
            try:
                result = compute_alpha(sym, skip_layers={"sentiment"})
                if result.alpha_score > 20:
                    scores[sym] = round(result.alpha_score, 2)
            except Exception as e:
                logger.debug("Alpha scorer failed for %s: %s", sym, e)
        return scores

    def _run_dip() -> dict[str, float]:
        from advisor.confluence.dip_analyzer import analyze_dip

        scores = {}
        for sym in symbols[:top_n]:
            try:
                result = analyze_dip(sym)
                if result.composite_score > 30:
                    scores[sym] = round(result.composite_score, 2)
            except Exception as e:
                logger.debug("Dip analyzer failed for %s: %s", sym, e)
        return scores

    smart_money = {}
    mispricing = {}
    alpha = {}
    dip = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_run_smart_money): "smart_money",
            pool.submit(_run_mispricing): "mispricing",
            pool.submit(_run_alpha): "alpha",
            pool.submit(_run_dip): "dip",
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if name == "smart_money":
                    smart_money = result
                elif name == "mispricing":
                    mispricing = result
                elif name == "alpha":
                    alpha = result
                elif name == "dip":
                    dip = result
            except Exception as e:
                errors.append(f"Scanner {name} failed: {e}")
                logger.warning("Scanner %s failed: %s", name, e)

    # Merge hits: count appearances across scanners
    all_symbols: set[str] = set()
    all_symbols.update(smart_money.keys(), mispricing.keys(), alpha.keys(), dip.keys())

    hits: list[ScannerHit] = []
    for sym in all_symbols:
        sources = []
        scores = []
        sm = smart_money.get(sym)
        mp = mispricing.get(sym)
        al = alpha.get(sym)
        dp = dip.get(sym)

        if sm is not None:
            sources.append("smart_money")
            scores.append(sm)
        if mp is not None:
            sources.append("mispricing")
            scores.append(mp)
        if al is not None:
            sources.append("alpha")
            scores.append(al)
        if dp is not None:
            sources.append("dip")
            scores.append(dp)

        best = max(scores) if scores else 0
        hits.append(
            ScannerHit(
                symbol=sym,
                sources=sources,
                best_score=best,
                smart_money_score=sm,
                mispricing_score=mp,
                alpha_score=al,
                dip_score=dp,
            )
        )

    # Gate: keep tickers in >=2 scanners OR score > 70
    passed = [h for h in hits if len(h.sources) >= 2 or h.best_score > 70]

    # If < 3 survive, include top scorer from each scanner
    if len(passed) < 3:
        top_per_scanner: set[str] = set()
        for scanner_scores in [smart_money, mispricing, alpha, dip]:
            if scanner_scores:
                top_sym = max(scanner_scores, key=scanner_scores.get)
                top_per_scanner.add(top_sym)
        existing = {h.symbol for h in passed}
        for h in hits:
            if h.symbol in top_per_scanner and h.symbol not in existing:
                passed.append(h)

    passed.sort(key=lambda h: h.best_score, reverse=True)
    return passed


# ── Lean Discovery ──────────────────────────────────────────────────────────


def run_lean_discovery(
    universes: list[str],
    watchlist: list[str],
    errors: list[str],
    top_n: int = 5,
) -> list[ScannerHit]:
    """Progressive funnel: free yfinance pre-screen → cheap alpha → top N."""
    symbols = _resolve_symbols(universes, watchlist)
    if not symbols:
        errors.append("No symbols resolved for discovery")
        return []

    # Step B: Batch technical pre-screen (FREE)
    matched = _swing_prescreen_batch(symbols)
    if not matched:
        errors.append(
            "WARNING: 0 candidates passed pre-screen — " "consider broadening universe or watchlist"
        )
        return []

    logger.info("Lean pre-screen: %d/%d symbols passed", len(matched), len(symbols))

    # Step C: Cheap alpha score (FREE, yfinance-only layers)
    alpha_scores: dict[str, float] = {}
    skip_layers = {"sentiment", "smart_money", "ml_signal"}

    for sym in matched:
        try:
            from advisor.confluence.alpha_scorer import compute_alpha

            result = compute_alpha(sym, skip_layers=skip_layers)
            alpha_scores[sym] = round(result.alpha_score, 2)
        except Exception as e:
            logger.debug("Cheap alpha failed for %s: %s", sym, e)

    # Rank by alpha descending, take top N
    ranked = sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True)
    top = ranked[:top_n]

    # Step D: Build ScannerHit objects
    hits: list[ScannerHit] = []
    for sym, score in top:
        patterns = matched[sym]
        hits.append(
            ScannerHit(
                symbol=sym,
                sources=patterns,
                best_score=score,
                alpha_score=score,
            )
        )

    return hits


def _swing_prescreen_batch(symbols: list[str]) -> dict[str, list[str]]:
    """Batch yfinance download + 2-pattern swing-trade check.

    Patterns:
      - momentum_breakout: breakout up with volume confirmation
      - swing_signal: oversold bounce (Bollinger Band gate + confirmations)

    Returns dict of symbol → list of matched pattern names.
    """
    import numpy as np
    import yfinance as yf

    # Single batch download — 3 months of daily OHLCV
    data = yf.download(symbols, period="3mo", group_by="ticker", progress=False)

    if data.empty:
        return {}

    matched: dict[str, list[str]] = {}
    single_ticker = len(symbols) == 1

    for sym in symbols:
        try:
            # Extract per-symbol DataFrame
            if single_ticker:
                df = data
            else:
                if sym not in data.columns.get_level_values(0):
                    continue
                df = data[sym]

            df = df.dropna(subset=["Close"])
            if len(df) < 50:
                continue

            close = df["Close"]
            volume = df["Volume"]

            # Pre-compute indicators
            sma20 = close.rolling(20).mean()
            sma200 = close.rolling(200).mean()
            avg_vol = volume.rolling(20).mean()

            # RSI (14-period)
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            # MACD histogram
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - macd_signal

            # Stochastic %K/%D (14,3,3)
            low = df["Low"]
            low14 = low.rolling(14).min()
            high14 = df["High"].rolling(14).max()
            stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
            stoch_d = stoch_k.rolling(3).mean()

            last_close = close.iloc[-1]
            last_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
            last_vol = volume.iloc[-1]
            last_avg_vol = avg_vol.iloc[-1]
            last_sma20 = sma20.iloc[-1]
            last_sma200 = sma200.iloc[-1] if len(close) >= 200 else np.nan

            patterns: list[str] = []

            # 1. momentum_breakout: price > SMA-20 AND volume > 1.5x avg
            if last_close > last_sma20 and last_vol > 1.5 * last_avg_vol:
                patterns.append("momentum_breakout")

            # 2. swing_signal: Bollinger Band gate + confirmation layers
            bb_std = close.rolling(20).std()
            lower_band = last_sma20 - 2 * bb_std.iloc[-1]

            if not np.isnan(lower_band) and bb_std.iloc[-1] > 0:
                # Gate: price <= lower band OR within 1% of it
                gate = last_close <= lower_band or last_close <= lower_band * 1.01

                if gate:
                    confirmations: list[str] = []

                    # C1: RSI(14) < 35
                    if last_rsi < 35:
                        confirmations.append("rsi")

                    # C2: Volume > 1.5x average
                    if last_avg_vol > 0 and last_vol > 1.5 * last_avg_vol:
                        confirmations.append("volume")

                    # C3: Price > SMA-200 (long-term uptrend intact)
                    if not np.isnan(last_sma200) and last_close > last_sma200:
                        confirmations.append("sma200")

                    # C4: MACD histogram turning up
                    if len(macd_hist) >= 2 and not np.isnan(macd_hist.iloc[-1]):
                        if macd_hist.iloc[-1] > macd_hist.iloc[-2]:
                            confirmations.append("macd")

                    # C5: Stochastic %K crosses above %D while %K < 20
                    if len(stoch_k) >= 2 and not np.isnan(stoch_k.iloc[-1]):
                        k_now = stoch_k.iloc[-1]
                        k_prev = stoch_k.iloc[-2]
                        d_now = stoch_d.iloc[-1]
                        d_prev = stoch_d.iloc[-2]
                        if (
                            not np.isnan(d_now)
                            and not np.isnan(d_prev)
                            and k_now < 20
                            and k_prev <= d_prev
                            and k_now > d_now
                        ):
                            confirmations.append("stoch")

                    # Pass: gate met AND at least 2 confirmations
                    if len(confirmations) >= 2:
                        detail = ",".join(confirmations)
                        patterns.append(f"swing_signal({detail})")

            if patterns:
                matched[sym] = patterns

        except Exception as e:
            logger.debug("Pre-screen failed for %s: %s", sym, e)

    return matched


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_symbols(universes: list[str], watchlist: list[str]) -> list[str]:
    """Resolve universe names + watchlist into a deduplicated symbol list."""
    from advisor.market.options_scanner import UNIVERSES

    symbols: list[str] = []
    for u in universes:
        u_lower = u.lower()
        if u_lower in UNIVERSES:
            symbols.extend(UNIVERSES[u_lower])
        elif u_lower == "sp500":
            try:
                from advisor.data.universe import fetch_universe

                stocks = fetch_universe("sp500")
                symbols.extend([s.symbol for s in stocks])
            except Exception as e:
                logger.warning("Failed to fetch sp500 universe: %s", e)
        elif u_lower == "semiconductors":
            try:
                from advisor.data.universe import SEMICONDUCTORS

                symbols.extend([sym for sym, _name in SEMICONDUCTORS])
            except Exception as e:
                logger.warning("Failed to fetch semiconductors universe: %s", e)

    symbols.extend([w.upper() for w in watchlist])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for s in symbols:
        s = s.upper()
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """Get existing event loop or create a new one."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
