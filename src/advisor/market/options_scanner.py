"""Options scanner — find best naked puts and put credit spreads via yfinance."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import yfinance as yf
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Universes ──────────────────────────────────────────────────────────────────

UNIVERSES: dict[str, list[str]] = {
    "leveraged": ["TQQQ", "SOXL", "UPRO", "SPXL", "TNA", "LABU", "FNGU", "TECL"],
    "wheel": ["PLTR", "SOFI", "AMD", "MARA", "RIVN", "NIO", "LCID", "HOOD", "SNAP", "DKNG"],
    "blue_chip": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
}


# ── Result models ──────────────────────────────────────────────────────────────


class NakedPutResult(BaseModel):
    symbol: str
    strike: float
    expiry: date
    bid: float
    ask: float
    mid: float
    iv: float = 0.0
    iv_rank: float | None = None
    oi: int = 0
    volume: int = 0
    dte: int = 0
    otm_pct: float = 0.0
    margin_req: float = 0.0
    annualized_yield: float = 0.0
    exceeds_account_limit: bool = False


class SpreadResult(BaseModel):
    symbol: str
    short_strike: float
    long_strike: float
    expiry: date
    net_credit: float
    max_loss: float
    width: float
    dte: int = 0
    otm_pct: float = 0.0
    return_on_risk: float = 0.0
    annualized_return: float = 0.0
    iv_rank: float | None = None
    exceeds_account_limit: bool = False


class OptionsScanResult(BaseModel):
    scanned_at: datetime = Field(default_factory=datetime.now)
    account_size: float
    universe: str
    tickers_scanned: int = 0
    naked_puts: list[NakedPutResult] = Field(default_factory=list)
    spreads: list[SpreadResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ── Chain analysis ─────────────────────────────────────────────────────────────


class OptionLeg(BaseModel):
    """Single option contract from a chain analysis."""

    strike: float
    bid: float
    ask: float
    mid: float
    iv: float = 0.0
    oi: int = 0
    volume: int = 0
    otm_pct: float = 0.0
    annualized_yield: float | None = None  # puts only


class ChainAnalysis(BaseModel):
    """Result of analyzing an options chain for a single symbol + expiry."""

    symbol: str
    price: float
    expiry: str
    dte: int
    expirations: list[str] = Field(default_factory=list)
    puts: list[OptionLeg] = Field(default_factory=list)
    calls: list[OptionLeg] = Field(default_factory=list)


def analyze_chain(
    symbol: str,
    expiry: str | None = None,
    otm_range: tuple[float, float] = (0.02, 0.25),
    min_bid: float = 0.05,
) -> ChainAnalysis:
    """Fetch and analyze an options chain for a symbol.

    Returns structured data with OTM puts/calls, yields, and IV.
    Raises ValueError if no data is available.
    """
    ticker = yf.Ticker(symbol.upper())
    price = _get_price(ticker)
    if price is None or price <= 0:
        raise ValueError(f"No price data for {symbol}")

    expirations = list(ticker.options or [])
    if not expirations:
        raise ValueError(f"No options available for {symbol}")

    if expiry and expiry in expirations:
        target_exp = expiry
    elif expiry:
        # Snap to closest available expiry and warn
        closest = min(
            expirations, key=lambda e: abs(date.fromisoformat(e) - date.fromisoformat(expiry))
        )
        logger.warning(f"Expiry {expiry} not available for {symbol}, using closest: {closest}")
        target_exp = closest
    else:
        target_exp = expirations[0]

    try:
        chain = ticker.option_chain(target_exp)
    except Exception as e:
        raise ValueError(f"Failed to fetch option chain for {symbol} exp {target_exp}: {e}")

    exp_date = date.fromisoformat(target_exp)
    dte = (exp_date - date.today()).days

    result = ChainAnalysis(
        symbol=symbol.upper(),
        price=price,
        expiry=target_exp,
        dte=dte,
        expirations=expirations[:8],
    )

    otm_min, otm_max = otm_range

    # Puts
    if not chain.puts.empty:
        for _, row in chain.puts.iterrows():
            strike = float(row["strike"])
            if strike >= price:
                continue
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            if bid < min_bid:
                continue
            otm = (price - strike) / price
            if not (otm_min <= otm <= otm_max):
                continue
            mid = (bid + ask) / 2
            otm_amount = price - strike
            premium_100 = mid * 100
            margin = max(0.20 * price - otm_amount, 0.10 * strike) * 100 + premium_100
            ann_yield = (premium_100 / margin) * (365 / dte) if margin > 0 and dte > 0 else 0
            result.puts.append(
                OptionLeg(
                    strike=strike,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    iv=float(row.get("impliedVolatility", 0) or 0),
                    oi=int(row.get("openInterest", 0) or 0),
                    volume=int(row.get("volume", 0) or 0),
                    otm_pct=otm,
                    annualized_yield=ann_yield,
                )
            )

    # Calls
    if not chain.calls.empty:
        for _, row in chain.calls.iterrows():
            strike = float(row["strike"])
            if strike <= price:
                continue
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            if bid < min_bid:
                continue
            otm = (strike - price) / price
            if not (otm_min <= otm <= otm_max):
                continue
            result.calls.append(
                OptionLeg(
                    strike=strike,
                    bid=bid,
                    ask=ask,
                    mid=(bid + ask) / 2,
                    iv=float(row.get("impliedVolatility", 0) or 0),
                    oi=int(row.get("openInterest", 0) or 0),
                    volume=int(row.get("volume", 0) or 0),
                    otm_pct=otm,
                )
            )

    return result


# ── Scanner ────────────────────────────────────────────────────────────────────


def scan_options(
    tickers: list[str],
    account_size: float = 5000.0,
    universe_name: str = "custom",
    min_dte: int = 25,
    max_dte: int = 45,
    otm_min: float = 0.05,
    otm_max: float = 0.15,
    min_bid: float = 0.10,
    max_spread_pct: float = 0.30,
    iv_overrides: dict[str, dict] | None = None,
) -> OptionsScanResult:
    """Scan tickers for naked put and put credit spread opportunities.

    Args:
        iv_overrides: Optional per-symbol IV data from TastyTrade, keyed by
            symbol with values like {"iv_index": 0.45, "iv_rank": 0.7, ...}.
            When provided, the scanner logs the live IV alongside yfinance data.
    """
    result = OptionsScanResult(account_size=account_size, universe=universe_name)
    iv_data = iv_overrides or {}
    today = date.today()
    target_min = today + timedelta(days=min_dte)
    target_max = today + timedelta(days=max_dte)

    for symbol in tickers:
        try:
            ticker = yf.Ticker(symbol)
            price = _get_price(ticker)
            if price is None or price <= 0:
                result.errors.append(f"{symbol}: no price data")
                continue

            expirations = ticker.options
            if not expirations:
                result.errors.append(f"{symbol}: no options chain")
                continue

            result.tickers_scanned += 1
            sym_iv = iv_data.get(symbol)

            # Filter expirations in DTE window
            valid_expiries = _filter_expirations(expirations, target_min, target_max)
            if not valid_expiries:
                continue

            for exp_str in valid_expiries:
                exp_date = date.fromisoformat(exp_str)
                dte = (exp_date - today).days

                try:
                    chain = ticker.option_chain(exp_str)
                except Exception:
                    continue

                puts = chain.puts
                if puts.empty:
                    continue

                # Scan naked puts
                _scan_naked_puts(
                    puts,
                    symbol,
                    price,
                    exp_date,
                    dte,
                    otm_min,
                    otm_max,
                    min_bid,
                    max_spread_pct,
                    account_size,
                    result,
                    sym_iv,
                )

                # Scan put credit spreads
                _scan_credit_spreads(
                    puts,
                    symbol,
                    price,
                    exp_date,
                    dte,
                    otm_min,
                    otm_max,
                    min_bid,
                    account_size,
                    result,
                    sym_iv,
                )

        except Exception as e:
            result.errors.append(f"{symbol}: {e}")
            logger.warning(f"Error scanning {symbol}: {e}")

    # Sort results
    result.naked_puts.sort(key=lambda x: x.annualized_yield, reverse=True)
    result.spreads.sort(key=lambda x: x.annualized_return, reverse=True)

    return result


def _get_price(ticker: yf.Ticker) -> float | None:
    """Get current price from ticker."""
    info = ticker.info or {}
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price:
        return float(price)
    hist = ticker.history(period="1d")
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    return None


def _filter_expirations(
    expirations: tuple[str, ...], target_min: date, target_max: date
) -> list[str]:
    """Filter expiration dates within DTE range."""
    valid = []
    for exp_str in expirations:
        try:
            exp = date.fromisoformat(exp_str)
            if target_min <= exp <= target_max:
                valid.append(exp_str)
        except ValueError:
            continue
    return valid


def _scan_naked_puts(
    puts,
    symbol: str,
    price: float,
    exp_date: date,
    dte: int,
    otm_min: float,
    otm_max: float,
    min_bid: float,
    max_spread_pct: float,
    account_size: float,
    result: OptionsScanResult,
    sym_iv: dict | None = None,
):
    """Find qualifying naked put candidates from a puts chain."""
    for _, row in puts.iterrows():
        strike = float(row["strike"])
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)

        if bid < min_bid:
            continue

        otm_pct = (price - strike) / price
        if not (otm_min <= otm_pct <= otm_max):
            continue

        # Spread filter
        if ask > 0 and (ask - bid) / ask > max_spread_pct:
            continue

        mid = (bid + ask) / 2
        iv = float(row.get("impliedVolatility", 0) or 0)
        oi = int(row.get("openInterest", 0) or 0)
        vol = int(row.get("volume", 0) or 0)

        # Margin: max(20% underlying - OTM amount, 10% strike) * 100 + premium
        otm_amount = price - strike
        premium_collected = mid * 100
        margin_req = max(0.20 * price - otm_amount, 0.10 * strike) * 100 + premium_collected

        ann_yield = (
            (premium_collected / margin_req) * (365 / dte) if margin_req > 0 and dte > 0 else 0
        )

        result.naked_puts.append(
            NakedPutResult(
                symbol=symbol,
                strike=strike,
                expiry=exp_date,
                bid=bid,
                ask=ask,
                mid=mid,
                iv=iv,
                iv_rank=sym_iv.get("iv_rank") if sym_iv else None,
                oi=oi,
                volume=vol,
                dte=dte,
                otm_pct=otm_pct,
                margin_req=margin_req,
                annualized_yield=ann_yield,
                exceeds_account_limit=margin_req > account_size * 0.20,
            )
        )


def _scan_credit_spreads(
    puts,
    symbol: str,
    price: float,
    exp_date: date,
    dte: int,
    otm_min: float,
    otm_max: float,
    min_bid: float,
    account_size: float,
    result: OptionsScanResult,
    sym_iv: dict | None = None,
):
    """Find qualifying put credit spread candidates."""
    # Get strikes in OTM range
    candidates = []
    for _, row in puts.iterrows():
        strike = float(row["strike"])
        otm_pct = (price - strike) / price
        if otm_min <= otm_pct <= otm_max + 0.10:  # slightly wider for long leg
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            candidates.append((strike, bid, ask))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # Pair up: short (higher strike) + long (lower strike)
    for i in range(len(candidates) - 1):
        short_strike, short_bid, _ = candidates[i]
        long_strike, _, long_ask = candidates[i + 1]

        short_otm = (price - short_strike) / price
        if not (otm_min <= short_otm <= otm_max):
            continue

        if short_bid < min_bid:
            continue

        width = short_strike - long_strike
        if width <= 0:
            continue

        net_credit = short_bid - long_ask
        if net_credit <= 0:
            continue

        max_loss = (width - net_credit) * 100
        credit_collected = net_credit * 100

        ror = credit_collected / max_loss if max_loss > 0 else 0
        ann_return = ror * (365 / dte) if dte > 0 else 0

        result.spreads.append(
            SpreadResult(
                symbol=symbol,
                short_strike=short_strike,
                long_strike=long_strike,
                expiry=exp_date,
                net_credit=net_credit,
                max_loss=max_loss,
                width=width,
                dte=dte,
                otm_pct=short_otm,
                return_on_risk=ror,
                annualized_return=ann_return,
                iv_rank=sym_iv.get("iv_rank") if sym_iv else None,
                exceeds_account_limit=max_loss > account_size * 0.20,
            )
        )
