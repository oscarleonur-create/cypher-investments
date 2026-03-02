"""TastyTrade API client for live options trading."""

import asyncio
import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from tastytrade import Account, Session
from tastytrade.dxfeed import Greeks, Quote
from tastytrade.order import (
    InstrumentType,
    Leg,
    NewOrder,
    OrderAction,
    OrderTimeInForce,
    OrderType,
    PriceEffect,
)
from tastytrade.streamer import DXLinkStreamer

# Load env from project root .env
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


async def get_session():
    """Create authenticated TastyTrade session."""
    return Session(
        os.environ["TASTYTRADE_CLIENT_SECRET"],
        os.environ["TASTYTRADE_REFRESH_TOKEN"],
    )


async def get_account(session):
    """Get primary account."""
    accounts = await Account.get(session)
    return accounts[0]


async def get_balances(session=None):
    """Get account balances."""
    if not session:
        session = await get_session()
    account = await get_account(session)
    balances = await account.get_balances(session)
    return {
        "account": account.account_number,
        "net_liq": float(balances.net_liquidating_value),
        "cash": float(balances.cash_balance),
        "buying_power": float(balances.derivative_buying_power),
    }


async def get_positions(session=None):
    """Get open positions."""
    if not session:
        session = await get_session()
    account = await get_account(session)
    positions = await account.get_positions(session)
    return [
        {
            "symbol": p.symbol,
            "quantity": p.quantity,
            "instrument_type": p.instrument_type,
            "average_open_price": float(p.average_open_price) if p.average_open_price else 0,
            "close_price": float(p.close_price) if p.close_price else 0,
        }
        for p in positions
    ]


async def get_option_chain(session, symbol: str, min_dte: int = 25, max_dte: int = 45):
    """Get option chain for symbol filtered by DTE."""
    data = await session._get(f"/option-chains/{symbol}/nested")
    items = data.get("items", [])
    if not items:
        return []

    today = date.today()
    filtered = []
    for chain in items:
        for exp in chain.get("expirations", []):
            exp_date = datetime.strptime(exp["expiration-date"], "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                for strike_data in exp.get("strikes", []):
                    filtered.append(
                        {
                            "expiration": exp["expiration-date"],
                            "dte": dte,
                            "strike": float(strike_data["strike-price"]),
                            "put_symbol": strike_data.get("put", ""),
                            "call_symbol": strike_data.get("call", ""),
                            "put_streamer": strike_data.get("put-streamer-symbol", ""),
                            "call_streamer": strike_data.get("call-streamer-symbol", ""),
                        }
                    )
    return filtered


async def get_market_metrics(session, symbols: list[str]):
    """Get IV and market metrics for symbols."""
    sym_str = ",".join(symbols)
    data = await session._get(f"/market-metrics?symbols={sym_str}")
    results = {}
    for item in data.get("items", []):
        results[item["symbol"]] = {
            "iv_index": float(item.get("implied-volatility-index", 0) or 0),
            "iv_rank": float(item.get("implied-volatility-index-rank", 0) or 0),
            "iv_percentile": float(item.get("implied-volatility-percentile", 0) or 0),
        }
    return results


async def build_naked_put_order(
    symbol: str, strike: float, expiration: str, premium: float, quantity: int = 1
):
    """Build a naked put sell order (doesn't place it)."""
    # Format option symbol: TQQQ  260327P00048500
    exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
    exp_str = exp_dt.strftime("%y%m%d")
    strike_str = f"{int(strike * 1000):08d}"
    occ_symbol = f"{symbol:<6}{exp_str}P{strike_str}"

    leg = Leg(
        action=OrderAction.SELL_TO_OPEN,
        symbol=occ_symbol,
        quantity=quantity,
        instrument_type=InstrumentType.EQUITY_OPTION,
    )
    order = NewOrder(
        time_in_force=OrderTimeInForce.DAY,
        order_type=OrderType.LIMIT,
        legs=[leg],
        price=Decimal(str(round(premium, 2))),
        price_effect=PriceEffect.CREDIT,
    )
    return order


async def build_put_spread_order(
    symbol: str,
    sell_strike: float,
    buy_strike: float,
    expiration: str,
    credit: float,
    quantity: int = 1,
):
    """Build a put credit spread order (doesn't place it)."""
    exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
    exp_str = exp_dt.strftime("%y%m%d")

    sell_strike_str = f"{int(sell_strike * 1000):08d}"
    buy_strike_str = f"{int(buy_strike * 1000):08d}"

    sell_symbol = f"{symbol:<6}{exp_str}P{sell_strike_str}"
    buy_symbol = f"{symbol:<6}{exp_str}P{buy_strike_str}"

    legs = [
        Leg(
            action=OrderAction.SELL_TO_OPEN,
            symbol=sell_symbol,
            quantity=quantity,
            instrument_type=InstrumentType.EQUITY_OPTION,
        ),
        Leg(
            action=OrderAction.BUY_TO_OPEN,
            symbol=buy_symbol,
            quantity=quantity,
            instrument_type=InstrumentType.EQUITY_OPTION,
        ),
    ]
    order = NewOrder(
        time_in_force=OrderTimeInForce.DAY,
        order_type=OrderType.LIMIT,
        legs=legs,
        price=Decimal(str(round(credit, 2))),
        price_effect=PriceEffect.CREDIT,
    )
    return order


async def place_order(session, order: NewOrder, dry_run: bool = True):
    """Place an order. Set dry_run=False to actually execute."""
    account = await get_account(session)

    if dry_run:
        return {
            "status": "DRY_RUN",
            "order": json.loads(order.model_dump_json()),
            "message": "Order built but NOT placed (dry_run=True)",
        }

    response = await account.place_order(session, order)
    return {
        "status": "PLACED",
        "order_id": response.order.id if response.order else None,
        "response": json.loads(response.model_dump_json()),
    }


async def close_position(session, symbol: str, quantity: int, premium: float, dry_run: bool = True):
    """Close an open option position (buy to close)."""
    leg = Leg(
        action=OrderAction.BUY_TO_CLOSE,
        symbol=symbol,
        quantity=quantity,
        instrument_type=InstrumentType.EQUITY_OPTION,
    )
    order = NewOrder(
        time_in_force=OrderTimeInForce.DAY,
        order_type=OrderType.LIMIT,
        legs=[leg],
        price=Decimal(str(round(premium, 2))),
        price_effect=PriceEffect.DEBIT,
    )
    return await place_order(session, order, dry_run=dry_run)


# ── Streamer-based quote & greeks functions ────────────────────────────────

logger = logging.getLogger(__name__)


async def get_option_quotes(
    session, streamer_symbols: list[str], timeout: float = 5.0
) -> dict[str, dict]:
    """Subscribe to Quote events via DXLinkStreamer.

    Returns {symbol: {bid, ask, mid, bid_size, ask_size}} per symbol.
    """
    results = {}
    pending = set(streamer_symbols)

    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Quote, streamer_symbols)

        deadline = asyncio.get_event_loop().time() + timeout
        while pending and asyncio.get_event_loop().time() < deadline:
            try:
                quote = await asyncio.wait_for(
                    streamer.get_event(Quote),
                    timeout=max(0.1, deadline - asyncio.get_event_loop().time()),
                )
                sym = quote.event_symbol
                bid = float(quote.bid_price or 0)
                ask = float(quote.ask_price or 0)
                results[sym] = {
                    "bid": bid,
                    "ask": ask,
                    "mid": round((bid + ask) / 2, 4) if (bid + ask) > 0 else 0,
                    "bid_size": int(quote.bid_size or 0),
                    "ask_size": int(quote.ask_size or 0),
                }
                pending.discard(sym)
            except asyncio.TimeoutError:
                break

    return results


async def get_option_greeks(
    session, streamer_symbols: list[str], timeout: float = 5.0
) -> dict[str, dict]:
    """Subscribe to Greeks events via DXLinkStreamer.

    Returns {symbol: {delta, gamma, theta, vega, iv}} per symbol.
    """
    results = {}
    pending = set(streamer_symbols)

    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Greeks, streamer_symbols)

        deadline = asyncio.get_event_loop().time() + timeout
        while pending and asyncio.get_event_loop().time() < deadline:
            try:
                greeks = await asyncio.wait_for(
                    streamer.get_event(Greeks),
                    timeout=max(0.1, deadline - asyncio.get_event_loop().time()),
                )
                sym = greeks.event_symbol
                results[sym] = {
                    "delta": float(greeks.delta or 0),
                    "gamma": float(greeks.gamma or 0),
                    "theta": float(greeks.theta or 0),
                    "vega": float(greeks.vega or 0),
                    "iv": float(greeks.volatility or 0),
                }
                pending.discard(sym)
            except asyncio.TimeoutError:
                break

    return results


async def get_enriched_chain(
    session, symbol: str, min_dte: int = 25, max_dte: int = 45
) -> list[dict]:
    """Fetch full enriched chain: skeleton + quotes + greeks merged.

    Each record contains: symbol, expiration, dte, strike, put_symbol,
    bid, ask, mid, delta, gamma, theta, vega, iv, underlying_price.
    """
    # Get skeleton chain
    chain = await get_option_chain(session, symbol, min_dte, max_dte)
    if not chain:
        return []

    # Get underlying price via streamer Quote for the equity symbol
    underlying_price = 0.0
    try:
        async with DXLinkStreamer(session) as eq_streamer:
            await eq_streamer.subscribe(Quote, [symbol])
            eq_quote = await asyncio.wait_for(eq_streamer.get_event(Quote), timeout=3.0)
            bid = float(eq_quote.bid_price or 0)
            ask = float(eq_quote.ask_price or 0)
            underlying_price = round((bid + ask) / 2, 2) if (bid + ask) > 0 else 0.0
    except Exception:
        pass

    # Fallback to yfinance if streamer failed
    if underlying_price <= 0:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                underlying_price = float(hist["Close"].iloc[-1])
        except Exception:
            underlying_price = 0.0

    # Extract put streamer symbols
    put_streamers = [r["put_streamer"] for r in chain if r.get("put_streamer")]
    if not put_streamers:
        return []

    # Fetch quotes and greeks in parallel
    quotes, greeks = await asyncio.gather(
        get_option_quotes(session, put_streamers),
        get_option_greeks(session, put_streamers),
    )

    # Merge into enriched records
    enriched = []
    for rec in chain:
        streamer = rec.get("put_streamer", "")
        if not streamer:
            continue

        q = quotes.get(streamer, {})
        g = greeks.get(streamer, {})

        enriched.append(
            {
                "symbol": symbol,
                "expiration": rec["expiration"],
                "dte": rec["dte"],
                "strike": rec["strike"],
                "put_symbol": rec.get("put_symbol", ""),
                "put_streamer": streamer,
                "bid": q.get("bid", 0),
                "ask": q.get("ask", 0),
                "mid": q.get("mid", 0),
                "delta": g.get("delta", 0),
                "gamma": g.get("gamma", 0),
                "theta": g.get("theta", 0),
                "vega": g.get("vega", 0),
                "iv": g.get("iv", 0.30),
                "underlying_price": underlying_price,
            }
        )

    return enriched
