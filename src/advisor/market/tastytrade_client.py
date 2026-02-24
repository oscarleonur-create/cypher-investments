"""TastyTrade API client for live options trading."""

import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from tastytrade import Account, Session
from tastytrade.order import (
    InstrumentType,
    Leg,
    NewOrder,
    OrderAction,
    OrderTimeInForce,
    OrderType,
    PriceEffect,
)

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
