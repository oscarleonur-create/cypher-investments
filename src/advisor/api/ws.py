"""WebSocket: stream live TastyTrade equity quotes for the user's holdings.

The browser computes live P&L from these ticks + the cost-basis it already has
from `/api/portfolio/holdings`. One DXLink streamer per socket (single-user app).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from advisor.api import deps

logger = logging.getLogger(__name__)
router = APIRouter()


async def _held_equity_symbols() -> tuple[list[str], dict[str, dict]]:
    """Return (symbols, snapshot) of equity holdings across accounts."""
    from advisor.market.tastytrade_client import get_positions
    from advisor.research.portfolio_review import DEFAULT_ACCOUNTS

    session = await deps.get_tt_session()
    snapshot: dict[str, dict] = {}
    for acct in DEFAULT_ACCOUNTS:
        try:
            positions = await get_positions(session, acct)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ws positions fetch failed for %s: %s", acct, exc)
            continue
        for p in positions:
            itype = str(p.get("instrument_type", "")).upper()
            if "OPTION" in itype or "EQUITY" not in itype:
                continue
            sym = (p.get("symbol") or "").upper()
            if sym and sym not in snapshot:
                snapshot[sym] = {
                    "mark": float(p.get("mark_price", 0) or 0),
                    "close": float(p.get("close_price", 0) or 0),
                }
    return sorted(snapshot), snapshot


@router.websocket("/ws/quotes")
async def ws_quotes(ws: WebSocket) -> None:
    await ws.accept()

    try:
        symbols, snapshot = await _held_equity_symbols()
    except Exception as exc:  # noqa: BLE001
        await ws.send_json({"type": "error", "message": f"holdings unavailable: {exc}"})
        symbols, snapshot = [], {}

    await ws.send_json({"type": "snapshot", "quotes": snapshot, "symbols": symbols})

    if not symbols:
        # Nothing to stream; keep the socket open until the client leaves.
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            return

    stop = asyncio.Event()

    async def producer() -> None:
        from tastytrade.dxfeed import Quote
        from tastytrade.streamer import DXLinkStreamer

        try:
            session = await deps.get_tt_session()
            async with DXLinkStreamer(session) as streamer:
                await streamer.subscribe(Quote, symbols)
                while not stop.is_set():
                    quote = await streamer.get_event(Quote)
                    bid = float(quote.bid_price or 0)
                    ask = float(quote.ask_price or 0)
                    mid = round((bid + ask) / 2, 4) if (bid + ask) > 0 else 0.0
                    await ws.send_json(
                        {
                            "type": "quote",
                            "symbol": quote.event_symbol,
                            "bid": bid,
                            "ask": ask,
                            "mid": mid,
                            "ts": datetime.now().isoformat(),
                        }
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("quote stream ended: %s", exc)

    prod = asyncio.create_task(producer())
    try:
        while True:
            await ws.receive_text()  # block until the client disconnects
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        prod.cancel()
        try:
            await prod
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
