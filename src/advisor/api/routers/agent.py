"""Interactive research-agent endpoints: SSE chat + per-ticker conversations."""

from __future__ import annotations

import json
import logging
from typing import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from advisor.api import deps

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["agent"])


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@router.post("/research/{symbol}/chat")
async def chat(symbol: str, req: ChatRequest) -> StreamingResponse:
    """Stream a tool-using agent turn as Server-Sent Events.

    The new user message and the agent's final answer are persisted to the
    conversation when the stream completes.
    """
    from research_agent.config import ResearchConfig

    from advisor.agent.loop import run_agent
    from advisor.agent.tools import ToolContext
    from advisor.research.store import ResearchStore

    sym = symbol.upper()
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")

    requested_conv_id = req.conversation_id
    ctx = ToolContext(symbol=sym, config=ResearchConfig())

    def generate() -> Iterator[str]:
        # The store owns a SQLite connection that can't cross threads, so open,
        # use, and close it all inside this generator (it runs in a threadpool
        # thread, distinct from the request handler's thread).
        store = ResearchStore(deps.db_path())
        try:
            report = store.load_latest_report(sym)

            conv_id = requested_conv_id
            if conv_id and store.load_conversation(conv_id) is None:
                conv_id = None
            if conv_id is None:
                conv_id = store.create_conversation(sym, message[:80])
            conv = store.load_conversation(conv_id)
            history = conv["messages"] if conv else []

            yield _sse({"type": "meta", "conversation_id": conv_id})

            final_text = ""
            tools_used: list[str] = []
            try:
                for event in run_agent(ctx, history, message, report):
                    if event.get("type") == "tool_call":
                        tools_used.append(event.get("name", ""))
                    if event.get("type") == "done":
                        final_text = event.get("text", "")
                    yield _sse(event)
            except Exception as exc:  # noqa: BLE001
                logger.exception("agent chat failed for %s", sym)
                yield _sse({"type": "error", "message": str(exc)})

            store.append_messages(
                conv_id,
                [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": final_text, "tools": tools_used},
                ],
            )
        finally:
            store.close()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/research/{symbol}/recommend")
async def recommend(symbol: str) -> dict:
    """Decision support only -- no order is ever placed here. Checks the
    live position, pulls the cached research report if one exists, and
    returns a BUY/SELL/INCREASE/DECREASE/HOLD recommendation with reasoning
    grounded in that data."""
    from advisor.research.recommendation import build_recommendation, fetch_position_context
    from advisor.research.store import ResearchStore

    sym = symbol.upper()
    position = await fetch_position_context(sym)

    store = ResearchStore(deps.db_path())
    try:
        report = store.load_latest_report(sym)
    finally:
        store.close()

    result = build_recommendation(sym, report, position)
    return result.model_dump(mode="json")


@router.get("/research/{symbol}/conversations")
async def list_conversations(symbol: str, limit: int = 50) -> dict:
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        return {"conversations": store.list_conversations(symbol.upper(), limit=limit)}
    finally:
        store.close()


@router.get("/agent/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict:
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        conv = store.load_conversation(conversation_id)
    finally:
        store.close()
    if conv is None:
        raise HTTPException(status_code=404, detail="Unknown conversation id")
    return conv
