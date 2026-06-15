"""Investment thesis endpoints: write, store, and revise long-form research.

Theses are markdown documents that optionally attach to a ticker (``symbol``);
a blank symbol means a thematic / macro thesis. Backed by the ``theses`` table
in ``data/research.db`` (see ``advisor.research.store.ResearchStore``).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from advisor.api import deps

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["theses"])


class ThesisBody(BaseModel):
    symbol: str = ""
    title: str = ""
    content: str = ""
    tags: list[str] = []
    conviction: str = "MEDIUM"  # HIGH | MEDIUM | LOW
    status: str = "DRAFT"  # DRAFT | ACTIVE | ARCHIVED


@router.get("/theses")
async def list_theses(symbol: str | None = None) -> dict:
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        sym = symbol.strip().upper() if symbol else None
        return {"theses": store.list_theses(sym)}
    finally:
        store.close()


@router.get("/theses/{thesis_id}")
async def get_thesis(thesis_id: str) -> dict:
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        thesis = store.get_thesis(thesis_id)
        if thesis is None:
            raise HTTPException(status_code=404, detail="Thesis not found.")
        return thesis
    finally:
        store.close()


@router.post("/theses")
async def create_thesis(body: ThesisBody) -> dict:
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        return store.save_thesis(
            symbol=body.symbol,
            title=body.title,
            content=body.content,
            tags=body.tags,
            conviction=body.conviction,
            status=body.status,
        )
    finally:
        store.close()


@router.put("/theses/{thesis_id}")
async def update_thesis(thesis_id: str, body: ThesisBody) -> dict:
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        if store.get_thesis(thesis_id) is None:
            raise HTTPException(status_code=404, detail="Thesis not found.")
        return store.save_thesis(
            thesis_id=thesis_id,
            symbol=body.symbol,
            title=body.title,
            content=body.content,
            tags=body.tags,
            conviction=body.conviction,
            status=body.status,
        )
    finally:
        store.close()


@router.delete("/theses/{thesis_id}")
async def delete_thesis(thesis_id: str) -> dict:
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        store.delete_thesis(thesis_id)
    finally:
        store.close()
    return {"id": thesis_id, "removed": True}
