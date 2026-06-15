"""FastAPI application factory — JSON API + WebSocket + built React SPA."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FRONTEND_DIST = _REPO_ROOT / "frontend" / "dist"


def create_app() -> FastAPI:
    from advisor.api.routers import agent, portfolio, research, theses, watchlist
    from advisor.api.ws import router as ws_router

    app = FastAPI(title="Advisor — Investment Frontend", version="0.1.0")

    # Dev: Vite dev server origin. Prod is same-origin (SPA served below).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "spa_built": _FRONTEND_DIST.exists()}

    app.include_router(portfolio.router)
    app.include_router(research.router)
    app.include_router(agent.router)
    app.include_router(watchlist.router)
    app.include_router(theses.router)
    app.include_router(ws_router)

    _mount_spa(app)
    return app


def _mount_spa(app: FastAPI) -> None:
    """Serve the built SPA with a client-side-routing fallback to index.html."""
    if not _FRONTEND_DIST.exists():
        return

    assets = _FRONTEND_DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index_html = _FRONTEND_DIST / "index.html"

    @app.get("/{full_path:path}")
    async def spa(full_path: str) -> FileResponse:
        candidate = _FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_html)
