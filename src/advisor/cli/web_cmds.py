"""CLI command for the investment web app (FastAPI + React SPA)."""

from __future__ import annotations

import socket
from pathlib import Path

import typer

app = typer.Typer(help="Investment web app — live portfolio + research (FastAPI + React)")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FRONTEND_DIST = _REPO_ROOT / "frontend" / "dist"


def _lan_ip() -> str | None:
    """Best-effort local network IP (the address a phone on the same WiFi uses)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))  # no traffic sent; just resolves the route
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:  # noqa: BLE001
        return None


@app.command("serve")
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address. Use 0.0.0.0 to reach the app from your phone on the same WiFi.",
    ),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev)."),
) -> None:
    """Run the API + serve the built frontend on one port.

    Production: build the SPA first with `pnpm --dir frontend build`, then run this.
    Development: run `pnpm --dir frontend dev` separately (Vite proxies /api and /ws).
    """
    import uvicorn

    typer.echo(f"Local URL:   http://localhost:{port}")
    if host == "0.0.0.0":
        ip = _lan_ip()
        if ip:
            typer.echo(f"Network URL: http://{ip}:{port}  ← open this on your phone (same WiFi)")
        typer.echo(
            "Note: macOS may prompt to allow incoming connections; "
            "only use 0.0.0.0 on trusted networks (no auth)."
        )

    if not _FRONTEND_DIST.exists():
        typer.echo(
            "⚠ frontend not built yet — API works, but the UI is empty. "
            "Run `pnpm --dir frontend build` (prod) or `pnpm --dir frontend dev` (dev)."
        )

    uvicorn.run(
        "advisor.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )
