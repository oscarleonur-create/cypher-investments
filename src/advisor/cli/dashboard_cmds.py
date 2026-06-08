"""CLI commands for the multi-page Streamlit research dashboard."""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="Multi-page Streamlit research dashboard")


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


@app.command("ui")
def launch_ui(
    port: int = typer.Option(8502, "--port", "-p", help="Streamlit server port"),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address. Use 0.0.0.0 to reach the dashboard from your phone on the same WiFi.",
    ),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically"),
) -> None:
    """Launch the Streamlit research dashboard."""
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]
    if no_browser:
        cmd += ["--server.headless", "true"]

    typer.echo(f"Local URL:   http://localhost:{port}")
    if host == "0.0.0.0":
        ip = _lan_ip()
        if ip:
            typer.echo(f"Network URL: http://{ip}:{port}  ← open this on your phone (same WiFi)")
        typer.echo(
            "Note: macOS may prompt to allow incoming connections; "
            "only use 0.0.0.0 on trusted networks (no auth)."
        )

    subprocess.run(cmd)
