"""CLI commands for the Monte Carlo PCS simulator UI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="Monte Carlo PCS simulator tools")


@app.command("ui")
def launch_ui(
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit server port"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically"),
) -> None:
    """Launch the Streamlit simulator GUI."""
    app_path = Path(__file__).resolve().parents[1] / "simulator" / "app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
    ]
    if no_browser:
        cmd += ["--server.headless", "true"]

    subprocess.run(cmd)
