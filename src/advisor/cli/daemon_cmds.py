"""CLI commands for the always-on daemon."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from advisor.cli.formatters import console, output_json

app = typer.Typer(name="daemon", help="Always-on watcher: scheduled jobs and the event stream")


def _store():
    from advisor.daemon.store import DaemonStore
    from advisor.research.config import get_settings

    return DaemonStore(get_settings().db_path)


@app.command("run")
def daemon_run(
    tick: Annotated[int, typer.Option("--tick", help="Seconds between scheduler ticks")] = 30,
) -> None:
    """Run the daemon in the foreground until Ctrl-C."""
    import asyncio
    import logging

    from advisor.daemon.supervisor import serve

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    store = _store()
    try:
        asyncio.run(serve(store, tick_seconds=tick))
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")
    finally:
        store.close()


@app.command("once")
def daemon_once(
    job: Annotated[
        Optional[str], typer.Option("--job", help="Run only this job, ignoring its schedule")
    ] = None,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Run one scheduler tick (or force a single job) and exit."""
    import asyncio

    from advisor.daemon.supervisor import Supervisor

    store = _store()
    sup = Supervisor(store)
    try:
        if job:
            match = next((j for j in sup.registry if j.name == job), None)
            if match is None:
                names = ", ".join(j.name for j in sup.registry)
                raise typer.BadParameter(f"unknown job {job!r}; available: {names}")
            results = [asyncio.run(sup.run_job(match))]
        else:
            results = asyncio.run(sup.tick())

        if output == "json":
            output_json([r.model_dump(mode="json") for r in results])
            return
        if not results:
            console.print("[dim]nothing due[/dim]")
        for r in results:
            mark = "[green]ok[/green]" if r.ok else "[red]FAIL[/red]"
            console.print(f"{mark} {r.job} ({r.duration_ms}ms) — {r.detail}")
    finally:
        store.close()


@app.command("status")
def daemon_status(
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Show job heartbeats, ingest watermarks and event counts."""
    from rich.table import Table

    from advisor.daemon import market_calendar as mc
    from advisor.daemon.supervisor import build_registry

    store = _store()
    try:
        registry = build_registry()
        heartbeats = {h.job: h for h in store.all_heartbeats()}
        now = mc.now_et()

        if output == "json":
            output_json(
                {
                    "now_et": now.isoformat(),
                    "market_open": mc.is_market_open(now),
                    "jobs": [
                        {
                            "name": j.name,
                            "schedule": j.trigger.describe(),
                            **(
                                heartbeats[j.name].model_dump(mode="json")
                                if j.name in heartbeats
                                else {"run_count": 0}
                            ),
                        }
                        for j in registry
                    ],
                    "watermarks": [
                        w.model_dump(mode="json") for w in store.all_watermarks() if w.last_seen_ts
                    ],
                    "events_by_tier": store.event_counts_by_tier(),
                }
            )
            return

        state = "[green]OPEN[/green]" if mc.is_market_open(now) else "[dim]closed[/dim]"
        console.print(f"\n{now.strftime('%Y-%m-%d %H:%M:%S %Z')} — market {state}\n")

        table = Table(title="Jobs")
        table.add_column("job")
        table.add_column("schedule")
        table.add_column("last run")
        table.add_column("runs", justify="right")
        table.add_column("errors", justify="right")
        table.add_column("last error", overflow="fold")
        for j in registry:
            hb = heartbeats.get(j.name)
            table.add_row(
                j.name,
                j.trigger.describe(),
                hb.last_run_at.strftime("%m-%d %H:%M") if hb and hb.last_run_at else "—",
                str(hb.run_count) if hb else "0",
                f"[red]{hb.error_count}[/red]" if hb and hb.error_count else "0",
                (hb.last_error if hb else "") or "",
            )
        console.print(table)

        counts = store.event_counts_by_tier()
        if counts:
            summary = "  ".join(f"tier {k}: {v}" for k, v in sorted(counts.items()))
            console.print(f"\nEvents — {summary}")
        else:
            console.print("\n[dim]No events yet.[/dim]")

        marks = [w for w in store.all_watermarks() if w.last_seen_ts]
        if marks:
            console.print("\nWatermarks")
            for w in marks:
                console.print(f"  {w.source.value:<10} {w.last_seen_ts:%Y-%m-%d %H:%M}")
    finally:
        store.close()


@app.command("events")
def daemon_events(
    limit: Annotated[int, typer.Option("--limit", help="How many to show")] = 20,
    tier: Annotated[Optional[str], typer.Option("--tier", help="Filter: A, B or C")] = None,
    symbol: Annotated[Optional[str], typer.Option("--symbol", help="Filter by ticker")] = None,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """List recent events from the stream."""
    from rich.table import Table

    from advisor.daemon.models import EventTier

    store = _store()
    try:
        events = store.recent_events(
            limit=limit,
            tier=EventTier(tier.upper()) if tier else None,
            symbol=symbol,
        )
        if output == "json":
            output_json([e.model_dump(mode="json") for e in events])
            return
        if not events:
            console.print("[dim]No events.[/dim]")
            return
        table = Table(title=f"Events (latest {len(events)})")
        table.add_column("when")
        table.add_column("tier")
        table.add_column("source")
        table.add_column("symbol")
        table.add_column("kind")
        for e in events:
            table.add_row(
                e.ts.strftime("%m-%d %H:%M"),
                e.tier.value,
                e.source.value,
                e.symbol or "—",
                e.kind,
            )
        console.print(table)
    finally:
        store.close()


@app.command("exposure")
def daemon_exposure(
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Show the book's macro factor exposure and who carries it."""
    from rich.table import Table

    store = _store()
    try:
        exposure = store.load_latest_exposure()
        if exposure is None:
            console.print(
                "[dim]No exposure yet — run [/dim]advisor daemon once --job macro_refresh"
            )
            return

        if output == "json":
            output_json(exposure.model_dump(mode="json"))
            return

        console.print(
            f"\nBook exposure as of {exposure.asof} — net liq "
            f"${exposure.net_liq:,.0f}, {exposure.covered_weight:.0%} of notional covered"
        )
        if exposure.uncovered:
            console.print(
                f"[yellow]No estimate for {', '.join(exposure.uncovered)}[/yellow] "
                "(too little price history)"
            )

        table = Table(title="Factor exposure — largest bets first")
        table.add_column("factor")
        table.add_column("net loading", justify="right")
        table.add_column("top contributors")
        for entry in exposure.ranked():
            tops = "  ".join(f"{s} {v:+.2f}" for s, v in entry.top_contributors)
            colour = "red" if entry.net_loading < 0 else "green"
            table.add_row(entry.factor, f"[{colour}]{entry.net_loading:+.2f}[/{colour}]", tops)
        console.print(table)
        console.print(
            "\n[dim]Loadings are ridge estimates over correlated factors: read them "
            "as relative bets and use expected moves for prediction, not as "
            "standalone causal betas.[/dim]"
        )
    finally:
        store.close()
