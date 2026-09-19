"""CLI for structured thesis claims.

Writing a claim has to be faster than not writing one, or it will not happen —
the live evidence being that the repo's prose template has sat unfilled on
both existing theses for months. So a claim is one command and one sentence,
and the trigger is optional.
"""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from advisor.cli.formatters import console, output_json

app = typer.Typer(name="thesis", help="Structured thesis claims the daemon can test")


def _store():
    from advisor.daemon.store import DaemonStore
    from advisor.research.config import get_settings

    return DaemonStore(get_settings().db_path)


@app.command("add")
def add_claim(
    symbol: Annotated[str, typer.Argument(help="Ticker the claim belongs to")],
    text: Annotated[str, typer.Argument(help="The claim, in your own words")],
    kind: Annotated[
        str,
        typer.Option(
            "--kind", "-k", help="DRIVER, INVALIDATION, KPI, MACRO_DRIVER, CATALYST, RISK"
        ),
    ] = "INVALIDATION",
    on: Annotated[
        Optional[str], typer.Option("--on", help="Event kinds that test it, comma separated")
    ] = None,
    field: Annotated[Optional[str], typer.Option("--field", help="Payload field to read")] = None,
    above: Annotated[
        Optional[float], typer.Option("--above", help="Trips when field exceeds this")
    ] = None,
    below: Annotated[
        Optional[float], typer.Option("--below", help="Trips when field falls under this")
    ] = None,
    factor: Annotated[
        Optional[str], typer.Option("--factor", help="MACRO_DRIVER: factor name")
    ] = None,
    then: Annotated[
        Optional[str],
        typer.Option("--then", help="What you will do if this trips, in your own words"),
    ] = None,
) -> None:
    """Add a claim. With --on it is checked against every matching event."""
    from advisor.thesis.models import Claim, ClaimKind, Comparator, Trigger

    if above is not None and below is not None:
        raise typer.BadParameter("use --above or --below, not both")

    comparator = Comparator.HAPPENS
    threshold = None
    if above is not None:
        comparator, threshold = Comparator.ABOVE, above
    elif below is not None:
        comparator, threshold = Comparator.BELOW, below

    trigger = Trigger(
        event_kinds=[k.strip().upper() for k in on.split(",")] if on else [],
        field=field,
        comparator=comparator,
        threshold=threshold,
        factor=factor.upper() if factor else None,
    )
    claim = Claim(kind=ClaimKind(kind.upper()), text=text, trigger=trigger, response=then or "")

    store = _store()
    try:
        claim_id = store.save_claim(symbol, claim)
        console.print(f"[green]added[/green] {claim_id}  {claim.kind.value}: {text}")
        console.print(
            f"  [dim]tested by: {trigger.describe()}[/dim]"
            if trigger.is_testable
            else "  [yellow]recorded but not machine-testable — no event will check it[/yellow]"
        )
        if claim.response:
            console.print(f"  [green]if it trips:[/green] {claim.response}")
        elif trigger.is_testable:
            console.print(
                "  [dim]no --then, so when this trips the card will say to review "
                "rather than what you decided to do[/dim]"
            )
    finally:
        store.close()


@app.command("list")
def list_claims(
    symbol: Annotated[Optional[str], typer.Argument(help="Ticker, or omit for all")] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Show claims and whether the daemon can actually check them."""
    from rich.table import Table

    from advisor.thesis.repo import load_thesis

    store = _store()
    try:
        symbols = [symbol.upper()] if symbol else store.symbols_with_claims()
        if not symbols:
            console.print('[dim]No claims yet — try[/dim] advisor thesis add AAOI "..."')
            return

        rows = []
        for sym in symbols:
            thesis = load_thesis(store, sym)
            if thesis is None:
                continue
            for claim in thesis.claims:
                rows.append((sym, claim, thesis.blocked.get(claim.id or "") or ""))

        if output == "json":
            output_json(
                [
                    {
                        "symbol": sym,
                        **claim.model_dump(mode="json"),
                        "reachable": not blocked,
                        "blocked_reason": blocked,
                    }
                    for sym, claim, blocked in rows
                ]
            )
            return

        table = Table(title="Thesis claims")
        table.add_column("sym")
        table.add_column("id")
        table.add_column("kind")
        table.add_column("claim", overflow="fold")
        table.add_column("checked by", overflow="fold")
        for sym, claim, blocked in rows:
            if blocked:
                checked = f"[red]{blocked}[/red]"
            elif claim.monitored:
                checked = f"[green]{claim.trigger.describe()}[/green]"
            else:
                checked = "[yellow]not machine-testable[/yellow]"
            table.add_row(sym, claim.id or "—", claim.kind.value, claim.text[:60], checked)
        console.print(table)
    finally:
        store.close()


@app.command("status")
def status(
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Which holdings have a written thesis, and how much of it is testable."""
    import asyncio

    from rich.table import Table

    from advisor.daemon.book import fetch_book
    from advisor.thesis.repo import load_thesis

    store = _store()
    try:
        book = asyncio.run(fetch_book())
        rows = []
        for sym in book.symbols:
            thesis = load_thesis(store, sym)
            position = next((p for p in book.positions if p.underlying.upper() == sym), None)
            weight = (position.signed_notional / book.net_liq) if position and book.net_liq else 0.0
            rows.append(
                {
                    "symbol": sym,
                    "weight": weight,
                    "written": bool(thesis and thesis.substantive),
                    "claims": len(thesis.claims) if thesis else 0,
                    "monitored": len(thesis.monitored_claims) if thesis else 0,
                }
            )
        rows.sort(key=lambda r: -abs(r["weight"]))

        if output == "json":
            output_json(rows)
            return

        table = Table(title="Thesis coverage — by weight in the book")
        table.add_column("sym")
        table.add_column("weight", justify="right")
        table.add_column("thesis")
        table.add_column("claims", justify="right")
        table.add_column("checked", justify="right")
        for r in rows:
            table.add_row(
                r["symbol"],
                f"{r['weight'] * 100:.1f}%",
                "[green]written[/green]" if r["written"] else "[red]none[/red]",
                str(r["claims"]),
                f"{r['monitored']}" if r["claims"] else "—",
            )
        console.print(table)
        uncovered = [r for r in rows if not r["written"]]
        if uncovered:
            share = sum(abs(r["weight"]) for r in uncovered)
            console.print(
                f"\n[yellow]{len(uncovered)} of {len(rows)} positions have no thesis — "
                f"{share:.0%} of the book by weight.[/yellow]"
            )
    finally:
        store.close()
