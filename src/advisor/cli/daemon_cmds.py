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
    from datetime import datetime

    from advisor.daemon.market_calendar import MARKET_TZ
    from advisor.daemon.supervisor import serve

    # `force=True` matters: the CLI callback has already called basicConfig at
    # WARNING, and a second call without it is a silent no-op. The daemon then
    # ran completely mute — no "daemon up", no job results, no events — which
    # for an always-on process is the difference between working and only
    # appearing to.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s ET | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # The daemon reasons entirely in market time; logging in the machine's
    # local zone makes every line a translation exercise and has already
    # produced three timezone bugs in this project. Stamp ET, and say so.
    logging.Formatter.converter = lambda *args: datetime.now(MARKET_TZ).timetuple()
    # Third-party debug chatter would bury the daemon's own lines.
    for noisy in ("tastytrade", "httpx", "httpcore", "urllib3", "yfinance", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
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
    from advisor.daemon.summarize import summarize

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
        table.add_column("symbol")
        table.add_column("kind")
        table.add_column("detail", overflow="fold")
        tier_colour = {"A": "red", "B": "yellow", "C": "dim"}
        for e in events:
            colour = tier_colour.get(e.tier.value, "white")
            table.add_row(
                e.ts.strftime("%m-%d %H:%M"),
                f"[{colour}]{e.tier.value}[/{colour}]",
                e.symbol or "—",
                e.kind,
                summarize(e),
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


@app.command("sources")
def sources_cmd(
    symbol: Annotated[
        Optional[str], typer.Option("--symbol", "-s", help="Filter to one symbol")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Archived source items — what the advisor has read, and where it came from."""
    from rich.table import Table

    from advisor.news.lead import lead_for

    store = _store()
    try:
        items = store.recent_source_items(symbol, limit=limit)
        if output == "json":
            output_json([i.model_dump(mode="json") for i in items])
            return
        if not items:
            console.print("[dim]No source items yet — run[/dim] advisor daemon once --job brief")
            return

        table = Table(title=f"Source items{f' — {symbol.upper()}' if symbol else ''}")
        table.add_column("published (UTC)")
        table.add_column("sym")
        table.add_column("tier")
        table.add_column("type")
        table.add_column("match")
        table.add_column("title", overflow="fold")
        tier_colour = {
            "PRIMARY": "green",
            "BROKER": "green",
            "AGGREGATOR": "yellow",
            "UNTAGGED": "dim",
        }
        for item in items:
            colour = tier_colour.get(item.tier.value, "white")
            table.add_row(
                item.published_at.strftime("%Y-%m-%d %H:%M"),
                item.entity.symbol,
                f"[{colour}]{item.tier.value}[/{colour}]",
                item.doc_type or "-",
                f"{item.entity.method.value} {item.entity.confidence:.1f}",
                item.title[:70] + (f"\n[dim]{lead}[/dim]" if (lead := lead_for(item)) else ""),
            )
        console.print(table)
        console.print(
            "\n[dim]Tier caps what an item may do: PRIMARY/BROKER can interrupt, "
            "AGGREGATOR reaches the digest, UNTAGGED is context only.[/dim]"
        )
    finally:
        store.close()


@app.command("reading")
def reading_cmd(
    symbol: Annotated[str, typer.Argument(help="Ticker")],
    days: Annotated[int, typer.Option("--days", help="Window of events to read")] = 45,
    refresh: Annotated[bool, typer.Option("--refresh", help="Ignore the cached reading")] = False,
    language: Annotated[str, typer.Option("--language", help="Language to write in")] = "Spanish",
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """What the recent facts on one ticker imply together — model-written, fact-checked."""
    from advisor.story.reading import read_symbol

    store = _store()
    try:
        reading = read_symbol(store, symbol, days=days, refresh=refresh, language=language)
    finally:
        store.close()
    if output == "json":
        output_json(reading.model_dump(mode="json"))
        return
    console.print(f"[bold]{reading.symbol}[/bold]  {reading.status.value}", end="")
    if reading.stance:
        console.print(f"  ·  stance [bold]{reading.stance.value}[/bold]", end="")
    console.print(f"  ·  {len(reading.facts)} facts, last {reading.window_days} days")
    for sentence in reading.sentences:
        console.print(f"  {sentence.text} [dim]{' '.join(sentence.facts)}[/dim]")
    for problem in reading.problems:
        console.print(f"  [yellow]{problem}[/yellow]")
    console.print()
    for fact in reading.facts:
        console.print(f"  [dim]{fact.id}[/dim] {fact.text}", overflow="fold")


@app.command("angles")
def angles_cmd(
    symbol: Annotated[str, typer.Argument(help="Ticker")],
    confirm: Annotated[
        Optional[list[str]], typer.Option("--confirm", help="Search this term daily")
    ] = None,
    reject: Annotated[
        Optional[list[str]], typer.Option("--reject", help="Never suggest it")
    ] = None,
    scan: Annotated[bool, typer.Option("--scan", help="Search confirmed angles now")] = False,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Products and segments a holding is a bet on — suggested from your claims, confirmed by you.

    Only confirmed angles are searched: daily by the review job, or now with --scan.
    """
    from advisor.daemon.handlers import _company_name
    from advisor.news.angles import AngleStatus, refresh_suggestions, scan_angles

    sym = symbol.upper()
    store = _store()
    try:
        refresh_suggestions(store, sym, company_name=_company_name(sym))
        for term in confirm or []:
            store.set_angle_status(sym, term, AngleStatus.CONFIRMED.value)
        for term in reject or []:
            store.set_angle_status(sym, term, AngleStatus.REJECTED.value)
        result = None
        if scan:
            result = scan_angles(store, [sym], company_names={sym: _company_name(sym)})
        angles = store.list_angles(sym)
    finally:
        store.close()

    if output == "json":
        payload: dict = {"symbol": sym, "angles": angles}
        if result is not None:
            payload["scan"] = {
                "queries": result.queries,
                "items_stored": result.items_stored,
                "events": [e.payload for e in result.events],
                "errors": result.errors,
            }
        output_json(payload)
        return
    colour = {"CONFIRMED": "green", "SUGGESTED": "yellow", "REJECTED": "dim"}
    for angle in angles:
        c = colour.get(angle["status"], "white")
        console.print(
            f"  [{c}]{angle['status']:<9}[/{c}] {angle['term']}  [dim]{angle['source'] or ''}[/dim]"
        )
    if result is not None:
        console.print(
            f"\nscan: {result.queries} queries, {result.items_stored} new items, "
            f"{len(result.events)} new events"
        )
        from rich.markup import escape

        for event in result.events:
            # Escaped: rich reads "[xAI]" as a style tag and prints nothing.
            console.print(escape(f"  [{event.payload['angle']}] {event.payload['title']}"))
        for error in result.errors:
            console.print(f"  [red]{error}[/red]")


@app.command("backfill-leads")
def backfill_leads_cmd(
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Read the lead of every archived 8-K that has none, and attach it to its events.

    Idempotent: only empty fields are filled, so a second run changes nothing.
    """
    from advisor.news.ingest import backfill_leads

    store = _store()
    try:
        result = backfill_leads(store)
    finally:
        store.close()
    if output == "json":
        output_json(
            {
                "items_read": result.items_read,
                "items_filled": result.items_filled,
                "events_filled": result.events_filled,
                "errors": result.errors,
            }
        )
        return
    console.print(
        f"8-Ks read: {result.items_read}  ·  leads stored: {result.items_filled}  ·  "
        f"events given a lead: {result.events_filled}"
    )
    for error in result.errors:
        console.print(f"[red]{error}[/red]")


@app.command("reconcile")
def reconcile_cmd(
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Check every data input against an independent source."""
    import asyncio

    from rich.table import Table

    from advisor.daemon.book import fetch_book
    from advisor.daemon.reconcile import run_reconciliation

    async def _run():
        book = await fetch_book()
        return await run_reconciliation(book)

    report = asyncio.run(_run())
    if output == "json":
        output_json(
            {
                "asof": report.asof.isoformat(),
                "ok": report.ok,
                "findings": [f.__dict__ for f in report.findings],
            }
        )
        return

    table = Table(title=f"Data quality — {report.asof}")
    table.add_column("check")
    table.add_column("symbol")
    table.add_column("result")
    table.add_column("detail", overflow="fold")
    colour = {"OK": "green", "WARN": "yellow", "FAIL": "red"}
    for finding in report.findings:
        table.add_row(
            finding.check,
            finding.symbol or "-",
            f"[{colour[finding.severity]}]{finding.severity}[/{colour[finding.severity]}]",
            finding.detail[:80],
        )
    console.print(table)
    console.print(f"\n{report.summary()}")


@app.command("story")
def story_cmd(
    symbol: Annotated[str, typer.Argument(help="Ticker to assemble stories for")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many events back")] = 1,
    output: Annotated[str, typer.Option("--output", "-o")] = "text",
) -> None:
    """Assemble the story behind a ticker's recent events. No LLM."""
    from advisor.story.assemble import stories_for_symbol
    from advisor.story.render import render

    store = _store()
    try:
        stories = stories_for_symbol(store, symbol, limit=limit)
        if output == "json":
            output_json([s.model_dump(mode="json") for s in stories])
            return
        if not stories:
            console.print(
                f"[dim]No events for {symbol.upper()} yet — run[/dim] "
                "advisor daemon once --job brief"
            )
            return
        for story in stories:
            console.print(render(story))
            console.print()
    finally:
        store.close()


@app.command("valuation")
def valuation_cmd(
    symbol: Annotated[str, typer.Argument(help="Ticker to value")],
    price: Annotated[
        Optional[float], typer.Option("--price", help="Override the last price")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """What the current price requires the business to deliver. Not a fair value."""
    import asyncio

    from rich.table import Table

    from advisor.daemon.book import fetch_book
    from advisor.valuation.fundamentals import latest_fundamentals
    from advisor.valuation.implied import build_snapshot

    sym = symbol.upper()
    if price is None:
        book = asyncio.run(fetch_book())
        held = next((p for p in book.positions if p.underlying.upper() == sym), None)
        if held is None or not held.price:
            console.print(f"[red]No price for {sym} — pass --price[/red]")
            raise typer.Exit(1)
        price = held.price

    # A blank-cheque company has no revenue by design. Running it through the
    # revenue model returns "no usable filing", which is true and useless; its
    # trust is a real valuation with a real floor.
    from advisor.valuation.spac import latest_trust_value

    trust = latest_trust_value(sym, price)
    if trust is not None:
        console.print(f"\n[bold]{sym}[/bold] — blank-cheque company, valued on its trust")
        console.print(f"  filing {trust.source_accession}, period to {trust.asof}")
        console.print(
            f"  trust ${trust.trust_total:,.0f} across {trust.public_shares:,.0f} public shares"
        )
        colour = "red" if trust.premium > 0.15 else "yellow" if trust.premium > 0 else "green"
        console.print(
            f"  [bold]${trust.price:,.2f}[/bold] vs [bold]${trust.per_share:,.2f}[/bold] "
            f"of trust per share = [{colour}]{trust.premium:+.0%}[/{colour}]"
        )
        console.print(
            f"\n[dim]The trust is a floor: holders may redeem at ${trust.per_share:,.2f} "
            "if they decline the merger. Everything above it is what the market is "
            "paying for the deal. Revenue multiples do not apply to a company that "
            "has none by construction.[/dim]"
        )
        return

    fundamentals = latest_fundamentals(sym)
    if fundamentals is None:
        console.print(f"[red]No usable filing for {sym}[/red]")
        raise typer.Exit(1)

    snapshot = build_snapshot(fundamentals, price)
    if snapshot is None:
        console.print(f"[red]Cannot value {sym} — missing {', '.join(fundamentals.missing)}[/red]")
        raise typer.Exit(1)

    if output == "json":
        output_json(snapshot.model_dump(mode="json"))
        return

    console.print(
        f"\n[bold]{sym}[/bold] at ${snapshot.price:,.2f} — "
        f"filing {snapshot.source_accession}, period to {snapshot.period_end}"
    )
    if snapshot.is_stale():
        console.print(
            f"  [yellow]⚠ the figures are {snapshot.period_age_days()} days old — "
            f"the run-rate may describe a different business[/yellow]"
        )
    console.print(
        f"  market cap ${snapshot.market_cap / 1e9:,.0f}bn  "
        f"net cash ${(snapshot.net_cash or 0) / 1e9:,.0f}bn  "
        f"[bold]EV ${snapshot.enterprise_value / 1e9:,.0f}bn[/bold]"
    )
    if snapshot.ev_to_revenue:
        console.print(
            f"  EV / revenue {snapshot.ev_to_revenue:,.1f}x on "
            f"${(snapshot.revenue_runrate or 0) / 1e9:,.1f}bn run-rate"
        )

    table = Table(title="What the price requires, over 10 years")
    table.add_column("terminal EV/FCF", justify="right")
    table.add_column("FCF margin", justify="right")
    table.add_column("revenue needed", justify="right")
    table.add_column("implied CAGR", justify="right")
    for scenario in snapshot.scenarios:
        colour = (
            "red"
            if scenario.implied_cagr > 0.25
            else "yellow"
            if scenario.implied_cagr > 0.15
            else "green"
        )
        table.add_row(
            f"{scenario.terminal_multiple:g}x",
            f"{scenario.fcf_margin:.0%}",
            f"${scenario.required_revenue / 1e9:,.0f}bn",
            f"[{colour}]{scenario.implied_cagr:.1%}[/{colour}]",
        )
    console.print(table)
    console.print(
        "\n[dim]This is arithmetic, not advice: it says what would have to happen, "
        "not whether it will. No business above $100bn of revenue has sustained "
        "25% growth for a decade.[/dim]"
    )


@app.command("action")
def action_cmd(
    symbol: Annotated[Optional[str], typer.Argument(help="Ticker, or omit for the book")] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "text",
) -> None:
    """What follows from what is stored. Reports your own rules, proposes no trade."""
    import asyncio

    from advisor.action.assemble import build_all, build_card
    from advisor.action.render import one_line, render, summary_note
    from advisor.daemon.book import fetch_book

    store = _store()
    try:
        book = asyncio.run(fetch_book())
        if symbol:
            cards = [build_card(store, symbol, book)]
        else:
            cards = build_all(store, book)

        if output == "json":
            output_json([c.model_dump(mode="json") for c in cards])
            return

        if symbol:
            console.print(render(cards[0]))
            return

        console.print()
        for card in cards:
            colour = {
                "REVIEW_NOW": "red",
                "REVIEW": "yellow",
                "WRITE_THESIS": "cyan",
                "HOLD": "green",
                "CANNOT_SAY": "magenta",
            }[card.action.value]
            console.print(f"[{colour}]{one_line(card)}[/{colour}]")
        console.print(f"\n{summary_note(cards)}")
        console.print(
            "\n[dim]advisor daemon action <TICKER> for the full card. This reports "
            "conditions you defined in advance; it holds no view of its own.[/dim]"
        )
    finally:
        store.close()


@app.command("decide")
def decide_cmd(
    symbol: Annotated[str, typer.Argument(help="Ticker the decision is about")],
    subject: Annotated[str, typer.Argument(help="Claim id, or an event's dedup key")],
    verdict: Annotated[
        str, typer.Option("--verdict", "-v", help="ACKNOWLEDGED, ACTED, DISMISSED, THESIS_REVISED")
    ] = "ACKNOWLEDGED",
    note: Annotated[str, typer.Option("--note", "-n", help="Why, in your own words")] = "",
    output: Annotated[str, typer.Option("--output", "-o")] = "text",
) -> None:
    """Answer something a card raised, so it stops being raised.

    The reading at the moment you decide is stored with the decision, and the
    card stays quiet until the number moves materially past it.
    """
    import asyncio

    from advisor.action.assemble import build_card
    from advisor.action.decisions import (
        Decision,
        Direction,
        SubjectKind,
        Verdict,
        worse_direction,
    )
    from advisor.daemon.book import fetch_book
    from advisor.thesis.repo import load_thesis

    store = _store()
    try:
        try:
            chosen = Verdict(verdict.upper())
        except ValueError:
            console.print(
                f"[red]unknown verdict {verdict!r}[/red] — one of "
                f"{', '.join(v.value for v in Verdict)}"
            )
            raise typer.Exit(1) from None

        thesis = load_thesis(store, symbol)
        claim = next((c for c in (thesis.claims if thesis else []) if c.id == subject), None)

        # The value now is what the decision is *about*; without it the
        # decision would silence a 14% concentration and a 25% one alike.
        card = build_card(store, symbol, asyncio.run(fetch_book()))
        verdicts = {c.claim_id: c for c in card.claims if c.claim_id}
        observed = verdicts[subject].observed if subject in verdicts else None
        text = verdicts[subject].text if subject in verdicts else subject

        if claim is None and subject not in {t.kind for t in card.triggers} and not verdicts:
            console.print(f"[yellow]nothing on {symbol.upper()} matches {subject!r}[/yellow]")
            raise typer.Exit(1)

        decision = Decision(
            symbol=symbol.upper(),
            subject_kind=SubjectKind.CLAIM if claim is not None else SubjectKind.EVENT,
            subject_id=subject,
            verdict=chosen,
            note=note,
            observed=observed,
            worse_is=worse_direction(claim) if claim is not None else Direction.NEITHER,
        )
        store.record_decision(decision)

        if output == "json":
            output_json(decision.model_dump(mode="json"))
            return
        console.print(f"[green]recorded[/green] {decision.describe()}")
        console.print(f"  [dim]{text}[/dim]")
        if note:
            console.print(f"  [dim]“{note}”[/dim]")
        if chosen is Verdict.ACKNOWLEDGED and observed is not None:
            console.print(
                "  [dim]quiet until it moves 10% past this reading, "
                "then raised again with both numbers[/dim]"
            )
        elif chosen is Verdict.ACTED:
            console.print(
                "  [dim]quiet for 3 days; if the condition still stands after that "
                "the card says so[/dim]"
            )
    finally:
        store.close()


@app.command("decisions")
def decisions_cmd(
    symbol: Annotated[str, typer.Argument(help="Ticker")],
    output: Annotated[str, typer.Option("--output", "-o")] = "text",
) -> None:
    """Every decision recorded on a ticker, newest first."""
    from rich.table import Table

    store = _store()
    try:
        history = store.decision_history(symbol)
        if output == "json":
            output_json([d.model_dump(mode="json") for d in history])
            return
        if not history:
            console.print(f"[dim]no decisions recorded on {symbol.upper()}[/dim]")
            return
        table = Table(title=f"Decisions — {symbol.upper()}")
        table.add_column("when")
        table.add_column("verdict")
        table.add_column("subject")
        table.add_column("at", justify="right")
        table.add_column("note", overflow="fold")
        for d in history:
            table.add_row(
                d.decided_at.date().isoformat(),
                d.verdict.value,
                d.subject_id,
                f"{d.observed:,.4g}" if d.observed is not None else "—",
                d.note,
            )
        console.print(table)
    finally:
        store.close()
