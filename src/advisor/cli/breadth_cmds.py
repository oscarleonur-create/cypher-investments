"""CLI for the breadth layer: sync the market-wide stores and show the universe they cut."""

from __future__ import annotations

from typing import Annotated, Optional

import typer
from rich.table import Table

from advisor.cli.formatters import console, output_error, output_json

app = typer.Typer(name="breadth", help="The whole listed market: universe, bars and SEC facts")


def _db_path():
    from advisor.research.config import get_settings

    return get_settings().db_path


@app.command("sync")
def sync(
    bars: Annotated[bool, typer.Option("--bars/--no-bars", help="Pull daily bars")] = True,
    facts: Annotated[bool, typer.Option("--facts/--no-facts", help="Pull SEC frames")] = True,
    limit: Annotated[
        Optional[int], typer.Option("--limit", help="Only the first N common stocks (a sample)")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Directory → bars → E0 snapshot → SEC facts. The first run backfills four years."""
    from advisor.breadth.run import run_sync
    from advisor.daemon.market_calendar import now_et

    summary = run_sync(_db_path(), now_et(), bars=bars, facts=facts, limit=limit)
    if output == "json":
        output_json(summary)
        return
    if "error" in summary:
        output_error(summary["error"])
        return
    _print_summary(summary)
    if not summary["ok"]:
        console.print("[yellow]Incomplete: the bar source refused part of the pull.[/yellow]")
        raise typer.Exit(1)


def _print_summary(s: dict) -> None:
    d = s.get("directory", {})
    console.print(f"Directory: {d.get('listings')} listings, {d.get('common')} common stock")
    if "bars" in s:
        b = s["bars"]
        console.print(
            f"Bars: {b['fetched']} fetched, {b['current']} already current, {b['empty']} empty, "
            f"{b['skipped_empty']} skipped, {len(b['rebased'])} rebased, "
            f"{b['bars_written']} bars written"
            + (f" — RATE-LIMITED, {b['remaining']} left" if b["rate_limited"] else "")
        )
    _print_funnel(s.get("universe", {}))
    if "facts" in s:
        f = s["facts"]
        fr, gp = f["frames"], f["gaps"]
        console.print(
            f"SEC frames: {fr['frames_fetched']} read ({fr['frames_settled']} settled, "
            f"{fr['frames_missing']} absent), {fr['rows']} rows; "
            f"{gp['gaps_filled']} companies gap-filled"
        )
        console.print(f"Revenue coverage of eligible: {f['coverage_of_eligible']}")
        for e in fr["errors"] + gp["errors"]:
            console.print(f"[red]  {e}[/red]")
    console.print(f"[dim]{s.get('seconds')} s[/dim]")


def _print_funnel(u: dict) -> None:
    if not u:
        return
    console.print(
        f"Universe {u['day']} (rules {u['rules']}): {u['listings']} listings → "
        f"{u['common_stock_sec_filers']} common stock of SEC filers → "
        f"[bold]{u['eligible']} eligible[/bold]"
    )
    table = Table(title="Excluded, by reason")
    table.add_column("reason")
    table.add_column("names", justify="right")
    for reason, n in u["excluded"].items():
        table.add_row(reason, str(n))
    console.print(table)


@app.command("replay")
def replay(
    years: Annotated[int, typer.Option("--years", help="Sessions judged: the last N years")] = 2,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Run families P and F over history and judge them against matched controls (B1)."""
    from advisor.breadth.measure import replay as run_replay
    from advisor.breadth.ruleset import signal_rules
    from advisor.breadth.run import _register
    from advisor.breadth.store import BreadthStore, breadth_path
    from advisor.daemon.market_calendar import now_et

    _register(_db_path(), signal_rules())
    with BreadthStore(breadth_path(_db_path())) as store:
        summary = run_replay(store, now_et(), years=years)
    if output == "json":
        output_json(summary)
        return
    if not summary.get("ok"):
        output_error(summary.get("error", "replay failed"))
        return
    _print_study(summary)


@app.command("plan-replay")
def plan_replay(
    years: Annotated[int, typer.Option("--years", help="Sessions judged: the last N years")] = 3,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Replay a pick's entry plan: entered 0/5/10/15 sessions in, with and without its stop."""
    from advisor.breadth.plan_replay import replay_plan
    from advisor.breadth.store import BreadthStore, breadth_path
    from advisor.daemon.market_calendar import now_et

    with BreadthStore(breadth_path(_db_path())) as store:
        s = replay_plan(store, now_et(), years=years)
    if output == "json":
        output_json(s)
        return
    if not s.get("ok"):
        output_error(s.get("error", "plan replay failed"))
        return
    console.print(
        f"Plan replay {s['run_id']} (rules {s['rules']}): {s['from']} → {s['to']}, "
        f"{s['records']} records, {s['trades']} trades, skipped {s['skipped']}"
    )
    table = Table(title=f"The plan over {s['hold']} sessions, beyond matched peers")
    for col in ("group", "entry", "n", "peers", "no stop", "vs peers", "CI", "verdict",
                "with stop", "vs peers ", "CI ", "verdict ", "stopped", "stop avg"):  # fmt: skip
        table.add_column(col, justify="left" if col in ("group", "entry") else "right")
    for c in s["cells"]:
        if not c["n"]:
            table.add_row(c["group"], f"day +{c['offset']}", "0", *[""] * 11)
            continue
        p, w = c["plain"], c["with_stop"]

        def ci(x):
            return f"{_pct(x['ci'][0])} … {_pct(x['ci'][1])}" if x["ci"] else "—"

        table.add_row(
            c["group"], f"day +{c['offset']}", str(c["n"]), _pct(c["peers"]),
            _pct(p["mean"]), _pct(p["excess"]), ci(p), p["verdict"],
            _pct(w["mean"]), _pct(w["excess"]), ci(w), w["verdict"],
            f"{c['stopped'] * 100:.0f}%", _pct(-c["stop_pct"]),
        )  # fmt: skip
    console.print(table)


@app.command("value-replay")
def value_replay(
    years: Annotated[int, typer.Option("--years", help="The signal replay window to value")] = 3,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Value each pick record as of its day; judge picks by where the price sat vs value."""
    from advisor.breadth.store import breadth_path
    from advisor.breadth.value_replay import replay_value
    from advisor.daemon.market_calendar import now_et

    s = replay_value(
        breadth_path(_db_path()),
        now_et(),
        years=years,
        progress=(
            None
            if output == "json"  # stdout carries the JSON only
            else lambda k, n: console.print(f"valued {k:,}/{n:,}") if k % 1000 == 0 else None
        ),
    )
    if output == "json":
        output_json(s)
        return
    if not s.get("ok"):
        output_error(s.get("error", "value replay failed"))
        return
    console.print(f"Value test {s['id']} over {s['signal_run']}: {s['records']:,} records")
    table = Table(title="Excess over matched peers by where the price sat against value")
    for col in ("group", "bucket", "horizon", "n", "excess", "95% CI", "beat", "verdict"):
        table.add_column(col, justify="left" if col in ("group", "bucket", "verdict") else "right")
    for c in s["cells"]:
        if not c["n"]:
            continue
        ci = c.get("ci")
        table.add_row(
            c["group"], c["bucket"], c["horizon"], str(c["n"]), _pct(c["excess"]),
            f"{_pct(ci[0])} … {_pct(ci[1])}" if ci else "—", f"{c['beat'] * 100:.0f}%",
            c["verdict"],
        )  # fmt: skip
    console.print(table)


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:+.2f}%"


def _print_study(s: dict) -> None:
    e = s["eligible_per_session"]
    console.print(
        f"Replay {s['run_id']} (rules {s['rules']}, code {s['code']}): {s['from']} → {s['to']}, "
        f"{s['sessions']} sessions, {e['median']} eligible names per session "
        f"({e['min']}–{e['max']})"
    )
    console.print(
        "Records: "
        + ", ".join(f"{g} {n} ({s['per_session'][g]}/session)" for g, n in s["records"].items())
    )
    table = Table(title="Return beyond matched controls (same day, same industry and size)")
    for col in ("group", "horizon", "n", "windows", "mean", "controls", "excess", "95% CI",
                "vs same trend", "trend CI", "beat", "tail", "verdict"):  # fmt: skip
        table.add_column(col, justify="left" if col in ("group", "horizon", "verdict") else "right")
    for c in s["cells"]:
        if not c["n"]:
            table.add_row(c["group"], c["horizon"], "0", *[""] * 9, c["verdict"])
            continue
        ci = c["ci"]
        tci = c.get("ci_trend")
        table.add_row(
            c["group"],
            c["horizon"],
            str(c["n"]),
            str(c["windows"]),
            _pct(c["mean"]),
            _pct(c["control"]),
            _pct(c["excess"]),
            f"{_pct(ci[0])} … {_pct(ci[1])}" if ci else "—",
            _pct(c.get("excess_trend")),
            f"{_pct(tci[0])} … {_pct(tci[1])}" if tci else "—",
            f"{c['beat'] * 100:.0f}%",
            _pct(c["tail"]),
            c["verdict"],
        )
    console.print(table)
    console.print(
        f"[dim]{s['cells_tested']} cells tested: at 95% about "
        f"{s['cells_tested'] * 0.025:.1f} would clear zero by chance alone.[/dim]"
    )
    for caveat in s["caveats"]:
        console.print(f"[dim]· {caveat}[/dim]")


@app.command("report")
def report(
    run_id: Annotated[
        Optional[str], typer.Argument(help="A replay run id; the latest if omitted")
    ] = None,
    live: Annotated[bool, typer.Option("--live", help="Judge the live records instead")] = False,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """A stored replay's verdicts, or the live records judged so far."""
    import json

    from advisor.breadth.measure import CAVEATS, evaluate, load_records
    from advisor.breadth.signals import GROUPS
    from advisor.breadth.store import BreadthStore, breadth_path

    with BreadthStore(breadth_path(_db_path())) as store:
        if live:
            recs = load_records(store, "live")
            cells = evaluate(recs)
            summary = {
                "run_id": "live",
                "records": {g: sum(1 for r in recs if r["grp"] == g) for g in GROUPS},
                "cells": cells,
                "cells_tested": len([c for c in cells if c["n"]]),
                "caveats": list(CAVEATS[2:]),
            }
            if output == "json":
                output_json(summary)
                return
            console.print(f"Live records: {summary['records']}")
            for c in cells:
                console.print(
                    f"  {c['group']:4} {c['horizon']:5} n={c['n']:<4} {c['verdict']}"
                    f" — {c.get('reason', '')}"
                )
            return
        row = store.conn.execute(
            "SELECT summary_json FROM breadth_replay_runs "
            + ("WHERE id = ? " if run_id else "")
            + "ORDER BY started_at DESC LIMIT 1",
            (run_id,) if run_id else (),
        ).fetchone()
    if row is None:
        output_error("No replay on file: run `advisor breadth replay`.")
        return
    summary = json.loads(row[0])
    if output == "json":
        output_json(summary)
        return
    _print_study(summary)


@app.command("picks")
def picks(
    build: Annotated[
        bool, typer.Option("--build", help="Rebuild for the last closed session first")
    ] = False,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Names where two or more families agree, ordered by evidence, each with its reasons."""
    from advisor.breadth.picks import latest_picks, refresh_picks
    from advisor.breadth.store import BreadthStore, breadth_path
    from advisor.daemon.market_calendar import now_et

    with BreadthStore(breadth_path(_db_path())) as store:
        if build:
            built = refresh_picks(store, now_et(), _db_path())
            if not built.get("ok"):
                output_error(built.get("error", "could not build picks"))
                return
        data = latest_picks(store)
    if output == "json":
        output_json(data)
        return
    if not data["picks"]:
        console.print("No picks on file: run `advisor breadth picks --build`.")
        return
    tag = " — PROVISIONAL, live prices" if data.get("provisional") else ""
    console.print(f"Picks for {data['day']}{tag} (rules {data['rules']}), ordered by evidence:")
    for p in data["picks"]:
        flag = " [held]" if p.get("held") else ""
        console.print(
            f"\n[bold]{p['rank']}. {p['symbol']}[/bold]{flag} — {p.get('name') or ''} "
            f"({p.get('sector') or 'sector n/a'}) · families {'+'.join(p['families'])}"
        )
        for r in p["reasons"]:
            console.print(f"   · {r['text']} [dim]({r['source']})[/dim]")
        if p.get("invalidates"):
            console.print(f"   [dim]would undo it: {'; '.join(p['invalidates'])}[/dim]")


@app.command("universe")
def universe(
    symbol: Annotated[Optional[str], typer.Argument(help="Show one name's standing")] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """The latest E0 snapshot: the funnel, or one name's standing and why."""
    from advisor.breadth.store import BreadthStore, breadth_path
    from advisor.breadth.universe import funnel

    with BreadthStore(breadth_path(_db_path())) as store:
        day = store.latest_universe_day()
        if day is None:
            output_error("No universe yet: run `advisor breadth sync`.")
            return
        rows = store.universe(day)
        last = store.last_run()
        counts = store.counts()
    if symbol:
        match = [r for r in rows if r["symbol"] == symbol.strip().upper().replace(".", "-")]
        if not match:
            output_error(f"{symbol.upper()} is not in the {day} symbol directory.")
            return
        r = match[0]
        if output == "json":
            output_json(r)
            return
        verdict = "eligible" if r["eligible"] else f"excluded: {r['reason']}"
        console.print(f"{r['symbol']} ({r['name']}, {r['exchange']}) on {day}: {verdict}")
        if r["price"] is not None:
            dv = r["dollar_volume"]
            console.print(
                f"  close ${r['price']:.2f}, median dollar volume "
                f"{'n/a' if dv is None else f'${dv / 1e6:,.1f}M'}, {r['sessions']} sessions"
            )
        return
    f = {"day": day.isoformat(), "rules": rows[0]["rules"] if rows else None, **funnel(rows)}
    if output == "json":
        output_json({"universe": f, "store": counts, "last_run": last})
        return
    _print_funnel(f)
    console.print(f"Store: {counts}")
    if last:
        console.print(f"Last sync {last['finished_at']} — {'ok' if last['ok'] else 'INCOMPLETE'}")
