"""CLI for the learning loop: which rule versions exist and what each one produced."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Optional

import typer
from rich.table import Table

from advisor.cli.formatters import console, output_error, output_json

app = typer.Typer(name="learn", help="Rule versions and, later, what each one has earned")


def _current() -> dict[str, str]:
    """``{ruleset: version}`` of the rules this code runs now."""
    from advisor.entry.ruleset import entry_rules
    from advisor.scanner.ruleset import premarket_rules, session_rules

    return {s.ruleset: s.version for s in (session_rules(), premarket_rules(), entry_rules())}


def _current_params(ruleset: str) -> dict | None:
    from advisor.entry.ruleset import entry_rules
    from advisor.scanner.ruleset import premarket_rules, session_rules

    for s in (session_rules(), premarket_rules(), entry_rules()):
        if s.ruleset == ruleset:
            return s.params
    return None


@app.command("rules")
def rules(
    version: Annotated[
        Optional[str], typer.Argument(help="A version (or prefix) to show in full")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Rule versions on file, how many records each produced, and which one runs now."""
    from advisor.learning.rules import PRE_REGISTRY
    from advisor.learning.store import RuleStore
    from advisor.research.config import get_settings

    conn = sqlite3.connect(str(get_settings().db_path))
    try:
        store = RuleStore(conn)
        if version is not None:
            _show(store, version, output)
            return
        versions = store.list()
        counts = store.record_counts()
    finally:
        conn.close()
    current = _current()

    rows = []
    for v in versions:
        rows.append(
            {
                "ruleset": v.ruleset,
                "version": v.version,
                "current": current.get(v.ruleset) == v.version,
                "first_seen_at": v.first_seen_at.isoformat(),
                "first_code": v.first_code,
                "records": {label: c.get(v.version, 0) for label, c in counts.items()},
            }
        )
    unstamped = {label: c[PRE_REGISTRY] for label, c in counts.items() if PRE_REGISTRY in c}
    not_yet = sorted(rs for rs, ver in current.items() if ver not in {v.version for v in versions})
    if output == "json":
        output_json(
            {
                "versions": rows,
                "running": current,
                "not_yet_recorded": not_yet,
                "pre_registry": unstamped,
            }
        )
        return

    table = Table(title="Rule versions")
    for col in ("ruleset", "version", "now", "first seen", "first code", "records"):
        table.add_column(col)
    for r in rows:
        recs = ", ".join(f"{n} {label}" for label, n in r["records"].items() if n) or "—"
        table.add_row(
            r["ruleset"],
            r["version"],
            "●" if r["current"] else "",
            r["first_seen_at"][:16],
            r["first_code"],
            recs,
        )
    console.print(table)
    for rs in not_yet:
        console.print(f"[dim]{rs} {current[rs]} runs now but has produced no record yet[/dim]")
    if unstamped:
        text = ", ".join(f"{n} {label}" for label, n in unstamped.items())
        console.print(f"[dim]{PRE_REGISTRY}: {text} written before versions were stamped[/dim]")


def _show(store, version: str, output: str) -> None:
    v = store.get(version)
    if v is None:
        output_error(f"no single rule version matches {version!r}")
        return
    now = _current_params(v.ruleset) or {}
    changed = sorted(k for k in set(v.params) | set(now) if v.params.get(k) != now.get(k))
    if output == "json":
        output_json({**v.model_dump(mode="json"), "differs_from_running": changed})
        return
    table = Table(title=f"{v.ruleset} {v.version} (first run on {v.first_code})")
    for col in ("parameter", "value", "kind", "running now"):
        table.add_column(col)
    for name in sorted(set(v.params) | set(now)):
        here = v.params.get(name, "—")
        there = now.get(name, "—")
        table.add_row(
            name,
            str(here),
            v.kinds.get(name, "—"),
            "" if name not in changed else str(there),
        )
    console.print(table)
    if not changed:
        console.print("[dim]identical to the rules running now[/dim]")
