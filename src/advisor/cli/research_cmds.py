"""CLI commands for the fundamental research engine."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Optional

import typer

from advisor.cli.formatters import console, output_error, output_json

app = typer.Typer(name="research", help="Deep fundamental stock research")


@app.command("statements")
def research_statements(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    n_years: Annotated[int, typer.Option("--years", help="Years of history")] = 5,
    source: Annotated[
        str,
        typer.Option("--source", help="Data source: 'auto' | 'edgar' | 'yfinance'"),
    ] = "auto",
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Pull 5-yr financial statements (income, balance, cash flow)."""
    from advisor.research.edgar import EdgarClient
    from advisor.research.statements import extract_statements

    edgar_client = None
    if source in ("auto", "edgar"):
        try:
            edgar_client = EdgarClient()
        except Exception as exc:  # noqa: BLE001
            if source == "edgar":
                output_error(f"EDGAR client init failed: {exc}")
                return
            edgar_client = None

    try:
        bundle = extract_statements(
            symbol,
            edgar_client=edgar_client,
            n_years=n_years,
            fallback_to_yfinance=(source != "edgar"),
        )
    except Exception as exc:  # noqa: BLE001
        output_error(f"Failed to fetch statements for {symbol}: {exc}")
        return

    if output == "json":
        output_json(bundle)
        return

    _render_statements(bundle)


@app.command("ratios")
def research_ratios(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    n_years: Annotated[int, typer.Option("--years", help="Years of history")] = 5,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
    flag: Annotated[
        bool,
        typer.Option("--red-flags/--no-red-flags", help="Also emit detected red flags"),
    ] = True,
    save: Annotated[
        bool,
        typer.Option("--save/--no-save", help="Persist the (partial) report to the research store"),
    ] = False,
) -> None:
    """Compute ratios + optional red flags from the latest statements."""
    from advisor.research.config import get_settings
    from advisor.research.edgar import EdgarClient
    from advisor.research.models import ResearchReport
    from advisor.research.ratios import compute_ratios
    from advisor.research.red_flags import detect_red_flags
    from advisor.research.statements import extract_statements
    from advisor.research.store import ResearchStore

    try:
        edgar_client = EdgarClient()
    except Exception:  # noqa: BLE001
        edgar_client = None

    try:
        bundle = extract_statements(symbol, edgar_client=edgar_client, n_years=n_years)
        ratios = compute_ratios(bundle)
    except Exception as exc:  # noqa: BLE001
        output_error(f"Failed to compute ratios for {symbol}: {exc}")
        return

    red_flags = detect_red_flags(bundle, ratios) if flag else None

    if save:
        report = ResearchReport(
            symbol=symbol.upper(),
            as_of=date.today(),
            statements=bundle,
            ratios=ratios,
            red_flags=red_flags,
        )
        ResearchStore(get_settings().db_path).save_report(report)

    if output == "json":
        payload = {
            "symbol": symbol.upper(),
            "ratios": ratios.model_dump(mode="json"),
            "red_flags": red_flags.model_dump(mode="json") if red_flags else None,
        }
        output_json(payload)
        return

    _render_ratios(ratios)
    if red_flags is not None:
        _render_red_flags(red_flags)


# ── Renderers ────────────────────────────────────────────────────────────────


def _render_statements(bundle) -> None:  # type: ignore[no-untyped-def]
    from rich.table import Table

    console.print(
        f"[cyan]Statements for {bundle.symbol}[/cyan] "
        f"(source: {bundle.source}, periods: {len(bundle.income)})"
    )

    def _row(p, fields: list[str]) -> list[str]:
        return [str(p.fiscal_year)] + [_fmt(getattr(p, f)) for f in fields]

    inc_fields = [
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "eps_diluted",
    ]
    income_table = Table(title="Income Statement")
    income_table.add_column("FY", style="cyan")
    for f in inc_fields:
        income_table.add_column(f, justify="right")
    for p in bundle.income:
        income_table.add_row(*_row(p, inc_fields))
    console.print(income_table)

    bs_fields = [
        "cash_and_equivalents",
        "current_assets",
        "total_assets",
        "long_term_debt",
        "total_equity",
    ]
    bs_table = Table(title="Balance Sheet")
    bs_table.add_column("FY", style="cyan")
    for f in bs_fields:
        bs_table.add_column(f, justify="right")
    for p in bundle.balance:
        bs_table.add_row(*_row(p, bs_fields))
    console.print(bs_table)

    cf_fields = [
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "dividends_paid",
        "share_repurchases",
    ]
    cf_table = Table(title="Cash Flow Statement")
    cf_table.add_column("FY", style="cyan")
    for f in cf_fields:
        cf_table.add_column(f, justify="right")
    for p in bundle.cashflow:
        cf_table.add_row(*_row(p, cf_fields))
    console.print(cf_table)


def _render_ratios(ratios) -> None:  # type: ignore[no-untyped-def]
    from rich.table import Table

    def _table(title: str, cols: list[tuple[str, str, str]]) -> Table:
        t = Table(title=title, show_lines=False)
        t.add_column("FY", style="cyan", width=6)
        for _, header, _ in cols:
            t.add_column(header, justify="right", min_width=8)
        for p in ratios.periods:
            row = [str(p.fiscal_year)]
            for field, _, unit in cols:
                row.append(_fmt(getattr(p, field), unit))
            t.add_row(*row)
        return t

    console.print(
        _table(
            f"Profitability — {ratios.symbol}",
            [
                ("gross_margin", "Gross Mgn", "pct"),
                ("operating_margin", "Op Mgn", "pct"),
                ("net_margin", "Net Mgn", "pct"),
                ("roe", "ROE", "pct"),
                ("roic", "ROIC", "pct"),
            ],
        )
    )
    console.print(
        _table(
            "Liquidity & Leverage",
            [
                ("current_ratio", "Current", "x"),
                ("quick_ratio", "Quick", "x"),
                ("debt_to_equity", "D/E", "x"),
                ("debt_to_ebitda", "D/EBITDA", "x"),
                ("interest_coverage", "Int Cov", "x"),
            ],
        )
    )
    console.print(
        _table(
            "Efficiency & Cash Quality",
            [
                ("asset_turnover", "Asset TO", "x"),
                ("inventory_turns", "Inv Turns", "x"),
                ("dso", "DSO", "d"),
                ("fcf_margin", "FCF Mgn", "pct"),
                ("capex_intensity", "Capex/Rev", "pct"),
                ("fcf_to_net_income", "FCF/NI", "x"),
            ],
        )
    )
    if ratios.share_count_cagr_3y is not None:
        cagr = ratios.share_count_cagr_3y
        color = "red" if cagr > 0.03 else "green" if cagr < -0.01 else "yellow"
        console.print(f"Share count 3y CAGR: [{color}]{cagr:+.1%}[/{color}]")


def _render_red_flags(red_flags) -> None:  # type: ignore[no-untyped-def]
    from rich.table import Table

    if not red_flags.flags:
        console.print("[green]No red flags detected.[/green]")
        return

    table = Table(title=f"Red Flags — {red_flags.symbol}")
    table.add_column("Severity", style="bold")
    table.add_column("Code", style="cyan")
    table.add_column("Title")
    table.add_column("Detail")
    for f in red_flags.flags:
        color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "dim"}.get(f.severity.value, "white")
        table.add_row(f"[{color}]{f.severity.value}[/{color}]", f.code, f.title, f.detail)
    console.print(table)


def _fmt(value, unit: str = "raw") -> str:
    if value is None:
        return "—"
    if unit == "pct":
        return f"{value * 100:+.1f}%"
    if unit == "x":
        return f"{value:.2f}x"
    if unit == "d":
        return f"{value:.0f}d"
    # raw number — auto-scale
    abs_v = abs(value)
    if abs_v >= 1e9:
        return f"{value / 1e9:,.2f}B"
    if abs_v >= 1e6:
        return f"{value / 1e6:,.1f}M"
    if abs_v >= 1e3:
        return f"{value / 1e3:,.1f}K"
    return f"{value:,.2f}"
