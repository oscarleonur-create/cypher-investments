"""CLI commands for confluence scanning."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from advisor.cli.formatters import console, output_error, output_json

app = typer.Typer(name="confluence", help="Confluence scanner")


def _dip_row(table, check: str, value, threshold: str, ok: bool | None) -> None:
    """Add a row to the dip screener detail table."""
    val_str = str(value) if value is not None else "N/A"
    if ok is None:
        status = "[dim]-[/dim]"
    elif ok:
        status = "[green]PASS[/green]"
    else:
        status = "[red]FAIL[/red]"
    table.add_row(check, val_str, str(threshold), status)


@app.command("scan")
def confluence_scan(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol to scan")],
    strategy: Annotated[
        str, typer.Option("--strategy", "-s", help="Strategy to use for the technical check")
    ] = "momentum_breakout",
    output: Annotated[
        Optional[str], typer.Option("--output", help="Output format (json)")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed agent results")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Run sentiment + fundamental even without breakout (for dip analysis)")
    ] = False,
) -> None:
    """Run all three confluence checks (technical, sentiment, fundamental) on a symbol."""
    from rich.panel import Panel
    from rich.table import Table

    from advisor.confluence.orchestrator import run_confluence

    try:
        result = run_confluence(symbol.upper(), strategy_name=strategy, force_all=force)
    except Exception as e:
        output_error(f"Confluence scan failed: {e}")
        return

    if output == "json":
        output_json(result)
        return

    # Verdict panel
    verdict_colors = {"ENTER": "green", "CAUTION": "yellow", "PASS": "red"}
    color = verdict_colors.get(result.verdict.value, "white")
    console.print(
        Panel(
            f"[bold {color}]{result.verdict.value}[/bold {color}]\n\n"
            f"{result.reasoning}\n\n"
            f"Suggested hold: {result.suggested_hold_days} days",
            title=f"Confluence: {result.symbol} ({result.strategy_name})",
        )
    )

    # Agent results table
    table = Table(title="Agent Results")
    table.add_column("Agent", style="cyan")
    table.add_column("Status")
    table.add_column("Detail")

    # Technical
    tech_color = "green" if result.technical.is_bullish else "red"
    tech_status = f"[{tech_color}]{'BULLISH' if result.technical.is_bullish else 'BEARISH'}[/{tech_color}]"
    tech_detail = f"Signal: {result.technical.signal}, Price: ${result.technical.price:,.2f}"
    table.add_row("Technical", tech_status, tech_detail)

    # Sentiment
    sent_color = "green" if result.sentiment.is_bullish else "red"
    sent_status = f"[{sent_color}]{'BULLISH' if result.sentiment.is_bullish else 'BEARISH'}[/{sent_color}]"
    sent_detail = f"Score: {result.sentiment.score:.0f}/100, Positive: {result.sentiment.positive_pct:.0f}%"
    table.add_row("Sentiment", sent_status, sent_detail)

    # Fundamental
    ds = result.fundamental.dip_screener
    fund_color = "green" if result.fundamental.is_clear else "red"
    fund_status = f"[{fund_color}]{'CLEAR' if result.fundamental.is_clear else 'RISK'}[/{fund_color}]"

    if ds is not None:
        # Dip screener: show overall score instead of simple detail
        score_colors = {
            "STRONG_BUY": "bold green", "BUY": "green", "LEAN_BUY": "cyan",
            "WATCH": "yellow", "WEAK": "dim", "FAIL": "red",
        }
        sc = score_colors.get(ds.overall_score, "white")
        fund_detail = f"Dip Score: [{sc}]{ds.overall_score}[/{sc}]"
        if ds.rejection_reason:
            fund_detail += f" ({ds.rejection_reason})"
        table.add_row("Fundamental", fund_status, fund_detail)

        # Dip screener layer rows
        safety = ds.safety
        s_color = "green" if safety.passes else "red"
        s_parts = []
        s_parts.append(f"CR: {safety.current_ratio or 'N/A'}")
        s_parts.append(f"D/E: {safety.debt_to_equity or 'N/A'}")
        fcf_label = "OK" if safety.fcf_ok else "FAIL"
        s_parts.append(f"FCF: {fcf_label}")
        table.add_row(
            "  Safety Gate",
            f"[{s_color}]{'PASS' if safety.passes else 'FAIL'}[/{s_color}]",
            ", ".join(s_parts),
        )

        if ds.value_trap is not None:
            vt = ds.value_trap
            v_color = "green" if vt.is_value else "dim"
            v_parts = []
            if vt.pe_discount_pct is not None:
                v_parts.append(f"P/E discount: {vt.pe_discount_pct:+.1f}%")
            if vt.rsi_divergence:
                v_parts.append("RSI divergence")
            if not v_parts:
                v_parts.append("No value signal")
            table.add_row(
                "  Value Trap",
                f"[{v_color}]{'VALUE' if vt.is_value else 'NO SIGNAL'}[/{v_color}]",
                ", ".join(v_parts),
            )

        if ds.fast_fundamentals is not None:
            ff = ds.fast_fundamentals
            f_color = "green" if ff.has_confirmation else "dim"
            f_parts = []
            if ff.c_suite_buying:
                f_parts.append("C-suite buying")
            elif ff.insider_buying:
                f_parts.append("Insider buying")
            if ff.analyst_upside_pct is not None:
                f_parts.append(f"Analyst upside: {ff.analyst_upside_pct:+.1f}% ({ff.n_analysts} analysts)")
            if not f_parts:
                f_parts.append("No timing signal")
            table.add_row(
                "  Fast Fundamentals",
                f"[{f_color}]{'CONFIRMED' if ff.has_confirmation else 'NO SIGNAL'}[/{f_color}]",
                ", ".join(f_parts),
            )
    else:
        fund_parts = []
        if result.fundamental.earnings_within_7_days:
            fund_parts.append(f"Earnings: {result.fundamental.earnings_date}")
        else:
            fund_parts.append("No imminent earnings")
        if result.fundamental.insider_buying_detected:
            fund_parts.append("Insider buying detected")
        table.add_row("Fundamental", fund_status, ", ".join(fund_parts))

    console.print(table)

    # Verbose: show headlines and sources
    if verbose:
        if result.sentiment.key_headlines:
            console.print("\n[bold]Key Headlines:[/bold]")
            for headline in result.sentiment.key_headlines:
                console.print(f"  - {headline}")

        if result.sentiment.sources:
            source_table = Table(title="Sources")
            source_table.add_column("ID", style="dim")
            source_table.add_column("Tier", justify="center")
            source_table.add_column("Title")
            source_table.add_column("URL", style="dim")
            for src in result.sentiment.sources:
                tier_color = {1: "green", 2: "yellow"}.get(src.tier, "dim")
                source_table.add_row(
                    src.source_id,
                    f"[{tier_color}]{src.tier}[/{tier_color}]",
                    src.title,
                    src.url,
                )
            console.print(source_table)

        if ds is not None:
            dip_table = Table(title="Dip Screener Detail")
            dip_table.add_column("Check", style="cyan")
            dip_table.add_column("Value")
            dip_table.add_column("Threshold")
            dip_table.add_column("Status")

            s = ds.safety
            _dip_row(dip_table, "Current Ratio", s.current_ratio, "> 1.5", s.current_ratio_ok)
            _dip_row(dip_table, "Debt/Equity", s.debt_to_equity, "< 2.0", s.debt_to_equity_ok)
            _dip_row(dip_table, "FCF (4Q positive)", len([v for v in s.fcf_values if v > 0]), "4/4", s.fcf_ok)

            if ds.value_trap is not None:
                vt = ds.value_trap
                _dip_row(dip_table, "Current P/E", vt.current_pe, "-", None)
                _dip_row(dip_table, "5yr Avg P/E", vt.five_year_avg_pe, "-", None)
                _dip_row(dip_table, "P/E Discount", f"{vt.pe_discount_pct}%" if vt.pe_discount_pct else "N/A", ">= 20%", vt.pe_on_sale)
                _dip_row(dip_table, "Price Change", f"{vt.price_change_pct}%" if vt.price_change_pct else "N/A", "<= -10%", None)
                _dip_row(dip_table, "RSI Divergence", vt.rsi_divergence, "True", vt.rsi_divergence)

            if ds.fast_fundamentals is not None:
                ff = ds.fast_fundamentals
                _dip_row(dip_table, "Insider Buying", ff.insider_buying, "True", ff.insider_buying)
                _dip_row(dip_table, "C-Suite Buying", ff.c_suite_buying, "True", ff.c_suite_buying)
                _dip_row(dip_table, "Analyst Upside", f"{ff.analyst_upside_pct}%" if ff.analyst_upside_pct else "N/A", ">= 15%", ff.analyst_bullish)
                _dip_row(dip_table, "# Analysts", ff.n_analysts, ">= 3", ff.n_analysts >= 3)

                if ff.insider_details:
                    for detail in ff.insider_details:
                        name = detail.get("name", "")
                        title = detail.get("title", "")
                        shares = detail.get("shares", "")
                        _dip_row(dip_table, f"  {name}", title, f"{shares} shares", None)

            console.print(dip_table)
