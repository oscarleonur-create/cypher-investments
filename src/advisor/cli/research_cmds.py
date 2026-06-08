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


@app.command("valuation")
def research_valuation(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    peers: Annotated[
        Optional[str],
        typer.Option("--peers", help="Comma-separated peer tickers (e.g. MSFT,GOOGL,META)"),
    ] = None,
    n_years: Annotated[int, typer.Option("--years", help="Years for historical multiples")] = 5,
    output: Annotated[Optional[str], typer.Option("--output")] = None,
    save: Annotated[bool, typer.Option("--save/--no-save")] = False,
) -> None:
    """Peer comp table, 5-yr own multiple history, and DCF model."""
    from advisor.research.edgar import EdgarClient
    from advisor.research.peers import build_peer_set
    from advisor.research.statements import extract_statements
    from advisor.research.valuation.dcf import build_dcf
    from advisor.research.valuation.multiples import build_multiples
    from advisor.research.valuation.reverse_dcf import solve_implied_growth

    explicit = [p.strip().upper() for p in peers.split(",")] if peers else None

    console.print(f"[cyan]Building peer set for {symbol.upper()}…[/cyan]")
    peer_list = build_peer_set(symbol, explicit_peers=explicit)
    console.print(f"Peers: {', '.join(peer_list) or 'none found'}")

    try:
        edgar = EdgarClient()
    except Exception:  # noqa: BLE001
        edgar = None

    try:
        bundle = extract_statements(symbol, edgar_client=edgar, n_years=n_years)
    except Exception:  # noqa: BLE001
        bundle = None

    console.print("[cyan]Fetching multiples…[/cyan]")
    multiples = build_multiples(symbol, peer_list, statements=bundle)

    console.print("[cyan]Building DCF…[/cyan]")
    dcf = build_dcf(symbol, statements=bundle)
    dcf.implied_growth_rate = solve_implied_growth(dcf)

    if save:
        from advisor.research.config import get_settings
        from advisor.research.models import ResearchReport
        from advisor.research.store import ResearchStore

        report = ResearchReport(
            symbol=symbol.upper(),
            as_of=date.today(),
            statements=bundle,
            multiples=multiples,
            dcf=dcf,
        )
        ResearchStore(get_settings().db_path).save_report(report)

    if output == "json":
        output_json(
            {"multiples": multiples.model_dump(mode="json"), "dcf": dcf.model_dump(mode="json")}
        )
        return

    _render_multiples(multiples)
    _render_dcf(dcf)


@app.command("ecosystem")
def research_ecosystem(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    output: Annotated[Optional[str], typer.Option("--output")] = None,
    with_customers: Annotated[bool, typer.Option("--customers/--no-customers")] = True,
) -> None:
    """Institutional holders, insider activity, and key customers."""
    from advisor.research.ecosystem.holders import get_holders
    from advisor.research.ecosystem.insiders import get_insiders
    from advisor.research.models import EcosystemResult

    console.print(f"[cyan]Fetching ecosystem data for {symbol.upper()}…[/cyan]")

    holders = get_holders(symbol)
    insiders = get_insiders(symbol)

    customers = []
    if with_customers:
        from advisor.research.ecosystem.customers import get_customers

        console.print("[dim]Searching for customer disclosures (may take a moment)…[/dim]")
        customers = get_customers(symbol)

    ecosystem = EcosystemResult(
        symbol=symbol.upper(),
        holders=holders,
        insiders=insiders,
        top_customers=customers,
    )

    if output == "json":
        output_json(ecosystem)
        return

    _render_holders(holders)
    _render_insiders(insiders)
    if customers:
        _render_customers(customers)


@app.command("compare")
def research_compare(
    symbols: Annotated[str, typer.Argument(help="Comma-separated tickers (e.g. AAPL,MSFT,GOOGL)")],
    output: Annotated[Optional[str], typer.Option("--output")] = None,
) -> None:
    """Side-by-side multiples comparison for a manually specified peer set."""
    from advisor.research.valuation.multiples import _snapshot

    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if len(tickers) < 2:
        output_error("Provide at least 2 tickers separated by commas (e.g. AAPL,MSFT)")
        return

    console.print(f"[cyan]Fetching snapshots for {', '.join(tickers)}…[/cyan]")
    snaps = [_snapshot(t) for t in tickers]

    if output == "json":
        output_json([s.model_dump(mode="json") for s in snaps])
        return

    _render_compare_table(snaps)


@app.command("ticker")
def research_ticker(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    peers: Annotated[
        Optional[str],
        typer.Option("--peers", help="Comma-separated peer tickers"),
    ] = None,
    n_years: Annotated[int, typer.Option("--years", help="Years of statements")] = 5,
    output: Annotated[Optional[str], typer.Option("--output")] = None,
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Ignore cached report and rebuild"),
    ] = False,
) -> None:
    """Full 7-layer research report for a ticker (cached to SQLite)."""
    from advisor.research.render import render_report
    from advisor.research.report import build_report

    explicit_peers = [p.strip().upper() for p in peers.split(",")] if peers else None

    console.print(f"[cyan]Building full research report for {symbol.upper()}…[/cyan]")
    console.print("[dim]This may take 1-3 minutes for the first run.[/dim]")
    report = build_report(
        symbol, n_years=n_years, explicit_peers=explicit_peers, force_refresh=force
    )

    if output == "json":
        output_json(report)
        return

    md = render_report(report)
    console.print(md)


@app.command("refresh")
def research_refresh(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    peers: Annotated[Optional[str], typer.Option("--peers")] = None,
    n_years: Annotated[int, typer.Option("--years")] = 5,
    output: Annotated[Optional[str], typer.Option("--output")] = None,
) -> None:
    """Force-rebuild the research report, bypassing the cache."""
    from advisor.research.render import render_report
    from advisor.research.report import build_report

    explicit_peers = [p.strip().upper() for p in peers.split(",")] if peers else None
    console.print(f"[cyan]Force-refreshing report for {symbol.upper()}…[/cyan]")
    report = build_report(
        symbol, n_years=n_years, explicit_peers=explicit_peers, force_refresh=True
    )

    if output == "json":
        output_json(report)
        return

    console.print(render_report(report))


@app.command("view")
def research_view(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    output: Annotated[Optional[str], typer.Option("--output")] = None,
) -> None:
    """Render the latest cached research report for a ticker."""
    from advisor.research.config import get_settings
    from advisor.research.render import render_report
    from advisor.research.store import ResearchStore

    report = ResearchStore(get_settings().db_path).load_latest_report(symbol.upper())
    if report is None:
        output_error(
            f"No cached report for {symbol.upper()}. Run: advisor research ticker {symbol}"
        )
        return

    if output == "json":
        output_json(report)
        return

    console.print(render_report(report))


@app.command("list")
def research_list(
    limit: Annotated[int, typer.Option("--limit", help="Max rows to show")] = 20,
    output: Annotated[Optional[str], typer.Option("--output")] = None,
) -> None:
    """List all cached research reports."""
    from advisor.research.config import get_settings
    from advisor.research.store import ResearchStore

    rows = ResearchStore(get_settings().db_path).list_reports(limit=limit)
    if not rows:
        console.print("[dim]No research reports cached yet.[/dim]")
        return

    if output == "json":
        output_json(rows)
        return

    from rich.table import Table

    table = Table(title="Cached Research Reports")
    table.add_column("Symbol", style="cyan", width=8)
    table.add_column("As Of", width=12)
    table.add_column("Saved At", width=22)
    for row in rows:
        table.add_row(row.get("symbol", ""), row.get("as_of", ""), row.get("saved_at", ""))
    console.print(table)


@app.command("monitor")
def research_monitor(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild/--no-rebuild",
            help="Re-generate KPI definitions from the thesis (re-runs LLM)",
        ),
    ] = False,
    output: Annotated[Optional[str], typer.Option("--output")] = None,
) -> None:
    """Re-check KPI values for a ticker and show thesis health status."""
    from advisor.research.kpi_tracker import build_thesis_monitor, re_check_kpis

    sym = symbol.upper()
    console.print(f"[cyan]{'Rebuilding' if rebuild else 'Refreshing'} KPIs for {sym}…[/cyan]")

    if rebuild:
        from advisor.research.config import get_settings
        from advisor.research.store import ResearchStore

        report = ResearchStore(get_settings().db_path).load_latest_report(sym)
        if report is None:
            output_error(f"No cached report for {sym}. Run: advisor research ticker {sym}")
            return
        result = build_thesis_monitor(report)
        store = ResearchStore(get_settings().db_path)
        try:
            store.save_kpi_watchlist(sym, result.kpis)
            store.save_kpi_check(sym, result)
        finally:
            store.close()
    else:
        result = re_check_kpis(sym)

    if output == "json":
        output_json(result.model_dump(mode="json"))
        return

    _render_monitor(result)


@app.command("memo")
def research_memo(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    account: Annotated[
        Optional[float],
        typer.Option("--account", help="Account size in USD (reserved for future dollar sizing)"),
    ] = None,
    output: Annotated[Optional[str], typer.Option("--output")] = None,
) -> None:
    """Render the IC-format Investment Memo from the cached research report."""
    from advisor.research.config import get_settings
    from advisor.research.investment_memo import build_investment_memo
    from advisor.research.management_quality import build_management_quality
    from advisor.research.store import ResearchStore

    sym = symbol.upper()
    report = ResearchStore(get_settings().db_path).load_latest_report(sym)
    if report is None:
        output_error(f"No cached report for {sym}. Run: advisor research ticker {sym}")
        return

    # Compute management quality if not already present
    if report.management_quality is None and report.ecosystem:
        report.management_quality = build_management_quality(
            sym,
            ratios=report.ratios,
            holders=report.ecosystem.holders,
            insiders=report.ecosystem.insiders,
        )

    memo = build_investment_memo(report, account_size=account)

    if output == "json":
        output_json(memo.model_dump(mode="json"))
        return

    _render_memo(memo)


@app.command("network")
def research_network(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    output: Annotated[Optional[str], typer.Option("--output")] = None,
) -> None:
    """Map Level-2 connections: peers, public holders, competitors + comparison table."""
    from advisor.research.network import build_network

    console.print(f"[cyan]Building company network for {symbol.upper()}…[/cyan]")
    net = build_network(symbol)

    if output == "json":
        output_json(net.model_dump(mode="json"))
        return

    _render_network(net)


@app.command("catalysts")
def research_catalysts(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    output: Annotated[Optional[str], typer.Option("--output")] = None,
) -> None:
    """Upcoming catalysts and ranked risk register."""
    from advisor.research.catalysts import build_catalysts
    from advisor.research.risks import build_risks

    console.print(f"[cyan]Building catalyst/risk register for {symbol.upper()}…[/cyan]")
    result = build_catalysts(symbol)
    result = build_risks(symbol, catalyst_result=result)

    if output == "json":
        output_json(result)
        return

    _render_catalysts(result)


# ── Phase 2 renderers ────────────────────────────────────────────────────────


def _render_multiples(m) -> None:  # type: ignore[no-untyped-def]
    from rich.table import Table

    table = Table(title="Peer Comp Table")
    table.add_column("Ticker", style="cyan", width=8)
    table.add_column("Name", max_width=20)
    cols = [
        ("market_cap", "Mkt Cap", "raw"),
        ("revenue_growth_yoy", "Rev Grw", "pct"),
        ("gross_margin", "Gr Mgn", "pct"),
        ("operating_margin", "Op Mgn", "pct"),
        ("ev_to_sales", "EV/S", "x"),
        ("ev_to_ebitda", "EV/EBITDA", "x"),
        ("pe_trailing", "P/E", "x"),
        ("pe_forward", "Fwd P/E", "x"),
        ("p_to_fcf", "P/FCF", "x"),
        ("peg", "PEG", "x"),
    ]
    for _, header, _ in cols:
        table.add_column(header, justify="right", min_width=7)

    for snap in m.all_snapshots:
        style = "bold" if snap.symbol == m.subject.symbol else ""
        row = [snap.symbol, snap.name[:20]]
        for field, _, unit in cols:
            row.append(_fmt(getattr(snap, field), unit))
        table.add_row(*row, style=style)

    console.print(table)

    if m.own_history:
        hist = Table(title=f"{m.subject.symbol} — Own Multiple History")
        hist.add_column("FY", style="cyan")
        for h in ["EV/Sales", "EV/EBITDA", "P/E", "P/FCF"]:
            hist.add_column(h, justify="right")
        for p in m.own_history:
            hist.add_row(
                str(p.fiscal_year),
                _fmt(p.ev_to_sales, "x"),
                _fmt(p.ev_to_ebitda, "x"),
                _fmt(p.pe, "x"),
                _fmt(p.p_to_fcf, "x"),
            )
        console.print(hist)


def _render_dcf(dcf) -> None:  # type: ignore[no-untyped-def]
    from rich.table import Table

    console.print(
        f"\n[bold]DCF — {dcf.symbol}[/bold]  "
        f"Current: [cyan]{_fmt(dcf.current_price)}[/cyan]  "
        f"WACC: {dcf.wacc:.1%}  Rf: {dcf.risk_free_rate:.1%}  β: {dcf.beta:.2f}"
    )

    table = Table(title="Scenario Analysis")
    table.add_column("Scenario", style="cyan")
    table.add_column("Growth Yr1-3", justify="right")
    table.add_column("FCF Margin", justify="right")
    table.add_column("Implied Price", justify="right")
    table.add_column("Upside", justify="right")

    if dcf.implied_growth_rate is not None:
        console.print(
            f"[yellow]Implied growth rate (reverse-DCF): " f"{dcf.implied_growth_rate:.1%}[/yellow]"
        )

    for scenario in [dcf.base, dcf.bull, dcf.bear]:
        if scenario is None:
            continue
        a = scenario.assumptions
        color = (
            "green"
            if scenario.upside_pct > 0.10
            else "red"
            if scenario.upside_pct < -0.10
            else "yellow"
        )
        upside_str = f"[{color}]{scenario.upside_pct:+.1%}[/{color}]"
        table.add_row(
            a.scenario.upper(),
            f"{a.revenue_growth_yr1_3:.1%}",
            f"{a.target_fcf_margin:.1%}",
            _fmt(scenario.implied_price),
            upside_str,
        )

    console.print(table)


def _render_holders(holders) -> None:  # type: ignore[no-untyped-def]
    from rich.table import Table

    console.print(
        f"\n[bold]Institutional Holders — {holders.symbol}[/bold]  "
        f"Inst: {_fmt_pct(holders.pct_institutional)}  "
        f"Insider: {_fmt_pct(holders.pct_insider)}"
    )
    if not holders.top_holders:
        console.print("[dim]No holder data available.[/dim]")
        return

    table = Table()
    table.add_column("#", style="dim", width=3)
    table.add_column("Holder", min_width=25)
    table.add_column("Shares", justify="right")
    table.add_column("% Out", justify="right")
    table.add_column("Value", justify="right")
    for i, h in enumerate(holders.top_holders[:15], 1):
        table.add_row(
            str(i),
            h.name,
            _fmt(h.shares),
            _fmt_pct(h.pct_held),
            _fmt(h.value_usd),
        )
    console.print(table)


def _render_insiders(insiders) -> None:  # type: ignore[no-untyped-def]
    from rich.table import Table

    net = insiders.net_buying_usd
    color = "green" if net > 0 else "red"
    console.print(
        f"\n[bold]Insider Activity — {insiders.symbol}[/bold] "
        f"(trailing {insiders.lookback_days}d)  "
        f"Net: [{color}]{_fmt(net)}[/{color}]  "
        f"C-suite buying: {'[green]Yes[/green]' if insiders.c_suite_buying else '[dim]No[/dim]'}"
    )
    if not insiders.transactions:
        console.print("[dim]No insider transactions found.[/dim]")
        return

    table = Table()
    table.add_column("Date", width=12)
    table.add_column("Insider", min_width=20)
    table.add_column("Title", min_width=12)
    table.add_column("Type", width=12)
    table.add_column("Shares", justify="right")
    table.add_column("Value", justify="right")
    for tx in insiders.transactions[:12]:
        color = "green" if "purchase" in tx.transaction_type.lower() else "red"
        table.add_row(
            str(tx.transaction_date or ""),
            tx.insider_name[:20],
            tx.title[:12],
            f"[{color}]{tx.transaction_type}[/{color}]",
            _fmt(tx.shares),
            _fmt(tx.value_usd),
        )
    console.print(table)


def _render_customers(customers) -> None:  # type: ignore[no-untyped-def]
    from rich.table import Table

    console.print("\n[bold]Key Customers[/bold]")
    table = Table()
    table.add_column("Customer", min_width=25)
    table.add_column("Revenue %", justify="right")
    table.add_column("Note")
    for c in customers:
        table.add_row(
            c.name, _fmt_pct(c.concentration_pct / 100 if c.concentration_pct else None), c.note
        )
    console.print(table)


def _render_compare_table(snaps) -> None:  # type: ignore[no-untyped-def]
    from rich.table import Table

    table = Table(title="Comparison")
    table.add_column("Ticker", style="cyan", width=8)
    table.add_column("Name", max_width=22)
    for header in [
        "Mkt Cap",
        "Rev Grw",
        "Gr Mgn",
        "Op Mgn",
        "EV/S",
        "EV/EBITDA",
        "P/E",
        "P/FCF",
        "PEG",
        "Beta",
    ]:
        table.add_column(header, justify="right", min_width=7)
    for s in snaps:
        table.add_row(
            s.symbol,
            s.name[:22],
            _fmt(s.market_cap, "raw"),
            _fmt(s.revenue_growth_yoy, "pct"),
            _fmt(s.gross_margin, "pct"),
            _fmt(s.operating_margin, "pct"),
            _fmt(s.ev_to_sales, "x"),
            _fmt(s.ev_to_ebitda, "x"),
            _fmt(s.pe_trailing, "x"),
            _fmt(s.p_to_fcf, "x"),
            _fmt(s.peg, "x"),
            _fmt(s.beta, "x"),
        )
    console.print(table)


def _render_catalysts(result) -> None:  # type: ignore[no-untyped-def]
    from rich.table import Table

    if result.catalysts:
        cat_table = Table(title=f"Catalysts — {result.symbol}")
        cat_table.add_column("Type", style="cyan", width=14)
        cat_table.add_column("Near-term", width=10)
        cat_table.add_column("Date")
        cat_table.add_column("Description")
        for c in result.catalysts:
            cat_table.add_row(
                c.catalyst_type.value,
                "[green]Yes[/green]" if c.is_near_term else "[dim]No[/dim]",
                c.expected_date,
                c.description,
            )
        console.print(cat_table)

    if result.risks:
        risk_table = Table(title="Risk Register")
        risk_table.add_column("Sev", width=8)
        risk_table.add_column("Category", width=12)
        risk_table.add_column("P", justify="right", width=6)
        risk_table.add_column("I", justify="right", width=6)
        risk_table.add_column("Description")
        for r in result.risks:
            color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "dim"}.get(r.severity.value, "white")
            risk_table.add_row(
                f"[{color}]{r.severity.value}[/{color}]",
                r.category,
                f"{r.probability:.0%}" if r.probability is not None else "—",
                f"{r.impact:.0%}" if r.impact is not None else "—",
                r.description[:80],
            )
        console.print(risk_table)
    elif not result.catalysts:
        console.print("[dim]No catalyst/risk data found.[/dim]")


def _render_monitor(result) -> None:  # type: ignore[no-untyped-def]
    from rich.panel import Panel
    from rich.table import Table

    status_color = {"ON_TRACK": "green", "CAUTION": "yellow", "INVALIDATED": "red"}.get(
        result.thesis_status, "white"
    )
    console.print(
        Panel(
            f"[{status_color}]{result.thesis_status.replace('_', ' ')}[/{status_color}]",
            title=f"Thesis Monitor — {result.symbol}",
            border_style=status_color,
        )
    )

    table = Table(title=f"KPI Dashboard  ({result.checked_at:%Y-%m-%d %H:%M})")
    table.add_column("KPI", min_width=24)
    table.add_column("Current", justify="right", width=12)
    table.add_column("Bull ≥", justify="right", width=10)
    table.add_column("Bear <", justify="right", width=10)
    table.add_column("Status", width=12)

    for kpi in result.kpis:
        kpi_color = {
            "on_track": "green",
            "caution": "yellow",
            "breached": "red",
            "unknown": "dim",
        }.get(kpi.status.value, "white")
        table.add_row(
            kpi.metric_name,
            _fmt_kpi(kpi.current_value, kpi.unit),
            _fmt_kpi(kpi.bull_threshold, kpi.unit),
            _fmt_kpi(kpi.bear_threshold, kpi.unit),
            f"[{kpi_color}]{kpi.status.value.upper().replace('_', ' ')}[/{kpi_color}]",
        )
    console.print(table)

    if result.alerts:
        console.print("\n[bold]Alerts[/bold]")
        for alert in result.alerts:
            is_breach = "breached" in alert.lower() or alert.startswith("⚠")
            color = "red" if is_breach else "yellow"
            console.print(f"  [{color}]{alert}[/{color}]")


def _fmt_kpi(v: float | None, unit: str) -> str:
    if v is None:
        return "—"
    if unit == "pct":
        return f"{v:.1%}"
    if unit in ("x", "ratio"):
        return f"{v:.2f}x"
    if unit == "usd":
        abs_v = abs(v)
        if abs_v >= 1e9:
            return f"${v / 1e9:,.2f}B"
        if abs_v >= 1e6:
            return f"${v / 1e6:,.1f}M"
        return f"${v:,.0f}"
    return f"{v:.3f}"


def _render_memo(memo) -> None:  # type: ignore[no-untyped-def]
    from rich.panel import Panel
    from rich.table import Table

    conv_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "dim"}.get(memo.conviction, "white")

    # Header
    console.print(
        Panel(
            f"[bold cyan]{memo.symbol}[/bold cyan]"
            + (f" — {memo.company_name}" if memo.company_name != memo.symbol else ""),
            title="Investment Memo",
            border_style="cyan",
        )
    )

    # §0 Executive Summary
    if memo.executive_summary:
        console.print(f"\n[italic]{memo.executive_summary}[/italic]")
    if memo.key_insight:
        console.print(f"\n[bold]Edge:[/bold] {memo.key_insight}")
    if memo.mispricing_type and memo.mispricing_type not in ("none", ""):
        console.print(
            f"[dim]Mispricing type: {memo.mispricing_type.replace('_', ' ').title()}[/dim]"
        )

    # §1 Business Quality
    console.print(f"\n[bold]Moat:[/bold] {memo.moat_type.replace('_', ' ').title()}")
    if memo.moat_description:
        console.print(f"  {memo.moat_description}")
    if memo.management_quality_score is not None:
        mq_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(
            memo.management_quality_tier, "white"
        )
        mq_label = f"{memo.management_quality_score:.0f}/100 — {memo.management_quality_tier}"
        console.print(f"[bold]Management Quality:[/bold] [{mq_color}]{mq_label}[/{mq_color}]")

    # §2 Valuation
    val_table = Table(title="Valuation")
    val_table.add_column("Scenario", style="cyan")
    val_table.add_column("Target", justify="right")
    val_table.add_column("Upside", justify="right")
    if memo.bear_target:
        color = "red" if (memo.bear_upside or 0) < 0 else "green"
        val_table.add_row(
            "Bear",
            f"${memo.bear_target:,.2f}",
            f"[{color}]{memo.bear_upside:+.1%}[/{color}]" if memo.bear_upside is not None else "—",
        )
    if memo.base_target:
        color = "green" if (memo.base_upside or 0) > 0 else "red"
        val_table.add_row(
            "Base",
            f"${memo.base_target:,.2f}",
            f"[{color}]{memo.base_upside:+.1%}[/{color}]" if memo.base_upside is not None else "—",
        )
    if memo.bull_target:
        val_table.add_row(
            "Bull",
            f"${memo.bull_target:,.2f}",
            f"[green]{memo.bull_upside:+.1%}[/green]" if memo.bull_upside is not None else "—",
        )
    if memo.consensus_target:
        val_table.add_row(
            f"Consensus ({memo.n_analysts} analysts)",
            f"${memo.consensus_target:,.2f}",
            f"{memo.consensus_upside:+.1%}" if memo.consensus_upside is not None else "—",
        )
    console.print(val_table)

    # §3 Catalysts
    if memo.key_catalysts:
        console.print("\n[bold]Key Catalysts[/bold]")
        for cat in memo.key_catalysts:
            console.print(f"  • {cat}")

    # §4 Downside
    if memo.max_loss_scenario:
        console.print(f"\n[bold red]Max-Loss Scenario:[/bold red] {memo.max_loss_scenario}")

    # §5 Position Sizing
    console.print(
        f"\n[bold]Position Sizing:[/bold] "
        f"[{conv_color}]{memo.conviction}[/{conv_color}] conviction → "
        f"[bold]{memo.position_size_pct_low:.1f}%–"
        f"{memo.position_size_pct_high:.1f}%[/bold] of portfolio"
    )
    if memo.sizing_rationale:
        console.print(f"  [dim]{memo.sizing_rationale}[/dim]")

    # §6 Exit Triggers
    if memo.exit_triggers:
        console.print("\n[bold yellow]Exit Triggers[/bold yellow]")
        for t in memo.exit_triggers:
            console.print(f"  ⚠  {t}")


def _render_network(net) -> None:  # type: ignore[no-untyped-def]
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree

    sym = net.subject_symbol
    subj = net.subject_snapshot

    # ── Level 1: subject summary panel ───────────────────────────────────────
    subj_lines = [f"[bold cyan]{sym}[/bold cyan]"]
    if subj:
        if subj.name:
            subj_lines.append(subj.name)
        parts = []
        if subj.market_cap:
            parts.append(f"Mkt Cap {_fmt(subj.market_cap, 'raw')}")
        if subj.pe_trailing:
            parts.append(f"P/E {subj.pe_trailing:.1f}x")
        if subj.ev_to_ebitda:
            parts.append(f"EV/EBITDA {subj.ev_to_ebitda:.1f}x")
        if subj.operating_margin:
            parts.append(f"Op Mgn {subj.operating_margin:.1%}")
        if parts:
            subj_lines.append("  ".join(parts))
    console.print(Panel("\n".join(subj_lines), title="Level 1 — Subject", border_style="cyan"))

    if not net.nodes:
        console.print(
            "[dim]No Level-2 connections found. Run `advisor research ticker` first.[/dim]"
        )
        return

    # ── Level 2: connection tree ──────────────────────────────────────────────
    rel_labels = {
        "peer": ("Peers", "yellow"),
        "competitor": ("Competitors", "red"),
        "top_holder": ("Public Institutional Holders", "green"),
        "customer": ("Customers", "blue"),
        "supplier": ("Suppliers", "magenta"),
    }

    tree = Tree(f"[bold]{sym}[/bold] — Level 2 connections")
    by_rel = net.by_relationship
    for rel_key, (label, color) in rel_labels.items():
        nodes = by_rel.get(rel_key, [])
        if not nodes:
            continue
        branch = tree.add(f"[{color}]{label}[/{color}] ({len(nodes)})")
        for node in nodes:
            snap = node.snapshot
            details = []
            if snap:
                if snap.market_cap:
                    details.append(f"Mkt Cap {_fmt(snap.market_cap, 'raw')}")
                if snap.pe_trailing:
                    details.append(f"P/E {snap.pe_trailing:.1f}x")
                if snap.revenue_growth_yoy:
                    details.append(f"Rev Grw {snap.revenue_growth_yoy:+.1%}")
            note = f" [{node.note}]" if node.note else ""
            detail_str = f"  ({', '.join(details)})" if details else ""
            branch.add(
                f"[bold]{node.symbol}[/bold] {node.name[:28]}" f"[dim]{note}[/dim]{detail_str}"
            )
    console.print(tree)

    # ── Comparison table (subject + all Level-2 nodes) ───────────────────────
    all_snaps = []
    if subj:
        all_snaps.append(("SUBJECT", subj))
    for node in net.nodes:
        if node.snapshot:
            all_snaps.append((node.relationship.value.upper(), node.snapshot))

    if all_snaps:
        table = Table(
            title=f"Level 2 Comparison — {sym} + Connected Companies",
            show_lines=False,
        )
        table.add_column("Rel", style="dim", width=10)
        table.add_column("Ticker", style="cyan", width=8)
        table.add_column("Name", max_width=22)
        for hdr in ["Mkt Cap", "Rev Grw", "Gr Mgn", "Op Mgn", "EV/S", "EV/EBITDA", "P/E", "P/FCF"]:
            table.add_column(hdr, justify="right", min_width=7)

        for rel_label, snap in all_snaps:
            style = "bold" if rel_label == "SUBJECT" else ""
            table.add_row(
                rel_label,
                snap.symbol,
                snap.name[:22],
                _fmt(snap.market_cap, "raw"),
                _fmt(snap.revenue_growth_yoy, "pct"),
                _fmt(snap.gross_margin, "pct"),
                _fmt(snap.operating_margin, "pct"),
                _fmt(snap.ev_to_sales, "x"),
                _fmt(snap.ev_to_ebitda, "x"),
                _fmt(snap.pe_trailing, "x"),
                _fmt(snap.p_to_fcf, "x"),
                style=style,
            )
        console.print(table)

    # ── Level 3: hint ────────────────────────────────────────────────────────
    all_syms = [n.symbol for n in net.nodes]
    console.print(
        f"\n[dim]Level 3 → drill into any connection:[/dim]\n"
        f"  advisor research ticker [bold]{all_syms[0]}[/bold]  "
        f"   # full 7-layer report\n"
        f"  advisor research network [bold]{all_syms[0]}[/bold]  "
        f"  # its own Level-2 map"
    )


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.1%}"


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
