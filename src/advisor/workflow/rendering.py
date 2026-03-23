"""Rich rendering helpers for workflow output."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from advisor.workflow.models import (
    AccountSnapshot,
    AfternoonPositionStatus,
    AfternoonResult,
    BrierScoreSummary,
    DeepDiveResult,
    EarningsAlert,
    MorningResult,
    PositionAction,
    PositionHealth,
    RegimeContext,
    ScannerHit,
    ValidatedCandidate,
    WatchCandidate,
)

console = Console()


# ── Phase separators ─────────────────────────────────────────────────────────


def phase_rule(name: str, number: int) -> None:
    console.print()
    console.print(Rule(f"[bold]Phase {number} — {name}[/bold]"))


def gate_note(msg: str) -> None:
    console.print(f"  [dim]Gate: {msg}[/dim]")


# ── Regime ───────────────────────────────────────────────────────────────────


def render_regime(ctx: RegimeContext) -> None:
    phase_rule("Market Regime", 1)
    regime_color = {"low_vol": "green", "normal": "yellow", "high_vol": "red"}.get(
        ctx.regime_name, "white"
    )
    console.print(
        f"  Regime: [{regime_color}]{ctx.regime_name}[/{regime_color}]"
        f"  |  VIX: {ctx.vix:.1f}  |  SPY Vol: {ctx.spy_vol:.4f}"
    )
    if ctx.warning:
        console.print(Panel(f"[bold red]{ctx.warning}[/bold red]", style="red"))


# ── Account ──────────────────────────────────────────────────────────────────


def render_account(snap: AccountSnapshot, phase_num: int = 2) -> None:
    phase_rule("Account Snapshot", phase_num)
    console.print(
        f"  Net Liq: [bold]${snap.net_liq:,.2f}[/bold]"
        f"  |  Cash: ${snap.cash:,.2f}"
        f"  |  Buying Power: ${snap.buying_power:,.2f}"
    )
    if snap.warning:
        console.print(Panel(f"[bold yellow]{snap.warning}[/bold yellow]", style="yellow"))


# ── Position Health ──────────────────────────────────────────────────────────


def render_position_health(positions: list[PositionHealth], phase_num: int = 3) -> None:
    phase_rule("Position Health Check", phase_num)
    if not positions:
        console.print("  [dim]No open positions[/dim]")
        return

    table = Table()
    table.add_column("Symbol", style="cyan")
    table.add_column("Qty", justify="right")
    table.add_column("Avg Price", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("P&L %", justify="right")
    table.add_column("Action", style="bold")
    table.add_column("Reason")

    action_colors = {
        PositionAction.CONSIDER_PROFIT: "green",
        PositionAction.TAKE_PROFIT: "green",
        PositionAction.REVIEW: "red",
        PositionAction.CONSIDER_CLOSING: "red",
        PositionAction.TREND_WARNING: "yellow",
        PositionAction.APPROACHING_EXPIRY: "yellow",
        PositionAction.DELTA_WARNING: "yellow",
        PositionAction.HOLD: "dim",
    }

    for ph in positions:
        pnl_color = "green" if ph.pnl_pct >= 0 else "red"
        action_color = action_colors.get(ph.action, "white")
        sym = ph.underlying or ph.symbol
        table.add_row(
            sym,
            str(ph.quantity),
            f"${ph.avg_open_price:.2f}",
            f"${ph.current_price:.2f}",
            f"[{pnl_color}]{ph.pnl_pct:+.1f}%[/{pnl_color}]",
            f"[{action_color}]{ph.action.value}[/{action_color}]",
            ph.reason,
        )

    console.print(table)


# ── Scanner Hits ─────────────────────────────────────────────────────────────


def render_scanner_hits(hits: list[ScannerHit], phase_num: int = 4, lean: bool = False) -> None:
    label = "Opportunity Discovery (lean)" if lean else "Opportunity Discovery"
    phase_rule(label, phase_num)
    if not hits:
        console.print("  [dim]No scanner hits[/dim]")
        return

    table = Table()
    table.add_column("Symbol", style="cyan bold")
    table.add_column("Sources")
    table.add_column("Best Score", justify="right", style="bold")
    table.add_column("Smart$", justify="right")
    table.add_column("Misprice", justify="right")
    table.add_column("Alpha", justify="right")
    table.add_column("Dip", justify="right")

    for h in hits:
        table.add_row(
            h.symbol,
            ", ".join(h.sources),
            f"{h.best_score:.0f}",
            f"{h.smart_money_score:.0f}" if h.smart_money_score else "-",
            f"{h.mispricing_score:.0f}" if h.mispricing_score else "-",
            f"{h.alpha_score:.0f}" if h.alpha_score else "-",
            f"{h.dip_score:.0f}" if h.dip_score else "-",
        )

    console.print(table)
    gate_note(f"{len(hits)} candidates pass → advancing to validation")


# ── Validated Candidates ─────────────────────────────────────────────────────


def render_validated(candidates: list[ValidatedCandidate], phase_num: int = 5) -> None:
    phase_rule("Candidate Validation", phase_num)
    if not candidates:
        console.print("  [dim]No candidates validated[/dim]")
        return

    table = Table()
    table.add_column("Symbol", style="cyan")
    table.add_column("Verdict", style="bold")
    table.add_column("Confluence")
    table.add_column("ML Signal")
    table.add_column("Win Prob", justify="right")
    table.add_column("Strategy Signals")
    table.add_column("Action")

    verdict_colors = {"DEEP_DIVE": "green", "WATCH": "yellow", "SKIP": "red"}

    for c in candidates:
        v_color = verdict_colors.get(c.verdict.value, "white")
        wp = f"{c.ml_win_prob:.0%}" if c.ml_win_prob is not None else "-"
        table.add_row(
            c.symbol,
            f"[{v_color}]{c.verdict.value}[/{v_color}]",
            c.confluence_verdict,
            c.ml_signal or "-",
            wp,
            ", ".join(c.strategy_signals) if c.strategy_signals else "-",
            c.reason,
        )

    console.print(table)
    dd_count = sum(1 for c in candidates if c.verdict == "DEEP_DIVE")
    watch_count = sum(1 for c in candidates if c.verdict == "WATCH")
    gate_note(f"{dd_count} DEEP_DIVE, {watch_count} WATCH")


# ── Deep Dive ────────────────────────────────────────────────────────────────


def render_deep_dives(dives: list[DeepDiveResult], mode: str, phase_num: int = 6) -> None:
    phase_rule("Deep Dive", phase_num)
    if not dives:
        console.print("  [dim]No deep dives (--quick mode or no DEEP_DIVE candidates)[/dim]")
        return

    for d in dives:
        parts = [f"[bold cyan]{d.symbol}[/bold cyan]"]

        if mode == "stocks":
            if d.scenario_expected_return is not None:
                ret_color = "green" if d.scenario_expected_return > 0 else "red"
                parts.append(
                    f"Scenario E[R]: [{ret_color}]{d.scenario_expected_return:+.1f}%[/{ret_color}]"
                )
            if d.scenario_prob_profit is not None:
                parts.append(f"Prob(+): {d.scenario_prob_profit:.0%}")
            if d.best_strategy:
                parts.append(f"Best: {d.best_strategy}")
        else:
            if d.strikes:
                parts.append(f"Strikes: {d.strikes}")
            if d.conviction is not None:
                parts.append(f"Conv: {d.conviction:.0f}")
            if d.pop is not None:
                parts.append(f"POP: {d.pop:.0%}")
            if d.ev is not None:
                parts.append(f"EV: ${d.ev:.2f}")

        if d.research_verdict:
            parts.append(f"Research: {d.research_verdict}")
        if d.research_conviction is not None:
            parts.append(f"({d.research_conviction:.0f} conviction)")

        console.print("  " + "  |  ".join(parts))

        if d.bull_case:
            console.print(f"    [green]Bull:[/green] {d.bull_case[0]}")
        if d.bear_case:
            console.print(f"    [red]Bear:[/red] {d.bear_case[0]}")


# ── Action Items ─────────────────────────────────────────────────────────────


def render_morning_actions(result: MorningResult) -> None:
    console.print()
    console.print(Rule("[bold green]Action Items[/bold green]"))

    # Positions to review
    review_positions = [p for p in result.position_health if p.action != PositionAction.HOLD]
    if review_positions:
        console.print("\n[bold]Positions to Review:[/bold]")
        for p in review_positions:
            sym = p.underlying or p.symbol
            console.print(f"  {sym}: {p.action.value} — {p.reason}")

    # New opportunities (DEEP_DIVE)
    deep = [v for v in result.validated if v.verdict == "DEEP_DIVE"]
    if deep:
        console.print("\n[bold]New Opportunities:[/bold]")
        for c in deep:
            wp = f"{c.ml_win_prob:.0%}" if c.ml_win_prob else "?"
            console.print(f"  {c.symbol}: alpha {c.alpha_score or '?'}, ML {c.ml_signal} ({wp})")
            # Add deep dive details if available
            for d in result.deep_dives:
                if d.symbol == c.symbol and d.scenario_expected_return is not None:
                    console.print(
                        f"    → scenario E[R] {d.scenario_expected_return:+.1f}%, "
                        f"best: {d.best_strategy}"
                    )

    # Research verdicts
    researched = [d for d in result.deep_dives if d.research_verdict]
    if researched:
        console.print("\n[bold]Research Verdicts:[/bold]")
        for d in researched:
            conv = f" ({d.research_conviction:.0f} conviction)" if d.research_conviction else ""
            dtype = f" — {d.research_dip_type}" if d.research_dip_type else ""
            console.print(f"  {d.symbol}: {d.research_verdict}{conv}{dtype}")

    # Watchlist adds
    watch = [v for v in result.validated if v.verdict == "WATCH"]
    if watch:
        console.print("\n[bold]Watchlist Adds:[/bold]")
        for c in watch:
            wp = f"ML {c.ml_signal} at {c.ml_win_prob:.0%}" if c.ml_win_prob else ""
            console.print(f"  {c.symbol} — confluence {c.confluence_verdict} {wp}")


def render_afternoon_actions(result: AfternoonResult, mode: str) -> None:
    console.print()
    console.print(Rule("[bold green]Action Items[/bold green]"))

    # Positions requiring action
    actions = [p for p in result.position_statuses if p.action != PositionAction.HOLD]
    if actions:
        action_groups = {
            "Close": [
                p for p in actions if p.action in (PositionAction.CLOSE, PositionAction.TAKE_PROFIT)
            ],
            "Roll": [p for p in actions if p.action == PositionAction.ROLL],
            "Adjust Stops": [p for p in actions if p.action == PositionAction.TRAIL_STOP],
            "Review Thesis": [
                p
                for p in actions
                if p.action in (PositionAction.TREND_BREAK, PositionAction.REVIEW)
            ],
        }
        for group_name, items in action_groups.items():
            if items:
                console.print(f"\n[bold]{group_name}:[/bold]")
                for p in items:
                    price_note = f" — limit at ${p.limit_price:.2f}" if p.limit_price else ""
                    console.print(f"  {p.symbol}: {p.reason}{price_note}")

    # Earnings alerts
    if result.earnings_alerts:
        console.print("\n[bold]Earnings Alerts:[/bold]")
        for ea in result.earnings_alerts:
            pos_note = " [bold red](HAS POSITION)[/bold red]" if ea.has_position else ""
            console.print(f"  {ea.symbol} reports in {ea.days_until} days{pos_note}")

    # Tomorrow's watchlist
    if result.tomorrow_watchlist:
        console.print("\n[bold]Tomorrow's Watchlist:[/bold]")
        for sym in result.tomorrow_watchlist:
            # Find matching watch update for context
            update = next((w for w in result.watch_updates if w.symbol == sym), None)
            if update and update.status:
                console.print(f"  {sym} — {update.status}")
            else:
                console.print(f"  {sym}")

    # Brier score warning (options only)
    if result.brier_summary and result.brier_summary.warning:
        console.print()
        console.print(
            Panel(
                f"[bold yellow]{result.brier_summary.warning}[/bold yellow]",
                style="yellow",
            )
        )


# ── Afternoon-specific renderers ─────────────────────────────────────────────


def render_account_comparison(
    current: AccountSnapshot,
    morning: dict | None,
    phase_num: int = 1,
) -> None:
    phase_rule("Account Update", phase_num)
    console.print(
        f"  Net Liq: [bold]${current.net_liq:,.2f}[/bold]"
        f"  |  Cash: ${current.cash:,.2f}"
        f"  |  Buying Power: ${current.buying_power:,.2f}"
    )
    if morning:
        morning_liq = morning.get("net_liq", 0)
        if morning_liq > 0:
            day_change = current.net_liq - morning_liq
            day_pct = day_change / morning_liq * 100
            color = "green" if day_change >= 0 else "red"
            console.print(f"  Day's P&L: [{color}]{day_change:+,.2f} ({day_pct:+.2f}%)[/{color}]")


def render_afternoon_positions(
    statuses: list[AfternoonPositionStatus],
    phase_num: int = 2,
) -> None:
    phase_rule("Position Management", phase_num)
    if not statuses:
        console.print("  [dim]No open positions[/dim]")
        return

    table = Table()
    table.add_column("Symbol", style="cyan")
    table.add_column("P&L %", justify="right")
    table.add_column("Action", style="bold")
    table.add_column("Reason")
    table.add_column("Limit", justify="right")

    for s in statuses:
        pnl_color = "green" if s.pnl_pct >= 0 else "red"
        action_color = {"CLOSE": "green", "ROLL": "yellow", "HOLD": "dim"}.get(
            s.action.value, "white"
        )
        limit = f"${s.limit_price:.2f}" if s.limit_price else "-"
        table.add_row(
            s.symbol,
            f"[{pnl_color}]{s.pnl_pct:+.1f}%[/{pnl_color}]",
            f"[{action_color}]{s.action.value}[/{action_color}]",
            s.reason,
            limit,
        )

    console.print(table)


def render_brier_summary(summary: BrierScoreSummary | None, phase_num: int = 3) -> None:
    phase_rule("Prediction Validation", phase_num)
    if summary is None:
        console.print("  [dim]No predictions to validate[/dim]")
        return

    if summary.n_samples == 0:
        console.print("  [dim]No resolved predictions in lookback window[/dim]")
        return

    console.print(f"  Resolved: {summary.n_resolved}  |  Samples: {summary.n_samples}")
    if summary.pop_brier is not None:
        quality = (
            "excellent"
            if summary.pop_brier < 0.10
            else "good"
            if summary.pop_brier < 0.20
            else "fair"
            if summary.pop_brier < 0.25
            else "poor"
        )
        console.print(f"  POP Brier: {summary.pop_brier:.3f} ({quality})")
    if summary.warning:
        console.print(Panel(f"[bold yellow]{summary.warning}[/bold yellow]", style="yellow"))


def render_watch_updates(updates: list[WatchCandidate], phase_num: int = 3) -> None:
    phase_rule("Signal Re-Check", phase_num)
    if not updates:
        console.print("  [dim]No watch candidates to re-check[/dim]")
        return

    table = Table()
    table.add_column("Symbol", style="cyan")
    table.add_column("New BUY Signals")
    table.add_column("Alpha", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Status")

    for w in updates:
        change_str = ""
        if w.score_change is not None:
            color = "green" if w.score_change > 0 else "red"
            change_str = f"[{color}]{w.score_change:+.0f}[/{color}]"
        table.add_row(
            w.symbol,
            ", ".join(w.new_buy_signals) if w.new_buy_signals else "-",
            f"{w.alpha_score:.0f}" if w.alpha_score else "-",
            change_str,
            w.status,
        )

    console.print(table)


def render_next_day_prep(
    earnings: list[EarningsAlert],
    tomorrow_watchlist: list[str],
    phase_num: int = 4,
) -> None:
    phase_rule("Next-Day Prep", phase_num)

    if earnings:
        console.print("\n  [bold]Earnings Calendar (next 7 days):[/bold]")
        for ea in earnings:
            console.print(f"    {ea.symbol} — {ea.earnings_date} ({ea.days_until}d)")

    if tomorrow_watchlist:
        console.print(f"\n  [bold]Tomorrow's Watchlist:[/bold] {', '.join(tomorrow_watchlist)}")

    if not earnings and not tomorrow_watchlist:
        console.print("  [dim]Nothing notable for next day[/dim]")
