"""CLI commands for ML signal tracking, training, and prediction."""

from __future__ import annotations

import logging
from typing import Annotated, Optional

import typer

from advisor.cli.formatters import console, output_error, output_json

logger = logging.getLogger(__name__)

app = typer.Typer(name="ml", help="ML signal tracking, training, and prediction")


@app.command("log")
def ml_log(
    ticker: Annotated[str, typer.Argument(help="Ticker symbol")],
    scanner: Annotated[str, typer.Argument(help="Scanner name")],
    score: Annotated[float, typer.Argument(help="Signal score")],
    verdict: Annotated[str, typer.Option("--verdict", "-v", help="Signal verdict")] = "",
    price: Annotated[float, typer.Option("--price", "-p", help="Entry price")] = 0.0,
) -> None:
    """Manually log a scanner signal."""
    from advisor.ml.outcome_tracker import log_signal

    rec = log_signal(ticker, scanner, score, verdict, price)
    typer.echo(
        f"Logged signal {rec['id']} — {rec['ticker']} via {rec['scanner']} "
        f"(score={rec['score']})"
    )


@app.command("resolve")
def ml_resolve() -> None:
    """Resolve all pending signals older than 30 days."""
    from advisor.ml.outcome_tracker import resolve_signals

    typer.echo("Resolving signals...")
    n = resolve_signals()
    typer.echo(f"Resolved {n} signal(s)")


@app.command("stats")
def ml_stats(
    scanner: Annotated[
        Optional[str], typer.Option("--scanner", "-s", help="Filter by scanner")
    ] = None,
) -> None:
    """Show win rates and stats."""
    from rich.table import Table

    from advisor.ml.outcome_tracker import get_stats

    stats = get_stats(scanner)
    label = scanner or "all scanners"

    table = Table(title=f"ML Stats: {label}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Resolved signals", str(stats["count"]))
    table.add_row("Win rate", f"{stats['win_rate']}%")
    table.add_row("Avg return", f"{stats['avg_return']}%")
    table.add_row("Avg drawdown", f"{stats['avg_drawdown']}%")
    console.print(table)


@app.command("train")
def ml_train(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model type: logistic, lightgbm, ensemble"),
    ] = "lightgbm",
    symbols: Annotated[
        Optional[str],
        typer.Option("--symbols", "-s", help="Comma-separated symbols"),
    ] = None,
    lookback: Annotated[str, typer.Option("--lookback", "-l", help="Data lookback period")] = "5y",
    threshold: Annotated[
        float, typer.Option("--threshold", "-t", help="Win threshold (% return)")
    ] = 3.0,
    horizon: Annotated[
        int, typer.Option("--horizon", help="Forward return horizon in trading days")
    ] = 10,
    label_mode: Annotated[
        str,
        typer.Option("--label-mode", help="Label mode: barrier (vol-adjusted) or fixed"),
    ] = "barrier",
    decay: Annotated[
        int,
        typer.Option("--decay", help="Sample weight half-life in days (0=uniform)"),
    ] = 365,
    barrier_ratio: Annotated[
        float,
        typer.Option(
            "--barrier-ratio",
            help="Asymmetric barrier ratio (>1 = tighter stop, e.g. 2.0 for 2:1 R:R)",
        ),
    ] = 1.0,
    cutoff: Annotated[
        Optional[str],
        typer.Option("--cutoff", help="Train cutoff date (YYYY-MM-DD) for OOS split"),
    ] = None,
    compare: Annotated[bool, typer.Option("--compare", help="Compare all model types")] = False,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Train the ML model on historical OHLCV data."""
    from rich.table import Table

    from advisor.ml.models import ModelType
    from advisor.ml.pipeline import MLPipeline

    symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None
    pipeline = MLPipeline(
        symbols=symbol_list,
        lookback=lookback,
        threshold=threshold,
        horizon=horizon,
        label_mode=label_mode,
        decay=decay,
        barrier_ratio=barrier_ratio,
    )

    console.print(
        f"[dim]Config: horizon={horizon}d, threshold={threshold}%, "
        f"lookback={lookback}, labels={label_mode}, decay={decay}d, "
        f"barrier_ratio={barrier_ratio}, cutoff={cutoff or 'auto'}[/dim]"
    )

    if compare:
        console.print("[bold]Comparing all model types...[/bold]")
        try:
            comparison = pipeline.compare_models(train_cutoff=cutoff)
        except Exception as e:
            output_error(f"Training failed: {e}")
            return

        if output == "json":
            output_json(comparison.reset_index().to_dict(orient="records"))
            return

        table = Table(title="Model Comparison (CV Metrics)")
        table.add_column("Model", style="cyan")
        table.add_column("AUC", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("Brier", justify="right")

        for model_name, row in comparison.iterrows():
            table.add_row(
                str(model_name),
                f"{row['cv_auc_mean']:.4f}",
                f"{row['cv_f1_mean']:.4f}",
                f"{row['cv_accuracy_mean']:.4f}",
                f"{row['cv_precision_mean']:.4f}",
                f"{row['cv_recall_mean']:.4f}",
                f"{row['cv_brier_mean']:.4f}",
            )
        console.print(table)

        # Train the best model and save it
        best_model = comparison["cv_auc_mean"].idxmax()
        console.print(f"\n[bold]Saving best model: {best_model}[/bold]")
        mt = ModelType(best_model)
        result = pipeline.train_and_evaluate(model_type=mt, train_cutoff=cutoff)
        console.print(f"Model saved to {result['model_path']}")
    else:
        try:
            mt = ModelType(model)
        except ValueError:
            output_error(f"Unknown model type: {model}. Use logistic, lightgbm, or ensemble.")
            return

        console.print(f"[bold]Training {mt} model...[/bold]")
        try:
            result = pipeline.train_and_evaluate(model_type=mt, train_cutoff=cutoff)
        except Exception as e:
            output_error(f"Training failed: {e}")
            return

        if output == "json":
            output_json(result)
            return

        cv = result["cv_metrics"]
        meta = result["metadata"]
        table = Table(title=f"Training Results: {mt}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Samples", str(meta["n_samples"]))
        table.add_row("Features", str(meta["n_features"]))
        table.add_row("Symbols", str(meta.get("n_symbols", "?")))
        table.add_row("Horizon", f"{meta.get('horizon', '?')}d")
        table.add_row("Threshold", f"{meta.get('threshold', '?')}%")
        table.add_row("Label mode", meta.get("label_mode", "fixed"))
        table.add_row("Decay", f"{meta.get('decay', '?')}d")
        table.add_row("Train cutoff", meta.get("train_cutoff", "?"))
        table.add_row("Win rate", f"{meta['win_rate']}%")
        table.add_row(
            "CV AUC",
            f"{cv.get('cv_auc_mean', 0):.4f} +/- {cv.get('cv_auc_std', 0):.4f}",
        )
        table.add_row(
            "CV F1",
            f"{cv.get('cv_f1_mean', 0):.4f} +/- {cv.get('cv_f1_std', 0):.4f}",
        )
        table.add_row(
            "CV Accuracy",
            f"{cv.get('cv_accuracy_mean', 0):.4f} +/- " f"{cv.get('cv_accuracy_std', 0):.4f}",
        )
        table.add_row("CV Brier", f"{cv.get('cv_brier_mean', 0):.4f}")
        console.print(table)

        # Feature importance
        importance = result.get("feature_importance", {})
        if importance:
            imp_table = Table(title="Top Feature Importances")
            imp_table.add_column("Feature", style="cyan")
            imp_table.add_column("Importance", justify="right")
            for name, imp in list(importance.items())[:10]:
                imp_table.add_row(name, f"{imp:.4f}")
            console.print(imp_table)

        console.print(f"\nModel saved to {result['model_path']}")


@app.command("train-meta")
def ml_train_meta(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model type: logistic, lightgbm, ensemble"),
    ] = "lightgbm",
    symbols: Annotated[
        Optional[str],
        typer.Option("--symbols", "-s", help="Comma-separated symbols"),
    ] = None,
    lookback: Annotated[str, typer.Option("--lookback", "-l", help="Data lookback period")] = "5y",
    threshold: Annotated[
        float, typer.Option("--threshold", "-t", help="Win threshold (% return)")
    ] = 3.0,
    horizon: Annotated[
        int, typer.Option("--horizon", help="Forward return horizon in trading days")
    ] = 10,
    label_mode: Annotated[
        str,
        typer.Option("--label-mode", help="Label mode: barrier or fixed"),
    ] = "fixed",
    decay: Annotated[
        int,
        typer.Option("--decay", help="Sample weight half-life in days"),
    ] = 365,
    barrier_ratio: Annotated[
        float,
        typer.Option("--barrier-ratio", help="Asymmetric barrier ratio (>1 = tighter stop)"),
    ] = 1.0,
    cutoff: Annotated[
        Optional[str],
        typer.Option("--cutoff", help="Train cutoff date (YYYY-MM-DD)"),
    ] = None,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Train primary + meta-labeling model and compare precision curves."""
    from rich.table import Table

    from advisor.ml.models import ModelType
    from advisor.ml.pipeline import MLPipeline

    try:
        mt = ModelType(model)
    except ValueError:
        output_error(f"Unknown model type: {model}")
        return

    symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None
    pipeline = MLPipeline(
        symbols=symbol_list,
        lookback=lookback,
        threshold=threshold,
        horizon=horizon,
        label_mode=label_mode,
        decay=decay,
        barrier_ratio=barrier_ratio,
    )

    console.print(
        f"[bold]Training primary ({mt}) + meta-labeling model...[/bold]\n"
        f"[dim]horizon={horizon}d, threshold={threshold}%, labels={label_mode}[/dim]"
    )

    try:
        result = pipeline.train_with_meta(model_type=mt, train_cutoff=cutoff)
    except Exception as e:
        output_error(f"Training failed: {e}")
        return

    if output == "json":
        output_json(result)
        return

    # Primary model summary
    cv = result["cv_metrics"]
    meta = result["metadata"]
    console.print(
        f"\n[bold]Primary model:[/bold] {meta['n_samples']} samples, "
        f"{meta['n_features']} features, "
        f"CV AUC={cv.get('cv_auc_mean', 0):.4f}"
    )

    # Meta-labeling summary
    ml_res = result.get("meta_labeling", {})
    ml_metrics = ml_res.get("metrics", {})
    console.print(
        f"[bold]Meta-model:[/bold] "
        f"AUC={ml_metrics.get('meta_auc', 0):.4f}, "
        f"primary correct rate={ml_metrics.get('primary_correct_rate', 0):.1%}"
    )

    # Precision comparison table
    pc = result.get("precision_comparison", {})
    if pc:
        base_rate = pc.get("base_rate", 0)
        table = Table(title="Precision Comparison: Primary vs Meta-Labeled (OOS pooled CV)")
        table.add_column("Threshold", style="cyan", justify="right")
        table.add_column("Primary Prec", justify="right")
        table.add_column("Primary N", justify="right")
        table.add_column("Combined Prec", justify="right")
        table.add_column("Combined N", justify="right")
        table.add_column("Lift Gain", justify="right")

        primary_rows = pc.get("primary", [])
        combined_rows = pc.get("combined", [])
        for p_row, c_row in zip(primary_rows, combined_rows):
            t = p_row["threshold"]
            p_prec = p_row["precision"]
            c_prec = c_row["precision"]
            delta = c_prec - p_prec

            delta_color = "green" if delta > 0 else "red" if delta < 0 else "white"
            p_color = "green" if p_prec > base_rate else "red"
            c_color = "green" if c_prec > base_rate else "red"

            table.add_row(
                f"{'[bold]' if t >= 0.70 else ''}{t:.0%}{'[/bold]' if t >= 0.70 else ''}",
                f"[{p_color}]{p_prec:.1%}[/{p_color}]",
                str(p_row["n_signals"]),
                f"[{c_color}]{c_prec:.1%}[/{c_color}]",
                str(c_row["n_signals"]),
                f"[{delta_color}]{delta:+.1%}[/{delta_color}]",
            )

        console.print(f"\n[dim]Base win rate: {base_rate:.1%}[/dim]")
        console.print(table)

    console.print(f"\nPrimary saved to {result['model_path']}")
    console.print(f"Meta-model saved to {result.get('meta_model_path', '?')}")


@app.command("predict")
def ml_predict(
    ticker: Annotated[str, typer.Argument(help="Ticker symbol")],
    explain: Annotated[
        bool, typer.Option("--explain", "-e", help="Show feature breakdown")
    ] = False,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Run ML model prediction with live features."""
    from rich.panel import Panel
    from rich.table import Table

    from advisor.ml.signal_generator import MLSignalGenerator

    gen = MLSignalGenerator()
    if not gen.is_available():
        output_error("No trained ML model found. Run 'advisor ml train' first.")
        return

    if explain:
        explanation = gen.explain_prediction(ticker)
        if "error" in explanation:
            output_error(explanation["error"])
            return

        if output == "json":
            output_json(explanation)
            return

        signal = explanation["signal"]
        prob = explanation["win_probability"]
        signal_colors = {"BUY": "green", "SELL": "red", "NEUTRAL": "yellow"}
        color = signal_colors.get(signal, "white")

        console.print(
            Panel(
                f"[bold {color}]{signal}[/bold {color}] — "
                f"Win probability: {prob:.1%}\n"
                f"Model: {explanation['model_type']}\n"
                f"{explanation['reason']}",
                title=f"ML Prediction: {ticker.upper()}",
            )
        )

        table = Table(title="Feature Breakdown")
        table.add_column("Feature", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Importance", justify="right")

        for feat in explanation["features"]:
            table.add_row(
                feat["feature"],
                f"{feat['value']:.4f}",
                f"{feat['importance']:.4f}",
            )
        console.print(table)
    else:
        signal = gen.generate_signal(ticker)
        if signal is None:
            output_error(f"Could not generate signal for {ticker}")
            return

        if output == "json":
            output_json(signal)
            return

        signal_colors = {
            "BUY": "green",
            "SELL": "red",
            "NEUTRAL": "yellow",
            "HOLD": "yellow",
        }
        color = signal_colors.get(signal.action.value, "white")
        console.print(
            f"[bold {color}]{signal.action.value}[/bold {color}] "
            f"— {signal.reason} (price: ${signal.price:,.2f})"
        )


@app.command("features")
def ml_features(
    ticker: Annotated[str, typer.Argument(help="Ticker symbol")],
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Inspect computed ML features for a symbol."""
    from rich.table import Table

    from advisor.ml.features import FeatureEngine

    engine = FeatureEngine()
    features = engine.compute_features(ticker)

    if not features:
        output_error(f"Could not compute features for {ticker}")
        return

    if output == "json":
        output_json(features)
        return

    table = Table(title=f"ML Features: {ticker.upper()}")
    table.add_column("Feature", style="cyan")
    table.add_column("Value", justify="right")

    groups = {
        "Momentum": [
            "ret_1d",
            "ret_5d",
            "ret_10d",
            "ret_20d",
            "ret_60d",
            "momentum_rank",
        ],
        "Trend": ["sma_20_dist", "sma_50_dist", "ema_12_26_diff", "macd_signal_diff"],
        "Mean-reversion": ["rsi_14", "bb_width", "bb_pct"],
        "Volatility": ["atr_14", "realized_vol_20", "realized_vol_5", "vol_ratio"],
        "Volume": ["volume_zscore", "volume_trend", "obv_slope"],
        "Cross-asset": [
            "vix_level",
            "vix_change_5d",
            "vix_regime",
            "sector_rel_strength",
        ],
        "Cross-sectional": ["ret_20d_rank"],
        "Microstructure": [
            "volume_price_divergence",
            "high_low_range",
            "gap_pct",
            "up_volume_ratio",
            "consecutive_direction",
        ],
        "Relative": ["dist_from_52w_high"],
        "Alpha Library": [
            "alpha_mom_12_1",
            "alpha_short_reversal",
            "alpha_acceleration",
            "alpha_ret_consistency",
            "alpha_overnight_sentiment",
            "alpha_close_location",
            "alpha_volume_surprise",
            "alpha_price_volume_corr",
            "alpha_intraday_intensity",
            "alpha_volume_price_trend",
            "alpha_vol_of_vol",
            "alpha_skewness",
            "alpha_kurtosis",
            "alpha_down_vol_ratio",
            "alpha_amihud_illiquidity",
            "alpha_high_low_momentum",
            "alpha_trend_strength",
            "alpha_max_drawdown_20d",
            "alpha_vwap_deviation",
        ],
        "Swing Patterns": [
            "swing_adx_14",
            "swing_di_spread",
            "swing_higher_highs_ratio",
            "swing_range_contraction",
            "swing_bb_squeeze",
            "swing_consolidation_days",
            "swing_pullback_to_sma20",
            "swing_pullback_to_sma50",
            "swing_support_proximity",
            "swing_weekly_trend",
            "swing_daily_weekly_align",
            "swing_trend_maturity",
            "swing_volume_at_breakout",
            "swing_volume_dryup",
            "swing_accumulation_score",
        ],
        "Fractional Diff": [
            "fracdiff_d03",
            "fracdiff_d04",
            "fracdiff_d05",
        ],
        "Fundamental": [
            "earnings_proximity",
            "analyst_upside",
            "pe_vs_sector",
            "recommendation_score",
            "earnings_growth",
        ],
        "Options": [
            "iv_rv_ratio",
            "put_call_oi_ratio",
            "iv_skew",
            "options_volume_ratio",
        ],
    }

    for group_name, feature_names in groups.items():
        table.add_row(f"[bold]{group_name}[/bold]", "")
        for name in feature_names:
            value = features.get(name, 0.0)
            table.add_row(f"  {name}", f"{value:.6f}")

    console.print(table)
    console.print(f"\nTotal features: {len(features)}")


@app.command("backtest")
def ml_backtest(
    ticker: Annotated[str, typer.Argument(help="Ticker symbol")],
    start: Annotated[Optional[str], typer.Option("--start", help="Start date (YYYY-MM-DD)")] = None,
    buy_threshold: Annotated[
        float,
        typer.Option("--buy-threshold", "-b", help="Min probability to trigger BUY"),
    ] = 0.65,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Backtest ML signals on out-of-sample data only."""
    from rich.table import Table

    from advisor.ml.pipeline import MLPipeline

    pipeline = MLPipeline()
    result = pipeline.backtest_signals(ticker, start=start, buy_threshold=buy_threshold)

    if "error" in result:
        output_error(result["error"])
        return

    if output == "json":
        output_json(result)
        return

    table = Table(title=f"ML Backtest (OOS): {ticker.upper()}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    # OOS metadata
    table.add_row("OOS period", f"{result['oos_start']} to {result['oos_end']}")
    table.add_row("Train cutoff", result.get("train_cutoff", "?"))
    table.add_row("Horizon", f"{result['horizon']}d")
    table.add_row("Threshold", f"{result['threshold']}%")
    in_train = result.get("in_training_set", False)
    in_train_str = "[yellow]Yes[/yellow]" if in_train else "[green]No (true OOS)[/green]"
    table.add_row("In training set?", in_train_str)
    table.add_row("", "")

    table.add_row("Total bars", str(result["total_bars"]))
    table.add_row("Buy signals", str(result["buy_signals"]))
    table.add_row("Sell signals", str(result["sell_signals"]))
    table.add_row("Neutral signals", str(result["neutral_signals"]))

    bh = result["baseline_avg_return"]
    bh_color = "green" if bh >= 0 else "red"
    table.add_row(
        "Baseline (buy & hold)",
        f"[{bh_color}]{bh:+.2f}%[/{bh_color}]",
    )

    if result.get("buy_avg_return") is not None:
        ret = result["buy_avg_return"]
        color = "green" if ret >= 0 else "red"
        table.add_row("Buy avg return", f"[{color}]{ret:+.2f}%[/{color}]")
        table.add_row("Buy win rate", f"{result.get('buy_win_rate', 0):.1f}%")
        if result.get("buy_avg_win"):
            table.add_row(
                "Buy avg win",
                f"[green]+{result['buy_avg_win']:.2f}%[/green]",
            )
        if result.get("buy_avg_loss"):
            table.add_row(
                "Buy avg loss",
                f"[red]{result['buy_avg_loss']:.2f}%[/red]",
            )
        edge = result.get("buy_edge", 0)
        edge_color = "green" if edge > 0 else "red"
        table.add_row(
            "Buy edge vs baseline",
            f"[{edge_color}]{edge:+.2f}pp[/{edge_color}]",
        )

    if result.get("sell_avg_return") is not None:
        table.add_row("Sell avg return", f"{result['sell_avg_return']:.2f}%")
        table.add_row("Sell accuracy", f"{result.get('sell_accuracy', 0):.1f}%")

    console.print(table)


@app.command("precision")
def ml_precision(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model type: logistic, lightgbm, ensemble"),
    ] = "lightgbm",
    symbols: Annotated[
        Optional[str],
        typer.Option("--symbols", "-s", help="Comma-separated symbols"),
    ] = None,
    lookback: Annotated[str, typer.Option("--lookback", "-l", help="Data lookback period")] = "5y",
    horizon: Annotated[
        int, typer.Option("--horizon", help="Forward return horizon in trading days")
    ] = 10,
    label_mode: Annotated[
        str,
        typer.Option("--label-mode", help="Label mode: barrier or fixed"),
    ] = "barrier",
    decay: Annotated[
        int,
        typer.Option("--decay", help="Sample weight half-life in days"),
    ] = 365,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Precision at high probability thresholds (pooled CV)."""
    from rich.table import Table

    from advisor.ml.models import ModelType
    from advisor.ml.pipeline import MLPipeline

    try:
        mt = ModelType(model)
    except ValueError:
        output_error(f"Unknown model type: {model}")
        return

    symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None
    pipeline = MLPipeline(
        symbols=symbol_list,
        lookback=lookback,
        horizon=horizon,
        label_mode=label_mode,
        decay=decay,
    )

    console.print(f"[bold]Precision curve ({mt}, {horizon}d horizon)...[/bold]")
    try:
        result = pipeline.precision_curve(model_type=mt)
    except Exception as e:
        output_error(f"Precision analysis failed: {e}")
        return

    if output == "json":
        output_json(result)
        return

    console.print(
        f"[dim]Base win rate: {result['base_win_rate']}% | "
        f"Symbols: {result['n_symbols']} | "
        f"Model: {result['model_type']}[/dim]\n"
    )

    table = Table(title="Precision at Probability Thresholds (OOS pooled CV)")
    table.add_column("Threshold", style="cyan", justify="right")
    table.add_column("Signals", justify="right")
    table.add_column("% Universe", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Lift", justify="right")

    for row in result["thresholds"]:
        t = row["threshold"]
        n = row["n_signals"]
        prec = row["precision"]

        prec_color = "green" if prec > result["base_win_rate"] / 100 else "red"
        t_style = "bold" if t >= 0.70 else ""

        table.add_row(
            f"[{t_style}]{t:.0%}[/{t_style}]" if t_style else f"{t:.0%}",
            str(n),
            f"{row['pct_universe']:.1f}%",
            f"[{prec_color}]{prec:.1%}[/{prec_color}]",
            f"{row['recall']:.1%}",
            f"{row['lift']:.1f}x",
        )

    console.print(table)

    # Summary insight
    high_rows = [r for r in result["thresholds"] if r["threshold"] >= 0.70 and r["n_signals"] > 0]
    if high_rows:
        best = max(high_rows, key=lambda r: r["precision"])
        console.print(
            f"\n[bold]Best high-conviction:[/bold] At {best['threshold']:.0%} threshold -> "
            f"{best['precision']:.1%} precision on {best['n_signals']} signals "
            f"({best['lift']:.1f}x lift over base rate)"
        )
    else:
        console.print("\n[yellow]No signals at >= 70% threshold[/yellow]")


@app.command("walk-forward")
def ml_walk_forward(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model type: logistic, lightgbm, ensemble"),
    ] = "lightgbm",
    symbols: Annotated[
        Optional[str],
        typer.Option("--symbols", "-s", help="Comma-separated symbols"),
    ] = None,
    horizon: Annotated[
        int, typer.Option("--horizon", help="Forward return horizon in trading days")
    ] = 10,
    label_mode: Annotated[
        str,
        typer.Option("--label-mode", help="Label mode: barrier (vol-adjusted) or fixed"),
    ] = "barrier",
    decay: Annotated[
        int,
        typer.Option("--decay", help="Sample weight half-life in days (0=uniform)"),
    ] = 365,
    windows: Annotated[
        int,
        typer.Option("--windows", "-w", help="Number of walk-forward windows"),
    ] = 5,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Walk-forward validation showing IS vs OOS gap."""
    from rich.table import Table

    from advisor.ml.models import ModelType
    from advisor.ml.pipeline import MLPipeline

    try:
        mt = ModelType(model)
    except ValueError:
        output_error(f"Unknown model type: {model}")
        return

    symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None
    pipeline = MLPipeline(
        symbols=symbol_list,
        horizon=horizon,
        label_mode=label_mode,
        decay=decay,
    )

    console.print(
        f"[bold]Walk-forward validation " f"({mt}, {windows} windows, {horizon}d horizon)...[/bold]"
    )
    try:
        result = pipeline.walk_forward(model_type=mt, n_windows=windows)
    except Exception as e:
        output_error(f"Walk-forward failed: {e}")
        return

    if "error" in result:
        output_error(result["error"])
        return

    if output == "json":
        output_json(result)
        return

    table = Table(title="Walk-Forward Results")
    table.add_column("Metric", style="cyan")
    table.add_column("In-Sample", justify="right")
    table.add_column("Out-of-Sample", justify="right")
    table.add_column("Gap", justify="right")

    auc_gap = result["auc_gap"]
    gap_color = "green" if auc_gap < 0.05 else "yellow" if auc_gap < 0.10 else "red"

    table.add_row(
        "AUC",
        f"{result['is_auc']:.4f}",
        f"{result['oos_auc']:.4f}",
        f"[{gap_color}]{auc_gap:.4f}[/{gap_color}]",
    )
    table.add_row(
        "Accuracy",
        f"{result['is_accuracy']:.4f}",
        f"{result['oos_accuracy']:.4f}",
        f"{result['is_accuracy'] - result['oos_accuracy']:.4f}",
    )
    table.add_row(
        "F1",
        f"{result['is_f1']:.4f}",
        f"{result['oos_f1']:.4f}",
        f"{result['is_f1'] - result['oos_f1']:.4f}",
    )

    console.print(table)
    console.print(f"\nWindows evaluated: {result['n_windows']}")

    if auc_gap < 0.05:
        console.print("[green]AUC gap < 5% — model generalizes well[/green]")
    elif auc_gap < 0.10:
        console.print("[yellow]AUC gap 5-10% — minor overfitting, acceptable[/yellow]")
    else:
        console.print("[red]AUC gap > 10% — significant overfitting detected[/red]")


@app.command("regime")
def ml_regime(
    date: Annotated[
        Optional[str],
        typer.Option("--date", "-d", help="Date to detect regime for (YYYY-MM-DD)"),
    ] = None,
    fit: Annotated[bool, typer.Option("--fit", help="Force re-fit the HMM model")] = False,
    lookback: Annotated[str, typer.Option("--lookback", "-l", help="Lookback for fitting")] = "5y",
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Detect current market regime using HMM."""
    from rich.table import Table

    from advisor.ml.regime import RegimeDetector

    detector = RegimeDetector()

    if fit or not RegimeDetector.model_exists():
        console.print(f"[bold]Fitting HMM regime model (lookback={lookback})...[/bold]")
        try:
            summary = detector.fit(lookback=lookback)
            detector.save()
            console.print(
                f"[dim]Fit on {summary['n_observations']} observations, "
                f"log-likelihood={summary['log_likelihood']}[/dim]"
            )
        except Exception as e:
            output_error(f"HMM fitting failed: {e}")
            return
    else:
        detector = RegimeDetector.load()

    try:
        result = detector.detect_regime(date=date)
    except Exception as e:
        output_error(f"Regime detection failed: {e}")
        return

    if output == "json":
        output_json(result)
        return

    regime_colors = {0: "green", 1: "yellow", 2: "red"}
    regime = result["regime"]
    color = regime_colors.get(regime, "white")

    table = Table(title=f"Market Regime: {result['date']}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Regime", f"[bold {color}]{result['regime_name'].upper()}[/bold {color}]")
    table.add_row("Calm probability", f"{result['regime_prob'][0]:.1%}")
    table.add_row("Normal probability", f"{result['regime_prob'][1]:.1%}")
    table.add_row("Stressed probability", f"{result['regime_prob'][2]:.1%}")
    table.add_row("SPY realized vol", f"{result['spy_vol']:.1%}")
    table.add_row("VIX", f"{result['vix']:.1f}")
    console.print(table)


@app.command("allocate")
def ml_allocate(
    symbols: Annotated[str, typer.Argument(help="Comma-separated symbols")],
    lookback: Annotated[int, typer.Option("--lookback", "-l", help="Lookback days")] = 252,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Compute HRP portfolio weights for given symbols."""
    from rich.table import Table

    from advisor.ml.hrp import HRPAllocator

    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    allocator = HRPAllocator()

    console.print(f"[bold]Computing HRP weights for {len(symbol_list)} symbols...[/bold]")
    signals = [(s, 1.0) for s in symbol_list]  # Equal conviction

    try:
        weights = allocator.compute_weights_from_signals(signals, lookback_days=lookback)
    except Exception as e:
        output_error(f"HRP allocation failed: {e}")
        return

    if output == "json":
        output_json(weights)
        return

    table = Table(title="HRP Portfolio Weights")
    table.add_column("Symbol", style="cyan")
    table.add_column("Weight", justify="right")
    table.add_column("Allocation", justify="right")

    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for sym, w in sorted_weights:
        bar = "#" * int(w * 50)
        table.add_row(sym, f"{w:.1%}", f"[dim]{bar}[/dim]")

    console.print(table)
    console.print(f"\nTotal: {sum(weights.values()):.4f} (should be 1.0)")


@app.command("versions")
def ml_versions(
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """List all saved model versions."""
    from rich.table import Table

    from advisor.ml.model_store import list_versions

    versions = list_versions()

    if not versions:
        console.print(
            "[dim]No model versions found. Versions are created on each training run.[/dim]"
        )
        return

    if output == "json":
        output_json(versions)
        return

    table = Table(title=f"Model Versions ({len(versions)} saved)")
    table.add_column("Version", style="cyan")
    table.add_column("Created", justify="right")
    table.add_column("Tag", justify="left")
    table.add_column("CV AUC", justify="right")
    table.add_column("Meta AUC", justify="right")
    table.add_column("Cutoff", justify="right")
    table.add_column("Files", justify="right")

    for i, v in enumerate(versions):
        vid = v["version_id"]
        created = v.get("created_at", "?")
        if created and created != "?":
            created = created[:19].replace("T", " ")
        tag = v.get("tag", "")
        cv_metrics = v.get("cv_metrics", {})
        cv_auc = cv_metrics.get("cv_auc_mean", 0)
        meta_auc = v.get("metrics", {}).get("meta_auc")
        model_meta = v.get("model_metadata", {})
        cutoff = model_meta.get("train_cutoff", "")
        n_files = len(v.get("files", []))

        label = f"[bold]{vid}[/bold]" if i == 0 else vid
        auc_str = f"{cv_auc:.4f}" if cv_auc else "[dim]-[/dim]"
        meta_str = f"{meta_auc:.4f}" if meta_auc else "[dim]-[/dim]"

        table.add_row(label, created, tag, auc_str, meta_str, cutoff or "", str(n_files))

    console.print(table)
    console.print(
        "\n[dim]Latest version is shown in bold."
        " Use 'advisor ml rollback <version>' to restore.[/dim]"
    )


@app.command("rollback")
def ml_rollback(
    version: Annotated[
        str, typer.Argument(help="Version ID to rollback to (e.g., 20260223_040000)")
    ],
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Rollback to a previous model version."""
    from advisor.ml.model_store import rollback

    try:
        result = rollback(version)
    except FileNotFoundError as e:
        output_error(str(e))
        return

    if output == "json":
        output_json(result)
        return

    console.print(f"[bold green]Rolled back to version {result['rolled_back_to']}[/bold green]")
    console.print(f"  Restored files: {', '.join(result['restored_files'])}")
    if result.get("backup_version"):
        console.print(f"  Previous state backed up as: {result['backup_version']}")


@app.command("status")
def ml_status() -> None:
    """Show signal database and model status."""
    from rich.table import Table

    from advisor.ml.models import MLModelTrainer
    from advisor.ml.outcome_tracker import status_summary

    s = status_summary()

    table = Table(title="ML Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="right")

    table.add_row("Total signals", str(s["total"]))
    table.add_row("Resolved", str(s["resolved"]))
    table.add_row("Unresolved", str(s["unresolved"]))

    if MLModelTrainer.model_exists():
        try:
            trainer = MLModelTrainer.load()
            meta = trainer.metadata
            table.add_row(
                "Model",
                f"[green]{meta.get('model_type', 'unknown')}[/green]",
            )
            table.add_row("Samples", str(meta.get("n_samples", "?")))
            table.add_row("Features", str(meta.get("n_features", "?")))
            table.add_row("Symbols", str(meta.get("n_symbols", "?")))
            table.add_row("Horizon", f"{meta.get('horizon', '?')}d")
            table.add_row("Threshold", f"{meta.get('threshold', '?')}%")
            table.add_row("Label mode", meta.get("label_mode", "fixed"))
            table.add_row("Decay", f"{meta.get('decay', '?')}d")
            table.add_row("Train cutoff", meta.get("train_cutoff", "?"))
            table.add_row("Trained", meta.get("trained_at", "?"))

            auc = trainer.metrics.get("cv_auc_mean", 0)
            table.add_row("CV AUC", f"{auc:.4f}")
        except Exception as e:
            logger.debug("Model loading failed: %s", e)
            table.add_row("Model", "[yellow]error loading[/yellow]")
    else:
        table.add_row("Model", "[dim]not trained[/dim]")

    # Legacy meta-ensemble status
    from advisor.ml.meta_ensemble import MetaEnsemble

    ensemble = MetaEnsemble()
    table.add_row("Meta-ensemble", ensemble.model_status)

    # HMM regime status
    from advisor.ml.regime import RegimeDetector

    if RegimeDetector.model_exists():
        try:
            det = RegimeDetector.load()
            result = det.detect_regime()
            regime_colors = {0: "green", 1: "yellow", 2: "red"}
            color = regime_colors.get(result["regime"], "white")
            table.add_row(
                "HMM regime",
                f"[{color}]{result['regime_name']}[/{color}] " f"(VIX={result['vix']:.1f})",
            )
        except Exception as e:
            logger.debug("HMM regime detection failed: %s", e)
            table.add_row("HMM regime", "[yellow]error[/yellow]")
    else:
        table.add_row("HMM regime", "[dim]not fitted[/dim]")

    # Quantile model status
    from advisor.ml.quantile import QuantileModel

    if QuantileModel.model_exists():
        table.add_row("Quantile model", "[green]trained[/green]")
    else:
        table.add_row("Quantile model", "[dim]not trained[/dim]")

    # Meta-labeling status
    from advisor.ml.meta_label import MetaLabeler

    if MetaLabeler.model_exists():
        table.add_row("Meta-labeler", "[green]trained[/green]")
    else:
        table.add_row("Meta-labeler", "[dim]not trained[/dim]")

    # Model versions
    from advisor.ml.model_store import list_versions

    versions = list_versions()
    if versions:
        latest = versions[0]["version_id"]
        table.add_row("Versions", f"{len(versions)} saved (latest: {latest})")
    else:
        table.add_row("Versions", "[dim]none[/dim]")

    console.print(table)


@app.command("train-quantile")
def ml_train_quantile(
    symbols: Annotated[
        Optional[str],
        typer.Option("--symbols", "-s", help="Comma-separated symbols"),
    ] = None,
    lookback: Annotated[str, typer.Option("--lookback", "-l", help="Data lookback period")] = "5y",
    horizon: Annotated[
        int, typer.Option("--horizon", help="Forward return horizon in trading days")
    ] = 10,
    decay: Annotated[
        int,
        typer.Option("--decay", help="Sample weight half-life in days"),
    ] = 365,
    cutoff: Annotated[
        Optional[str],
        typer.Option("--cutoff", help="Train cutoff date (YYYY-MM-DD)"),
    ] = None,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Train quantile regression models for return distribution prediction."""
    from rich.table import Table

    from advisor.ml.pipeline import MLPipeline

    symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None
    pipeline = MLPipeline(
        symbols=symbol_list,
        lookback=lookback,
        horizon=horizon,
        decay=decay,
    )

    console.print(
        f"[bold]Training quantile regression models...[/bold]\n"
        f"[dim]horizon={horizon}d, lookback={lookback}, decay={decay}d, "
        f"cutoff={cutoff or 'auto'}[/dim]"
    )

    try:
        result = pipeline.train_quantile(train_cutoff=cutoff)
    except Exception as e:
        output_error(f"Quantile training failed: {e}")
        return

    if output == "json":
        output_json(result)
        return

    meta = result["metadata"]
    cv = result["cv_metrics"]

    table = Table(title="Quantile Model Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Samples", str(meta["n_samples"]))
    table.add_row("Features", str(meta["n_features"]))
    table.add_row("Symbols", str(meta.get("n_symbols", "?")))
    table.add_row("Horizon", f"{meta['horizon']}d")
    table.add_row("Quantiles", ", ".join(f"{q:.0%}" for q in meta["quantiles"]))
    table.add_row("Train cutoff", meta.get("train_cutoff", "?"))

    console.print(table)

    # Coverage metrics
    coverage = cv.get("coverage", {})
    if coverage:
        cov_table = Table(title="Interval Coverage (OOS pooled CV)")
        cov_table.add_column("Interval", style="cyan")
        cov_table.add_column("Target", justify="right")
        cov_table.add_column("Actual", justify="right")
        cov_table.add_column("Calibration", justify="right")

        for interval, actual in [
            ("25th-75th", coverage.get("50_pct", 0)),
            ("10th-90th", coverage.get("80_pct", 0)),
            ("5th-95th", coverage.get("90_pct", 0)),
        ]:
            target = {"25th-75th": 50.0, "10th-90th": 80.0, "5th-95th": 90.0}[interval]
            diff = actual - target
            color = "green" if abs(diff) < 5 else "yellow" if abs(diff) < 10 else "red"
            cov_table.add_row(
                interval,
                f"{target:.0f}%",
                f"{actual:.1f}%",
                f"[{color}]{diff:+.1f}pp[/{color}]",
            )
        console.print(cov_table)

    # Pinball loss
    pinball = cv.get("pinball_loss", {})
    if pinball:
        pin_table = Table(title="Pinball Loss by Quantile (lower = better)")
        pin_table.add_column("Quantile", style="cyan")
        pin_table.add_column("Loss", justify="right")
        for q_name, loss in pinball.items():
            pin_table.add_row(q_name, f"{loss:.4f}")
        console.print(pin_table)

    # Feature importance
    importance = result.get("feature_importance", {})
    if importance:
        imp_table = Table(title="Top Feature Importances (aggregated)")
        imp_table.add_column("Feature", style="cyan")
        imp_table.add_column("Importance", justify="right")
        for name, imp in list(importance.items())[:10]:
            imp_table.add_row(name, f"{imp:.4f}")
        console.print(imp_table)

    console.print(f"\nModel saved to {result['model_path']}")


@app.command("distribution")
def ml_distribution(
    ticker: Annotated[str, typer.Argument(help="Ticker symbol")],
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Predict return distribution for a symbol using quantile regression."""
    import pandas as pd
    from rich.panel import Panel
    from rich.table import Table

    from advisor.ml.features import FeatureEngine
    from advisor.ml.quantile import QuantileModel
    from advisor.ml.signal_generator import MLSignalGenerator

    if not QuantileModel.model_exists():
        output_error("No quantile model found. Run 'advisor ml train-quantile' first.")
        return

    ticker = ticker.upper()
    engine = FeatureEngine()
    features = engine.compute_features(ticker)
    if not features:
        output_error(f"Could not compute features for {ticker}")
        return

    qm = QuantileModel.load()
    feature_names = qm.feature_names or FeatureEngine.feature_names()

    # Build feature row with sensible defaults (same as signal_generator)
    row = {}
    for col in feature_names:
        if col.endswith("_cs_rank"):
            default = 0.5
        elif col == "hmm_regime":
            default = 1.0
        elif col.startswith("hmm_") and col.endswith("_prob"):
            default = 1.0 / 3.0
        else:
            default = 0.0
        row[col] = features.get(col, default)

    features_df = pd.DataFrame([row])

    # Fetch OHLCV for trend filter
    ohlcv_df = _fetch_ohlcv(ticker)
    dist = qm.predict_distribution(features_df, ohlcv_df=ohlcv_df)

    if output == "json":
        output_json(dist)
        return

    horizon = qm.metadata.get("horizon", 10)
    median_ret = dist["median_return"]
    med_color = "green" if median_ret > 0 else "red" if median_ret < 0 else "white"

    # Trend info line
    trend_line = ""
    if dist.get("trend_score") is not None:
        ts = dist["trend_score"]
        ts_color = "green" if ts >= 2 else "yellow" if ts == 1 else "red"
        fk = dist["falling_knife_warning"]
        fk_str = "  [bold red]FALLING KNIFE[/bold red]" if fk else ""
        ras = dist["risk_adjusted_score"]
        ras_color = "green" if ras > 0 else "red" if ras < 0 else "white"
        trend_line = (
            f"Trend score: [{ts_color}]{ts}/3[/{ts_color}]{fk_str}  |  "
            f"Risk-adj score: [{ras_color}]{ras:+.2f}%[/{ras_color}]\n"
        )

    # Header panel
    skew = dist["distribution_skew"]
    skew_icon = {
        "right": "[green]>>>[/green]",
        "left": "[red]<<<[/red]",
        "symmetric": "[white]<->[/white]",
    }
    console.print(
        Panel(
            f"{trend_line}"
            f"Median {horizon}d return: [{med_color}]{median_ret:+.2f}%[/{med_color}]\n"
            f"80% range: [{med_color}]{dist['expected_range'][0]:+.2f}%[/{med_color}] "
            f"to [{med_color}]{dist['expected_range'][1]:+.2f}%[/{med_color}]\n"
            f"Skew: {skew_icon.get(skew, skew)} ({skew})  |  "
            f"Upside ratio: {dist['upside_ratio']:.2f}  |  "
            f"Tail ratio: {dist['tail_ratio']:.2f}",
            title=f"Return Distribution: {ticker} ({horizon}d)",
        )
    )

    # Quantile table
    table = Table(title="Predicted Quantiles")
    table.add_column("Quantile", style="cyan", justify="right")
    table.add_column("Return (%)", justify="right")
    table.add_column("Interpretation", style="dim")

    labels = {
        0.05: "Worst case (1 in 20)",
        0.10: "VaR 90%",
        0.25: "Bearish scenario",
        0.50: "Median outcome",
        0.75: "Bullish scenario",
        0.90: "VaR 10% upside",
        0.95: "Best case (1 in 20)",
    }

    for q, val in dist["quantiles"].items():
        color = "green" if val > 0 else "red" if val < 0 else "white"
        label = labels.get(q, "")
        table.add_row(
            f"{q:.0%}",
            f"[{color}]{val:+.2f}%[/{color}]",
            label,
        )

    console.print(table)

    # Risk metrics
    risk_table = Table(title="Risk Metrics")
    risk_table.add_column("Metric", style="cyan")
    risk_table.add_column("Value", justify="right")

    var5 = dist["var_5"]
    var10 = dist["var_10"]
    cvar5 = dist["cvar_5"]
    risk_table.add_row("VaR 95%", f"[red]{var5:+.2f}%[/red]")
    risk_table.add_row("VaR 90%", f"[red]{var10:+.2f}%[/red]")
    risk_table.add_row("CVaR 95% (approx)", f"[red]{cvar5:+.2f}%[/red]")
    risk_table.add_row("Upside/Downside ratio", f"{dist['upside_ratio']:.2f}")

    # Trend detail rows
    if dist.get("trend_score") is not None:
        risk_table.add_row("", "")
        ts = dist["trend_score"]
        ts_color = "green" if ts >= 2 else "yellow" if ts == 1 else "red"
        risk_table.add_row("Trend score", f"[{ts_color}]{ts}/3[/{ts_color}]")
        risk_table.add_row(
            "  SMA20 > SMA50",
            "[green]Yes[/green]" if dist.get("sma20_above_sma50") else "[red]No[/red]",
        )
        risk_table.add_row(
            "  Price > SMA50",
            "[green]Yes[/green]" if dist.get("price_above_sma50") else "[red]No[/red]",
        )
        risk_table.add_row(
            "  SMA50 slope up",
            "[green]Yes[/green]" if dist.get("sma50_slope_positive") else "[red]No[/red]",
        )
        if dist["falling_knife_warning"]:
            risk_table.add_row("Falling knife", "[bold red]WARNING[/bold red]")
        risk_table.add_row("Risk-adj score", f"{dist['risk_adjusted_score']:+.2f}%")

    console.print(risk_table)

    # Also show classifier signal if available
    gen = MLSignalGenerator()
    if gen.is_available():
        signal = gen.generate_signal(ticker)
        if signal:
            signal_colors = {"BUY": "green", "SELL": "red", "NEUTRAL": "yellow", "HOLD": "yellow"}
            color = signal_colors.get(signal.action.value, "white")
            console.print(
                f"\n[dim]Classifier signal:[/dim] "
                f"[bold {color}]{signal.action.value}[/bold {color}] — {signal.reason}"
            )


def _fetch_ohlcv(ticker: str, period: str = "6mo") -> "pd.DataFrame | None":  # noqa: F821
    """Fetch OHLCV data for trend scoring. Returns None on failure."""
    try:
        import yfinance as yf

        df = yf.download(ticker, period=period, progress=False)
        if df is not None and len(df) >= 50:
            # Flatten multi-level columns from yfinance
            if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception as e:
        logger.debug("OHLCV fetch failed for %s: %s", ticker, e)
    return None


@app.command("scan")
def ml_scan(
    symbols: Annotated[
        Optional[str],
        typer.Option("--symbols", "-s", help="Comma-separated tickers to scan"),
    ] = None,
    top: Annotated[int, typer.Option("--top", "-n", help="Number of top results to show")] = 10,
    include_falling_knives: Annotated[
        bool,
        typer.Option("--include-falling-knives", help="Include trend_score < 2 stocks"),
    ] = False,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Scan tickers with quantile model, ranked by risk-adjusted score.

    By default filters out falling knives (trend_score < 2).
    Use --include-falling-knives to see all.
    """
    import pandas as pd
    from rich.table import Table

    from advisor.ml.features import FeatureEngine
    from advisor.ml.quantile import QuantileModel

    if not QuantileModel.model_exists():
        output_error("No quantile model found. Run 'advisor ml train-quantile' first.")
        return

    if symbols:
        ticker_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        output_error("Provide symbols with --symbols (e.g., --symbols AAPL,NVDA,GOOGL)")
        return

    qm = QuantileModel.load()
    feature_names = qm.feature_names or FeatureEngine.feature_names()
    engine = FeatureEngine()
    horizon = qm.metadata.get("horizon", 10)

    results: list[dict] = []
    for ticker in ticker_list:
        try:
            features = engine.compute_features(ticker)
            if not features:
                console.print(f"[dim]Skipping {ticker}: no features[/dim]")
                continue

            row = {}
            for col in feature_names:
                if col.endswith("_cs_rank"):
                    default = 0.5
                elif col == "hmm_regime":
                    default = 1.0
                elif col.startswith("hmm_") and col.endswith("_prob"):
                    default = 1.0 / 3.0
                else:
                    default = 0.0
                row[col] = features.get(col, default)

            features_df = pd.DataFrame([row])
            ohlcv_df = _fetch_ohlcv(ticker)
            dist = qm.predict_distribution(features_df, ohlcv_df=ohlcv_df)
            dist["ticker"] = ticker
            results.append(dist)
        except Exception as e:
            console.print(f"[dim]Skipping {ticker}: {e}[/dim]")

    if not results:
        output_error("No results produced.")
        return

    # Filter falling knives unless opted in
    if not include_falling_knives:
        filtered = [r for r in results if not r.get("falling_knife_warning", False)]
        n_removed = len(results) - len(filtered)
        if n_removed > 0:
            console.print(
                f"[dim]Filtered {n_removed} falling knife(s). "
                f"Use --include-falling-knives to see all.[/dim]"
            )
        results = filtered

    # Rank by risk-adjusted score
    results.sort(key=lambda r: r.get("risk_adjusted_score", r["median_return"]), reverse=True)
    results = results[:top]

    if output == "json":
        output_json(results)
        return

    table = Table(title=f"ML Scan — Top {len(results)} by Risk-Adjusted Score ({horizon}d)")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Median", justify="right")
    table.add_column("Risk-Adj", justify="right")
    table.add_column("VaR 10%", justify="right")
    table.add_column("80% Range", justify="right")
    table.add_column("Trend", justify="center")
    table.add_column("Skew", justify="center")

    for i, r in enumerate(results, 1):
        med = r["median_return"]
        med_color = "green" if med > 0 else "red"
        ras = r.get("risk_adjusted_score", med)
        ras_color = "green" if ras > 0 else "red"
        var10 = r["var_10"]
        rng = r["expected_range"]
        ts = r.get("trend_score")
        fk = r.get("falling_knife_warning", False)

        if ts is not None:
            ts_color = "green" if ts >= 2 else "yellow" if ts == 1 else "red"
            trend_str = f"[{ts_color}]{ts}/3[/{ts_color}]"
            if fk:
                trend_str += " [red]FK[/red]"
        else:
            trend_str = "[dim]?[/dim]"

        skew = r.get("distribution_skew", "?")
        skew_map = {
            "right": "[green]R[/green]",
            "left": "[red]L[/red]",
            "symmetric": "[white]S[/white]",
        }

        table.add_row(
            str(i),
            r["ticker"],
            f"[{med_color}]{med:+.2f}%[/{med_color}]",
            f"[{ras_color}]{ras:+.2f}%[/{ras_color}]",
            f"[red]{var10:+.2f}%[/red]",
            f"{rng[0]:+.1f} / {rng[1]:+.1f}",
            trend_str,
            skew_map.get(skew, skew),
        )

    console.print(table)
