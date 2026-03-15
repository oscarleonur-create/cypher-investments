"""Plotly chart factory functions for Monte Carlo PCS simulator results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from advisor.simulator.models import SimResult

if TYPE_CHECKING:
    from advisor.simulator.exit_sensitivity import ExitSensitivityResult


def pnl_distribution_chart(results: list[SimResult]) -> go.Figure:
    """Box plot of P&L distributions (p5/p25/p50/p75/p95) across top candidates."""
    fig = go.Figure()
    for r in results:
        label = f"{r.symbol} ${r.short_strike}/{r.long_strike}"
        fig.add_trace(
            go.Box(
                lowerfence=[r.pnl_p5],
                q1=[r.pnl_p25],
                median=[r.pnl_p50],
                q3=[r.pnl_p75],
                upperfence=[r.pnl_p95],
                name=label,
                marker_color="#636EFA",
                boxpoints=False,
            )
        )

    fig.update_layout(
        title="P&L Distribution by Candidate",
        yaxis_title="P&L ($)",
        showlegend=False,
        template="plotly_dark",
        height=400,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def exit_breakdown_chart(results: list[SimResult]) -> go.Figure:
    """Horizontal stacked bar of exit reason fractions across candidates."""
    labels = [f"{r.symbol} ${r.short_strike}/{r.long_strike}" for r in results]
    categories = [
        ("Profit Target", [r.exit_profit_target for r in results], "#00CC96"),
        ("Stop Loss", [r.exit_stop_loss for r in results], "#EF553B"),
        ("DTE Close", [r.exit_dte for r in results], "#636EFA"),
        ("Trailing Stop", [r.exit_trailing_stop for r in results], "#FFA15A"),
        ("Expiration", [r.exit_expiration for r in results], "#AB63FA"),
    ]

    fig = go.Figure()
    for name, values, color in categories:
        fig.add_trace(
            go.Bar(
                y=labels,
                x=[v * 100 for v in values],
                name=name,
                orientation="h",
                marker_color=color,
                text=[f"{v:.0%}" for v in values],
                textposition="inside",
            )
        )

    fig.update_layout(
        title="Exit Breakdown",
        xaxis_title="Percentage (%)",
        barmode="stack",
        template="plotly_dark",
        height=max(300, len(results) * 60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def risk_comparison_chart(results: list[SimResult]) -> go.Figure:
    """Grouped bar comparing POP, touch prob, and stop prob across candidates."""
    labels = [f"{r.symbol} ${r.short_strike}/{r.long_strike}" for r in results]
    metrics = [
        ("POP", [r.pop * 100 for r in results], "#00CC96"),
        ("Touch Prob", [r.touch_prob * 100 for r in results], "#FFA15A"),
        ("Stop Prob", [r.stop_prob * 100 for r in results], "#EF553B"),
    ]

    fig = go.Figure()
    for name, values, color in metrics:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                name=name,
                marker_color=color,
                text=[f"{v:.0f}%" for v in values],
                textposition="outside",
            )
        )

    fig.update_layout(
        title="Risk Comparison",
        yaxis_title="Probability (%)",
        barmode="group",
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def risk_return_scatter(results: list[SimResult]) -> go.Figure:
    """Scatter of EV/BP (x) vs CVaR95 (y), bubble size = credit."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[r.ev_per_bp for r in results],
            y=[r.cvar_95 for r in results],
            mode="markers+text",
            marker=dict(
                size=[max(r.net_credit * 40, 10) for r in results],
                color=[r.pop * 100 for r in results],
                colorscale="RdYlGn",
                colorbar=dict(title="POP %"),
                showscale=True,
                line=dict(width=1, color="white"),
            ),
            text=[f"{r.symbol} ${r.short_strike}/{r.long_strike}" for r in results],
            textposition="top center",
            textfont=dict(size=10),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "EV/BP: %{x:.4f}<br>"
                "CVaR95: $%{y:.2f}<br>"
                "Credit: $%{customdata:.2f}<extra></extra>"
            ),
            customdata=[r.net_credit for r in results],
        )
    )

    fig.update_layout(
        title="Risk-Return Profile",
        xaxis_title="EV / Buying Power",
        yaxis_title="CVaR 95% ($)",
        template="plotly_dark",
        height=450,
    )
    return fig


# ── Validation charts ─────────────────────────────────────────────────────────


def calibration_curve_chart(records: list[dict]) -> go.Figure:
    """Reliability diagram: predicted probability vs observed frequency.

    Separate traces for POP, touch, and stop. Diagonal = perfect calibration.
    Marker size scales with bin count.
    """
    from advisor.simulator.validation import _compute_calibration_buckets

    fig = go.Figure()

    # Perfect calibration diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray", width=1),
            name="Perfect",
            showlegend=True,
        )
    )

    # Build traces for each metric
    metrics = [
        ("POP", "predicted_pop", "actual_profit", "#00CC96"),
        ("Touch", "predicted_touch", "actual_touch", "#FFA15A"),
        ("Stop", "predicted_stop", "actual_stop", "#EF553B"),
    ]

    for label, pred_key, act_key, color in metrics:
        predicted = [r[pred_key] for r in records if pred_key in r and act_key in r]
        actual = [r[act_key] for r in records if pred_key in r and act_key in r]

        if not predicted:
            continue

        buckets = _compute_calibration_buckets(predicted, actual)
        non_empty = [b for b in buckets if b["count"] > 0]

        if not non_empty:
            continue

        sizes = [max(b["count"] * 3, 8) for b in non_empty]

        fig.add_trace(
            go.Scatter(
                x=[b["predicted_mean"] for b in non_empty],
                y=[b["actual_mean"] for b in non_empty],
                mode="markers+lines",
                marker=dict(size=sizes, color=color, line=dict(width=1, color="white")),
                line=dict(color=color, width=1),
                name=label,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Predicted: %{x:.2f}<br>"
                    "Actual: %{y:.2f}<br>"
                    "Count: %{customdata}<extra></extra>"
                ),
                customdata=[b["count"] for b in non_empty],
            )
        )

    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_dark",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def brier_trend_chart(records: list[dict], window: int = 30) -> go.Figure:
    """Rolling Brier score over time with reference quality lines.

    Separate traces for POP, touch, and stop.
    """
    fig = go.Figure()

    if len(records) < 2:
        fig.update_layout(
            title="Brier Score Trend (insufficient data)",
            template="plotly_dark",
            height=400,
            annotations=[
                dict(
                    text="Need more resolved predictions for trend analysis",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                )
            ],
        )
        return fig

    # Sort by created_at
    sorted_records = sorted(records, key=lambda r: r.get("created_at", ""))

    metrics = [
        ("POP", "predicted_pop", "actual_profit", "#00CC96"),
        ("Touch", "predicted_touch", "actual_touch", "#FFA15A"),
        ("Stop", "predicted_stop", "actual_stop", "#EF553B"),
    ]

    dates = [r.get("created_at", "")[:10] for r in sorted_records]

    for label, pred_key, act_key, color in metrics:
        # Compute per-record squared error
        errors = []
        for r in sorted_records:
            pred = r.get(pred_key)
            act = r.get(act_key)
            if pred is not None and act is not None:
                errors.append((pred - act) ** 2)
            else:
                errors.append(np.nan)

        if all(np.isnan(e) for e in errors):
            continue

        # Rolling mean
        errors_arr = np.array(errors, dtype=float)
        rolling_window = min(window, len(errors_arr))
        if rolling_window < 1:
            continue

        rolling_brier = np.full_like(errors_arr, np.nan)
        for i in range(rolling_window - 1, len(errors_arr)):
            window_slice = errors_arr[i - rolling_window + 1 : i + 1]
            valid = window_slice[~np.isnan(window_slice)]
            if len(valid) > 0:
                rolling_brier[i] = float(np.mean(valid))

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_brier.tolist(),
                mode="lines",
                line=dict(color=color, width=2),
                name=label,
            )
        )

    # Reference lines
    fig.add_hline(
        y=0.10, line_dash="dot", line_color="green", opacity=0.5, annotation_text="Excellent (0.10)"
    )
    fig.add_hline(
        y=0.25, line_dash="dot", line_color="red", opacity=0.5, annotation_text="Poor (0.25)"
    )

    fig.update_layout(
        title=f"Rolling Brier Score ({window}-sample window)",
        xaxis_title="Date",
        yaxis_title="Brier Score",
        yaxis=dict(rangemode="tozero"),
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def prediction_scatter_chart(records: list[dict]) -> go.Figure:
    """Scatter of predicted EV vs actual P&L with OLS regression line."""
    fig = go.Figure()

    predicted_ev = [r["predicted_ev"] for r in records if "predicted_ev" in r and "actual_pnl" in r]
    actual_pnl = [r["actual_pnl"] for r in records if "predicted_ev" in r and "actual_pnl" in r]

    if not predicted_ev:
        fig.update_layout(
            title="Predicted EV vs Actual P&L (no data)",
            template="plotly_dark",
            height=450,
        )
        return fig

    pred = np.array(predicted_ev, dtype=float)
    actual = np.array(actual_pnl, dtype=float)

    # Scatter
    fig.add_trace(
        go.Scatter(
            x=pred.tolist(),
            y=actual.tolist(),
            mode="markers",
            marker=dict(
                color=actual.tolist(),
                colorscale="RdYlGn",
                colorbar=dict(title="P&L ($)"),
                showscale=True,
                size=8,
                line=dict(width=1, color="white"),
            ),
            name="Trades",
            hovertemplate=("Predicted EV: $%{x:.2f}<br>" "Actual P&L: $%{y:.2f}<extra></extra>"),
        )
    )

    # Perfect prediction diagonal
    all_vals = np.concatenate([pred, actual])
    val_min, val_max = float(np.min(all_vals)), float(np.max(all_vals))
    fig.add_trace(
        go.Scatter(
            x=[val_min, val_max],
            y=[val_min, val_max],
            mode="lines",
            line=dict(dash="dash", color="gray", width=1),
            name="Perfect",
        )
    )

    # OLS regression line
    if len(pred) >= 2:
        valid = ~(np.isnan(pred) | np.isnan(actual))
        if np.sum(valid) >= 2:
            coeffs = np.polyfit(pred[valid], actual[valid], 1)
            x_line = np.linspace(float(pred[valid].min()), float(pred[valid].max()), 50)
            y_line = np.polyval(coeffs, x_line)

            # R-squared
            y_hat = np.polyval(coeffs, pred[valid])
            ss_res = np.sum((actual[valid] - y_hat) ** 2)
            ss_tot = np.sum((actual[valid] - np.mean(actual[valid])) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            fig.add_trace(
                go.Scatter(
                    x=x_line.tolist(),
                    y=y_line.tolist(),
                    mode="lines",
                    line=dict(color="#636EFA", width=2),
                    name=f"OLS (R\u00b2={r_squared:.3f})",
                )
            )

    fig.update_layout(
        title="Predicted EV vs Actual P&L",
        xaxis_title="Predicted EV ($)",
        yaxis_title="Actual P&L ($)",
        template="plotly_dark",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Exit sensitivity charts ─────────────────────────────────────────────────


def exit_sensitivity_heatmap(
    result: "ExitSensitivityResult",
    metric: str = "ev",
    fixed_param: str = "close_at_dte",
    fixed_value: int | float | None = None,
) -> go.Figure:
    """2D heatmap of exit sensitivity results.

    Fixes one parameter dimension and shows the other two as axes.
    Default: fix DTE, show profit_target (x) vs stop_loss (y), color by metric.
    """

    points = result.points
    if not points:
        fig = go.Figure()
        fig.update_layout(title="No sensitivity data", template="plotly_dark")
        return fig

    # Determine fixed value (use most common if not specified)
    if fixed_value is None:
        vals = [getattr(p, fixed_param) for p in points]
        fixed_value = max(set(vals), key=vals.count)

    filtered = [p for p in points if getattr(p, fixed_param) == fixed_value]
    if not filtered:
        fig = go.Figure()
        fig.update_layout(title=f"No data for {fixed_param}={fixed_value}", template="plotly_dark")
        return fig

    # Determine axes based on fixed param
    if fixed_param == "close_at_dte":
        x_param, y_param = "profit_target_pct", "stop_loss_multiplier"
        x_label, y_label = "Profit Target (%)", "Stop Loss Multiplier"
    elif fixed_param == "profit_target_pct":
        x_param, y_param = "stop_loss_multiplier", "close_at_dte"
        x_label, y_label = "Stop Loss Multiplier", "Close at DTE"
    else:
        x_param, y_param = "profit_target_pct", "close_at_dte"
        x_label, y_label = "Profit Target (%)", "Close at DTE"

    x_vals = sorted(set(getattr(p, x_param) for p in filtered))
    y_vals = sorted(set(getattr(p, y_param) for p in filtered))

    # Build Z matrix
    z = np.full((len(y_vals), len(x_vals)), np.nan)
    for p in filtered:
        xi = x_vals.index(getattr(p, x_param))
        yi = y_vals.index(getattr(p, y_param))
        z[yi, xi] = getattr(p, metric, 0)

    # Format x labels
    x_text = (
        [f"{v:.0%}" for v in x_vals] if x_param == "profit_target_pct" else [str(v) for v in x_vals]
    )
    y_text = (
        [f"{v:.0%}" for v in y_vals] if y_param == "profit_target_pct" else [str(v) for v in y_vals]
    )

    metric_labels = {
        "ev": "EV ($)",
        "pop": "POP",
        "cvar_95": "CVaR95 ($)",
        "sharpe_approx": "Sharpe",
        "avg_hold_days": "Avg Hold (days)",
    }

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_text,
            y=y_text,
            colorscale="RdYlGn",
            colorbar=dict(title=metric_labels.get(metric, metric)),
            text=np.round(z, 2),
            texttemplate="%{text}",
            hovertemplate=(
                f"{x_label}: %{{x}}<br>"
                f"{y_label}: %{{y}}<br>"
                f"{metric}: %{{z:.2f}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            f"Exit Sensitivity: {metric_labels.get(metric, metric)} "
            f"({fixed_param}={fixed_value})"
        ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_dark",
        height=500,
    )
    return fig


def exit_sensitivity_parallel_coords(result: "ExitSensitivityResult") -> go.Figure:
    """Parallel coordinates plot for 3D exit parameter exploration."""

    points = result.points
    if not points:
        fig = go.Figure()
        fig.update_layout(title="No sensitivity data", template="plotly_dark")
        return fig

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=[p.ev for p in points],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="EV ($)"),
            ),
            dimensions=[
                dict(
                    range=[
                        min(p.profit_target_pct for p in points),
                        max(p.profit_target_pct for p in points),
                    ],
                    label="Profit Target",
                    values=[p.profit_target_pct for p in points],
                    ticktext=[f"{v:.0%}" for v in sorted(set(p.profit_target_pct for p in points))],
                    tickvals=sorted(set(p.profit_target_pct for p in points)),
                ),
                dict(
                    range=[
                        min(p.stop_loss_multiplier for p in points),
                        max(p.stop_loss_multiplier for p in points),
                    ],
                    label="Stop Loss",
                    values=[p.stop_loss_multiplier for p in points],
                ),
                dict(
                    range=[
                        min(p.close_at_dte for p in points),
                        max(p.close_at_dte for p in points),
                    ],
                    label="DTE Exit",
                    values=[p.close_at_dte for p in points],
                ),
                dict(
                    range=[min(p.pop for p in points), max(p.pop for p in points)],
                    label="POP",
                    values=[p.pop for p in points],
                ),
                dict(
                    range=[min(p.cvar_95 for p in points), max(p.cvar_95 for p in points)],
                    label="CVaR95",
                    values=[p.cvar_95 for p in points],
                ),
                dict(
                    range=[
                        min(p.avg_hold_days for p in points),
                        max(p.avg_hold_days for p in points),
                    ],
                    label="Avg Hold",
                    values=[p.avg_hold_days for p in points],
                ),
            ],
        )
    )

    fig.update_layout(
        title="Exit Parameter Exploration (Parallel Coordinates)",
        template="plotly_dark",
        height=500,
    )
    return fig
