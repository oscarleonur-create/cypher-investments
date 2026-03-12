"""Plotly chart factory functions for scenario simulation results."""

from __future__ import annotations

import plotly.graph_objects as go

from advisor.scenario.models import CompositeStrategyResult, ScenarioSimResult


def strategy_ranking_bar(result: ScenarioSimResult) -> go.Figure:
    """Horizontal bar chart of strategy risk-adjusted scores."""
    composites = sorted(result.composites, key=lambda c: c.risk_adjusted_score)
    names = [c.strategy_name for c in composites]
    scores = [c.risk_adjusted_score for c in composites]
    colors = ["#00CC96" if s > 0 else "#EF553B" for s in scores]

    fig = go.Figure(
        go.Bar(
            y=names,
            x=scores,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.1f}" for s in scores],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Risk-Adjusted Score by Strategy",
        xaxis_title="Score",
        template="plotly_dark",
        height=max(300, len(composites) * 55),
        margin=dict(l=140),
    )
    return fig


def return_distribution_box(result: ScenarioSimResult) -> go.Figure:
    """Box plot of return distributions across scenarios for each strategy."""
    fig = go.Figure()

    for comp in result.composites:
        for sr in comp.scenario_results:
            fig.add_trace(
                go.Box(
                    lowerfence=[sr.p5_return_pct],
                    q1=[sr.p25_return_pct],
                    median=[sr.median_return_pct],
                    q3=[sr.p75_return_pct],
                    upperfence=[sr.p95_return_pct],
                    name=f"{comp.strategy_name}",
                    legendgroup=comp.strategy_name,
                    showlegend=(sr == comp.scenario_results[0]),
                    boxpoints=False,
                    x=[sr.scenario_name],
                )
            )

    fig.update_layout(
        title="Return Distribution by Strategy & Scenario",
        yaxis_title="Return (%)",
        xaxis_title="Scenario",
        template="plotly_dark",
        height=450,
        boxmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def risk_return_scatter(result: ScenarioSimResult) -> go.Figure:
    """Scatter of E[Return] vs E[MaxDD] for each strategy, sized by Prob(+)."""
    fig = go.Figure()

    for comp in result.composites:
        fig.add_trace(
            go.Scatter(
                x=[comp.expected_max_dd],
                y=[comp.expected_return],
                mode="markers+text",
                marker=dict(
                    size=max(comp.prob_positive * 50, 10),
                    opacity=0.8,
                ),
                text=[comp.strategy_name],
                textposition="top center",
                name=comp.strategy_name,
                hovertemplate=(
                    f"<b>{comp.strategy_name}</b><br>"
                    f"E[Return]: {comp.expected_return:+.2f}%<br>"
                    f"E[MaxDD]: {comp.expected_max_dd:.2f}%<br>"
                    f"Prob(+): {comp.prob_positive:.0%}<br>"
                    f"Score: {comp.risk_adjusted_score:.1f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Risk-Return Profile (bubble size = Prob(+))",
        xaxis_title="Expected Max Drawdown (%)",
        yaxis_title="Expected Return (%)",
        template="plotly_dark",
        height=450,
        showlegend=False,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    return fig


def scenario_breakdown_bar(comp: CompositeStrategyResult) -> go.Figure:
    """Grouped bar showing mean return and drawdown per scenario for one strategy."""
    scenarios = [sr.scenario_name for sr in comp.scenario_results]
    returns = [sr.mean_return_pct for sr in comp.scenario_results]
    drawdowns = [-sr.mean_max_dd_pct for sr in comp.scenario_results]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=returns,
            name="Mean Return",
            marker_color="#00CC96",
            text=[f"{r:+.1f}%" for r in returns],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=drawdowns,
            name="Avg Drawdown",
            marker_color="#EF553B",
            text=[f"{d:.1f}%" for d in drawdowns],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"{comp.strategy_name} — Return vs Drawdown by Scenario",
        yaxis_title="Percentage (%)",
        template="plotly_dark",
        height=400,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    return fig


def prob_positive_heatmap(result: ScenarioSimResult) -> go.Figure:
    """Heatmap of Prob(+) for each strategy x scenario combination."""
    strategies = [c.strategy_name for c in result.composites]
    scenarios = [sr.scenario_name for sr in result.composites[0].scenario_results]

    z = []
    text = []
    for comp in result.composites:
        row = [sr.prob_positive * 100 for sr in comp.scenario_results]
        z.append(row)
        text.append([f"{v:.0f}%" for v in row])

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=scenarios,
            y=strategies,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, "#EF553B"], [0.5, "#FFA15A"], [1, "#00CC96"]],
            zmin=0,
            zmax=100,
            colorbar=dict(title="Prob(+) %"),
        )
    )

    fig.update_layout(
        title="Probability of Positive Return (Strategy x Scenario)",
        template="plotly_dark",
        height=max(300, len(strategies) * 45 + 100),
    )
    return fig


def scenario_weights_pie(weights: dict[str, float]) -> go.Figure:
    """Pie chart of scenario probability weights."""
    colors = {
        "bull": "#00CC96",
        "sideways": "#636EFA",
        "bear": "#FFA15A",
        "crash": "#EF553B",
    }

    names = list(weights.keys())
    values = list(weights.values())
    marker_colors = [colors.get(n, "#AB63FA") for n in names]

    fig = go.Figure(
        go.Pie(
            labels=names,
            values=values,
            marker=dict(colors=marker_colors),
            textinfo="label+percent",
            hole=0.35,
        )
    )

    fig.update_layout(
        title="Scenario Weights",
        template="plotly_dark",
        height=350,
        showlegend=False,
    )
    return fig
