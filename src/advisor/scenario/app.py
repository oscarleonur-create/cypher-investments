"""Streamlit GUI for Scenario Simulator."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from advisor.scenario.charts import (
    prob_positive_heatmap,
    return_distribution_box,
    risk_return_scatter,
    scenario_breakdown_bar,
    scenario_weights_pie,
    strategy_ranking_bar,
)
from advisor.scenario.models import BUILTIN_SCENARIOS, ScenarioConfig, ScenarioSimResult

st.set_page_config(
    page_title="Scenario Simulator",
    page_icon=":crystal_ball:",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("Scenario Simulator")
st.sidebar.caption("Forward price simulation + strategy evaluation")

symbol = st.sidebar.text_input("Ticker Symbol", value="AAPL", key="symbol").strip().upper()

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation")
dte = st.sidebar.slider("Horizon (trading days)", min_value=5, max_value=120, value=30, step=5)
n_paths = st.sidebar.slider("Paths per scenario", min_value=50, max_value=2000, value=200, step=50)
seed = st.sidebar.number_input("Seed (0 = random)", min_value=0, max_value=99999, value=42)
seed_val = seed if seed > 0 else None
max_workers = st.sidebar.slider(
    "Parallel workers",
    min_value=1,
    max_value=8,
    value=1,
    help="Keep at 1 for reliability. Higher values use multiprocessing.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Scenarios")
scenario_options = list(BUILTIN_SCENARIOS.keys())
selected_scenarios = st.sidebar.multiselect(
    "Select scenarios",
    options=scenario_options,
    default=scenario_options,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategies")
ALL_EQUITY = [
    "buy_hold",
    "sma_crossover",
    "momentum_breakout",
    "buy_the_dip",
    "mean_reversion",
    "pead",
]
selected_strategies = st.sidebar.multiselect(
    "Select strategies",
    options=ALL_EQUITY,
    default=ALL_EQUITY,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Signal Integration")
include_signals = st.sidebar.checkbox("Include alpha/confluence signals", value=False)

run_clicked = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

# ── Main ───────────────────────────────────────────────────────────────────────

tab_run, tab_results, tab_detail = st.tabs(["Overview", "Charts", "Strategy Detail"])

# ── Run simulation ─────────────────────────────────────────────────────────────

if run_clicked:
    if not symbol:
        st.error("Please enter a ticker symbol.")
        st.stop()
    if not selected_scenarios:
        st.error("Please select at least one scenario.")
        st.stop()
    if not selected_strategies:
        st.error("Please select at least one strategy.")
        st.stop()

    config = ScenarioConfig(
        dte=dte,
        n_paths=n_paths,
        seed=seed_val,
    )

    with st.status(f"Running scenario simulation for **{symbol}**...", expanded=True) as status:
        try:
            from advisor.scenario.pipeline import run_scenario_simulation

            st.write("Calibrating, generating paths, running strategies...")
            result = run_scenario_simulation(
                symbol=symbol,
                config=config,
                strategy_names=selected_strategies if selected_strategies != ALL_EQUITY else None,
                scenario_names=selected_scenarios
                if selected_scenarios != scenario_options
                else None,
                include_signals=include_signals,
                max_workers=max_workers,
            )

            total_paths = sum(sr.n_paths for c in result.composites for sr in c.scenario_results)
            n_sc = len(result.scenarios)
            st.write(f"Done: {total_paths} strategy-path evaluations across {n_sc} scenarios")

            st.session_state["scenario_result"] = result
            status.update(label="Simulation complete!", state="complete")
        except Exception as e:
            status.update(label="Simulation failed", state="error")
            st.error(f"Error: {e}")
            import traceback

            st.code(traceback.format_exc())
            st.stop()

# ── Display results ────────────────────────────────────────────────────────────

result: ScenarioSimResult | None = st.session_state.get("scenario_result")

if result is None:
    st.info("Configure parameters in the sidebar and click **Run Simulation** to start.")
    st.stop()

# ── Tab 1: Overview ────────────────────────────────────────────────────────────

with tab_run:
    st.header(f"Scenario Simulation: {result.symbol}")
    caption = (
        f"{result.config.dte}-day horizon | {result.config.n_paths} paths/scenario"
        f" | {len(result.scenarios)} scenarios | {len(result.strategies)} strategies"
    )
    st.caption(caption)

    # Signal context
    if result.signal_context and result.signal_context.alpha_score is not None:
        ctx = result.signal_context
        sig_cols = st.columns(3)
        with sig_cols[0]:
            st.metric("Alpha Score", f"{ctx.alpha_score:.0f}/100")
        with sig_cols[1]:
            st.metric("Alpha Signal", ctx.alpha_signal or "N/A")
        with sig_cols[2]:
            st.metric("Confluence", ctx.confluence_verdict or "N/A")

        if ctx.adjusted_weights:
            st.plotly_chart(scenario_weights_pie(ctx.adjusted_weights), use_container_width=True)
    else:
        # Show default weights
        weights = {s: BUILTIN_SCENARIOS[s].base_probability for s in result.scenarios}
        col_w1, col_w2 = st.columns([1, 2])
        with col_w1:
            st.plotly_chart(scenario_weights_pie(weights), use_container_width=True)
        with col_w2:
            scenario_df = pd.DataFrame(
                [
                    {
                        "Scenario": BUILTIN_SCENARIOS[s].name.title(),
                        "Drift": f"{BUILTIN_SCENARIOS[s].annual_drift:+.0%}",
                        "Vol Mult": f"{BUILTIN_SCENARIOS[s].vol_multiplier:.1f}x",
                        "Weight": f"{BUILTIN_SCENARIOS[s].base_probability:.0%}",
                    }
                    for s in result.scenarios
                ]
            )
            st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Best strategy callout
    if result.best_strategy:
        best = next(c for c in result.composites if c.strategy_name == result.best_strategy)
        st.success(
            f"**Best Strategy: {result.best_strategy}** — "
            f"Score: {best.risk_adjusted_score:.1f} | "
            f"E[Return]: {best.expected_return:+.2f}% | "
            f"E[MaxDD]: {best.expected_max_dd:.2f}% | "
            f"Prob(+): {best.prob_positive:.0%}"
        )

    # Strategy ranking table
    ranking_data = []
    for i, comp in enumerate(result.composites, 1):
        ranking_data.append(
            {
                "Rank": i,
                "Strategy": comp.strategy_name,
                "E[Return]": f"{comp.expected_return:+.2f}%",
                "E[MaxDD]": f"{comp.expected_max_dd:.2f}%",
                "Prob(+)": f"{comp.prob_positive:.0%}",
                "Worst p5": f"{comp.worst_case_return_p5:+.2f}%",
                "Score": f"{comp.risk_adjusted_score:.1f}",
            }
        )

    st.dataframe(
        pd.DataFrame(ranking_data),
        use_container_width=True,
        hide_index=True,
    )

# ── Tab 2: Charts ──────────────────────────────────────────────────────────────

with tab_results:
    if not result.composites:
        st.warning("No results to display.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(strategy_ranking_bar(result), use_container_width=True)

        with col2:
            st.plotly_chart(risk_return_scatter(result), use_container_width=True)

        st.plotly_chart(return_distribution_box(result), use_container_width=True)

        st.plotly_chart(prob_positive_heatmap(result), use_container_width=True)

# ── Tab 3: Strategy Detail ────────────────────────────────────────────────────

with tab_detail:
    if not result.composites:
        st.warning("No results to display.")
    else:
        selected_strat = st.selectbox(
            "Select strategy for detailed breakdown",
            options=[c.strategy_name for c in result.composites],
            key="detail_strategy",
        )

        comp = next(c for c in result.composites if c.strategy_name == selected_strat)

        # Metrics row
        m_cols = st.columns(5)
        with m_cols[0]:
            st.metric("E[Return]", f"{comp.expected_return:+.2f}%")
        with m_cols[1]:
            st.metric("E[MaxDD]", f"{comp.expected_max_dd:.2f}%")
        with m_cols[2]:
            st.metric("Prob(+)", f"{comp.prob_positive:.0%}")
        with m_cols[3]:
            st.metric("Worst p5", f"{comp.worst_case_return_p5:+.2f}%")
        with m_cols[4]:
            st.metric("Score", f"{comp.risk_adjusted_score:.1f}")

        # Scenario breakdown chart
        st.plotly_chart(scenario_breakdown_bar(comp), use_container_width=True)

        # Detailed scenario table
        detail_data = []
        for sr in comp.scenario_results:
            detail_data.append(
                {
                    "Scenario": sr.scenario_name.title(),
                    "Paths": sr.n_paths,
                    "Mean Ret": f"{sr.mean_return_pct:+.2f}%",
                    "Median Ret": f"{sr.median_return_pct:+.2f}%",
                    "p5": f"{sr.p5_return_pct:+.2f}%",
                    "p25": f"{sr.p25_return_pct:+.2f}%",
                    "p75": f"{sr.p75_return_pct:+.2f}%",
                    "p95": f"{sr.p95_return_pct:+.2f}%",
                    "Prob(+)": f"{sr.prob_positive:.0%}",
                    "Avg DD": f"{sr.mean_max_dd_pct:.2f}%",
                    "Avg Trades": f"{sr.avg_trades:.1f}",
                    "Win Rate": f"{sr.avg_win_rate:.1f}%" if sr.avg_win_rate is not None else "N/A",
                }
            )

        st.dataframe(
            pd.DataFrame(detail_data),
            use_container_width=True,
            hide_index=True,
        )
