"""Streamlit GUI for Monte Carlo PCS Simulator."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from advisor.market.options_scanner import UNIVERSES
from advisor.simulator.charts import (
    brier_trend_chart,
    calibration_curve_chart,
    exit_breakdown_chart,
    exit_sensitivity_heatmap,
    exit_sensitivity_parallel_coords,
    pnl_distribution_chart,
    prediction_scatter_chart,
    risk_comparison_chart,
    risk_return_scatter,
)
from advisor.simulator.db import SimulatorStore
from advisor.simulator.models import PipelineResult, SimConfig
from advisor.simulator.pipeline import SimulatorPipeline

st.set_page_config(
    page_title="PCS Monte Carlo Simulator",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("PCS Monte Carlo Simulator")

universe_options = list(UNIVERSES.keys()) + ["custom"]
universe = st.sidebar.selectbox("Universe", universe_options, index=1)

if universe == "custom":
    custom_input = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL, MSFT")
    tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
else:
    tickers = UNIVERSES[universe]
    st.sidebar.caption(f"Tickers: {', '.join(tickers)}")

max_bp = st.sidebar.number_input(
    "Max Buying Power ($)", min_value=500, max_value=100_000, value=5000, step=500
)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Paths")
quick_paths = st.sidebar.slider(
    "Quick sim paths", min_value=1_000, max_value=50_000, value=10_000, step=1_000
)
deep_paths = st.sidebar.slider(
    "Deep sim paths", min_value=10_000, max_value=500_000, value=100_000, step=10_000
)
top_n = st.sidebar.slider("Top N results", min_value=1, max_value=20, value=5)

st.sidebar.markdown("---")
st.sidebar.subheader("Exit Rules")
profit_target = st.sidebar.slider("Profit target (%)", min_value=10, max_value=90, value=50, step=5)
stop_loss_mult = st.sidebar.slider(
    "Stop loss multiplier", min_value=1.0, max_value=5.0, value=2.0, step=0.5
)
close_dte = st.sidebar.slider("Close at DTE", min_value=0, max_value=45, value=7)

st.sidebar.markdown("---")
st.sidebar.subheader("Spread Filters")
min_credit = st.sidebar.number_input(
    "Min credit ($)", min_value=0.05, max_value=2.0, value=0.10, step=0.05
)
min_width = st.sidebar.number_input(
    "Min width ($)", min_value=1.0, max_value=20.0, value=2.0, step=0.5
)
max_width = st.sidebar.number_input(
    "Max width ($)", min_value=1.0, max_value=50.0, value=10.0, step=1.0
)
delta_target = st.sidebar.number_input(
    "Delta target (0 = auto)", min_value=0.0, max_value=0.50, value=0.0, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced")
slippage = st.sidebar.number_input(
    "Slippage (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5
)

st.sidebar.markdown("---")
st.sidebar.subheader("Variance Reduction")
use_antithetic = st.sidebar.checkbox("Antithetic variates", value=True)
use_control_variate = st.sidebar.checkbox("Control variate (BSM)", value=True)
use_stratified = st.sidebar.checkbox("Stratified sampling", value=False)
use_importance_sampling = st.sidebar.checkbox("Importance sampling (tail risk)", value=False)


def _build_config() -> SimConfig:
    return SimConfig(
        n_paths=quick_paths,
        profit_target_pct=profit_target / 100,
        stop_loss_multiplier=stop_loss_mult,
        close_at_dte=close_dte,
        slippage_pct=slippage / 100,
        min_credit=min_credit,
        min_width=min_width,
        max_width=max_width,
        max_buying_power=max_bp,
        delta_target=delta_target if delta_target > 0 else None,
        use_antithetic=use_antithetic,
        use_control_variate=use_control_variate,
        use_stratified=use_stratified,
        use_importance_sampling=use_importance_sampling,
    )


run_clicked = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

# ── Main area ──────────────────────────────────────────────────────────────────

tab_run, tab_results, tab_history, tab_validation, tab_exit_sens = st.tabs(
    ["Run Sim", "Results", "History", "Validation", "Exit Sensitivity"]
)

# ── Run Sim tab ────────────────────────────────────────────────────────────────

with tab_run:
    if run_clicked:
        if not tickers:
            st.error("No tickers selected. Enter tickers or pick a universe.")
        else:
            config = _build_config()
            store = SimulatorStore()
            progress_placeholder = st.empty()

            def _update_progress(msg: str) -> None:
                progress_placeholder.text(msg)

            try:
                with st.status("Running Monte Carlo simulation...", expanded=True) as status:
                    st.write(f"Tickers: {', '.join(tickers)}")
                    st.write(f"Quick paths: {quick_paths:,} | Deep paths: {deep_paths:,}")

                    pipeline = SimulatorPipeline(
                        config=config,
                        store=store,
                        progress_callback=_update_progress,
                    )
                    result = pipeline.run(
                        symbols=tickers,
                        top_n=top_n,
                        quick_paths=quick_paths,
                        deep_paths=deep_paths,
                    )
                    st.session_state["pipeline_result"] = result
                    status.update(label="Simulation complete!", state="complete")
            except Exception as e:
                st.error(f"Simulation failed: {e}")
            finally:
                store.close()

    result: PipelineResult | None = st.session_state.get("pipeline_result")

    if result:
        # Calibration metrics row — cal_params is {symbol: {param: value}}
        cal = result.calibration_params
        if cal:
            # Extract first symbol's params
            first_sym = next(iter(cal))
            params = cal[first_sym] if isinstance(cal[first_sym], dict) else cal
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Calibration Symbol", first_sym if isinstance(cal[first_sym], dict) else "N/A"
            )
            c2.metric("Student-t df", f"{params.get('student_t_df', 0):.1f}")
            vol_source = params.get("vol_source", "historical")
            vol_label = "IV" if vol_source == "live_iv" else "HV"
            c3.metric("Vol Level", f"{params.get('vol_mean_level', 0):.0%} ({vol_label})")
            c4.metric("Leverage Effect", f"{params.get('leverage_effect', 0):.2f}")

        # Summary
        st.info(
            f"Scanned **{result.symbols_scanned}** symbols | "
            f"**{result.candidates_generated}** candidates generated | "
            f"**{result.candidates_simulated}** simulations run | "
            f"**{len(result.top_results)}** top results"
        )

        if not result.top_results:
            st.warning(
                "No candidates survived filtering. Try lowering min credit, "
                "widening spread filters, or adding more tickers."
            )
    elif not run_clicked:
        st.markdown("Configure parameters in the sidebar and click **Run Simulation** to start.")

# ── Results tab ────────────────────────────────────────────────────────────────

with tab_results:
    result = st.session_state.get("pipeline_result")
    if not result or not result.top_results:
        st.info("Run a simulation first to see results here.")
    else:
        results = result.top_results

        # Ranking table
        st.subheader("Ranking Table")
        df = pd.DataFrame(
            [
                {
                    "Symbol": r.symbol,
                    "Short": f"${r.short_strike:.2f}",
                    "Long": f"${r.long_strike:.2f}",
                    "DTE": r.dte,
                    "Credit": f"${r.net_credit:.2f}",
                    "EV": f"${r.ev:+.2f}",
                    "POP": f"{r.pop:.0%}",
                    "Touch%": f"{r.touch_prob:.0%}",
                    "CVaR95": f"${r.cvar_95:.2f}",
                    "Stop%": f"{r.stop_prob:.0%}",
                    "Hold (days)": f"{r.avg_hold_days:.0f}",
                    "EV/BP": f"{r.ev_per_bp:.4f}",
                }
                for r in results
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Charts in a 2x2 grid
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(pnl_distribution_chart(results), use_container_width=True)
        with col2:
            st.plotly_chart(exit_breakdown_chart(results), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(risk_comparison_chart(results), use_container_width=True)
        with col4:
            st.plotly_chart(risk_return_scatter(results), use_container_width=True)

# ── History tab ────────────────────────────────────────────────────────────────

with tab_history:
    st.subheader("Historical Simulation Results")

    hcol1, hcol2 = st.columns(2)
    with hcol1:
        start_date = st.date_input("Start date", value=date.today() - timedelta(days=30))
    with hcol2:
        end_date = st.date_input("End date", value=date.today())

    try:
        store = SimulatorStore()
        history = store.get_results_by_date_range(
            start_date.isoformat(),
            end_date.isoformat() + "T23:59:59",
        )
        # Brier score calibration metrics
        brier = store.compute_brier_scores(lookback_days=90)
        store.close()
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        history = []
        brier = {"pop_brier": None, "touch_brier": None, "stop_brier": None, "n_samples": 0}

    # Display Brier scores if we have calibration data
    if brier["n_samples"] > 0:
        st.subheader("Calibration Quality (Brier Scores)")
        st.caption("Lower is better: < 0.10 excellent, < 0.20 good, > 0.25 poor")
        bc1, bc2, bc3, bc4 = st.columns(4)

        def _brier_color(score: float | None) -> str:
            if score is None:
                return "off"
            if score < 0.10:
                return "normal"
            if score < 0.20:
                return "normal"
            return "inverse"

        bc1.metric(
            "POP Brier", f"{brier['pop_brier']:.3f}" if brier["pop_brier"] is not None else "N/A"
        )
        bc2.metric(
            "Touch Brier",
            f"{brier['touch_brier']:.3f}" if brier["touch_brier"] is not None else "N/A",
        )
        bc3.metric(
            "Stop Brier", f"{brier['stop_brier']:.3f}" if brier["stop_brier"] is not None else "N/A"
        )
        bc4.metric("Samples", brier["n_samples"])
        st.markdown("---")

    if not history:
        st.info("No historical results found for this date range.")
    else:
        hist_df = pd.DataFrame(history)

        display_cols = [
            "symbol",
            "short_strike",
            "long_strike",
            "dte",
            "net_credit",
            "ev",
            "pop",
            "touch_prob",
            "cvar_95",
            "stop_prob",
            "ev_per_bp",
            "created_at",
        ]
        available = [c for c in display_cols if c in hist_df.columns]
        st.dataframe(hist_df[available], use_container_width=True, hide_index=True)

        # EV/BP trend scatter over time
        if "created_at" in hist_df.columns and "ev_per_bp" in hist_df.columns:
            import plotly.express as px

            hist_df["created_at"] = pd.to_datetime(hist_df["created_at"])
            fig = px.scatter(
                hist_df,
                x="created_at",
                y="ev_per_bp",
                color="symbol",
                size="net_credit",
                hover_data=["short_strike", "long_strike", "dte"],
                title="EV/BP Over Time",
                template="plotly_dark",
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="EV / Buying Power",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Validation tab ────────────────────────────────────────────────────────────

with tab_validation:
    st.subheader("Prediction Validation")

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        val_symbol = st.text_input("Symbol filter (blank = all)", value="", key="val_symbol")
    with vcol2:
        val_lookback = st.slider(
            "Lookback (days)", min_value=30, max_value=365, value=90, key="val_lookback"
        )

    try:
        val_store = SimulatorStore()
        resolved = val_store.get_resolved_calibrations(
            symbol=val_symbol.upper() if val_symbol else None,
            lookback_days=val_lookback,
        )
        val_brier = val_store.compute_brier_scores(
            symbol=val_symbol.upper() if val_symbol else None,
            lookback_days=val_lookback,
        )
        val_store.close()
    except Exception as e:
        st.error(f"Failed to load validation data: {e}")
        resolved = []
        val_brier = {"pop_brier": None, "touch_brier": None, "stop_brier": None, "n_samples": 0}

    if not resolved:
        st.info(
            "No resolved predictions found. Run `advisor options validate` to resolve "
            "expired predictions against historical prices."
        )
    else:
        # Aggregate Brier score metrics row
        vb1, vb2, vb3, vb4 = st.columns(4)
        vb1.metric(
            "POP Brier",
            f"{val_brier['pop_brier']:.3f}" if val_brier["pop_brier"] is not None else "N/A",
        )
        vb2.metric(
            "Touch Brier",
            f"{val_brier['touch_brier']:.3f}" if val_brier["touch_brier"] is not None else "N/A",
        )
        vb3.metric(
            "Stop Brier",
            f"{val_brier['stop_brier']:.3f}" if val_brier["stop_brier"] is not None else "N/A",
        )
        vb4.metric("Samples", val_brier["n_samples"])

        st.caption("Lower is better: < 0.10 excellent, < 0.20 good, > 0.25 poor")
        st.markdown("---")

        # 2x1 chart grid: calibration curve + prediction scatter
        vc1, vc2 = st.columns(2)
        with vc1:
            st.plotly_chart(calibration_curve_chart(resolved), use_container_width=True)
        with vc2:
            st.plotly_chart(prediction_scatter_chart(resolved), use_container_width=True)

        # Full-width Brier trend chart below
        st.plotly_chart(brier_trend_chart(resolved, window=30), use_container_width=True)

# ── Exit Sensitivity tab ─────────────────────────────────────────────────────

with tab_exit_sens:
    st.subheader("Exit Parameter Sensitivity")
    st.caption(
        "Sweep exit parameters over shared MC paths to find optimal settings. "
        "Inherits simulation config from sidebar."
    )

    es_col1, es_col2, es_col3 = st.columns(3)
    with es_col1:
        es_pt_input = st.text_input(
            "Profit targets (comma-separated)", value="0.25,0.40,0.50,0.60,0.75", key="es_pt"
        )
    with es_col2:
        es_sl_input = st.text_input(
            "Stop losses (comma-separated)", value="1.0,1.5,2.0,3.0", key="es_sl"
        )
    with es_col3:
        es_dte_input = st.text_input(
            "DTE exits (comma-separated)", value="0,5,7,14,21", key="es_dte"
        )

    es_metric = st.selectbox(
        "Heatmap metric",
        ["ev", "pop", "cvar_95", "sharpe_approx", "avg_hold_days"],
        index=0,
        key="es_metric",
    )

    es_run = st.button("Run Sensitivity Sweep", key="es_run")

    if es_run:
        result = st.session_state.get("pipeline_result")
        if not result or not result.top_results:
            st.warning("Run a simulation first to generate candidates.")
        else:
            from advisor.simulator.engine import MonteCarloEngine
            from advisor.simulator.exit_sensitivity import ExitSensitivityAnalyzer
            from advisor.simulator.models import PCSCandidate

            config = _build_config()
            engine = MonteCarloEngine(config)

            # Use top candidate
            top = result.top_results[0]
            candidate = PCSCandidate(
                symbol=top.symbol,
                expiration="",
                dte=top.dte,
                short_strike=top.short_strike,
                long_strike=top.long_strike,
                width=top.short_strike - top.long_strike,
                short_bid=0,
                short_ask=0,
                long_bid=0,
                long_ask=0,
                net_credit=top.net_credit,
                mid_credit=top.net_credit,
                short_delta=0.25,
                short_gamma=0,
                short_theta=0,
                short_vega=0,
                short_iv=config.vol_mean_level,
                long_delta=0,
                long_iv=config.vol_mean_level,
                underlying_price=top.short_strike / 0.95,
                buying_power=(top.short_strike - top.long_strike - top.net_credit) * 100,
            )

            pt_list = [float(x.strip()) for x in es_pt_input.split(",")]
            sl_list = [float(x.strip()) for x in es_sl_input.split(",")]
            dte_list = [int(x.strip()) for x in es_dte_input.split(",")]

            with st.spinner(f"Sweeping {len(pt_list) * len(sl_list) * len(dte_list)} combos..."):
                analyzer = ExitSensitivityAnalyzer(engine, candidate)
                sens_result = analyzer.sweep(pt_list, sl_list, dte_list)
                st.session_state["exit_sensitivity_result"] = sens_result

    sens_result = st.session_state.get("exit_sensitivity_result")
    if sens_result and sens_result.points:
        st.success(
            f"{len(sens_result.points)} combos evaluated over " f"{sens_result.n_paths:,} paths"
        )

        # Heatmap
        st.plotly_chart(
            exit_sensitivity_heatmap(sens_result, metric=es_metric),
            use_container_width=True,
        )

        # Parallel coordinates
        st.plotly_chart(
            exit_sensitivity_parallel_coords(sens_result),
            use_container_width=True,
        )

        # Data table
        st.subheader("All Results")
        sens_df = pd.DataFrame([p.model_dump() for p in sens_result.points])
        sens_df = sens_df.sort_values("ev", ascending=False)
        st.dataframe(sens_df, use_container_width=True, hide_index=True)
    elif not es_run:
        st.info("Configure sweep ranges above and click **Run Sensitivity Sweep**.")
