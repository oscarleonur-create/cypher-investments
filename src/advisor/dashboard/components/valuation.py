"""Valuation tab — DCF fan, sensitivity heatmap, peer comp, multiples history.

Read-only in step 2. What-if sliders and "Save scenario" come in step 4 (which
also adds a pure `compute_dcf_scenario` helper to `valuation/dcf.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from advisor.dashboard.persistence import get_store
from advisor.dashboard.theme import PALETTE, PLOTLY_TEMPLATE, upside_color
from advisor.research.models import (
    DcfAssumptions,
    DcfResult,
    MultiplesResult,
    PeerSnapshot,
    ResearchReport,
)
from advisor.research.valuation.dcf import compute_dcf_scenario, dcf_inputs_from_report


def render_valuation(report: ResearchReport) -> None:
    if report.dcf is None and report.multiples is None:
        st.warning("No valuation data available.")
        return

    if report.dcf is not None:
        _render_dcf_fan(report.dcf)
        st.divider()
        _render_consensus_vs_dcf(report)
        st.divider()
        _render_whatif(report)
        st.divider()
        _render_reverse_dcf_gauge(report)
        st.divider()
        _render_sensitivity_heatmap(report.dcf)
        st.divider()

    if report.multiples is not None:
        _render_peer_comp(report.multiples)
        st.divider()
        _render_own_history(report.multiples)


# ── DCF fan chart ───────────────────────────────────────────────────────────


def _render_dcf_fan(dcf: DcfResult) -> None:
    st.subheader("DCF — implied price by scenario")
    scenarios = [
        ("Bear", dcf.bear, PALETTE["negative"]),
        ("Base", dcf.base, PALETTE["accent"]),
        ("Bull", dcf.bull, PALETTE["positive"]),
    ]
    rows = [(label, sc, color) for label, sc, color in scenarios if sc is not None]
    if not rows:
        st.info("No DCF scenarios populated.")
        return

    labels = [label for label, _, _ in rows]
    prices = [sc.implied_price for _, sc, _ in rows]
    upsides = [sc.upside_pct for _, sc, _ in rows]
    colors = [color for _, _, color in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=prices,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"${p:,.2f} ({u:+.1%})" for p, u in zip(prices, upsides)],
            textposition="outside",
            hovertemplate="%{y}: $%{x:,.2f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=dcf.current_price,
        line_dash="dash",
        line_color=PALETTE["neutral"],
        annotation_text=f"Current ${dcf.current_price:,.2f}",
        annotation_position="top",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=260,
        margin=dict(l=20, r=80, t=30, b=20),
        xaxis_title="Implied price ($)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(4)
    cols[0].metric("WACC", f"{dcf.wacc:.1%}")
    cols[1].metric("Risk-free rate", f"{dcf.risk_free_rate:.1%}")
    cols[2].metric("Beta", f"{dcf.beta:.2f}" if dcf.beta is not None else "—")
    cols[3].metric("Net debt", _money(dcf.net_debt))


# ── Consensus vs. DCF comparison ────────────────────────────────────────────


def _render_consensus_vs_dcf(report: ResearchReport) -> None:
    cn = report.consensus
    dcf = report.dcf
    if cn is None and (dcf is None or dcf.base is None):
        return

    st.subheader("Consensus vs. our DCF")

    current = (cn.current_price if cn else None) or (dcf.current_price if dcf else None)
    cons_target = cn.target_price_mean if cn else None
    our_base = dcf.base.implied_price if dcf and dcf.base else None

    cols = st.columns(4)

    if cons_target and current:
        cons_upside = (cons_target / current) - 1
        cols[0].metric(
            "Consensus target",
            f"${cons_target:,.2f}",
            delta=f"{cons_upside:+.1%}",
            delta_color="normal",
        )
    else:
        mean = cn.target_price_mean if cn else None
        cols[0].metric("Consensus target", f"${mean:,.2f}" if mean else "—")

    if our_base and current:
        our_upside = (our_base / current) - 1
        cols[1].metric(
            "Our DCF (base)",
            f"${our_base:,.2f}",
            delta=f"{our_upside:+.1%}",
            delta_color="normal",
        )
    else:
        cols[1].metric("Our DCF (base)", our_base and f"${our_base:,.2f}" or "—")

    if cn:
        cols[2].metric(
            "Analysts",
            str(cn.n_analysts) if cn.n_analysts else "—",
            delta=cn.recommendation_key.capitalize() if cn.recommendation_key else None,
            delta_color="off",
        )
        cols[3].metric(
            "Estimate revisions (30d)",
            cn.revision_trend.capitalize(),
            delta=f"↑{cn.eps_revisions_up_30d} / ↓{cn.eps_revisions_down_30d}",
            delta_color="off",
        )
    else:
        cols[2].metric("Analysts", "—")
        cols[3].metric("Estimate revisions", "—")

    # Short variant perception insight if available
    vp = report.variant_perception
    if vp and vp.our_key_insight:
        st.caption(f"**Edge:** {vp.our_key_insight} — see the **Edge** tab for full analysis.")
    elif cn is None:
        st.caption(
            "Run edge analysis to fetch analyst consensus. See sidebar → Refresh layers → Edge."
        )


# ── What-if scenario builder ────────────────────────────────────────────────


def _render_whatif(report: ResearchReport) -> None:
    st.subheader("What-if scenario builder")
    inputs = dcf_inputs_from_report(report)
    if inputs is None or report.dcf is None or report.dcf.base is None:
        st.info("DCF inputs aren't available — cannot run a what-if scenario.")
        return

    base_assump = report.dcf.base.assumptions

    # Show a warning if the base inputs weren't persisted (older cached reports)
    if report.dcf.base_revenue is None or report.dcf.seed_fcf is None:
        st.caption(
            f":warning: This report was built before the DCF inputs were persisted, "
            f"so the slider replay uses statement-derived values "
            f"(base revenue ${inputs.base_revenue:,.0f}, seed FCF ${inputs.seed_fcf:,.0f}). "
            f"Force-refresh the report to re-anchor exactly to the headline base."
        )

    # ── Sliders ─────────────────────────────────────────────────────────────
    with st.expander("Assumptions", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            g13 = st.slider(
                "Revenue growth Yr 1–3",
                min_value=-0.10,
                max_value=0.50,
                value=float(base_assump.revenue_growth_yr1_3),
                step=0.005,
                format="%.1f%%",
                key="whatif_g13",
                help="CAGR for the next three years.",
            )
            g410 = st.slider(
                "Revenue growth Yr 4–10",
                min_value=-0.05,
                max_value=0.30,
                value=float(base_assump.revenue_growth_yr4_10),
                step=0.005,
                format="%.1f%%",
                key="whatif_g410",
                help="Fade growth out to terminal.",
            )
            target_margin = st.slider(
                "Target FCF margin",
                min_value=0.0,
                max_value=0.60,
                value=float(base_assump.target_fcf_margin),
                step=0.005,
                format="%.1f%%",
                key="whatif_margin",
            )
            capex = st.slider(
                "Capex intensity",
                min_value=0.0,
                max_value=0.30,
                value=float(base_assump.capex_intensity),
                step=0.005,
                format="%.1f%%",
                key="whatif_capex",
            )
        with c2:
            wacc = st.slider(
                "WACC (discount rate)",
                min_value=0.04,
                max_value=0.18,
                value=float(base_assump.wacc),
                step=0.001,
                format="%.2f%%",
                key="whatif_wacc",
            )
            g_term = st.slider(
                "Terminal growth (g)",
                min_value=0.0,
                max_value=0.05,
                value=float(base_assump.terminal_growth_rate),
                step=0.0025,
                format="%.2f%%",
                key="whatif_gterm",
            )
            use_exit = st.checkbox(
                "Use exit multiple instead of Gordon terminal",
                value=base_assump.terminal_exit_multiple is not None,
                key="whatif_use_exit",
            )
            exit_default = (
                float(base_assump.terminal_exit_multiple)
                if base_assump.terminal_exit_multiple
                else 12.0
            )
            exit_mult = st.slider(
                "Exit EV/EBITDA multiple",
                min_value=5.0,
                max_value=40.0,
                value=exit_default,
                step=0.5,
                key="whatif_exit_mult",
                disabled=not use_exit,
            )

    # ── Live recompute ──────────────────────────────────────────────────────
    custom_assump = DcfAssumptions(
        scenario="what-if",
        revenue_growth_yr1_3=g13,
        revenue_growth_yr4_10=g410,
        target_fcf_margin=target_margin,
        capex_intensity=capex,
        terminal_growth_rate=g_term,
        terminal_exit_multiple=exit_mult if use_exit else None,
        wacc=wacc,
    )

    try:
        whatif = compute_dcf_scenario(
            custom_assump,
            inputs.base_revenue,
            inputs.seed_fcf,
            inputs.net_debt,
            inputs.shares,
            inputs.current_price,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Recompute failed: {exc}")
        return

    # ── Metric tiles ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Implied price", f"${whatif.implied_price:,.2f}")
    m2.metric("Upside vs current", f"{whatif.upside_pct:+.1%}")
    m3.metric("Equity value", _abbrev(whatif.equity_value))
    base_price = report.dcf.base.implied_price
    delta_str = f"{(whatif.implied_price / base_price - 1):+.1%}" if base_price else "n/a"
    m4.metric(
        "vs base",
        f"${whatif.implied_price - base_price:+,.2f}",
        delta=delta_str if delta_str != "n/a" else None,
    )

    # ── Compare chart: base vs. what-if vs. current ──────────────────────────
    _render_whatif_compare(report.dcf, whatif)

    # ── Save scenario ────────────────────────────────────────────────────────
    save_col1, save_col2 = st.columns([3, 1])
    with save_col1:
        name = st.text_input(
            "Scenario name",
            value="",
            placeholder="e.g. 'bull-case-AI', 'recession-stress'",
            key="whatif_name",
            label_visibility="collapsed",
        )
    with save_col2:
        if st.button("💾 Save scenario", use_container_width=True, type="primary"):
            if not name.strip():
                st.warning("Give the scenario a name first.")
            else:
                from advisor.dashboard import state as _state

                store = get_store()
                try:
                    store.save_scenario(report.symbol, name.strip(), custom_assump, whatif)
                    st.success(f"Saved '{name.strip()}'.")
                finally:
                    store.close()
                _state.log_event(
                    report.symbol,
                    "scenario_saved",
                    {
                        "name": name.strip(),
                        "implied_price": whatif.implied_price,
                        "upside_pct": whatif.upside_pct,
                    },
                )
                st.rerun()

    # ── Saved scenarios list ─────────────────────────────────────────────────
    store = get_store()
    try:
        saved = store.list_scenarios(report.symbol)
    finally:
        store.close()
    if saved:
        st.markdown("**Saved scenarios**")
        for sc in saved:
            cols = st.columns([3, 2, 2, 1, 1])
            cols[0].markdown(f"`{sc['name']}`")
            cols[1].caption(f"Implied: ${sc['result'].implied_price:,.2f}")
            cols[2].caption(f"Upside: {sc['result'].upside_pct:+.1%}")
            if cols[3].button("Load", key=f"load_{sc['id']}"):
                a = sc["assumptions"]
                st.session_state["whatif_g13"] = float(a.revenue_growth_yr1_3)
                st.session_state["whatif_g410"] = float(a.revenue_growth_yr4_10)
                st.session_state["whatif_margin"] = float(a.target_fcf_margin)
                st.session_state["whatif_capex"] = float(a.capex_intensity)
                st.session_state["whatif_wacc"] = float(a.wacc)
                st.session_state["whatif_gterm"] = float(a.terminal_growth_rate)
                st.session_state["whatif_use_exit"] = a.terminal_exit_multiple is not None
                if a.terminal_exit_multiple is not None:
                    st.session_state["whatif_exit_mult"] = float(a.terminal_exit_multiple)
                st.rerun()
            if cols[4].button("✕", key=f"del_{sc['id']}", help="Delete"):
                store = get_store()
                try:
                    store.delete_scenario(sc["id"])
                finally:
                    store.close()
                st.rerun()


def _render_whatif_compare(dcf: DcfResult, whatif) -> None:  # type: ignore[no-untyped-def]
    fig = go.Figure()
    rows: list[tuple[str, float, str]] = []
    if dcf.bear is not None:
        rows.append(("Bear", dcf.bear.implied_price, PALETTE["negative"]))
    if dcf.base is not None:
        rows.append(("Base", dcf.base.implied_price, PALETTE["accent"]))
    if dcf.bull is not None:
        rows.append(("Bull", dcf.bull.implied_price, PALETTE["positive"]))
    rows.append(("What-if", whatif.implied_price, "#a78bfa"))

    fig.add_trace(
        go.Bar(
            x=[v for _, v, _ in rows],
            y=[label for label, _, _ in rows],
            orientation="h",
            marker=dict(color=[c for _, _, c in rows]),
            text=[f"${v:,.2f}" for _, v, _ in rows],
            textposition="outside",
            hovertemplate="%{y}: $%{x:,.2f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=dcf.current_price,
        line_dash="dash",
        line_color=PALETTE["neutral"],
        annotation_text=f"Current ${dcf.current_price:,.2f}",
        annotation_position="top",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=260,
        margin=dict(l=20, r=80, t=30, b=20),
        xaxis_title="Implied price ($)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Reverse-DCF gauge ────────────────────────────────────────────────────────


def _render_reverse_dcf_gauge(report: ResearchReport) -> None:
    dcf = report.dcf
    if dcf is None or dcf.implied_growth_rate is None:
        return

    hist_cagr = _revenue_cagr(report)
    implied = dcf.implied_growth_rate

    st.subheader("Reverse-DCF — what's priced in")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=implied * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                delta={
                    "reference": (hist_cagr or 0) * 100,
                    "suffix": "% vs hist",
                    "valueformat": ".1f",
                },
                gauge={
                    "axis": {"range": [-5, 30], "tickformat": ".0f"},
                    "bar": {"color": PALETTE["accent"]},
                    "steps": [
                        {"range": [-5, 5], "color": "#1f2937"},
                        {"range": [5, 15], "color": "#374151"},
                        {"range": [15, 30], "color": "#4b5563"},
                    ],
                    "threshold": {
                        "line": {"color": PALETTE["warn"], "width": 3},
                        "thickness": 0.75,
                        "value": (hist_cagr or 0) * 100,
                    },
                },
                title={"text": "Implied revenue growth (%)"},
            )
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Implied growth (reverse-DCF)", f"{implied:.1%}")
        st.metric(
            "Historical revenue CAGR",
            f"{hist_cagr:.1%}" if hist_cagr is not None else "—",
        )
        gap = (implied - hist_cagr) if hist_cagr is not None else None
        if gap is not None:
            color = upside_color(gap)
            neutral = PALETTE["neutral"]
            st.markdown(
                f"<span style='color:{color};font-weight:600'>Gap: {gap:+.1%}</span> "
                f"<span style='color:{neutral}'>(market expectations vs. history)</span>",
                unsafe_allow_html=True,
            )


# ── Sensitivity heatmap (WACC × terminal growth) ─────────────────────────────


def _render_sensitivity_heatmap(dcf: DcfResult) -> None:
    base = dcf.base
    if base is None or not base.projected_fcf:
        return

    # Reproduce the DCF builder's logic exactly: if an exit multiple is set in
    # the base assumptions, the model uses that (Gordon ignored). So we vary
    # WACC × exit_multiple in that case; otherwise WACC × terminal_g (Gordon).
    using_exit = base.assumptions.terminal_exit_multiple is not None
    title = (
        "Sensitivity — implied price by WACC × terminal exit multiple"
        if using_exit
        else "Sensitivity — implied price by WACC × terminal growth (g)"
    )
    st.subheader(title)

    fcf = list(base.projected_fcf)
    n = len(fcf)
    base_wacc = dcf.wacc
    shares = dcf.shares_outstanding
    net_debt = dcf.net_debt
    current_price = dcf.current_price

    wacc_grid = np.linspace(max(0.04, base_wacc - 0.03), base_wacc + 0.03, 7)

    if using_exit:
        base_em = base.assumptions.terminal_exit_multiple
        target_margin = max(base.assumptions.target_fcf_margin, 0.01)
        em_grid = np.linspace(max(5.0, base_em - 8.0), base_em + 8.0, 7)
        y_grid = em_grid
        y_labels = [f"{em:.1f}×" for em in em_grid]
        y_axis_title = "Exit multiple (EV / EBITDA)"
    else:
        base_g = base.assumptions.terminal_growth_rate
        y_grid = np.linspace(max(0.0, base_g - 0.02), base_g + 0.02, 7)
        y_labels = [f"{g * 100:.1f}%" for g in y_grid]
        y_axis_title = "Terminal growth (g)"

    grid = np.zeros((len(y_grid), len(wacc_grid)))
    upside_grid = np.zeros_like(grid)
    text = np.full(grid.shape, "", dtype=object)

    for i, yv in enumerate(y_grid):
        for j, w in enumerate(wacc_grid):
            if using_exit:
                terminal_ebitda_proxy = fcf[-1] / target_margin * 0.25
                tv = terminal_ebitda_proxy * yv
            else:
                if w <= yv:
                    grid[i, j] = np.nan
                    upside_grid[i, j] = np.nan
                    text[i, j] = "n/a"
                    continue
                tv = fcf[-1] * (1 + yv) / (w - yv)

            pv_fcf = sum(f / (1 + w) ** (t + 1) for t, f in enumerate(fcf))
            pv_terminal = tv / (1 + w) ** n
            ev = pv_fcf + pv_terminal
            equity = max(ev - net_debt, 0)
            implied = equity / shares if shares else float("nan")
            grid[i, j] = implied
            upside_grid[i, j] = implied / current_price - 1
            text[i, j] = f"${implied:,.0f}"

    fig = go.Figure(
        data=go.Heatmap(
            z=upside_grid * 100,
            x=[f"{w * 100:.1f}%" for w in wacc_grid],
            y=y_labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=[
                [0.0, PALETTE["negative"]],
                [0.5, "#1f2937"],
                [1.0, PALETTE["positive"]],
            ],
            zmid=0,
            colorbar=dict(title="Upside %", tickformat=".0f"),
            hovertemplate="WACC %{x}, " + y_axis_title + " %{y} → %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="WACC",
        yaxis_title=y_axis_title,
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    method = "exit-multiple terminal value" if using_exit else "Gordon terminal value"
    st.caption(
        f"Replays the base scenario's projection through the {method} used by the DCF "
        f"builder (net debt ${net_debt:,.0f}, shares {shares:,.0f}). "
        f"Full what-if sliders arrive in step 4."
    )


# ── Peer comp table ──────────────────────────────────────────────────────────


def _render_peer_comp(multiples: MultiplesResult) -> None:
    st.subheader("Peer comp")
    snaps = multiples.all_snapshots
    if len(snaps) < 2:
        st.caption("No peers to compare against.")
        if multiples.subject:
            df = pd.DataFrame([_snap_row(multiples.subject, is_subject=True)])
            st.dataframe(df, use_container_width=True, hide_index=True)
        return

    rows = [
        _snap_row(snaps[0], is_subject=True),
        *(_snap_row(p, is_subject=False) for p in snaps[1:]),
    ]
    df = pd.DataFrame(rows)
    # Conditional formatting on multiples (cheap = green, expensive = red).
    multiple_cols = ["P/E", "Fwd P/E", "EV/EBITDA", "EV/Sales", "P/FCF", "PEG"]
    styler = df.style.apply(
        lambda col: _color_multiples(col, lower_is_cheap=True)
        if col.name in multiple_cols
        else [""] * len(col),
        axis=0,
    )
    # Bold the subject row
    styler = styler.apply(
        lambda row: ["font-weight: bold" if i == 0 else "" for i in range(len(row))],
        axis=1,
    )
    st.dataframe(styler, use_container_width=True, hide_index=True)


def _snap_row(snap: PeerSnapshot, is_subject: bool) -> dict:
    return {
        "Ticker": ("★ " if is_subject else "") + snap.symbol,
        "Name": snap.name[:30],
        "Mkt cap": _abbrev(snap.market_cap),
        "EV": _abbrev(snap.enterprise_value),
        "Rev grw": _pct(snap.revenue_growth_yoy),
        "Gr mgn": _pct(snap.gross_margin),
        "Op mgn": _pct(snap.operating_margin),
        "P/E": _ratio(snap.pe_trailing),
        "Fwd P/E": _ratio(snap.pe_forward),
        "EV/EBITDA": _ratio(snap.ev_to_ebitda),
        "EV/Sales": _ratio(snap.ev_to_sales),
        "P/FCF": _ratio(snap.p_to_fcf),
        "PEG": _ratio(snap.peg),
        "Beta": _ratio(snap.beta),
    }


def _color_multiples(col: pd.Series, lower_is_cheap: bool = True) -> list[str]:
    """Color a column of multiples (string form '12.3×') with green = cheap, red = expensive."""
    nums = []
    for v in col:
        try:
            nums.append(float(str(v).replace("×", "").replace("—", "nan")))
        except (ValueError, TypeError):
            nums.append(float("nan"))
    arr = np.array(nums)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return [""] * len(col)
    median = np.median(finite)
    spread = max(np.std(finite), 1e-6)
    styles = []
    for v in arr:
        if not np.isfinite(v):
            styles.append("")
            continue
        z = (v - median) / spread
        if not lower_is_cheap:
            z = -z
        # z<0 → cheap (green); z>0 → expensive (red)
        if z < -0.5:
            styles.append(
                f"background-color: {PALETTE['positive']}22; color: {PALETTE['positive']}"
            )
        elif z > 0.5:
            styles.append(
                f"background-color: {PALETTE['negative']}22; color: {PALETTE['negative']}"
            )
        else:
            styles.append("")
    return styles


# ── Own multiples history ────────────────────────────────────────────────────


def _render_own_history(multiples: MultiplesResult) -> None:
    st.subheader(f"{multiples.subject.symbol} — own multiples history")
    hist = multiples.own_history
    if not hist:
        st.caption("No historical multiples on file.")
        return

    years = [str(p.fiscal_year) for p in hist]
    series = [
        ("pe", "P/E", PALETTE["accent"]),
        ("ev_to_ebitda", "EV/EBITDA", PALETTE["positive"]),
        ("ev_to_sales", "EV/Sales", PALETTE["warn"]),
        ("p_to_fcf", "P/FCF", "#a78bfa"),
    ]
    fig = go.Figure()
    for attr, label, color in series:
        y = [getattr(p, attr, None) for p in hist]
        if not any(v is not None for v in y):
            continue
        fig.add_trace(
            go.Scatter(
                x=years,
                y=y,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
            )
        )
        # Mean line (dashed)
        vals = [v for v in y if v is not None]
        if vals:
            mean = float(np.mean(vals))
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=[mean] * len(years),
                    mode="lines",
                    name=f"{label} mean",
                    line=dict(color=color, dash="dash", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
        yaxis_title="Multiple (×)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _revenue_cagr(report: ResearchReport) -> float | None:
    if report.statements is None or not report.statements.income:
        return None
    series = [p.revenue for p in report.statements.income if p.revenue is not None]
    if len(series) < 2:
        return None
    latest, oldest = series[0], series[-1]
    years = len(series) - 1
    if oldest is None or oldest <= 0 or years <= 0:
        return None
    try:
        return (latest / oldest) ** (1.0 / years) - 1.0
    except (ZeroDivisionError, ValueError):
        return None


def _money(v: float | None) -> str:
    if v is None:
        return "—"
    return f"${v:,.0f}"


def _abbrev(v: float | None) -> str:
    if v is None:
        return "—"
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"${v / 1e12:,.2f}T"
    if abs_v >= 1e9:
        return f"${v / 1e9:,.2f}B"
    if abs_v >= 1e6:
        return f"${v / 1e6:,.1f}M"
    return f"${v:,.0f}"


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.1f}%"


def _ratio(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}×"
