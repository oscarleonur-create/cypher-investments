"""Options Flow tab — live options chain analytics for the research workstation."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from advisor.dashboard import data as _data
from advisor.dashboard import state as _state
from advisor.dashboard.theme import PALETTE, PLOTLY_TEMPLATE
from advisor.research.models import OptionsFlowResult, ResearchReport


def render_options_flow(report: ResearchReport) -> None:
    flow = report.options_flow

    if flow is None:
        _render_no_data(report)
        return

    _render_summary_metrics(flow)
    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        _render_expiry_volume_chart(flow)
    with col_right:
        _render_pcr_gauge(flow)

    st.divider()
    _render_unusual_activity(flow)
    st.divider()
    _render_refresh_footer(report)


# ── Empty state ───────────────────────────────────────────────────────────────


def _render_no_data(report: ResearchReport) -> None:
    st.info(
        "No options flow data yet. Click **↻ Refresh Options Flow** below to pull "
        "live chain data for this ticker."
    )
    _render_refresh_footer(report)


# ── Summary metrics row ───────────────────────────────────────────────────────


def _render_summary_metrics(flow: OptionsFlowResult) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)

    pcr = flow.put_call_ratio
    pcr_str = f"{pcr:.2f}" if pcr is not None else "—"
    pcr_delta = None
    if pcr is not None:
        if pcr > 1.2:
            pcr_delta = "Bearish"
        elif pcr < 0.7:
            pcr_delta = "Bullish"
        else:
            pcr_delta = "Neutral"

    c1.metric("Put/Call Ratio", pcr_str, delta=pcr_delta)

    total_vol = flow.total_call_volume + flow.total_put_volume
    c2.metric("Total Volume", _fmt_int(total_vol))
    c3.metric("Total OI", _fmt_int(flow.total_call_oi + flow.total_put_oi))

    atm = flow.atm_iv
    c4.metric("ATM IV", f"{atm:.1%}" if atm else "—")

    mp = flow.max_pain_price
    price = flow.current_price
    mp_delta = None
    if mp and price:
        pct = mp / price - 1
        mp_delta = f"{pct:+.1%} vs spot"
    c5.metric("Max Pain", f"${mp:,.2f}" if mp else "—", delta=mp_delta)

    # Sentiment bar
    if flow.total_call_volume > 0 or flow.total_put_volume > 0:
        call_pct = flow.total_call_volume / (flow.total_call_volume + flow.total_put_volume)
        put_pct = 1 - call_pct
        st.caption(
            f"Volume mix: **{call_pct:.0%} calls** · **{put_pct:.0%} puts** "
            f"({_fmt_int(flow.total_call_volume)} / {_fmt_int(flow.total_put_volume)})"
        )
        if flow.nearest_expiry:
            st.caption(f"Nearest expiry: **{flow.nearest_expiry}**")


# ── Expiration volume chart ───────────────────────────────────────────────────


def _render_expiry_volume_chart(flow: OptionsFlowResult) -> None:
    st.subheader("Volume by expiration")
    if not flow.expirations:
        st.caption("No expiration data.")
        return

    exps = [e.expiration for e in flow.expirations]
    call_vols = [e.call_volume for e in flow.expirations]
    put_vols = [e.put_volume for e in flow.expirations]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Calls",
            x=exps,
            y=call_vols,
            marker_color=PALETTE["positive"],
            hovertemplate="%{x}<br>Calls: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Puts",
            x=exps,
            y=[-v for v in put_vols],
            marker_color=PALETTE["negative"],
            hovertemplate="%{x}<br>Puts: %{customdata:,}<extra></extra>",
            customdata=put_vols,
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="relative",
        height=320,
        margin=dict(l=20, r=20, t=30, b=40),
        yaxis_title="Volume (calls +, puts −)",
        xaxis_tickangle=-30,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Put/Call ratio mini-gauge ─────────────────────────────────────────────────


def _render_pcr_gauge(flow: OptionsFlowResult) -> None:
    st.subheader("Open interest split")
    total_oi = flow.total_call_oi + flow.total_put_oi
    if total_oi == 0:
        st.caption("No open interest data.")
        return

    fig = go.Figure(
        go.Pie(
            labels=["Calls OI", "Puts OI"],
            values=[flow.total_call_oi, flow.total_put_oi],
            marker_colors=[PALETTE["positive"], PALETTE["negative"]],
            textinfo="label+percent",
            hole=0.55,
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        annotations=[
            dict(
                text=(
                    f"OI PCR<br>{flow.total_put_oi / flow.total_call_oi:.2f}"
                    if flow.total_call_oi > 0
                    else "OI PCR<br>—"
                ),
                x=0.5,
                y=0.5,
                font_size=14,
                showarrow=False,
            )
        ],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Call OI: {_fmt_int(flow.total_call_oi)} · Put OI: {_fmt_int(flow.total_put_oi)}")


# ── Unusual activity table ────────────────────────────────────────────────────


def _render_unusual_activity(flow: OptionsFlowResult) -> None:
    st.subheader("Unusual activity")
    st.caption(
        "Contracts where today's volume is ≥ 3× the existing open interest — a signal "
        "that new money is aggressively opening positions."
    )

    unusual = flow.unusual_activity
    if not unusual:
        st.info("No unusual options activity detected for this ticker today.")
        return

    import pandas as pd

    rows = []
    for u in unusual:
        sentiment = "Bullish" if u.option_type == "call" else "Bearish"
        rows.append(
            {
                "Contract": u.contract_symbol,
                "Exp": u.expiration,
                "Strike": f"${u.strike:,.2f}",
                "Type": u.option_type.upper(),
                "Vol": f"{u.volume:,}",
                "OI": f"{u.open_interest:,}",
                "Vol/OI": f"{u.volume_oi_ratio:.1f}×",
                "IV": f"{u.implied_volatility:.1%}",
                "Last": f"${u.last_price:.2f}",
                "Premium": _abbrev(u.estimated_premium),
                "Signal": sentiment,
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    total_prem = sum(u.estimated_premium for u in unusual)
    st.caption(
        f"{len(unusual)} unusual contract{'s' if len(unusual) != 1 else ''} · "
        f"Total notional premium: {_abbrev(total_prem)}"
    )


# ── Refresh footer ────────────────────────────────────────────────────────────


def _render_refresh_footer(report: ResearchReport) -> None:
    col1, col2 = st.columns([3, 1])
    col1.caption(
        "Options data fetched live from yfinance. "
        "Volume and open interest are as of the last market close. "
        "IV is mid-market implied volatility from the chain."
    )
    if col2.button(
        "↻ Refresh Options Flow",
        key="inline_refresh_options_flow",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Fetching options chain…"):
            try:
                updated = _data.refresh_layer(report, layer="options_flow")
                _state.set_report(updated)
                _state.log_event(report.symbol, "refresh", {"layer": "options_flow"})
            except Exception as exc:  # noqa: BLE001
                st.error(f"Options flow refresh failed: {exc}")
        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fmt_int(v: int) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return str(v)


def _abbrev(v: float | None) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:,.0f}"
