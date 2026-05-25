"""Sticky header card for the Research page."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from advisor.research.models import ResearchReport


def render_header(report: ResearchReport, last_run_at: datetime | None = None) -> None:
    """Render symbol + company info + market metrics row."""
    snap = report.multiples.subject if report.multiples and report.multiples.subject else None
    company_name = (snap.name if snap and snap.name else report.business_model) or report.symbol
    sector = snap.sector if snap else ""
    industry = snap.industry if snap else ""

    title_col, refresh_col = st.columns([4, 1])
    with title_col:
        sub = " · ".join([s for s in (sector, industry) if s])
        st.markdown(f"### {report.symbol} — {company_name}" + (f"  \n*{sub}*" if sub else ""))
    with refresh_col:
        st.caption(f"As of **{report.as_of.isoformat()}**")
        if last_run_at is not None:
            st.caption(f"Last refresh: {last_run_at.strftime('%H:%M:%S')}")

    cols = st.columns(5)
    cols[0].metric("Price", _money(snap.price if snap else None))
    cols[1].metric("Market cap", _abbrev(snap.market_cap if snap else None))
    cols[2].metric("EV", _abbrev(snap.enterprise_value if snap else None))
    cols[3].metric("P/E (TTM)", _ratio(snap.pe_trailing if snap else None))
    cols[4].metric("EV/EBITDA", _ratio(snap.ev_to_ebitda if snap else None))


def _money(v: float | None) -> str:
    if v is None:
        return "—"
    return f"${v:,.2f}"


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
    if abs_v >= 1e3:
        return f"${v / 1e3:,.0f}K"
    return f"${v:,.0f}"


def _ratio(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}×"
