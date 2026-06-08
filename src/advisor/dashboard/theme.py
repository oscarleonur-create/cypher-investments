"""Shared theme + page-config helpers for the research workstation."""

from __future__ import annotations

import streamlit as st

PLOTLY_TEMPLATE = "plotly_dark"

PALETTE = {
    "positive": "#19c37d",
    "negative": "#ef4444",
    "neutral": "#9ca3af",
    "accent": "#60a5fa",
    "warn": "#f59e0b",
}

SEVERITY_COLORS = {
    "LOW": "#9ca3af",
    "MEDIUM": "#f59e0b",
    "HIGH": "#ef4444",
}

RELATIONSHIP_COLORS = {
    "subject": "#60a5fa",
    "peer": "#19c37d",
    "customer": "#f59e0b",
    "supplier": "#ef4444",
    "holder": "#a78bfa",
    "insider": "#ec4899",
}


def apply_page_config(title: str, icon: str = ":bar_chart:") -> None:
    """Apply the dashboard's default Streamlit page config (idempotent)."""
    try:
        st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    except st.errors.StreamlitAPIException:
        # Already set on this run (e.g., re-entered via st.switch_page)
        pass


def upside_color(pct: float | None) -> str:
    if pct is None:
        return PALETTE["neutral"]
    if pct > 0.05:
        return PALETTE["positive"]
    if pct < -0.05:
        return PALETTE["negative"]
    return PALETTE["warn"]


def severity_chip(severity: str, label: str) -> str:
    """Render a colored badge as inline HTML for st.markdown(unsafe_allow_html=True)."""
    color = SEVERITY_COLORS.get(severity, PALETTE["neutral"])
    return (
        f'<span style="background:{color}22;color:{color};'
        f"padding:2px 8px;border-radius:6px;font-size:0.85em;"
        f'border:1px solid {color}55;margin-right:6px;">{label}</span>'
    )
