"""Ecosystem tab — interactive network graph + relationship tables.

The graph is built with pyvis (self-contained HTML, rendered inside Streamlit
via `components.v1.html`). Because pyvis click events stay inside the iframe,
we expose a *parallel* "Follow a connection" button row in Streamlit-native
widgets — that's the navigation surface. The graph is the picture; the
buttons drive `state.enter_research`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from pyvis.network import Network

from advisor.dashboard import state
from advisor.dashboard.theme import RELATIONSHIP_COLORS
from advisor.research.models import (
    NetworkNode,
    RelationshipType,
    ResearchReport,
)


def render_ecosystem(report: ResearchReport) -> None:
    nodes = _collect_nodes(report)
    if not nodes:
        st.warning("No ecosystem data available for this ticker.")
        return

    _render_graph(report.symbol, nodes)
    st.divider()
    _render_follow_buttons(nodes)
    st.divider()
    _render_holders_table(report)
    _render_insiders_table(report)
    _render_customers_suppliers_tables(report)


# ── Node collection ─────────────────────────────────────────────────────────


def _collect_nodes(report: ResearchReport) -> list[NetworkNode]:
    """Merge `report.network` (if present) with peers + insiders for a richer graph."""
    nodes: list[NetworkNode] = []
    seen: set[tuple[str, str]] = set()  # (symbol or name, relationship)

    def add(node: NetworkNode) -> None:
        key = (node.symbol or node.name, node.relationship.value)
        if key in seen:
            return
        seen.add(key)
        nodes.append(node)

    if report.network and report.network.nodes:
        for n in report.network.nodes:
            add(n)

    if report.multiples:
        for p in report.multiples.peers:
            add(
                NetworkNode(
                    symbol=p.symbol,
                    name=p.name or p.symbol,
                    relationship=RelationshipType.PEER,
                    note=f"Mkt cap {_abbrev(p.market_cap)}" if p.market_cap else "",
                )
            )

    eco = report.ecosystem
    if eco is not None:
        # Insiders are shown in their own table — keeping them out of the graph
        # avoids cluttering it with non-tradeable name nodes.
        for c in eco.top_customers:
            add(
                NetworkNode(
                    symbol="",
                    name=c.name,
                    relationship=RelationshipType.CUSTOMER,
                    note=(
                        f"~{c.concentration_pct:.0f}% of revenue"
                        if c.concentration_pct
                        else (c.note or "")
                    ),
                )
            )
        for s in eco.top_suppliers:
            add(
                NetworkNode(
                    symbol="",
                    name=s.name,
                    relationship=RelationshipType.SUPPLIER,
                    note=s.category or s.note or "",
                )
            )

    return nodes


# ── Graph rendering ─────────────────────────────────────────────────────────


def _render_graph(subject: str, nodes: list[NetworkNode]) -> None:
    st.subheader("Connections")
    net = Network(
        height="540px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        directed=False,
        notebook=False,
        cdn_resources="remote",
    )
    net.barnes_hut(
        gravity=-3500,
        central_gravity=0.25,
        spring_length=140,
        spring_strength=0.04,
        damping=0.9,
    )

    subject_color = RELATIONSHIP_COLORS["subject"]
    net.add_node(
        subject,
        label=subject,
        title=f"Subject: {subject}",
        color=subject_color,
        size=42,
        font={"size": 22, "color": "white"},
        shape="dot",
    )

    relation_label = {
        RelationshipType.PEER: ("peer", "peer"),
        RelationshipType.COMPETITOR: ("competitor", "peer"),
        RelationshipType.TOP_HOLDER: ("holds", "holder"),
        RelationshipType.CUSTOMER: ("buys from", "customer"),
        RelationshipType.SUPPLIER: ("supplies", "supplier"),
    }

    for i, n in enumerate(nodes):
        edge_label, color_key = relation_label.get(n.relationship, ("", "holder"))
        color = RELATIONSHIP_COLORS.get(color_key, "#9ca3af")
        node_id = n.symbol or f"{n.name}#{i}"
        label = n.symbol if n.symbol else _shorten(n.name, 18)
        tooltip = (
            f"<b>{n.name or n.symbol}</b><br>" f"<i>{n.relationship.value}</i><br>" f"{n.note}"
        )
        net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color=color,
            size=22 if n.symbol else 16,
            font={"size": 14},
            shape="dot",
        )
        net.add_edge(
            subject,
            node_id,
            color={"color": color, "opacity": 0.55},
            label=edge_label,
            font={"size": 10, "color": color},
            smooth=False,
        )

    # Render to a temp HTML file and embed
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        net.write_html(fh.name, notebook=False, open_browser=False)
        html = Path(fh.name).read_text(encoding="utf-8")
    st.components.v1.html(html, height=560, scrolling=False)

    legend = " · ".join(
        f"<span style='color:{c}'>● {label}</span>"
        for label, c in [
            ("subject", RELATIONSHIP_COLORS["subject"]),
            ("peer / competitor", RELATIONSHIP_COLORS["peer"]),
            ("holder / insider", RELATIONSHIP_COLORS["holder"]),
            ("customer", RELATIONSHIP_COLORS["customer"]),
            ("supplier", RELATIONSHIP_COLORS["supplier"]),
        ]
    )
    st.markdown(legend, unsafe_allow_html=True)


# ── Follow buttons ──────────────────────────────────────────────────────────


def _render_follow_buttons(nodes: list[NetworkNode]) -> None:
    st.subheader("Follow a connection")
    clickable = [n for n in nodes if n.symbol]
    if not clickable:
        st.caption(
            "No ticker-mapped connections to follow. Customers, suppliers, and "
            "private holders show up in the graph but aren't tradeable tickers."
        )
        return

    st.caption(
        "Clicking switches the active research subject; the previous ticker "
        "lands in History (sidebar) so you can step back."
    )
    # Group by relationship for a tidy layout
    by_rel: dict[str, list[NetworkNode]] = {}
    for n in clickable:
        by_rel.setdefault(n.relationship.value, []).append(n)

    relation_label = {
        "peer": "Peers",
        "competitor": "Competitors",
        "top_holder": "Top holders",
        "customer": "Customers",
        "supplier": "Suppliers",
    }
    for rel, group in by_rel.items():
        st.markdown(f"**{relation_label.get(rel, rel.title())}**")
        cols = st.columns(min(len(group), 5))
        for i, node in enumerate(group):
            with cols[i % len(cols)]:
                label = f"→ {node.symbol}"
                if st.button(
                    label, key=f"follow_{rel}_{node.symbol}_{i}", use_container_width=True
                ):
                    state.enter_research(node.symbol)
                    st.rerun()
                if node.note:
                    st.caption(node.note[:60])


# ── Holder / insider / customer / supplier tables ──────────────────────────


def _render_holders_table(report: ResearchReport) -> None:
    holders = report.ecosystem.holders if report.ecosystem else None
    if holders is None or not holders.top_holders:
        return
    st.subheader("Institutional holders")
    cols = st.columns(2)
    cols[0].metric(
        "% institutional",
        f"{holders.pct_institutional:.1%}" if holders.pct_institutional else "—",
    )
    cols[1].metric(
        "% insider",
        f"{holders.pct_insider:.1%}" if holders.pct_insider else "—",
    )

    rows = []
    for h in holders.top_holders[:15]:
        rows.append(
            {
                "Holder": h.name,
                "Shares": _abbrev(h.shares),
                "% held": f"{h.pct_held:.2%}" if h.pct_held else "—",
                "Value": _abbrev(h.value_usd),
                "Reported": str(h.date_reported) if h.date_reported else "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_insiders_table(report: ResearchReport) -> None:
    insiders = report.ecosystem.insiders if report.ecosystem else None
    if insiders is None or not insiders.transactions:
        return
    st.subheader("Insider activity")
    cols = st.columns(3)
    cols[0].metric(
        f"Net buying ({insiders.lookback_days}d)",
        _money(insiders.net_buying_usd),
        delta=_money(insiders.net_buying_usd) if insiders.net_buying_usd else None,
    )
    cols[1].metric("C-suite buying", "Yes" if insiders.c_suite_buying else "No")
    cols[2].metric("Transactions", len(insiders.transactions))

    rows = []
    for tx in insiders.transactions[:12]:
        rows.append(
            {
                "Date": str(tx.transaction_date) if tx.transaction_date else "—",
                "Insider": tx.insider_name,
                "Title": tx.title or "—",
                "Type": tx.transaction_type,
                "Shares": _abbrev(tx.shares),
                "Value": _money(tx.value_usd),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_customers_suppliers_tables(report: ResearchReport) -> None:
    eco = report.ecosystem
    if eco is None:
        return
    if eco.top_customers:
        st.subheader("Key customers")
        rows = [
            {
                "Customer": c.name,
                "Concentration": f"{c.concentration_pct:.1f}%" if c.concentration_pct else "—",
                "Note": c.note,
            }
            for c in eco.top_customers
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if eco.top_suppliers:
        st.subheader("Key suppliers")
        rows = [
            {
                "Supplier": s.name,
                "Category": s.category,
                "Note": s.note,
            }
            for s in eco.top_suppliers
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _shorten(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _abbrev(v: float | None) -> str:
    if v is None:
        return "—"
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"{v / 1e12:,.2f}T"
    if abs_v >= 1e9:
        return f"{v / 1e9:,.2f}B"
    if abs_v >= 1e6:
        return f"{v / 1e6:,.1f}M"
    if abs_v >= 1e3:
        return f"{v / 1e3:,.1f}K"
    return f"{v:,.0f}"


def _money(v: float | None) -> str:
    if v is None:
        return "—"
    return f"${_abbrev(v)}"
