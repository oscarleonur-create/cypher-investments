"""Advisor Dashboard — home page.

Multi-page Streamlit shell. Pages auto-discovered from `pages/` are listed in
the left sidebar. Launch with `advisor dashboard ui`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from advisor.dashboard import data, state
from advisor.dashboard.persistence import get_store
from advisor.dashboard.theme import apply_page_config

apply_page_config("Advisor Dashboard", icon=":bar_chart:")
state.init_state()
session_id = state.get_session_id()

st.title("Advisor Dashboard")
st.markdown(
    "A research workstation built on the advisor's fundamental research engine. "
    "Pick a ticker, follow the connections, save your thinking — sessions persist "
    "across browser restarts."
)

st.markdown("---")

# ── Sessions ────────────────────────────────────────────────────────────────

st.subheader("Sessions")
_store = get_store()
try:
    _sessions = _store.list_sessions(limit=10)
finally:
    _store.close()

cols = st.columns([1, 1, 4])
if cols[0].button("➕ New session", type="primary"):
    _store = get_store()
    try:
        _new_id = _store.new_session()
    finally:
        _store.close()
    state.set_session_id(_new_id)
    st.switch_page("pages/1_Research.py")
if cols[1].button("Browse all"):
    st.switch_page("pages/2_Sessions.py")

if _sessions:
    rows = []
    for s in _sessions:
        active = "● " if s["id"] == session_id else ""
        rows.append(
            {
                "Active": active,
                "Name": s["name"],
                "Current ticker": s.get("current_symbol") or "—",
                "Last opened": (s.get("last_opened_at") or "")[:19],
                "Events": s.get("event_count", 0),
                "Pinned": "📌" if s.get("pinned") else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Switch the active session via the sidebar selectbox on the Research page, "
        "or hit **Browse all** to manage them."
    )
else:
    st.caption("No sessions yet. Click **New session** to get started.")

st.markdown("---")

# ── Pages ───────────────────────────────────────────────────────────────────

st.subheader("Pages")
st.markdown(
    "- **Research** — single-ticker deep fundamentals: statements, ratios, valuation, "
    "ecosystem, catalysts, notes. The workstation lives here.\n"
    "- **Sessions** — browse every research session, see its event timeline, reopen "
    "or delete."
)

st.markdown("---")

# ── Recent cached reports ───────────────────────────────────────────────────

st.subheader("Recent cached reports")
rows = data.list_cached_reports(limit=20)
if not rows:
    st.caption(
        "No cached reports yet. Open the Research page from the sidebar and run "
        "a ticker — saved reports land here."
    )
else:
    df = pd.DataFrame(rows)
    df = df.rename(columns={"symbol": "Symbol", "as_of": "As of", "created_at": "Saved at"})
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Open the Research page → enter a symbol → cached reports load instantly.")

st.markdown("---")
st.caption(
    "Streamlit dashboard · `advisor dashboard ui --port 8502` · "
    "Reports + sessions + scenarios + notes persist to `data/research.db`."
)
