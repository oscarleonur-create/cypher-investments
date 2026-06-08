"""Sessions browser — manage every research session."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from advisor.dashboard import state
from advisor.dashboard.persistence import get_store
from advisor.dashboard.theme import apply_page_config

apply_page_config("Sessions")
state.init_state()
active_session = state.get_session_id()

st.title("Research sessions")
st.caption(
    "Every session is a thread of work — the tickers you visited, notes you wrote, "
    "scenarios you saved, layers you refreshed. Reopen one and the Research page "
    "lands back on the ticker you left."
)

store = get_store()
try:
    sessions = store.list_sessions(limit=100)
finally:
    store.close()

if not sessions:
    st.info("No sessions yet. Create one from the Home page or the Research sidebar.")
    st.stop()

# ── Summary table ───────────────────────────────────────────────────────────

rows = []
for s in sessions:
    rows.append(
        {
            "Active": "● " if s["id"] == active_session else "",
            "Pinned": "📌" if s.get("pinned") else "",
            "Name": s["name"],
            "Current ticker": s.get("current_symbol") or "—",
            "Created": (s.get("created_at") or "")[:19],
            "Last opened": (s.get("last_opened_at") or "")[:19],
            "Events": s.get("event_count", 0),
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ── Per-session expanders with reopen / pin / delete / event timeline ───────

for s in sessions:
    sid = s["id"]
    is_active = sid == active_session
    title = f"{'●  ' if is_active else ''}{s['name']}  ·  {(s.get('current_symbol') or '—')}"
    with st.expander(title, expanded=is_active):
        action_cols = st.columns(4)
        if action_cols[0].button(
            "Reopen", key=f"reopen_{sid}", type="primary", use_container_width=True
        ):
            state.set_session_id(sid)
            st.switch_page("pages/1_Research.py")
        pinned = bool(s.get("pinned"))
        if action_cols[1].button(
            "Unpin" if pinned else "Pin",
            key=f"pin_{sid}",
            use_container_width=True,
        ):
            store = get_store()
            try:
                store.set_pinned(sid, not pinned)
            finally:
                store.close()
            st.rerun()
        new_name = action_cols[2].text_input(
            "Rename",
            value=s["name"],
            key=f"rename_input_{sid}",
            label_visibility="collapsed",
            placeholder="Rename…",
        )
        if action_cols[2].button("Save name", key=f"save_name_{sid}", use_container_width=True):
            store = get_store()
            try:
                store.rename_session(sid, new_name)
            finally:
                store.close()
            st.rerun()
        if action_cols[3].button(
            "🗑 Delete",
            key=f"del_{sid}",
            help="Delete the session and all of its events + notes",
            use_container_width=True,
        ):
            store = get_store()
            try:
                store.delete_session(sid)
            finally:
                store.close()
            if is_active:
                st.session_state["session_id"] = None
            st.rerun()

        st.caption(f"Created {s.get('created_at')} · last opened {s.get('last_opened_at')}")

        # Event timeline
        store = get_store()
        try:
            events = store.list_events(sid, limit=50)
        finally:
            store.close()
        if events:
            ev_rows = [
                {
                    "When": e["ts"][:19],
                    "Symbol": e["symbol"],
                    "Event": e["event_type"],
                    "Detail": ", ".join(f"{k}={v}" for k, v in (e["payload"] or {}).items()),
                }
                for e in events
            ]
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("_No events logged for this session yet._")
