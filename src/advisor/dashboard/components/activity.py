"""Activity tab — chronological log of session events for the current ticker."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from advisor.dashboard.persistence import get_store

_EVENT_ICONS = {
    "view": "👁️",
    "follow": "→",
    "refresh": "↻",
    "note_added": "📝",
    "scenario_saved": "💾",
}


def render_activity(session_id: str, symbol: str) -> None:
    st.subheader(f"Activity — {symbol}")
    st.caption(
        "Everything you did in this session for this ticker. Comes from the "
        "`research_session_events` table; persists across browser restarts."
    )

    store = get_store()
    try:
        events_symbol = store.list_events(session_id, symbol, limit=200)
        events_all = store.list_events(session_id, limit=200)
    finally:
        store.close()

    cols = st.columns(3)
    cols[0].metric("This-ticker events", len(events_symbol))
    cols[1].metric("All-ticker events", len(events_all))
    tickers = sorted({e["symbol"] for e in events_all})
    cols[2].metric("Tickers visited", len(tickers))

    if not events_symbol:
        st.info(
            "No events logged for this ticker yet. Open it, refresh layers, save a "
            "scenario, or write a note — they'll all show up here."
        )
        return

    rows = []
    for e in events_symbol:
        icon = _EVENT_ICONS.get(e["event_type"], "•")
        payload = e["payload"]
        summary = ", ".join(f"{k}={v}" for k, v in payload.items()) if payload else ""
        rows.append(
            {
                "When": e["ts"][:19],
                "Event": f"{icon} {e['event_type']}",
                "Detail": summary,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
