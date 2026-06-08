"""Interactive thesis builder — multi-turn LLM chat grounded in the ResearchReport."""

from __future__ import annotations

import logging

import streamlit as st

from advisor.research.models import ResearchReport

logger = logging.getLogger(__name__)

_CHAT_KEY = "thesis_chat_messages"
_SUGGESTED_STARTERS = [
    "What's the strongest bull case given the data?",
    "What would invalidate this thesis?",
    "How does the DCF compare to the peer multiples?",
    "Draft a one-paragraph investment thesis summary.",
    "What are the top 3 risks I should be monitoring?",
    "What does the X.com sentiment tell us about market perception?",
]


def render_thesis_chat(report: ResearchReport, session_id: str | None = None) -> None:
    from advisor.dashboard import ai

    st.subheader("Thesis Builder")
    st.caption(
        "Chat with an analyst AI that has read the full research report. "
        "Ask it to stress-test your thesis, draft sections, or explain the data."
    )

    if not ai.is_configured():
        st.warning(
            "Set `RESEARCH_AGENT_OPENROUTER_API_KEY` in your environment "
            "to enable the Thesis Builder."
        )
        return

    # Per-ticker message history in session state
    msg_key = f"{_CHAT_KEY}_{report.symbol}"
    if msg_key not in st.session_state:
        st.session_state[msg_key] = []

    messages: list[dict] = st.session_state[msg_key]

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_cols = st.columns([1, 5])
    if ctrl_cols[0].button("🗑 Reset chat", key=f"reset_chat_{report.symbol}"):
        st.session_state[msg_key] = []
        st.rerun()

    # ── Starter suggestions (only when chat is empty) ─────────────────────────
    if not messages:
        st.markdown("**Suggested questions to start:**")
        starter_cols = st.columns(2)
        for i, suggestion in enumerate(_SUGGESTED_STARTERS):
            col = starter_cols[i % 2]
            if col.button(suggestion, key=f"starter_{report.symbol}_{i}"):
                _send_message(report, messages, msg_key, suggestion, session_id)
                st.rerun()
        st.divider()

    # ── Conversation history ──────────────────────────────────────────────────
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Input box ─────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        f"Ask about the {report.symbol} thesis…",
        key=f"chat_input_{report.symbol}",
    )
    if user_input and user_input.strip():
        _send_message(report, messages, msg_key, user_input.strip(), session_id)
        st.rerun()


# ── Internal helpers ──────────────────────────────────────────────────────────


def _send_message(
    report: ResearchReport,
    messages: list[dict],
    msg_key: str,
    user_text: str,
    session_id: str | None,
) -> None:
    from research_agent.config import ResearchConfig
    from research_agent.llm import OpenRouterLLM

    from advisor.dashboard import ai, state

    messages.append({"role": "user", "content": user_text})

    with st.chat_message("user"):
        st.markdown(user_text)

    system_prompt = ai.build_chat_context(report)

    try:
        config = ResearchConfig()
        llm = OpenRouterLLM(config)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = llm.chat(system_prompt=system_prompt, messages=messages)
            st.markdown(reply)

        messages.append({"role": "assistant", "content": reply})
        st.session_state[msg_key] = messages

        if session_id:
            state.log_event(
                report.symbol,
                "thesis_chat",
                {"role": "user", "content": user_text},
            )
            state.log_event(
                report.symbol,
                "thesis_chat",
                {"role": "assistant", "content": reply},
            )

    except Exception as exc:
        logger.warning("Thesis chat failed: %s", exc)
        error_msg = f"Sorry, the AI call failed: {exc}"
        messages.append({"role": "assistant", "content": error_msg})
        st.session_state[msg_key] = messages
        with st.chat_message("assistant"):
            st.error(error_msg)
