"""Tests for the interactive research agent: tool dispatch, loop, conversations."""

from __future__ import annotations

import json
from types import SimpleNamespace

from advisor.agent.loop import build_system_prompt, run_agent
from advisor.agent.tools import ToolContext, dispatch, tool_specs
from advisor.research.store import ResearchStore
from research_agent.config import ResearchConfig

# ── Tool registry / dispatch ──────────────────────────────────────────────────


def test_tool_specs_are_well_formed():
    specs = tool_specs()
    names = {s["function"]["name"] for s in specs}
    # Every category the user asked for is represented.
    assert {
        "get_report_section",
        "web_search",
        "compute_dcf",
        "recompute_bayesian",
        "get_holdings",
        "rebuild_report",
    } <= names
    for s in specs:
        assert s["type"] == "function"
        assert "parameters" in s["function"]


def test_dispatch_unknown_tool_returns_error():
    ctx = ToolContext(symbol="TEST", config=ResearchConfig())
    out = dispatch("nope", {}, ctx)
    assert "error" in out


def test_dispatch_get_report_section_invalid_section():
    ctx = ToolContext(symbol="TEST", config=ResearchConfig())
    out = dispatch("get_report_section", {"section": "bogus"}, ctx)
    assert "error" in out


def test_dispatch_web_search_without_key_is_graceful():
    cfg = ResearchConfig(tavily_api_key="")
    ctx = ToolContext(symbol="TEST", config=cfg)
    out = dispatch("web_search", {"query": "anything"}, ctx)
    assert "error" in out  # no key → graceful error, not a crash


# ── Agent loop (mocked LLM) ───────────────────────────────────────────────────


class _FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


class _FakeLLM:
    """Returns one tool call, then a final answer."""

    configured = True

    def __init__(self):
        self.calls = 0

    def chat_with_tools(self, messages, tools, temperature=0.3):
        self.calls += 1
        if self.calls == 1:
            return _msg(tool_calls=[_FakeToolCall("get_report_section", '{"section": "thesis"}')])
        return _msg(content="Final answer based on the thesis.")


def test_run_agent_emits_tool_then_answer():
    ctx = ToolContext(symbol="TEST", config=ResearchConfig())
    events = list(run_agent(ctx, [], "what's the thesis?", report=None, llm=_FakeLLM()))
    types = [e["type"] for e in events]

    assert "tool_call" in types
    assert "tool_result" in types
    assert types[-1] == "done"

    tool_call = next(e for e in events if e["type"] == "tool_call")
    assert tool_call["name"] == "get_report_section"
    assert tool_call["args"] == {"section": "thesis"}

    done = events[-1]
    assert "Final answer" in done["text"]


def test_run_agent_without_llm_key_errors_cleanly():
    class _Unconfigured(_FakeLLM):
        configured = False

    ctx = ToolContext(symbol="TEST", config=ResearchConfig())
    events = list(run_agent(ctx, [], "hi", report=None, llm=_Unconfigured()))
    assert events == [
        {"type": "error", "message": "LLM unavailable (no OpenRouter API key configured)."}
    ]


def test_build_system_prompt_handles_missing_report():
    prompt = build_system_prompt(None, "TEST")
    assert "TEST" in prompt
    assert "no cached report" in prompt


# ── Conversation persistence ──────────────────────────────────────────────────


def test_conversation_round_trip(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    cid = store.create_conversation("test", title="My question")

    assert store.load_conversation(cid)["messages"] == []

    store.append_messages(
        cid,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "tools": ["web_search"]},
        ],
    )

    conv = store.load_conversation(cid)
    assert conv["symbol"] == "TEST"
    assert conv["title"] == "My question"
    assert len(conv["messages"]) == 2
    assert conv["messages"][1]["tools"] == ["web_search"]

    listing = store.list_conversations("TEST")
    assert len(listing) == 1
    assert listing[0]["id"] == cid
    assert listing[0]["message_count"] == 2

    # JSON serialization of the stored payload stays intact.
    assert json.dumps(conv["messages"])
    store.close()
