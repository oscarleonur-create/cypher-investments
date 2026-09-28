"""Tests for research_agent.llm (mocked OpenAI SDK via OpenRouter, no network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
from research_agent.config import ResearchConfig
from research_agent.llm import OpenRouterLLM


def _make_config(**overrides) -> ResearchConfig:
    defaults = dict(
        _env_file=None,
        tavily_api_key="test-key",
        openrouter_api_key="test-key",
    )
    defaults.update(overrides)
    return ResearchConfig(**defaults)


class SimpleResponse(BaseModel):
    answer: str
    confidence: float = 0.0


def _mock_response(text: str) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


class TestOpenRouterLLM:
    @patch("research_agent.llm.OpenAI")
    def test_complete_plain_text(self, MockOpenAI):
        """complete() returns plain text when no response_model."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("Hello, world!")

        llm = OpenRouterLLM(_make_config())
        result = llm.complete("system", "user")
        assert result == "Hello, world!"

    @patch("research_agent.llm.OpenAI")
    def test_complete_structured_output(self, MockOpenAI):
        """complete() parses JSON into Pydantic model when response_model given."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"answer": "42", "confidence": 0.95}'
        )

        llm = OpenRouterLLM(_make_config())
        result = llm.complete("system", "user", response_model=SimpleResponse)
        assert isinstance(result, SimpleResponse)
        assert result.answer == "42"
        assert result.confidence == 0.95

    @patch("research_agent.llm.OpenAI")
    def test_complete_strips_code_fences(self, MockOpenAI):
        """complete() strips markdown code fences from JSON response."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '```json\n{"answer": "wrapped", "confidence": 0.5}\n```'
        )

        llm = OpenRouterLLM(_make_config())
        result = llm.complete("system", "user", response_model=SimpleResponse)
        assert isinstance(result, SimpleResponse)
        assert result.answer == "wrapped"

    @patch("research_agent.llm.OpenAI")
    def test_system_prompt_includes_schema(self, MockOpenAI):
        """When response_model is given, system prompt includes JSON schema."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"answer": "x", "confidence": 0.1}'
        )

        llm = OpenRouterLLM(_make_config())
        llm.complete("base system", "user", response_model=SimpleResponse)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_content = messages[0]["content"]
        assert "JSON" in system_content
        assert "answer" in system_content


class TestReasoningEffort:
    """GLM-5.3 at its default effort spent 82 s on one reading; "medium" took 4.7 s."""

    @pytest.fixture(autouse=True)
    def _no_effort_from_the_environment(self, monkeypatch):
        # _env_file=None keeps .env out of the config, but anything that ran
        # load_dotenv has already copied it into os.environ; a user's real
        # RESEARCH_AGENT_LLM_REASONING_EFFORT=medium then failed the "unset" tests.
        monkeypatch.delenv("RESEARCH_AGENT_LLM_REASONING_EFFORT", raising=False)

    def test_unset_sends_nothing_extra(self):
        assert _make_config().reasoning_body() == {}

    def test_set_is_sent_as_openrouter_reasoning(self):
        assert _make_config(llm_reasoning_effort=" Medium ").reasoning_body() == {
            "reasoning": {"effort": "medium"}
        }

    def test_a_typo_fails_loudly_instead_of_thinking_for_minutes(self):
        import pytest

        with pytest.raises(ValueError, match="low, medium, high"):
            _make_config(llm_reasoning_effort="meduim").reasoning_body()

    @patch("research_agent.llm.OpenAI")
    def test_complete_and_chat_carry_it(self, MockOpenAI):
        client = MagicMock()
        MockOpenAI.return_value = client
        client.chat.completions.create.return_value = _mock_response('{"answer": "a"}')
        llm = OpenRouterLLM(_make_config(llm_reasoning_effort="medium"))
        llm.complete("s", "u", response_model=SimpleResponse)
        llm.chat("s", [{"role": "user", "content": "hi"}])
        for call in client.chat.completions.create.call_args_list:
            assert call.kwargs["extra_body"] == {"reasoning": {"effort": "medium"}}

    @patch("research_agent.llm.OpenAI")
    def test_unset_leaves_the_call_as_before(self, MockOpenAI):
        client = MagicMock()
        MockOpenAI.return_value = client
        client.chat.completions.create.return_value = _mock_response("x")
        OpenRouterLLM(_make_config()).complete("s", "u")
        assert "extra_body" not in client.chat.completions.create.call_args.kwargs

    @patch("advisor.agent.llm.OpenAI")
    def test_the_tool_agent_carries_it(self, MockOpenAI):
        from advisor.agent.llm import AgentLLM

        client = MagicMock()
        MockOpenAI.return_value = client
        AgentLLM(_make_config(llm_reasoning_effort="low")).chat_with_tools([], [])
        assert client.chat.completions.create.call_args.kwargs["extra_body"] == {
            "reasoning": {"effort": "low"}
        }
