"""Tests for research_agent.llm (mocked OpenAI SDK via OpenRouter, no network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
