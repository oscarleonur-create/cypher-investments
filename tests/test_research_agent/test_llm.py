"""Tests for research_agent.llm (mocked Anthropic SDK, no network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import BaseModel
from research_agent.config import ResearchConfig
from research_agent.llm import ClaudeLLM


def _make_config(**overrides) -> ResearchConfig:
    defaults = dict(
        _env_file=None,
        tavily_api_key="test-key",
        anthropic_api_key="test-key",
    )
    defaults.update(overrides)
    return ResearchConfig(**defaults)


class SimpleResponse(BaseModel):
    answer: str
    confidence: float = 0.0


class TestClaudeLLM:
    @patch("research_agent.llm.anthropic.Anthropic")
    def test_complete_plain_text(self, MockAnthropic):
        """complete() returns plain text when no response_model."""
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="Hello, world!")]
        mock_client.messages.create.return_value = mock_msg

        llm = ClaudeLLM(_make_config())
        result = llm.complete("system", "user")
        assert result == "Hello, world!"

    @patch("research_agent.llm.anthropic.Anthropic")
    def test_complete_structured_output(self, MockAnthropic):
        """complete() parses JSON into Pydantic model when response_model given."""
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='{"answer": "42", "confidence": 0.95}')]
        mock_client.messages.create.return_value = mock_msg

        llm = ClaudeLLM(_make_config())
        result = llm.complete("system", "user", response_model=SimpleResponse)
        assert isinstance(result, SimpleResponse)
        assert result.answer == "42"
        assert result.confidence == 0.95

    @patch("research_agent.llm.anthropic.Anthropic")
    def test_complete_strips_code_fences(self, MockAnthropic):
        """complete() strips markdown code fences from JSON response."""
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client

        mock_msg = MagicMock()
        mock_msg.content = [
            MagicMock(text='```json\n{"answer": "wrapped", "confidence": 0.5}\n```')
        ]
        mock_client.messages.create.return_value = mock_msg

        llm = ClaudeLLM(_make_config())
        result = llm.complete("system", "user", response_model=SimpleResponse)
        assert isinstance(result, SimpleResponse)
        assert result.answer == "wrapped"

    @patch("research_agent.llm.anthropic.Anthropic")
    def test_system_prompt_includes_schema(self, MockAnthropic):
        """When response_model is given, system prompt includes JSON schema."""
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='{"answer": "x", "confidence": 0.1}')]
        mock_client.messages.create.return_value = mock_msg

        llm = ClaudeLLM(_make_config())
        llm.complete("base system", "user", response_model=SimpleResponse)

        call_args = mock_client.messages.create.call_args
        system = call_args.kwargs.get("system") or call_args[1].get("system")
        assert "JSON" in system
        assert "answer" in system
