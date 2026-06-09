"""Tool-calling LLM client over OpenRouter (OpenAI SDK).

Reuses :class:`research_agent.config.ResearchConfig` for the API key, base URL,
and model so the whole app speaks to a single provider. Unlike
``research_agent.llm.OpenRouterLLM`` (structured-output only) this exposes the
OpenAI-style ``tools=`` parameter needed for an agentic loop.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from research_agent.config import ResearchConfig
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class AgentLLM:
    """Minimal OpenRouter chat client with function/tool-calling support."""

    def __init__(self, config: ResearchConfig | None = None) -> None:
        self._config = config or ResearchConfig()
        self._client = OpenAI(
            api_key=self._config.openrouter_api_key,
            base_url=self._config.llm_base_url,
            timeout=self._config.llm_timeout_seconds,
        )

    @property
    def model(self) -> str:
        return self._config.llm_model

    @property
    def configured(self) -> bool:
        return bool(self._config.openrouter_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> Any:
        """One chat completion turn with tools available.

        Returns the raw OpenAI ``ChatCompletion``; the caller inspects
        ``choices[0].message`` for ``tool_calls`` vs. a final ``content``.

        ``max_tokens`` is kept modest: OpenRouter reserves prompt + max_tokens
        of credit up front, so an oversized cap can trip a 402 on tight budgets.
        """
        return self._client.chat.completions.create(
            model=self._config.llm_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
