"""Configuration via environment variables using pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class ResearchConfig(BaseSettings):
    """All research-agent settings, loaded from env vars with RESEARCH_AGENT_ prefix."""

    model_config = {"env_prefix": "RESEARCH_AGENT_", "extra": "ignore", "env_file": ".env"}

    # --- Tavily search ---
    tavily_api_key: str = ""
    search_endpoint: str = "https://api.tavily.com/search"
    tavily_search_depth: str = "advanced"

    # --- OpenRouter LLM ---
    openrouter_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "z-ai/glm-5.2"
    llm_timeout_seconds: int = 60
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.1
    # How hard a reasoning model thinks before answering, sent to OpenRouter as
    # ``reasoning.effort`` (low | medium | high); empty leaves the provider's
    # default. Measured 2026-09-27: GLM-5.3 at its default spent 9,218 reasoning
    # tokens and 82 s on one name reading (307 s with a retry); at "medium", 76
    # tokens and 4.7 s, and the reading still passed the number gate.
    llm_reasoning_effort: str = ""

    def reasoning_body(self) -> dict:
        """The ``extra_body`` a completion call sends: the effort, when one is set."""
        effort = self.llm_reasoning_effort.strip().lower()
        if not effort:
            return {}
        if effort not in ("low", "medium", "high"):
            raise ValueError(
                f"RESEARCH_AGENT_LLM_REASONING_EFFORT={effort!r}: one of low, medium, high"
            )
        return {"reasoning": {"effort": effort}}

    # --- Loop budgets ---
    max_iterations: int = 4
    max_queries_per_iteration: int = 4
    max_urls_per_query: int = 5
    max_sources_total: int = 30
    min_evidence_items: int = 8

    # --- Search policy ---
    search_recency_filter: str = "month"
    default_search_mode: str | None = None
    sec_search_enabled: bool = True
    transcript_search_enabled: bool = True
    curated_first: bool = True
    curated_domains: str = "sec.gov,reuters.com,bloomberg.com,wsj.com,ft.com"
    allow_fallback_web: bool = True

    # --- Paths ---
    output_dir: Path = Field(default=Path("out"))
    cache_dir: Path = Field(default=Path("data/research_cache"))
    db_path: Path = Field(default=Path("data/research.db"))

    # --- HTTP ---
    http_timeout_seconds: int = 10

    # --- Adaptive queries ---
    adaptive_queries_enabled: bool = True

    # --- Runtime ---
    offline_mode: bool = False

    @property
    def curated_domain_list(self) -> list[str]:
        return [d.strip() for d in self.curated_domains.split(",") if d.strip()]
