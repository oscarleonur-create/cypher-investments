"""Configuration via environment variables using pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class ResearchConfig(BaseSettings):
    """All research-agent settings, loaded from env vars with RESEARCH_AGENT_ prefix."""

    model_config = {"env_prefix": "RESEARCH_AGENT_", "extra": "ignore", "env_file": ".env"}

    # --- Perplexity Sonar search ---
    perplexity_api_key: str = ""
    search_endpoint: str = "https://api.perplexity.ai/chat/completions"
    perplexity_model: str = "sonar"

    # --- Anthropic LLM ---
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-5-20250929"
    llm_timeout_seconds: int = 60
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.1

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
    curated_first: bool = True
    curated_domains: str = "sec.gov,reuters.com,bloomberg.com,wsj.com,ft.com"
    allow_fallback_web: bool = True

    # --- Paths ---
    output_dir: Path = Field(default=Path("out"))
    cache_dir: Path = Field(default=Path("data/research_cache"))
    db_path: Path = Field(default=Path("data/research.db"))

    # --- HTTP ---
    http_timeout_seconds: int = 10

    # --- Runtime ---
    offline_mode: bool = False

    @property
    def curated_domain_list(self) -> list[str]:
        return [d.strip() for d in self.curated_domains.split(",") if d.strip()]
