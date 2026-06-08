"""Settings for the fundamental research engine.

Mirrors the pattern in `research_agent/config.py` so env loading is consistent
across the codebase. Env prefix is `ADVISOR_RESEARCH_` to avoid collision with
`RESEARCH_AGENT_*` (the dip-card pipeline).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class ResearchSettings(BaseSettings):
    model_config = {"env_prefix": "ADVISOR_RESEARCH_", "extra": "ignore", "env_file": ".env"}

    edgar_user_agent: str = "Advisor Research contact@example.com"

    cache_dir: Path = Field(default=Path("data/research_cache"))
    db_path: Path = Field(default=Path("data/research.db"))

    statement_ttl_days: int = 7
    filing_ttl_days: int = 30

    edgar_rate_limit_per_sec: int = 8


@lru_cache(maxsize=1)
def get_settings() -> ResearchSettings:
    return ResearchSettings()
