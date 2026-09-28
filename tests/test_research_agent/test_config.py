"""Tests for research_agent.config."""

from __future__ import annotations

from research_agent.config import ResearchConfig


def test_defaults(monkeypatch):
    """Config loads sensible defaults without env vars."""
    import os

    # _env_file=None keeps .env out, but a load_dotenv elsewhere in the run may
    # already have copied it into os.environ (the user's model is glm-5.3).
    for key in [k for k in os.environ if k.startswith("RESEARCH_AGENT_")]:
        monkeypatch.delenv(key)
    config = ResearchConfig(
        _env_file=None,
        tavily_api_key="test",
        openrouter_api_key="test",
    )
    assert config.max_iterations == 4
    assert config.max_queries_per_iteration == 4
    assert config.llm_temperature == 0.1
    assert config.curated_first is True
    assert config.offline_mode is False
    assert config.llm_model == "z-ai/glm-5.2"


def test_curated_domain_list():
    """curated_domain_list splits the comma-separated string."""
    config = ResearchConfig(
        _env_file=None,
        tavily_api_key="test",
        openrouter_api_key="test",
        curated_domains="sec.gov, reuters.com, bloomberg.com",
    )
    assert config.curated_domain_list == ["sec.gov", "reuters.com", "bloomberg.com"]


def test_env_prefix(monkeypatch):
    """Settings are loaded from RESEARCH_AGENT_ prefixed env vars."""
    monkeypatch.setenv("RESEARCH_AGENT_MAX_ITERATIONS", "8")
    monkeypatch.setenv("RESEARCH_AGENT_TAVILY_API_KEY", "tavily-key")
    monkeypatch.setenv("RESEARCH_AGENT_OPENROUTER_API_KEY", "or-key")
    config = ResearchConfig(_env_file=None)
    assert config.max_iterations == 8
    assert config.tavily_api_key == "tavily-key"
    assert config.openrouter_api_key == "or-key"
