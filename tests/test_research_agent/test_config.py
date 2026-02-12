"""Tests for research_agent.config."""

from __future__ import annotations

from research_agent.config import ResearchConfig


def test_defaults():
    """Config loads sensible defaults without env vars."""
    config = ResearchConfig(
        _env_file=None,
        tavily_api_key="test",
        anthropic_api_key="test",
    )
    assert config.max_iterations == 4
    assert config.max_queries_per_iteration == 4
    assert config.llm_temperature == 0.1
    assert config.curated_first is True
    assert config.offline_mode is False
    assert config.llm_model == "claude-sonnet-4-5-20250929"


def test_curated_domain_list():
    """curated_domain_list splits the comma-separated string."""
    config = ResearchConfig(
        _env_file=None,
        tavily_api_key="test",
        anthropic_api_key="test",
        curated_domains="sec.gov, reuters.com, bloomberg.com",
    )
    assert config.curated_domain_list == ["sec.gov", "reuters.com", "bloomberg.com"]


def test_env_prefix(monkeypatch):
    """Settings are loaded from RESEARCH_AGENT_ prefixed env vars."""
    monkeypatch.setenv("RESEARCH_AGENT_MAX_ITERATIONS", "8")
    monkeypatch.setenv("RESEARCH_AGENT_TAVILY_API_KEY", "tv-key")
    monkeypatch.setenv("RESEARCH_AGENT_ANTHROPIC_API_KEY", "ant-key")
    config = ResearchConfig(_env_file=None)
    assert config.max_iterations == 8
    assert config.tavily_api_key == "tv-key"
    assert config.anthropic_api_key == "ant-key"
