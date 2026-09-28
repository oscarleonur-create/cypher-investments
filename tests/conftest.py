"""Shared test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_news_verification_network(monkeypatch):
    """news.verify fetches publishers' pages; no test may reach the network through it.

    With every network step empty, an item can still be verified by
    corroboration among the items a test supplies, and otherwise is UNVERIFIED.
    Tests of the network steps themselves pass their own fakes.
    """
    from advisor.news import verify

    monkeypatch.setattr(verify, "fetch_page", lambda url: None)
    monkeypatch.setattr(verify, "resolve_google", lambda link: None)
    monkeypatch.setattr(verify, "title_search", lambda title: [])


@pytest.fixture(autouse=True)
def _no_news_rationale_network(monkeypatch):
    """An acting proposal pulls and judges news for its rationale; no test may.

    Tests of the news rationale pass their own ``news`` / ``puller``.
    """
    from advisor.entry import run

    monkeypatch.setattr(run, "default_news", lambda *a, **k: [])


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch):
    """No test may depend on the holder's ``.env`` or reach the live broker.

    ``market.tastytrade_client`` copies ``.env`` into ``os.environ`` when it is
    imported, so a developer's settings leaked into tests that pass
    ``_env_file=None`` (``RESEARCH_AGENT_LLM_MODEL`` failed the defaults test),
    and tests that forgot to fake the book passed only by calling the real
    TastyTrade account. Both now fail loudly instead of depending on the machine.
    Tests of the broker path pass their own fakes, which override this.
    """
    import os

    for name in list(os.environ):
        if name.startswith("RESEARCH_AGENT_"):
            monkeypatch.delenv(name, raising=False)

    async def no_broker(*_a, **_k):
        raise RuntimeError("tests must not reach the live broker; fake the session or book")

    monkeypatch.setattr("advisor.market.tastytrade_client.get_session", no_broker)
