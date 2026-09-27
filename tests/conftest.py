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
