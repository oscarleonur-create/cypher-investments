"""Every rule constant is versioned, or explicitly not a rule.

A constant added to a rule module and left out of its declaration would let
the rules change while the version stayed the same — the exact failure the
stamp exists to prevent. These tests read each module's source, so a new
constant fails here until someone decides what it is.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields, replace

import pytest
from advisor.breadth import bars as breadth_bars
from advisor.breadth import companies as breadth_companies
from advisor.breadth import facts as breadth_facts
from advisor.breadth import filings as breadth_filings
from advisor.breadth import insiders as breadth_insiders
from advisor.breadth import listings as breadth_listings
from advisor.breadth import measure as breadth_measure
from advisor.breadth import ruleset as breadth_ruleset
from advisor.breadth import signals as breadth_signals
from advisor.breadth import universe as breadth_universe
from advisor.entry import distress, exits, freshness, proposal, sheet, zone
from advisor.entry import ruleset as entry_ruleset
from advisor.learning.rules import Kind
from advisor.news import google_news, halts
from advisor.news import names as company_names
from advisor.scanner import detect, premarket, scan, sources
from advisor.scanner import ruleset as scanner_ruleset


def constants(module) -> set[str]:
    """Names assigned at module level in ALL_CAPS (a leading _ allowed)."""
    out = set()
    for node in ast.parse(inspect.getsource(module)).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for t in targets:
            for n in t.elts if isinstance(t, ast.Tuple) else [t]:
                if isinstance(n, ast.Name) and n.id.lstrip("_").isupper():
                    out.add(n.id)
    return out


SCANNER_MODULES = {"detect": detect, "premarket": premarket, "sources": sources, "scan": scan}
ENTRY_MODULES = {
    "proposal": proposal,
    "zone": zone,
    "sheet": sheet,
    "exits": exits,
    "distress": distress,
    "halts": halts,
    "google_news": google_news,
    "names": company_names,
    "freshness": freshness,
}


def _scanner_declared() -> set[str]:
    names = set(scanner_ruleset.session_rules().params) | set(
        scanner_ruleset.premarket_rules().params
    )
    return {n for n in names if not n.startswith("thresholds.")}


@pytest.mark.parametrize("label,module", SCANNER_MODULES.items())
def test_scanner_constants_are_declared_or_not_rules(label, module):
    known = _scanner_declared() | set(scanner_ruleset.NOT_RULES)
    missing = {f"{label}.{c}" for c in constants(module)} - known
    assert not missing, f"declare in scanner/ruleset.py or list in NOT_RULES: {sorted(missing)}"


@pytest.mark.parametrize("label,module", ENTRY_MODULES.items())
def test_entry_constants_are_declared_or_not_rules(label, module):
    declared = set(entry_ruleset.DECLARED.get(label, {}))
    known = {f"{label}.{n}" for n in declared} | set(entry_ruleset.NOT_RULES)
    missing = {f"{label}.{c}" for c in constants(module)} - known
    assert not missing, f"declare in entry/ruleset.py or list in NOT_RULES: {sorted(missing)}"


def test_no_stale_declarations():
    # A declared name that no longer exists would fail at stamp time; say so here.
    for label, names in entry_ruleset.DECLARED.items():
        for name in names:
            assert hasattr(ENTRY_MODULES[label], name), f"{label}.{name}"
    for key in scanner_ruleset.NOT_RULES:
        label, name = key.split(".", 1)
        assert hasattr(SCANNER_MODULES[label], name), key


def test_every_threshold_field_is_in_the_stamp():
    session = scanner_ruleset.session_rules()
    pre = scanner_ruleset.premarket_rules()
    assert {f"thresholds.{f.name}" for f in fields(detect.Thresholds)} <= set(session.params)
    assert {f"thresholds.{f.name}" for f in fields(premarket.PremarketThresholds)} <= set(
        pre.params
    )


class TestVersionFollowsTheRules:
    def test_a_different_threshold_is_a_different_session_version(self):
        a = scanner_ruleset.session_rules()
        b = scanner_ruleset.session_rules(replace(detect.DEFAULT, sigma_min=2.5))
        assert a.version != b.version
        assert b.params["thresholds.sigma_min"] == 2.5

    def test_a_different_premarket_threshold_is_a_different_version(self):
        a = scanner_ruleset.premarket_rules()
        b = scanner_ruleset.premarket_rules(replace(premarket.PM_DEFAULT, b_prev_day_min=0.1))
        assert a.version != b.version

    def test_a_module_constant_edit_is_a_different_entry_version(self, monkeypatch):
        before = entry_ruleset.entry_rules()
        monkeypatch.setattr(proposal, "TRADE_STOP_SIGMAS", 2.0)
        after = entry_ruleset.entry_rules()
        assert before.version != after.version
        assert after.params["proposal.TRADE_STOP_SIGMAS"] == 2.0

    def test_same_rules_same_version(self):
        assert entry_ruleset.entry_rules().version == entry_ruleset.entry_rules().version
        assert scanner_ruleset.session_rules().version == scanner_ruleset.session_rules().version

    def test_rulesets_do_not_collide(self):
        versions = {
            scanner_ruleset.session_rules().version,
            scanner_ruleset.premarket_rules().version,
            entry_ruleset.entry_rules().version,
        }
        assert len(versions) == 3


class TestKinds:
    """The user's decisions of 2026-09-25: learning may search thresholds, never limits."""

    def test_risk_budgets_and_book_limit_are_decided(self):
        kinds = entry_ruleset.entry_rules().kinds
        for name in (
            "TRADE_RISK",
            "POSITION_RISK",
            "POSITION_RISK_CHEAP",
            "MAX_TOTAL_RISK",
            "THESIS_BONUS",
            "MAX_TOTAL_RISK_THESIS",
            "BOOK_LIMIT",
        ):
            assert kinds[f"proposal.{name}"] is Kind.DECIDED, name
        assert kinds["zone.WINDOW_DAYS"] is Kind.DECIDED

    def test_stops_and_triggers_are_thresholds(self):
        kinds = entry_ruleset.entry_rules().kinds
        for name in ("TRADE_STOP_SIGMAS", "DIP_SIGMAS", "ENTRY_CONFIRM_SESSIONS"):
            assert kinds[f"proposal.{name}"] is Kind.THRESHOLD, name

    def test_nothing_that_sizes_risk_is_searchable(self):
        # Anything whose name says RISK or LIMIT must not be a threshold.
        kinds = entry_ruleset.entry_rules().kinds
        searchable = {n for n, k in kinds.items() if k is Kind.THRESHOLD}
        assert not {n for n in searchable if "RISK" in n or "LIMIT" in n or "BONUS" in n}


# ── breadth ────────────────────────────────────────────────────────────────

BREADTH_MODULES = {
    "universe": breadth_universe,
    "listings": breadth_listings,
    "bars": breadth_bars,
    "facts": breadth_facts,
}


@pytest.mark.parametrize("label,module", BREADTH_MODULES.items())
def test_breadth_constants_are_declared_or_not_rules(label, module):
    declared = set(breadth_ruleset.DECLARED.get(label, {}))
    known = {f"{label}.{n}" for n in declared} | set(breadth_ruleset.NOT_RULES)
    missing = {f"{label}.{c}" for c in constants(module)} - known
    assert not missing, f"declare in breadth/ruleset.py or list in NOT_RULES: {sorted(missing)}"


def test_breadth_declarations_exist():
    for label, names in breadth_ruleset.DECLARED.items():
        for name in names:
            assert hasattr(BREADTH_MODULES[label], name), f"{label}.{name}"
    modules = {**BREADTH_MODULES, **SIGNAL_MODULES}
    for label, names in breadth_ruleset.SIGNAL_DECLARED.items():
        for name in names:
            assert hasattr(modules[label], name), f"{label}.{name}"
    for key in breadth_ruleset.NOT_RULES:
        label, name = key.split(".", 1)
        assert hasattr(modules[label], name), key


SIGNAL_MODULES = {
    "signals": breadth_signals,
    "measure": breadth_measure,
    "companies": breadth_companies,
    "filings": breadth_filings,
    "insiders": breadth_insiders,
}


@pytest.mark.parametrize("label,module", SIGNAL_MODULES.items())
def test_breadth_signal_constants_are_declared_or_not_rules(label, module):
    declared = set(breadth_ruleset.SIGNAL_DECLARED.get(label, {}))
    known = {f"{label}.{n}" for n in declared} | set(breadth_ruleset.NOT_RULES)
    missing = {f"{label}.{c}" for c in constants(module)} - known
    assert not missing, f"declare in breadth/ruleset.py or list in NOT_RULES: {sorted(missing)}"


def test_every_signal_threshold_is_in_the_stamp_and_moves_the_version():
    s = breadth_ruleset.signal_rules()
    assert {f"thresholds.{f.name}" for f in fields(breadth_signals.Thresholds)} <= set(s.params)
    assert "universe.MIN_PRICE" in s.params  # the floor decides who can signal
    other = breadth_ruleset.signal_rules(replace(breadth_signals.DEFAULT, accel_min=0.1))
    assert other.version != s.version
    assert s.version != breadth_ruleset.universe_rules().version


def test_the_universe_floor_is_decided_and_versioned(monkeypatch):
    s = breadth_ruleset.universe_rules()
    assert s.kinds["universe.MIN_PRICE"] is Kind.DECIDED
    assert s.kinds["universe.MIN_DOLLAR_VOLUME"] is Kind.DECIDED
    monkeypatch.setattr(breadth_universe, "MIN_PRICE", 3.0)
    assert breadth_ruleset.universe_rules().version != s.version
