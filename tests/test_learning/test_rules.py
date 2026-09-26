"""Rule stamps: the version changes with the rules and with nothing else."""

from __future__ import annotations

import json
import subprocess
from datetime import time

import pytest
from advisor.learning import rules
from advisor.learning.rules import Kind, RuleStamp, stamp, version_of


class TestVersion:
    def test_order_of_parameters_does_not_matter(self):
        assert version_of("x", {"a": 1, "b": 2.0}) == version_of("x", {"b": 2.0, "a": 1})

    def test_any_value_change_is_a_new_version(self):
        assert version_of("x", {"a": 0.04}) != version_of("x", {"a": 0.0400001})

    def test_int_and_float_are_different_versions(self):
        # A constant retyped is a constant edited.
        assert version_of("x", {"a": 5}) != version_of("x", {"a": 5.0})

    def test_ruleset_name_is_part_of_the_version(self):
        # Two rulesets with identical parameters must not share a version:
        # it is the primary key of rule_versions.
        assert version_of("scanner.session", {"a": 1}) != version_of("entry", {"a": 1})

    def test_added_parameter_is_a_new_version(self):
        assert version_of("x", {"a": 1}) != version_of("x", {"a": 1, "b": None})

    def test_times_and_tuples_are_canonical(self):
        a = version_of("x", {"t": time(9, 35), "c": ((0, 0.0), (5, 0.05))})
        b = version_of("x", {"t": "09:35:00", "c": [[0, 0.0], [5, 0.05]]})
        assert a == b

    def test_kind_is_not_part_of_the_version(self):
        a = stamp("x", {"a": (1, Kind.THRESHOLD)})
        b = stamp("x", {"a": (1, Kind.DECIDED)})
        assert a.version == b.version


class TestStamp:
    def test_record_carries_the_version_not_the_parameters(self):
        s = stamp("x", {"a": (0.04, Kind.THRESHOLD)})
        dumped = json.loads(s.model_dump_json())
        assert set(dumped) == {"ruleset", "version", "code"}
        assert s.params == {"a": 0.04} and s.kinds == {"a": Kind.THRESHOLD}

    def test_a_stamp_read_back_has_no_parameters(self):
        s = stamp("x", {"a": (0.04, Kind.THRESHOLD)})
        back = RuleStamp.model_validate_json(s.model_dump_json())
        assert back.version == s.version and back.params == {}

    def test_parameters_are_stored_plain(self):
        s = stamp("x", {"t": (time(7, 0), Kind.MODEL)})
        assert s.params == {"t": "07:00:00"}


class TestCodeRev:
    @pytest.fixture(autouse=True)
    def _fresh(self):
        rules.code_rev.cache_clear()
        yield
        rules.code_rev.cache_clear()

    def _fake(self, monkeypatch, sha: str, status: str):
        def run(args, **kw):
            out = sha if "rev-parse" in args else status
            return subprocess.CompletedProcess(args, 0, stdout=out + "\n", stderr="")

        monkeypatch.setattr(rules.subprocess, "run", run)

    def test_clean_checkout(self, monkeypatch):
        self._fake(monkeypatch, "abc123def456", "")
        assert rules.code_rev() == "abc123def456"

    def test_uncommitted_source_is_dirty(self, monkeypatch):
        self._fake(monkeypatch, "abc123def456", " M src/advisor/entry/proposal.py")
        assert rules.code_rev() == "abc123def456+dirty"

    def test_no_git_is_unknown_not_a_crash(self, monkeypatch):
        def run(*a, **kw):
            raise FileNotFoundError("git")

        monkeypatch.setattr(rules.subprocess, "run", run)
        assert rules.code_rev() == "unknown"

    def test_git_failure_is_unknown(self, monkeypatch):
        def run(args, **kw):
            raise subprocess.CalledProcessError(128, args)

        monkeypatch.setattr(rules.subprocess, "run", run)
        assert rules.code_rev() == "unknown"

    def test_git_hanging_is_unknown(self, monkeypatch):
        def run(args, **kw):
            raise subprocess.TimeoutExpired(args, 5)

        monkeypatch.setattr(rules.subprocess, "run", run)
        assert rules.code_rev() == "unknown"

    def test_not_a_source_checkout_is_unknown(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rules, "_REPO", tmp_path)
        assert rules.code_rev() == "unknown"

    def test_asked_once_per_process(self, monkeypatch):
        calls = []

        def run(args, **kw):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout="abc\n", stderr="")

        monkeypatch.setattr(rules.subprocess, "run", run)
        rules.code_rev()
        rules.code_rev()
        assert len(calls) == 2  # rev-parse and status, once

    def test_real_checkout_gives_a_revision(self):
        # This test runs from a git checkout; the revision is real.
        rev = rules.code_rev()
        assert rev != "unknown" and len(rev.split("+")[0]) == 12
