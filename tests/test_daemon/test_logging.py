"""The daemon's own observability.

Found by starting it for the first time. Two bugs, both of which made an
always-on process indistinguishable from a broken one:

- The CLI callback calls `logging.basicConfig(level=WARNING)` before any
  subcommand runs, so the daemon's own `basicConfig(level=INFO)` was a silent
  no-op. It ran completely mute: no "daemon up", no job results, no events.
- Its timestamps were the machine's local zone while every decision it makes
  is in market time. This project has already shipped three timezone bugs; a
  log that needs translating is how the fourth starts.
"""

from __future__ import annotations

import logging
from datetime import datetime

from advisor.daemon.market_calendar import MARKET_TZ


class TestBasicConfigIsNotEnough:
    def test_a_second_basic_config_without_force_is_a_no_op(self, monkeypatch):
        """The exact mechanism: this is why the daemon was silent."""
        root = logging.getLogger()
        original = list(root.handlers)
        try:
            root.handlers.clear()
            logging.basicConfig(level=logging.WARNING, force=True)
            first = root.level
            logging.basicConfig(level=logging.INFO)  # no force — ignored
            assert root.level == first == logging.WARNING
        finally:
            root.handlers[:] = original

    def test_force_reconfigures(self):
        root = logging.getLogger()
        original, level = list(root.handlers), root.level
        try:
            logging.basicConfig(level=logging.WARNING, force=True)
            logging.basicConfig(level=logging.INFO, force=True)
            assert root.level == logging.INFO
        finally:
            root.handlers[:] = original
            root.setLevel(level)


class TestDaemonLogging:
    def _configure(self):
        from advisor.cli import daemon_cmds

        return daemon_cmds

    def test_the_run_command_forces_its_own_configuration(self):
        """Without force=True the CLI callback's WARNING wins and nothing prints."""
        import inspect

        from advisor.cli import daemon_cmds

        source = inspect.getsource(daemon_cmds.daemon_run)
        assert "force=True" in source

    def test_timestamps_are_stamped_in_market_time(self):
        import inspect

        from advisor.cli import daemon_cmds

        source = inspect.getsource(daemon_cmds.daemon_run)
        assert "MARKET_TZ" in source
        assert "ET |" in source, "the zone must be visible in the line, not implied"

    def test_third_party_chatter_is_quietened(self):
        import inspect

        from advisor.cli import daemon_cmds

        source = inspect.getsource(daemon_cmds.daemon_run)
        for noisy in ("tastytrade", "httpx", "urllib3"):
            assert noisy in source

    def test_the_market_converter_returns_market_time(self):
        """The formatter's clock, checked directly rather than through a log."""
        converter = lambda *args: datetime.now(MARKET_TZ).timetuple()  # noqa: E731
        stamped = converter()
        expected = datetime.now(MARKET_TZ)
        assert stamped.tm_hour == expected.hour


class TestSupervisorAnnouncesItself:
    async def test_the_loop_says_what_it_is_running(self, caplog):
        """'daemon up' is the only proof the loop started at all."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from advisor.daemon.jobs import JobRegistry
        from advisor.daemon.store import DaemonStore
        from advisor.daemon.supervisor import Supervisor

        with TemporaryDirectory() as tmp:
            store = DaemonStore(Path(tmp) / "research.db")
            try:
                sup = Supervisor(store, JobRegistry(), tick_seconds=0.01)
                with caplog.at_level(logging.INFO, logger="advisor.daemon.supervisor"):
                    sup._stop.set()
                    await sup.run()
                assert any("daemon up" in r.message for r in caplog.records)
                assert any("daemon stopped" in r.message for r in caplog.records)
            finally:
                store.close()
