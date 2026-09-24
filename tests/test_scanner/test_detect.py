"""Setup rules: boundaries, missing data, and the session clock."""

from __future__ import annotations

from datetime import datetime, time

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.jobs import EveryMinutes
from advisor.scanner import detect
from advisor.scanner.models import Mover, Setup


def et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=mc.MARKET_TZ)


NOON = et(2026, 9, 23, 12, 0)  # a Wednesday; ~48% of the day's volume by now


def gapper(**kw) -> Mover:
    base = dict(
        symbol="AMPG",
        price=10.8,
        prev_close=10.0,
        open=10.6,
        volume=3_000_000,
        avg_volume=1_000_000,
        market_cap=500e6,
    )
    base.update(kw)
    return Mover(**base)


def dipper(**kw) -> Mover:
    base = dict(
        symbol="AAPL",
        price=94.0,
        prev_close=100.0,
        open=97.0,
        volume=40_000_000,
        avg_volume=50_000_000,
        market_cap=3e12,
    )
    base.update(kw)
    return Mover(**base)


class TestVolumePace:
    def test_none_outside_session(self):
        assert detect.expected_volume_share(et(2026, 9, 23, 8, 0)) is None
        assert detect.expected_volume_share(et(2026, 9, 26, 11, 0)) is None  # Saturday
        assert detect.expected_volume_share(et(2026, 11, 26, 11, 0)) is None  # Thanksgiving

    def test_curve_is_monotonic_and_ends_at_one(self):
        shares = [
            detect.expected_volume_share(et(2026, 9, 23, h, m))
            for h, m in [(9, 31), (9, 35), (10, 0), (11, 0), (13, 0), (15, 0), (15, 59)]
        ]
        assert shares == sorted(shares)
        assert 0 < shares[0] < 0.02 and shares[-1] > 0.98

    def test_early_close_compresses_the_curve(self):
        """At 12:59 on a 13:00 close nearly the whole day has traded."""
        assert detect.expected_volume_share(et(2026, 11, 27, 12, 59)) > 0.98
        assert detect.expected_volume_share(
            et(2026, 11, 27, 12, 59)
        ) > detect.expected_volume_share(et(2026, 11, 25, 12, 59))

    def test_zero_average_volume_gives_no_rvol(self):
        assert detect.relative_volume(gapper(avg_volume=0), NOON) is None
        assert detect.relative_volume(gapper(avg_volume=None), NOON) is None

    def test_dst_boundary_is_wall_clock(self):
        """UTC input across the November DST change still maps to ET session time."""
        from datetime import timezone

        # 2026-11-02 is the first Monday after DST ends: 14:35 UTC == 09:35 EST.
        moment = datetime(2026, 11, 2, 14, 35, tzinfo=timezone.utc)
        assert detect.expected_volume_share(moment) == pytest.approx(0.05)


class TestSetupA:
    def test_qualifies(self):
        assert detect.is_catalyst_gap(gapper(), NOON)

    def test_exactly_at_thresholds_qualifies(self):
        m = gapper(open=10.4, price=10.4)  # gap and hold both exactly +4%
        assert detect.gap(m) == pytest.approx(0.04)
        assert detect.is_catalyst_gap(m, NOON)

    def test_just_below_gap_threshold(self):
        assert not detect.is_catalyst_gap(gapper(open=10.39), NOON)

    def test_gap_faded_does_not_qualify(self):
        """Opened +6% but gave it back: the setup the user trades is gone."""
        assert not detect.is_catalyst_gap(gapper(open=10.6, price=10.2), NOON)

    def test_thin_volume_rejected(self):
        assert not detect.is_catalyst_gap(gapper(volume=200_000), NOON)

    @pytest.mark.parametrize(
        "field", ["open", "prev_close", "price", "market_cap", "avg_volume", "volume"]
    )
    def test_missing_field_never_qualifies(self, field):
        assert not detect.is_catalyst_gap(gapper(**{field: None}), NOON)

    def test_zero_or_negative_prev_close(self):
        assert not detect.is_catalyst_gap(gapper(prev_close=0), NOON)
        assert not detect.is_catalyst_gap(gapper(prev_close=-5), NOON)

    def test_nan_price_rejected(self):
        assert not detect.is_catalyst_gap(gapper(price=float("nan")), NOON)

    def test_penny_and_microcap_rejected(self):
        assert not detect.is_catalyst_gap(gapper(price=1.5, open=1.45, prev_close=1.3), NOON)
        assert not detect.is_catalyst_gap(gapper(market_cap=100e6), NOON)

    def test_market_closed(self):
        assert not detect.is_catalyst_gap(gapper(), et(2026, 9, 23, 16, 30))


class TestSetupC:
    def test_qualifies_with_sigma(self):
        ok, z = detect.is_news_dip(dipper(), 0.02)
        assert ok and z == pytest.approx(3.0)

    def test_exactly_minus_four_percent(self):
        ok, _ = detect.is_news_dip(dipper(price=96.0), 0.01)
        assert ok

    def test_below_two_sigma_for_a_volatile_name(self):
        """-6% is a routine day for a stock with 5% daily vol."""
        ok, z = detect.is_news_dip(dipper(), 0.05)
        assert not ok and z == pytest.approx(1.2)

    def test_exactly_two_sigma(self):
        ok, z = detect.is_news_dip(dipper(), 0.03)
        assert ok and z == pytest.approx(2.0)

    @pytest.mark.parametrize("sigma", [None, 0.0, -0.01, float("nan")])
    def test_unusable_sigma_falls_back_and_says_so(self, sigma):
        ok, z = detect.is_news_dip(dipper(), sigma)
        assert ok and z is None

    def test_small_caps_excluded(self):
        """POET and PENG were the setup's losers; the size floor removes them."""
        assert not detect.is_news_dip(dipper(market_cap=2e9), 0.02)[0]

    def test_up_move_is_not_a_dip(self):
        assert not detect.is_news_dip(dipper(price=106.0), 0.02)[0]

    def test_missing_prev_close(self):
        assert detect.is_news_dip(dipper(prev_close=None), 0.02) == (False, None)


class TestGapKept:
    """Fix 3: a gap that has given back more than half is fading, not held."""

    def test_ionq_2026_09_23_is_rejected(self):
        m = gapper(prev_close=100.0, open=112.5, price=104.0)  # +12.5% gap, +4.0% now
        assert detect.gap_kept(m) == pytest.approx(0.32)
        assert not detect.is_catalyst_gap(m, NOON)

    def test_exactly_half_kept_qualifies(self):
        m = gapper(prev_close=100.0, open=110.0, price=105.0)
        assert detect.gap_kept(m) == pytest.approx(0.5)
        assert detect.is_catalyst_gap(m, NOON)

    def test_just_under_half_rejected(self):
        m = gapper(prev_close=100.0, open=110.0, price=104.9)
        assert not detect.is_catalyst_gap(m, NOON)

    def test_extending_the_gap_qualifies(self):
        m = gapper(prev_close=100.0, open=106.0, price=112.0)
        assert detect.gap_kept(m) == pytest.approx(2.0)
        assert detect.is_catalyst_gap(m, NOON)

    def test_no_gap_or_gap_down_has_no_ratio(self):
        assert detect.gap_kept(gapper(open=10.0)) is None
        assert detect.gap_kept(gapper(open=9.5)) is None
        assert detect.gap_kept(gapper(open=None)) is None


class TestOwnShare:
    """Fix 1: how much of a drop is the company's own."""

    def test_peers_flat_means_all_its_own(self):
        assert detect.own_share(-0.06, 0.0) == pytest.approx(1.0)

    def test_peers_fell_as_far(self):
        assert detect.own_share(-0.06, -0.06) == pytest.approx(0.0)

    def test_peers_rose(self):
        assert detect.own_share(-0.06, 0.02) > 1

    def test_exactly_half_passes(self):
        assert detect.passes_peer_test(-0.06, -0.03)

    def test_just_under_half_fails(self):
        assert not detect.passes_peer_test(-0.06, -0.0301)

    @pytest.mark.parametrize(
        "chg,peer",
        [(None, -0.01), (-0.06, None), (-0.06, float("nan")), (0.0, -0.01), (0.03, 0.0)],
    )
    def test_unjudgeable_passes(self, chg, peer):
        assert detect.own_share(chg, peer) is None
        assert detect.passes_peer_test(chg, peer)


class TestDetect:
    def test_a_stock_can_only_be_one_direction(self):
        assert [s for s, _ in detect.detect(gapper(market_cap=50e9), NOON, 0.02)] == [
            Setup.CATALYST_GAP
        ]
        assert [s for s, _ in detect.detect(dipper(), NOON, 0.02)] == [Setup.NEWS_DIP]


class TestScanTrigger:
    trigger = EveryMinutes(30, during_session_only=True, not_before=time(9, 35))

    def test_not_at_the_bell(self):
        assert not self.trigger.is_due(et(2026, 9, 23, 9, 30), None)

    def test_first_scan_at_0935(self):
        assert self.trigger.is_due(et(2026, 9, 23, 9, 35), None)

    def test_not_on_holidays_or_after_close(self):
        assert not self.trigger.is_due(et(2026, 11, 26, 10, 0), None)
        assert not self.trigger.is_due(et(2026, 11, 27, 13, 5), None)  # early close

    def test_laptop_asleep_resumes_on_wake(self):
        assert self.trigger.is_due(et(2026, 9, 23, 13, 12), et(2026, 9, 23, 9, 35))
