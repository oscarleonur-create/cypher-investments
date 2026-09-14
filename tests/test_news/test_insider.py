"""Insider transactions as a pattern, not a stream of filings.

A single Form 4 is noise, and the code said so by excluding Form 4 entirely —
a correct diagnosis with the wrong remedy. Bloom Energy files almost nothing
else: in a month with fourteen filings, every one was a Form 4, a 144 or a
13G amendment, and the system saw zero. What it missed was five insiders
selling $16.3m on the open market, four of them on the same day.

Transaction codes decide everything. The CEO's filing in that same window was
a 300,000-share gift and an option exercise; reading it as selling would be
flatly wrong.
"""

from __future__ import annotations

from datetime import date

import pytest
from advisor.daemon.models import EventTier
from advisor.news.insider import (
    MECHANICAL_CODES,
    MIN_CLUSTER_INSIDERS,
    InsiderActivity,
    InsiderTrade,
    by_insider,
    cluster_event,
)


def trade(insider="Aman Joshi", code="S", shares=4677.0, value=1_130_665.0, day=18) -> InsiderTrade:
    return InsiderTrade(
        filed=date(2026, 8, day),
        insider=insider,
        position="Chief Commercial Officer",
        code=code,
        shares=shares,
        value=value,
    )


def activity(*trades, window_days=45) -> InsiderActivity:
    return InsiderActivity(symbol="BE", window_days=window_days, trades=list(trades))


# Bloom Energy's real August open-market sales.
BE_REAL = activity(
    trade("Aman Joshi", value=1_130_665.0),
    trade("Jeffrey R Immelt", shares=30_000.0, value=7_167_300.0),
    trade("Satish Chitoori", shares=2_053.0, value=496_087.0),
    trade("Shawn Marie Soderberg", shares=2_895.0, value=676_272.0),
    trade("John T Chambers", shares=15_000.0, value=3_750_000.0, day=14),
    trade("John T Chambers", shares=15_000.0, value=3_083_700.0, day=4),
)


class TestTheRealCluster:
    def test_five_distinct_insiders_are_counted_not_six_trades(self):
        """Chambers sold twice; that is one person, not two."""
        assert len(BE_REAL.selling_insiders) == 5
        assert len(BE_REAL.sells) == 6

    def test_it_fires_a_selling_cluster(self):
        event = cluster_event(BE_REAL)
        assert event.kind == "INSIDER_SELLING_CLUSTER"
        assert event.payload["insider_count"] == 5

    def test_the_net_value_is_negative(self):
        assert BE_REAL.net_value == pytest.approx(-16_304_024.0)

    def test_it_is_tier_b_not_an_interrupt(self):
        """Insider behaviour develops over weeks; it is not a deadline."""
        assert cluster_event(BE_REAL).tier is EventTier.B

    def test_the_positions_travel_with_the_event(self):
        assert cluster_event(BE_REAL).payload["positions"]

    def test_per_insider_totals_sum_the_repeat_seller(self):
        totals = by_insider(BE_REAL)
        assert totals["John T Chambers"] == pytest.approx(-6_833_700.0)
        assert len(totals) == 5


class TestTransactionCodes:
    """Only open-market decisions express a view."""

    def test_a_gift_is_not_selling(self):
        """The CEO gifted 300,000 shares in the same window."""
        assert "G" in MECHANICAL_CODES

    def test_an_option_exercise_is_not_selling(self):
        assert "M" in MECHANICAL_CODES

    def test_a_grant_is_not_buying(self):
        assert "A" in MECHANICAL_CODES

    def test_tax_withholding_is_not_selling(self):
        assert "F" in MECHANICAL_CODES

    def test_every_excluded_code_states_why(self):
        for reason in MECHANICAL_CODES.values():
            assert reason.strip()


class TestClusterThreshold:
    def test_one_large_seller_is_not_a_cluster(self):
        """A director selling a block is a personal decision."""
        big = activity(trade("Jeffrey R Immelt", shares=100_000.0, value=25_000_000.0))
        assert big.cluster() is None
        assert cluster_event(big) is None

    def test_exactly_at_the_threshold_is_a_cluster(self):
        trades = [trade(f"Insider {i}") for i in range(MIN_CLUSTER_INSIDERS)]
        assert activity(*trades).cluster() == "SELLING"

    def test_one_below_the_threshold_is_not(self):
        trades = [trade(f"Insider {i}") for i in range(MIN_CLUSTER_INSIDERS - 1)]
        assert activity(*trades).cluster() is None

    def test_one_person_trading_many_times_is_not_a_cluster(self):
        repeat = activity(*[trade("Aman Joshi", day=d) for d in (4, 10, 14, 18)])
        assert repeat.cluster() is None

    def test_buying_is_reported_before_selling_when_both_qualify(self):
        """Insiders sell for many reasons and buy for one."""
        mixed = activity(
            *[trade(f"Seller {i}") for i in range(4)],
            *[trade(f"Buyer {i}", code="P") for i in range(3)],
        )
        assert mixed.cluster() == "BUYING"
        assert cluster_event(mixed).kind == "INSIDER_BUYING_CLUSTER"

    def test_an_empty_window_produces_nothing(self):
        empty = activity()
        assert empty.cluster() is None
        assert cluster_event(empty) is None
        assert "no open-market insider trades" in empty.summary()


class TestDedup:
    def test_the_same_week_dedups(self):
        a, b = cluster_event(BE_REAL), cluster_event(BE_REAL)
        assert a.dedup_hash() == b.dedup_hash()

    def test_buying_and_selling_clusters_do_not_collide(self):
        buying = activity(*[trade(f"B{i}", code="P") for i in range(3)])
        assert cluster_event(buying).dedup_hash() != cluster_event(BE_REAL).dedup_hash()

    def test_a_later_week_is_a_new_event(self):
        later = activity(*[trade(f"X{i}", day=31) for i in range(3)])
        assert cluster_event(later).dedup_hash() != cluster_event(BE_REAL).dedup_hash()


class TestMarketCapContext:
    def test_the_share_of_market_cap_is_reported_when_known(self):
        event = cluster_event(BE_REAL, market_cap=89_000_000_000.0)
        assert event.payload["pct_of_cap"] == pytest.approx(0.000183, abs=1e-5)

    def test_it_is_omitted_rather_than_guessed_when_unknown(self):
        assert "pct_of_cap" not in cluster_event(BE_REAL).payload
