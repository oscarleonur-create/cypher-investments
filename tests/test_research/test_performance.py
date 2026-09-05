"""Tests for deposit-adjusted portfolio performance calculations (all pure functions)."""

from __future__ import annotations

from datetime import date

from advisor.research.performance import (
    calculate_period_return,
    get_equity_curve,
    modified_dietz,
    standard_period_returns,
)

# ── modified_dietz ────────────────────────────────────────────────────────────


def test_modified_dietz_no_flows_is_simple_return():
    r = modified_dietz(
        v_start=100.0,
        v_end=110.0,
        cash_flows=[],
        period_start=date(2026, 1, 1),
        period_end=date(2026, 1, 31),
    )
    assert r == 0.1


def test_modified_dietz_mid_period_deposit_weighted():
    # $100 start, $10k deposit at the halfway mark, $10,110 end.
    # Gain is $10 on a $100+ (10000*0.5)=5100 weighted base -> ~0.00196.
    r = modified_dietz(
        v_start=100.0,
        v_end=10_110.0,
        cash_flows=[{"date": "2026-01-16", "amount": 10_000.0}],
        period_start=date(2026, 1, 1),
        period_end=date(2026, 1, 31),
    )
    assert r is not None
    assert 0.0 < r < 0.01  # small genuine gain, not inflated by the deposit


def test_modified_dietz_zero_or_negative_v_start_returns_none():
    assert modified_dietz(0.0, 100.0, [], date(2026, 1, 1), date(2026, 1, 31)) is None
    assert modified_dietz(-50.0, 100.0, [], date(2026, 1, 1), date(2026, 1, 31)) is None


def test_modified_dietz_zero_day_period_returns_none():
    d = date(2026, 1, 1)
    assert modified_dietz(100.0, 110.0, [], d, d) is None


def test_modified_dietz_ignores_flows_outside_period():
    r_with_out_of_range_flow = modified_dietz(
        v_start=100.0,
        v_end=110.0,
        cash_flows=[{"date": "2025-06-01", "amount": 999.0}],
        period_start=date(2026, 1, 1),
        period_end=date(2026, 1, 31),
    )
    assert r_with_out_of_range_flow == 0.1  # same as the no-flows case


# ── calculate_period_return ──────────────────────────────────────────────────


def test_calculate_period_return_no_snapshot_in_range_returns_none_shape():
    result = calculate_period_return(
        "1M",
        date(2026, 1, 1),
        date(2026, 1, 31),
        snapshots=[],
        cash_flows=[],
    )
    assert result["label"] == "1M"
    assert result["return_pct"] is None
    assert result["v_start"] is None
    assert result["v_end"] is None
    assert result["cf_net"] is None


def test_calculate_period_return_actual_start_after_end_returns_none_return():
    # Only one snapshot available on/after start and on/before end, and it's
    # the same point for both -> actual_start >= actual_end.
    snapshots = [{"date": "2026-01-15", "net_liq": 100.0}]
    result = calculate_period_return(
        "1M", date(2026, 1, 1), date(2026, 1, 10), snapshots, cash_flows=[]
    )
    assert result["return_pct"] is None
    assert result["cf_net"] is None


def test_calculate_period_return_computes_over_matched_snapshots():
    snapshots = [
        {"date": "2026-01-01", "net_liq": 100.0},
        {"date": "2026-01-31", "net_liq": 110.0},
    ]
    result = calculate_period_return(
        "1M", date(2026, 1, 1), date(2026, 1, 31), snapshots, cash_flows=[]
    )
    assert result["return_pct"] == 0.1
    assert result["v_start"] == 100.0
    assert result["v_end"] == 110.0
    assert result["cf_net"] == 0.0


# ── standard_period_returns ───────────────────────────────────────────────────


def test_standard_period_returns_includes_all_standard_labels():
    results = standard_period_returns([], [], reference_date=date(2026, 6, 15))
    labels = [r["label"] for r in results]
    assert labels == ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "All"]


def test_standard_period_returns_empty_snapshots_all_entry_is_none_shape():
    results = standard_period_returns([], [], reference_date=date(2026, 6, 15))
    all_entry = results[-1]
    assert all_entry["label"] == "All"
    assert all_entry["start_date"] is None
    assert all_entry["return_pct"] is None


def test_standard_period_returns_all_entry_uses_earliest_snapshot():
    snapshots = [
        {"date": "2026-01-01", "net_liq": 100.0},
        {"date": "2026-06-15", "net_liq": 120.0},
    ]
    results = standard_period_returns(snapshots, [], reference_date=date(2026, 6, 15))
    all_entry = results[-1]
    assert all_entry["start_date"] == "2026-01-01"
    assert all_entry["return_pct"] == 0.2


# ── get_equity_curve ──────────────────────────────────────────────────────────


def test_get_equity_curve_fewer_than_two_snapshots_is_flat():
    curve = get_equity_curve([{"date": "2026-01-01", "net_liq": 100.0}], [])
    assert curve == [
        {"date": "2026-01-01", "value": 1.0, "is_deposit": False, "deposit_amount": None}
    ]


def test_get_equity_curve_empty_snapshots():
    assert get_equity_curve([], []) == []


def test_get_equity_curve_compounds_across_sub_periods():
    snapshots = [
        {"date": "2026-01-01", "net_liq": 100.0},
        {"date": "2026-01-15", "net_liq": 110.0},
        {"date": "2026-01-31", "net_liq": 121.0},
    ]
    curve = get_equity_curve(snapshots, [])

    assert len(curve) == 3
    assert curve[0]["value"] == 1.0
    assert curve[0]["is_deposit"] is False
    # Two 10% legs compound to 21%, not 20% -> value ends at 1.21.
    assert round(curve[-1]["value"], 4) == 1.21


def test_get_equity_curve_flags_deposit_dates():
    snapshots = [
        {"date": "2026-01-01", "net_liq": 100.0},
        {"date": "2026-01-15", "net_liq": 10_100.0},  # jumps due to a deposit
    ]
    cash_flows = [{"date": "2026-01-15", "amount": 10_000.0}]

    curve = get_equity_curve(snapshots, cash_flows)

    assert curve[1]["is_deposit"] is True
    assert curve[1]["deposit_amount"] == 10_000.0
    # The deposit itself shouldn't be counted as investment gain.
    assert curve[1]["value"] < 1.1
