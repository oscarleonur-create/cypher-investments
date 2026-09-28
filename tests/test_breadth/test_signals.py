"""Families P and F and the records they make: boundaries, windows, and live = replay."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from advisor.breadth import signals as S
from advisor.breadth import universe as U
from advisor.breadth.panel import Panel, build_panel
from advisor.valuation.history import Point

N = 400
DAYS = pd.bdate_range("2025-01-01", periods=N)


def flat_panel(symbols, price=50.0, volume=1e6):
    close = pd.DataFrame(price, index=DAYS, columns=symbols)
    return Panel(close.copy(), close.copy(), pd.DataFrame(volume, index=DAYS, columns=symbols))


def all_eligible(panel):
    return pd.DataFrame(True, index=panel.close.index, columns=panel.close.columns)


# ── E0 over a panel equals E0 per name ─────────────────────────────────────


def test_eligibility_panel_matches_the_scalar_rule_on_random_data():
    rng = np.random.default_rng(7)
    symbols = [f"S{i}" for i in range(12)]
    close = pd.DataFrame(rng.uniform(3, 9, (N, 12)), index=DAYS, columns=symbols)
    volume = pd.DataFrame(rng.choice([0.0, 1e6, 3e6, np.nan], (N, 12)), index=DAYS, columns=symbols)
    # Holes: late listings, a halt, a name that stops trading.
    close.iloc[:150, 0] = np.nan
    close.iloc[200:215, 1] = np.nan
    close.iloc[300:, 2] = np.nan
    close.iloc[:, 3] = 5.0  # exactly at the price floor
    volume.iloc[:, 3] = U.MIN_DOLLAR_VOLUME / 5.0  # exactly at the dollar floor
    panel = U.eligibility_panel(close, volume)
    for j, s in enumerate(symbols):
        bars = [
            (d.date(), close.iat[i, j], None if np.isnan(volume.iat[i, j]) else volume.iat[i, j])
            for i, d in enumerate(DAYS)
            if not np.isnan(close.iat[i, j])
        ]
        for i in range(0, N, 7):
            day = DAYS[i].date()
            assert panel.iat[i, j] == U.eligibility(bars, day).eligible, (s, day)


# ── P ──────────────────────────────────────────────────────────────────────


def test_near_high_is_inclusive_at_the_boundary():
    p = flat_panel(["A", "B"])
    p.high.iloc[:, :] = 100.0
    p.close["A"] = 98.0  # exactly 2% under the high
    p.close["B"] = 97.99
    st = S.price_states(p, all_eligible(p))
    assert st["high"]["A"].iloc[-1] and not st["high"]["B"].iloc[-1]
    assert not st["high"]["A"].iloc[S.HIGH_WINDOW - 2]  # a 252-session high needs 252 sessions


def test_breakout_needs_a_big_move_on_heavy_volume():
    # 49.5, 50.5, 49.5, ... : about 1.4% daily sigma; yesterday closed at 50.5.
    base = np.where(np.arange(N) % 2 == 0, 49.5, 50.5)
    base[-2] = 50.5
    cases = {}
    for name, last, vol_mult in (
        ("ok", 55.0, 4.0),  # +8.9%, far beyond 2 sigma, above the 50-session high
        ("small", 51.0, 4.0),  # above the high, but +1% is inside 2 sigma
        ("quiet", 55.0, 1.0),  # the move without the volume
    ):
        c = base.astype(float).copy()
        c[-1] = last
        v = np.full(N, 1e6)
        v[-1] = 1e6 * vol_mult
        cases[name] = (c, v)
    close = pd.DataFrame({k: v[0] for k, v in cases.items()}, index=DAYS)
    volume = pd.DataFrame({k: v[1] for k, v in cases.items()}, index=DAYS)
    p = Panel(close, close.copy(), volume)
    st = S.price_states(p, all_eligible(p))["breakout"].iloc[-1]
    assert st["ok"] and not st["small"] and not st["quiet"]


def _trending(n):
    symbols = [f"S{i:02d}" for i in range(n)]
    close = pd.DataFrame({s: np.linspace(10, 10 + i, N) for i, s in enumerate(symbols)}, index=DAYS)
    return Panel(close, close.copy(), pd.DataFrame(1e6, index=DAYS, columns=symbols))


def test_momentum_is_the_top_decile_among_the_eligible_only():
    p = _trending(S.MIN_CROSS_SECTION)
    elig = all_eligible(p)
    top = S.price_states(p, elig)["momentum"].iloc[-1]
    assert set(top[top].index) == {"S45", "S46", "S47", "S48", "S49"}  # 10% of 50
    elig["S49"] = False
    top = S.price_states(p, elig)["momentum"].iloc[-1]
    assert "S49" not in set(top[top].index)  # no longer competes or signals


def test_momentum_needs_a_cross_section():
    p = _trending(S.MIN_CROSS_SECTION - 1)
    assert not S.price_states(p, all_eligible(p))["momentum"].to_numpy().any()


def test_nothing_signals_when_not_eligible():
    p = flat_panel(["A"])
    p.high.iloc[:, :] = 50.0
    st = S.price_states(p, pd.DataFrame(False, index=DAYS, columns=["A"]))
    assert not any(v.to_numpy().any() for v in st.values())


# ── records: P edges and cooldown ─────────────────────────────────────────


def _panel_with_p(on_rows, symbol="A"):
    """A panel whose only P state is 'near high' on the given rows."""
    p = flat_panel([symbol])
    p.close[symbol] = 50.0
    p.high[symbol] = 60.0
    for r in on_rows:
        p.close.iloc[r, 0] = 60.0
    return p


def test_p_is_recorded_when_it_turns_on_then_cools_down():
    on = list(range(300, 305)) + list(range(330, 332)) + list(range(370, 372))
    p = _panel_with_p(on)
    recs = S.records(p, all_eligible(p), [], 260, N - 1)
    assert [r["row"] for r in recs if r["grp"] == "P"] == [300, 370]  # 330 is inside the cooldown


def test_records_outside_the_range_are_not_returned_but_still_cool_down():
    p = _panel_with_p([300, 330])
    recs = S.records(p, all_eligible(p), [], 320, N - 1)
    assert [r["row"] for r in recs if r["grp"] == "P"] == []  # 330 cooled by 300 before the range


# ── F ──────────────────────────────────────────────────────────────────────


def q(end: str, value: float, known: str | None = None) -> Point:
    e = date.fromisoformat(end)
    return Point(
        end=e, value=value, known=date.fromisoformat(known) if known else e + timedelta(45)
    )


def quarters(latest: float, before: float, year_ago: float, year_ago_before: float, known=None):
    return {
        (2025, 1): q("2025-03-31", year_ago_before),
        (2025, 2): q("2025-06-30", year_ago),
        (2026, 1): q("2026-03-31", before),
        (2026, 2): q("2026-06-30", latest, known),
        (2025, 3): q("2025-09-30", 1.0),
        (2025, 4): q("2025-12-31", 1.0),
    }


SESSIONS = pd.bdate_range("2026-01-01", "2026-12-31")


@pytest.mark.parametrize(
    "latest,before,year_ago,year_ago_before,fires",
    [
        (110e6, 100e6, 100e6, 100e6, True),  # +10% vs 0%: both exactly at the thresholds
        (109.9e6, 100e6, 100e6, 100e6, False),  # growth just under 10%
        (130e6, 124.9e6, 100e6, 100e6, True),  # +30% vs +24.9%: acceleration 5.1 pts
        (130e6, 126e6, 100e6, 100e6, False),  # +30% vs +26%: only 4 pts faster
        (9e6, 5e6, 5e6, 5e6, False),  # below the revenue floor
        (110e6, 100e6, 0.0, 100e6, False),  # no year-ago base
        (110e6, 100e6, -5.0, 100e6, False),  # negative base (a restatement artefact)
    ],
)
def test_f_thresholds(latest, before, year_ago, year_ago_before, fires):
    ev = S.fundamental_events({"A": quarters(latest, before, year_ago, year_ago_before)}, SESSIONS)
    assert bool([e for e in ev if e.quarter == "2026Q2"]) is fires


def test_f_needs_every_quarter_it_compares():
    qs = quarters(130e6, 100e6, 100e6, 100e6)
    del qs[(2025, 1)]
    assert S.fundamental_events({"A": qs}, SESSIONS) == []


def test_f_lands_on_the_first_session_at_or_after_it_became_known():
    qs = quarters(130e6, 100e6, 100e6, 100e6, known="2026-08-08")  # a Saturday
    (ev,) = S.fundamental_events({"A": qs}, SESSIONS)
    assert SESSIONS[ev.row].date() == date(2026, 8, 10)
    qs = quarters(130e6, 100e6, 100e6, 100e6, known="2027-02-01")  # after the panel
    assert S.fundamental_events({"A": qs}, SESSIONS) == []


# ── F+P ───────────────────────────────────────────────────────────────────


def _fp(p_rows, f_row, eligible_at_f=True):
    p = _panel_with_p(p_rows)
    elig = all_eligible(p)
    if not eligible_at_f:
        elig.iloc[f_row : f_row + S.CONVERGE_SESSIONS + 1, 0] = False
    ev = S.FEvent("A", f_row, "2026Q2", 1e8, 0.3, 0.1)
    return [r for r in S.records(p, elig, [ev], 260, N - 1) if r["grp"] in ("F", "F+P")]


def test_p_in_the_month_before_f_converges_on_the_f_session():
    recs = _fp([340], 350)
    assert [(r["grp"], r["row"]) for r in recs] == [("F", 350), ("F+P", 350)]
    fp = recs[1]["detail"]
    assert fp["high_in_window"] and fp["f_row_lag"] == 0


def test_p_after_f_converges_when_it_arrives():
    recs = _fp([360], 350)
    assert ("F+P", 360) in [(r["grp"], r["row"]) for r in recs]


def test_p_too_long_before_f_does_not_converge():
    recs = _fp([350 - S.CONVERGE_SESSIONS], 350)
    assert [r["grp"] for r in recs] == ["F"]


def test_an_ineligible_name_records_nothing():
    assert _fp([340], 350, eligible_at_f=False) == []


# ── live and replay agree ─────────────────────────────────────────────────


def test_a_single_session_run_equals_the_replay_on_that_session():
    rng = np.random.default_rng(3)
    symbols = [f"S{i:02d}" for i in range(30)]
    close = pd.DataFrame(50 * np.cumprod(1 + rng.normal(0.0005, 0.02, (N, 30)), axis=0),
                         index=DAYS, columns=symbols)  # fmt: skip
    volume = pd.DataFrame(rng.uniform(5e5, 3e6, (N, 30)), index=DAYS, columns=symbols)
    p = Panel(close, close * 1.01, volume)
    elig = all_eligible(p)
    events = [S.FEvent("S05", 330, "2026Q2", 1e8, 0.3, 0.1)]
    full = S.records(p, elig, events, 260, N - 1)
    for row in range(300, N, 9):
        one = S.records(p, elig, events, row, row)
        assert sorted((r["grp"], r["symbol"]) for r in one) == sorted(
            (r["grp"], r["symbol"]) for r in full if r["row"] == row
        )


def test_share_classes_count_once():
    p = flat_panel(["GOOG", "GOOGL", "X"])
    p.volume["GOOGL"] = 2e6
    out = S.dedupe_share_classes(all_eligible(p), p, {"GOOG": 1, "GOOGL": 1, "X": 2})
    assert not out["GOOG"].any() and out["GOOGL"].all() and out["X"].all()


def test_empty_panel():
    p = build_panel([])
    assert S.fundamental_events({"A": quarters(1, 1, 1, 1)}, p.sessions) == []
