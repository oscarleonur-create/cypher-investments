"""Deposit-adjusted portfolio performance calculations.

All functions are pure — they accept pre-loaded lists of dicts and return
plain dicts/lists with no database or network calls.

Snapshots shape:  [{"date": "YYYY-MM-DD", "net_liq": float}, ...]  ascending
Cash flows shape: [{"date": "YYYY-MM-DD", "amount": float}, ...]   ascending
  amount > 0 = deposit, amount < 0 = withdrawal
"""

from __future__ import annotations

from datetime import date, timedelta

# ── Modified Dietz ────────────────────────────────────────────────────────────


def modified_dietz(
    v_start: float,
    v_end: float,
    cash_flows: list[dict],
    period_start: date,
    period_end: date,
) -> float | None:
    """Return the Modified Dietz return for a single sub-period.

    R = (V_end - V_start - CF_net) / (V_start + Σ(CF_i × W_i))
    W_i = days remaining in period after flow_i / total period days
    """
    if v_start <= 0:
        return None
    total_days = (period_end - period_start).days
    if total_days <= 0:
        return None

    cf_net = 0.0
    weighted_cf = 0.0
    for cf in cash_flows:
        try:
            flow_date = date.fromisoformat(cf["date"])
        except (ValueError, KeyError):
            continue
        if not (period_start <= flow_date <= period_end):
            continue
        amount = float(cf["amount"])
        cf_net += amount
        days_remaining = (period_end - flow_date).days
        weighted_cf += amount * (days_remaining / total_days)

    denominator = v_start + weighted_cf
    if denominator == 0:
        return None
    return (v_end - v_start - cf_net) / denominator


# ── Period return calculation ─────────────────────────────────────────────────


def _nearest_snapshot(snapshots: list[dict], target: date, before: bool = True) -> dict | None:
    """Return the snapshot on or before (before=True) / on or after (before=False) target."""
    if not snapshots:
        return None
    if before:
        best = None
        for s in snapshots:
            try:
                d = date.fromisoformat(s["date"])
            except (ValueError, KeyError):
                continue
            if d <= target:
                best = s
        return best
    else:
        for s in snapshots:
            try:
                d = date.fromisoformat(s["date"])
            except (ValueError, KeyError):
                continue
            if d >= target:
                return s
        return None


def calculate_period_return(
    label: str,
    start_date: date,
    end_date: date,
    snapshots: list[dict],
    cash_flows: list[dict],
) -> dict:
    snap_start = _nearest_snapshot(snapshots, start_date, before=False)
    snap_end = _nearest_snapshot(snapshots, end_date, before=True)

    if snap_start is None or snap_end is None:
        return {
            "label": label,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "return_pct": None,
            "v_start": None,
            "v_end": None,
            "cf_net": None,
        }

    actual_start = date.fromisoformat(snap_start["date"])
    actual_end = date.fromisoformat(snap_end["date"])

    if actual_start >= actual_end:
        return {
            "label": label,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "return_pct": None,
            "v_start": snap_start["net_liq"],
            "v_end": snap_end["net_liq"],
            "cf_net": None,
        }

    period_flows = [
        cf for cf in cash_flows if actual_start < date.fromisoformat(cf["date"]) <= actual_end
    ]
    cf_net = sum(float(cf["amount"]) for cf in period_flows)

    r = modified_dietz(
        snap_start["net_liq"],
        snap_end["net_liq"],
        period_flows,
        actual_start,
        actual_end,
    )

    return {
        "label": label,
        "start_date": actual_start.isoformat(),
        "end_date": actual_end.isoformat(),
        "return_pct": r,
        "v_start": snap_start["net_liq"],
        "v_end": snap_end["net_liq"],
        "cf_net": cf_net,
    }


def standard_period_returns(
    snapshots: list[dict],
    cash_flows: list[dict],
    reference_date: date | None = None,
) -> list[dict]:
    today = reference_date or date.today()
    periods = [
        ("1D", today - timedelta(days=1)),
        ("1W", today - timedelta(weeks=1)),
        ("1M", today - timedelta(days=30)),
        ("3M", today - timedelta(days=91)),
        ("6M", today - timedelta(days=182)),
        ("YTD", date(today.year, 1, 1)),
        ("1Y", today - timedelta(days=365)),
    ]
    results = [
        calculate_period_return(label, start, today, snapshots, cash_flows)
        for label, start in periods
    ]

    # "All" — from earliest snapshot to today
    if snapshots:
        earliest = date.fromisoformat(snapshots[0]["date"])
        results.append(calculate_period_return("All", earliest, today, snapshots, cash_flows))
    else:
        results.append(
            {
                "label": "All",
                "start_date": None,
                "end_date": today.isoformat(),
                "return_pct": None,
                "v_start": None,
                "v_end": None,
                "cf_net": None,
            }
        )

    return results


# ── Equity curve (chain-linked TWR) ───────────────────────────────────────────


def get_equity_curve(
    snapshots: list[dict],
    cash_flows: list[dict],
) -> list[dict]:
    """Chain-link Modified Dietz sub-periods broken at each cash flow date.

    Returns one point per snapshot date with an index level starting at 1.0.
    Deposit dates are flagged so the UI can annotate them.
    """
    if len(snapshots) < 2:
        return [
            {"date": s["date"], "value": 1.0, "is_deposit": False, "deposit_amount": None}
            for s in snapshots
        ]

    # Build a set of dates where cash flows occur (for annotation)
    flow_by_date: dict[str, float] = {}
    for cf in cash_flows:
        d = cf.get("date", "")
        flow_by_date[d] = flow_by_date.get(d, 0.0) + float(cf["amount"])

    curve: list[dict] = []
    index_level = 1.0
    prev_snap = snapshots[0]

    curve.append(
        {
            "date": prev_snap["date"],
            "value": round(index_level, 6),
            "is_deposit": prev_snap["date"] in flow_by_date,
            "deposit_amount": flow_by_date.get(prev_snap["date"]),
        }
    )

    for snap in snapshots[1:]:
        period_start = date.fromisoformat(prev_snap["date"])
        period_end = date.fromisoformat(snap["date"])

        # Cash flows strictly inside (period_start, period_end]
        period_flows = [
            cf for cf in cash_flows if period_start < date.fromisoformat(cf["date"]) <= period_end
        ]

        r = modified_dietz(
            prev_snap["net_liq"],
            snap["net_liq"],
            period_flows,
            period_start,
            period_end,
        )
        if r is not None:
            index_level *= 1.0 + r

        deposit_amount = flow_by_date.get(snap["date"])
        curve.append(
            {
                "date": snap["date"],
                "value": round(index_level, 6),
                "is_deposit": snap["date"] in flow_by_date,
                "deposit_amount": deposit_amount,
            }
        )
        prev_snap = snap

    return curve
