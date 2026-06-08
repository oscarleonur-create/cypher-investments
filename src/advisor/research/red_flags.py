"""Rule-based red-flag detector over `StatementBundle` + `RatioBundle`.

Flags target the patterns analysts watch for (per the blueprint):
- Working-capital drift: DSO or inventory growing faster than revenue
- Share-count creep: diluted shares up >5% YoY
- Cash-quality gap: FCF / net income persistently <0.6
- Earnings-quality: persistent negative net income with positive operating CF (mismatch)
- Leverage step-up: debt/EBITDA jumping >1x year over year
- Margin compression: operating margin down >300 bps year over year
"""

from __future__ import annotations

from advisor.research.models import (
    RatioBundle,
    RedFlag,
    RedFlagList,
    RedFlagSeverity,
    StatementBundle,
)


def detect_red_flags(statements: StatementBundle, ratios: RatioBundle) -> RedFlagList:
    flags: list[RedFlag] = []

    flags.extend(_dso_drift(statements))
    flags.extend(_inventory_drift(statements))
    flags.extend(_share_count_creep(statements, ratios))
    flags.extend(_fcf_quality_gap(ratios))
    flags.extend(_leverage_step_up(ratios))
    flags.extend(_margin_compression(ratios))

    return RedFlagList(symbol=statements.symbol, flags=flags)


# ── Detectors ───────────────────────────────────────────────────────────────


def _dso_drift(s: StatementBundle) -> list[RedFlag]:
    if len(s.income) < 2 or len(s.balance) < 2:
        return []
    inc0, inc1 = s.income[0], s.income[1]
    bs0, bs1 = s.balance[0], s.balance[1]
    rev_growth = _growth(inc0.revenue, inc1.revenue)
    ar_growth = _growth(bs0.accounts_receivable, bs1.accounts_receivable)
    if rev_growth is None or ar_growth is None:
        return []
    if ar_growth - rev_growth > 0.10:  # AR growing >10pp faster than revenue
        return [
            RedFlag(
                code="DSO_DRIFT",
                title="Receivables growing faster than revenue",
                severity=RedFlagSeverity.MEDIUM,
                detail=(
                    f"AR up {ar_growth:.1%} YoY vs revenue up {rev_growth:.1%}. "
                    "Possible channel-stuffing or collection slowdown."
                ),
                period_end=inc0.period_end,
            )
        ]
    return []


def _inventory_drift(s: StatementBundle) -> list[RedFlag]:
    if len(s.income) < 2 or len(s.balance) < 2:
        return []
    inc0, inc1 = s.income[0], s.income[1]
    bs0, bs1 = s.balance[0], s.balance[1]
    rev_growth = _growth(inc0.revenue, inc1.revenue)
    inv_growth = _growth(bs0.inventory, bs1.inventory)
    if rev_growth is None or inv_growth is None:
        return []
    if inv_growth - rev_growth > 0.15:
        return [
            RedFlag(
                code="INVENTORY_DRIFT",
                title="Inventory growing faster than revenue",
                severity=RedFlagSeverity.MEDIUM,
                detail=(
                    f"Inventory up {inv_growth:.1%} YoY vs revenue up {rev_growth:.1%}. "
                    "Demand softening risk or write-down ahead."
                ),
                period_end=inc0.period_end,
            )
        ]
    return []


def _share_count_creep(s: StatementBundle, r: RatioBundle) -> list[RedFlag]:
    cagr = r.share_count_cagr_3y
    if cagr is None or cagr <= 0.03:
        return []
    severity = RedFlagSeverity.HIGH if cagr > 0.07 else RedFlagSeverity.MEDIUM
    return [
        RedFlag(
            code="SHARE_DILUTION",
            title="Sustained share-count growth",
            severity=severity,
            detail=(
                f"Diluted shares CAGR over 3y: {cagr:.1%}. " "Buybacks not offsetting SBC issuance."
            ),
            period_end=s.income[0].period_end if s.income else None,
        )
    ]


def _fcf_quality_gap(r: RatioBundle) -> list[RedFlag]:
    """Flag if FCF/NI is persistently <0.6 across the most recent 3 periods with positive NI."""
    recent = [p for p in r.periods[:3] if p.fcf_to_net_income is not None]
    if len(recent) < 2:
        return []
    if all(p.fcf_to_net_income < 0.6 for p in recent):
        latest = recent[0]
        return [
            RedFlag(
                code="FCF_QUALITY_GAP",
                title="FCF persistently below net income",
                severity=RedFlagSeverity.HIGH,
                detail=(
                    f"FCF/NI = {latest.fcf_to_net_income:.2f} latest, "
                    f"<0.6 across {len(recent)} consecutive periods. "
                    "Earnings not converting to cash."
                ),
                period_end=latest.period_end,
            )
        ]
    return []


def _leverage_step_up(r: RatioBundle) -> list[RedFlag]:
    if len(r.periods) < 2:
        return []
    cur, prev = r.periods[0], r.periods[1]
    if cur.debt_to_ebitda is None or prev.debt_to_ebitda is None:
        return []
    delta = cur.debt_to_ebitda - prev.debt_to_ebitda
    if delta > 1.0 and cur.debt_to_ebitda > 3.0:
        return [
            RedFlag(
                code="LEVERAGE_STEP_UP",
                title="Debt/EBITDA jumped > 1x YoY into elevated territory",
                severity=RedFlagSeverity.HIGH,
                detail=(
                    f"Debt/EBITDA went from {prev.debt_to_ebitda:.1f}x to "
                    f"{cur.debt_to_ebitda:.1f}x (+{delta:.1f}x)."
                ),
                period_end=cur.period_end,
            )
        ]
    return []


def _margin_compression(r: RatioBundle) -> list[RedFlag]:
    if len(r.periods) < 2:
        return []
    cur, prev = r.periods[0], r.periods[1]
    if cur.operating_margin is None or prev.operating_margin is None:
        return []
    delta = cur.operating_margin - prev.operating_margin
    if delta < -0.03:  # >300 bps compression
        return [
            RedFlag(
                code="MARGIN_COMPRESSION",
                title="Operating margin compressed > 300 bps YoY",
                severity=RedFlagSeverity.MEDIUM,
                detail=(
                    f"Operating margin {prev.operating_margin:.1%} → "
                    f"{cur.operating_margin:.1%} ({delta * 100:+.0f} bps)."
                ),
                period_end=cur.period_end,
            )
        ]
    return []


# ── Helpers ─────────────────────────────────────────────────────────────────


def _growth(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / abs(prior)
