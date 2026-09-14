"""Insider transactions, as a pattern rather than a stream of filings.

A single Form 4 is noise, and the existing code said so by excluding Form 4
entirely — a correct diagnosis with the wrong remedy. Bloom Energy files
almost nothing else: in a month with fourteen filings, every one was a Form 4,
a Form 144 or a Schedule 13G amendment, and the system saw zero.

What it missed was a pattern. On 18 August four officers sold on the open
market on the same day — the Chief Commercial Officer, the Chief Operations
Officer, the Chief Legal Officer and a director — with a second director
selling twice more that month. Six insiders, about $19.8m, no purchases.

**Transaction codes decide everything.** The CEO's filing in the same window
was a 300,000-share gift and an option exercise; reading that as selling would
be flatly wrong. Only open-market purchases (P) and sales (S) express a view.
Grants, exercises, gifts and tax withholding are mechanical and are ignored.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# SEC Form 4 transaction codes that represent a decision to trade.
OPEN_MARKET_BUY = "P"
OPEN_MARKET_SELL = "S"

# Codes deliberately ignored, and why. Kept explicit so the exclusion is a
# stated judgement rather than an accident.
MECHANICAL_CODES = {
    "A": "grant or award — compensation, not a view",
    "M": "option exercise — mechanical",
    "G": "gift — no consideration",
    "F": "shares withheld for tax",
    "C": "conversion of a derivative",
    "D": "disposition to the issuer",
}

DEFAULT_WINDOW_DAYS = 45

# A cluster is several people acting the same way, not one person acting
# largely. One director selling a big block is a personal decision; four
# officers selling in a week is a pattern.
MIN_CLUSTER_INSIDERS = 3


@dataclass(frozen=True)
class InsiderTrade:
    filed: date
    insider: str
    position: str
    code: str
    shares: float
    value: float

    @property
    def is_buy(self) -> bool:
        return self.code == OPEN_MARKET_BUY


@dataclass
class InsiderActivity:
    """Open-market insider trading for one symbol over a window."""

    symbol: str
    window_days: int
    trades: list[InsiderTrade] = field(default_factory=list)

    @property
    def sells(self) -> list[InsiderTrade]:
        return [t for t in self.trades if not t.is_buy]

    @property
    def buys(self) -> list[InsiderTrade]:
        return [t for t in self.trades if t.is_buy]

    @property
    def selling_insiders(self) -> set[str]:
        return {t.insider for t in self.sells}

    @property
    def buying_insiders(self) -> set[str]:
        return {t.insider for t in self.buys}

    @property
    def net_value(self) -> float:
        """Buys minus sells, in dollars. Negative means net selling."""
        return sum(t.value for t in self.buys) - sum(t.value for t in self.sells)

    def cluster(self) -> str | None:
        """'SELLING', 'BUYING', or None when no side has enough people.

        Buying is the rarer and more informative signal, so it is tested
        first: insiders sell for many reasons and buy for one.
        """
        if len(self.buying_insiders) >= MIN_CLUSTER_INSIDERS:
            return "BUYING"
        if len(self.selling_insiders) >= MIN_CLUSTER_INSIDERS:
            return "SELLING"
        return None

    def summary(self) -> str:
        if not self.trades:
            return "no open-market insider trades in the window"
        parts = []
        if self.buys:
            parts.append(f"{len(self.buying_insiders)} buying")
        if self.sells:
            parts.append(f"{len(self.selling_insiders)} selling")
        return f"{', '.join(parts)} over {self.window_days} days, " f"net ${self.net_value:,.0f}"


def _trades_from_filing(filing) -> list[InsiderTrade]:
    """Open-market trades in one Form 4, ignoring mechanical transactions."""
    try:
        summary = filing.obj().get_ownership_summary()
    except Exception as exc:  # noqa: BLE001
        logger.debug("insider: could not parse %s: %s", filing.accession_no, exc)
        return []

    filed = filing.filing_date
    if isinstance(filed, str):
        filed = date.fromisoformat(filed[:10])

    out = []
    for transaction in getattr(summary, "transactions", []) or []:
        if transaction.code not in (OPEN_MARKET_BUY, OPEN_MARKET_SELL):
            continue
        out.append(
            InsiderTrade(
                filed=filed,
                insider=str(summary.insider_name or "unknown"),
                position=str(getattr(summary, "position", "") or ""),
                code=transaction.code,
                shares=float(transaction.shares or 0),
                value=float(transaction.value or 0),
            )
        )
    return out


def recent_activity(
    symbol: str,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    limit: int = 40,
    today: date | None = None,
) -> InsiderActivity:
    """Open-market insider trading for ``symbol``. Empty on any failure."""
    from advisor.news.edgar import company_for

    activity = InsiderActivity(symbol=symbol.upper(), window_days=window_days)
    company = company_for(symbol)
    if company is None:
        return activity

    cutoff = (today or date.today()) - timedelta(days=window_days)
    try:
        filings = company.get_filings(form="4").head(limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("insider: filing lookup failed for %s: %s", symbol, exc)
        return activity

    for filing in filings:
        filed = filing.filing_date
        if isinstance(filed, str):
            filed = date.fromisoformat(filed[:10])
        if filed < cutoff:
            break  # filings come newest first
        activity.trades.extend(_trades_from_filing(filing))

    return activity


def cluster_event(activity: InsiderActivity, market_cap: float | None = None):
    """A Tier B event when several insiders act the same way.

    Tier B, never A: insider behaviour is a standing condition that develops
    over weeks, not an action with a deadline. It belongs in a digest.
    """
    from advisor.daemon.models import Event, EventSource, EventTier

    side = activity.cluster()
    if side is None:
        return None

    insiders = activity.buying_insiders if side == "BUYING" else activity.selling_insiders
    trades = activity.buys if side == "BUYING" else activity.sells
    total = sum(t.value for t in trades)
    latest = max(t.filed for t in trades)

    payload = {
        "side": side,
        "insiders": sorted(insiders),
        "insider_count": len(insiders),
        "trade_count": len(trades),
        "total_value": round(total, 2),
        "net_value": round(activity.net_value, 2),
        "window_days": activity.window_days,
        "latest_filed": latest.isoformat(),
        "positions": sorted({t.position for t in trades if t.position}),
    }
    if market_cap:
        payload["market_cap"] = market_cap
        payload["pct_of_cap"] = round(total / market_cap, 6)

    return Event(
        source=EventSource.EDGAR,
        kind=f"INSIDER_{side}_CLUSTER",
        tier=EventTier.B,
        symbol=activity.symbol,
        # One cluster per side per week: the pattern develops slowly and a
        # daily reminder of the same six people would be noise.
        dedup_key=f"{side}:{latest.isocalendar().year}-W{latest.isocalendar().week:02d}",
        payload=payload,
    )


def by_insider(activity: InsiderActivity) -> dict[str, float]:
    """Net dollar value per insider, for showing who did what."""
    totals: dict[str, float] = defaultdict(float)
    for trade in activity.trades:
        totals[trade.insider] += trade.value if trade.is_buy else -trade.value
    return dict(totals)
