"""One scan: discover movers, apply the two rules, attach catalysts, record.

Sources are injected so tests can drive every failure mode without a network,
and so a live run and a test run exercise exactly the same logic.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from advisor.daemon import market_calendar as mc
from advisor.scanner import detect as rules
from advisor.scanner import sources
from advisor.scanner.models import Candidate, CatalystItem, Mover, Setup
from advisor.scanner.store import ScannerStore

logger = logging.getLogger(__name__)

# News lookups per session. Unlimited by the user's decision (2026-09-23):
# every candidate gets its news checked, whatever it costs in Tavily credits.
# A cap can still be passed; beyond it a candidate is recorded as "news not
# checked", which the report keeps apart from "no news".
DEFAULT_NEWS_BUDGET: int | None = None


@dataclass
class ScanResult:
    ran: bool
    reason: str = ""
    movers_seen: int = 0
    new: list[Candidate] = field(default_factory=list)
    already_recorded: int = 0
    news_lookups: int = 0
    news_empty: int = 0
    over_budget: int = 0
    source_error: str | None = None

    def summary(self) -> str:
        if not self.ran:
            return f"skipped: {self.reason}"
        by = {s: sum(1 for c in self.new if c.setup is s) for s in Setup}
        text = (
            f"{self.movers_seen} movers; new A={by[Setup.CATALYST_GAP]} "
            f"C={by[Setup.NEWS_DIP]}; {self.already_recorded} already recorded; "
            f"{self.news_lookups} news lookups ({self.news_empty} empty)"
        )
        if self.over_budget:
            text += f"; {self.over_budget} over news budget"
        if self.source_error:
            text += f"; source error: {self.source_error}"
        return text


def catalyst_window_start(now: datetime) -> datetime:
    """News counts as fresh from the previous session's close onward.

    An after-hours earnings release at 16:05 yesterday is today's catalyst;
    yesterday's 11:00 headline already had a whole session to be priced.
    """
    prev = mc.previous_trading_day(mc.to_et(now).date())
    close = mc.session_close(prev)
    return datetime.combine(prev, close, tzinfo=mc.MARKET_TZ)


def run_scan(
    store: ScannerStore,
    now: datetime,
    *,
    fetch_movers: Callable[[], tuple[list[Mover], str | None]] = sources.screen_movers,
    fetch_sigma: Callable[[list[str]], dict[str, float]] = sources.daily_sigma,
    fetch_catalysts: Callable[
        [str, str | None, datetime], tuple[list[CatalystItem], bool]
    ] = sources.find_catalysts,
    thresholds: rules.Thresholds = rules.DEFAULT,
    news_budget: int | None = DEFAULT_NEWS_BUDGET,
    check_news: bool = True,
) -> ScanResult:
    now = mc.to_et(now)
    if not mc.is_market_open(now):
        return ScanResult(ran=False, reason="market closed")
    if now.time() < rules.FIRST_SCAN:
        return ScanResult(ran=False, reason=f"before {rules.FIRST_SCAN.strftime('%H:%M')} ET")

    session = now.date()
    movers, error = fetch_movers()
    result = ScanResult(ran=True, movers_seen=len(movers), source_error=error)
    if not movers:
        return result

    # Volatility only for names that could be a setup-C dip; it is a batch
    # download and the gainers never need it to qualify.
    dip_names = [m.symbol for m in movers if rules.is_news_dip(m, None, thresholds)[0]]
    gap_names = [m.symbol for m in movers if rules.is_catalyst_gap(m, now, thresholds)]
    sigma = fetch_sigma(sorted(set(dip_names) | set(gap_names))) if (dip_names or gap_names) else {}

    since = catalyst_window_start(now)
    spent = store.news_checked_count(session)
    for m in movers:
        for setup, z in rules.detect(m, now, sigma.get(m.symbol), thresholds):
            if store.exists(session, setup, m.symbol):
                result.already_recorded += 1
                continue
            cand = Candidate(
                session=session,
                setup=setup,
                symbol=m.symbol,
                detected_at=now,
                name=m.name,
                price=m.price,
                prev_close=m.prev_close,
                open=m.open,
                change=rules.change(m),
                gap=rules.gap(m),
                rvol=rules.relative_volume(m, now),
                sigma=z,
                market_cap=m.market_cap,
            )
            if check_news and (news_budget is None or spent < news_budget):
                items, checked = fetch_catalysts(m.symbol, m.name, since)
                cand.catalysts, cand.news_checked = items, checked
                spent += 1
                result.news_lookups += 1
                result.news_empty += 0 if items else 1
            elif check_news:
                result.over_budget += 1
            if store.add(cand):
                result.new.append(cand)
            else:
                result.already_recorded += 1
    return result
