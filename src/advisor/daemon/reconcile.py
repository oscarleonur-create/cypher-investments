"""Cross-source reconciliation: checking the inputs before trusting the output.

Every finding in this repo's data-quality audit came from comparing two
sources rather than from reading code. Prices agreed across broker and
yfinance on all eleven positions; the news age filter turned out to be dead;
yfinance reported AMD's ex-dividend date as 1995-04-26 for a company that pays
no dividend; TastyTrade and yfinance disagreed on every earnings date because
one reports the *last* and the other the *next*.

None of that is discoverable from one source. So the checks live here, they
run on a schedule, and a failure becomes an event — the advisor's job includes
telling you when it can no longer trust its own inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from advisor.daemon.book import BookSnapshot
from advisor.daemon.market_calendar import previous_trading_day
from advisor.daemon.models import Event, EventSource, EventTier

logger = logging.getLogger(__name__)

# Two independent price sources should agree closely. Anything past this is
# either a corporate action one side has applied and the other has not, or a
# symbol mapping error — both worth knowing about before sizing a trade.
PRICE_TOLERANCE_PCT = 0.01

# Bars older than this mean a feed has quietly stopped.
MAX_PRICE_STALENESS_DAYS = 4


class Severity:
    FAIL = "FAIL"
    WARN = "WARN"
    OK = "OK"


@dataclass
class Finding:
    check: str
    severity: str
    symbol: str | None
    detail: str

    @property
    def failed(self) -> bool:
        return self.severity == Severity.FAIL


@dataclass
class ReconcileReport:
    asof: date
    findings: list[Finding] = field(default_factory=list)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.FAIL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARN]

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        checked = len(self.findings)
        return (
            f"{checked} checks, {len(self.failures)} failed, {len(self.warnings)} warned"
            if checked
            else "no checks run"
        )


def check_prices(book: BookSnapshot, quotes: dict[str, float]) -> list[Finding]:
    """Broker close against an independent price for every position."""
    findings: list[Finding] = []
    for position in book.positions:
        symbol = position.underlying.upper()
        broker = position.close_price
        other = quotes.get(symbol)
        if other is None:
            findings.append(
                Finding("price_coverage", Severity.FAIL, symbol, "no independent price available")
            )
            continue
        if broker <= 0:
            findings.append(
                Finding("price_positive", Severity.FAIL, symbol, f"broker close is {broker}")
            )
            continue
        drift = abs(other - broker) / broker
        if drift > PRICE_TOLERANCE_PCT:
            findings.append(
                Finding(
                    "price_agreement",
                    Severity.FAIL,
                    symbol,
                    f"broker {broker:.4f} vs independent {other:.4f} ({drift * 100:.2f}%)",
                )
            )
        else:
            findings.append(
                Finding("price_agreement", Severity.OK, symbol, f"{drift * 100:.3f}% apart")
            )
    return findings


def check_price_staleness(
    last_bars: dict[str, date], *, today: date | None = None
) -> list[Finding]:
    """Every symbol should have a bar from the last trading session."""
    today = today or date.today()
    expected = previous_trading_day(today)
    findings: list[Finding] = []
    for symbol, last in sorted(last_bars.items()):
        age = (expected - last).days
        if age > MAX_PRICE_STALENESS_DAYS:
            findings.append(
                Finding(
                    "price_staleness",
                    Severity.FAIL,
                    symbol,
                    f"last bar {last}, {age} days before the last session {expected}",
                )
            )
        elif age > 0:
            findings.append(
                Finding(
                    "price_staleness",
                    Severity.WARN,
                    symbol,
                    f"last bar {last} (expected {expected})",
                )
            )
        else:
            findings.append(Finding("price_staleness", Severity.OK, symbol, f"current to {last}"))
    return findings


def check_earnings(
    broker_dates: dict[str, tuple[date | None, bool]],
    yahoo_dates: dict[str, date | None],
    *,
    today: date | None = None,
) -> list[Finding]:
    """Earnings dates from two sources that answer different questions.

    TastyTrade returns the *last* report with an ``estimated`` flag; yfinance
    returns the *next*, unflagged. Neither is wrong and using either alone is:
    a naive "when is earnings" against the broker returns a date already in
    the past. This check makes that explicit rather than letting a caller
    discover it in production.
    """
    today = today or date.today()
    findings: list[Finding] = []
    for symbol in sorted(set(broker_dates) | set(yahoo_dates)):
        broker, estimated = broker_dates.get(symbol, (None, False))
        yahoo = yahoo_dates.get(symbol)
        if broker is None and yahoo is None:
            findings.append(
                Finding(
                    "earnings_coverage",
                    Severity.WARN,
                    symbol,
                    "no earnings date from either source",
                )
            )
            continue
        if broker is not None and broker < today:
            findings.append(
                Finding(
                    "earnings_direction",
                    Severity.OK,
                    symbol,
                    f"broker {broker} is the last report{' (estimated)' if estimated else ''}; "
                    f"next per yfinance {yahoo}",
                )
            )
            continue
        if broker is not None and yahoo is not None and abs((broker - yahoo).days) > 7:
            findings.append(
                Finding(
                    "earnings_agreement",
                    Severity.WARN,
                    symbol,
                    f"upcoming dates disagree: broker {broker} vs yfinance {yahoo}",
                )
            )
        else:
            findings.append(
                Finding("earnings_agreement", Severity.OK, symbol, f"{broker or yahoo}")
            )
    return findings


def reconcile_events(report: ReconcileReport) -> list[Event]:
    """One event per failing check — the advisor reporting its own blindness.

    Deliberately book-level and deduped by check and session, so a feed that
    breaks for eleven symbols wakes you once rather than eleven times.
    """
    events: list[Event] = []
    by_check: dict[str, list[Finding]] = {}
    for finding in report.failures:
        by_check.setdefault(finding.check, []).append(finding)

    for check, findings in sorted(by_check.items()):
        events.append(
            Event(
                source=EventSource.DAEMON,
                kind="DATA_QUALITY_FAILURE",
                tier=EventTier.A,
                symbol=None,
                dedup_key=f"{check}:{report.asof.isoformat()}",
                payload={
                    "check": check,
                    "failed": len(findings),
                    "symbols": [f.symbol for f in findings if f.symbol],
                    "detail": [f.detail for f in findings][:5],
                },
            )
        )
    return events


async def run_reconciliation(book: BookSnapshot, *, today: date | None = None) -> ReconcileReport:
    """Gather every source live and run all checks. Never raises."""
    report = ReconcileReport(asof=today or date.today())
    symbols = book.symbols
    if not symbols:
        return report

    quotes: dict[str, float] = {}
    last_bars: dict[str, date] = {}
    try:
        from advisor.macro.factors import fetch_prices

        prices = fetch_prices(symbols, period="1mo")
        for symbol in symbols:
            if symbol in prices.columns:
                column = prices[symbol].dropna()
                if not column.empty:
                    quotes[symbol] = float(column.iloc[-1])
                    last_bars[symbol] = column.index[-1].date()
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconcile: independent prices unavailable: %s", exc)
        report.findings.append(
            Finding("price_coverage", Severity.FAIL, None, f"price source unreachable: {exc}")
        )

    if quotes:
        report.findings.extend(check_prices(book, quotes))
    if last_bars:
        report.findings.extend(check_price_staleness(last_bars, today=report.asof))

    broker_dates: dict[str, tuple[date | None, bool]] = {}
    try:
        from tastytrade.metrics import get_market_metrics

        from advisor.api import deps

        session = await deps.get_tt_session()
        for metric in await get_market_metrics(session, symbols):
            earnings = metric.earnings
            broker_dates[metric.symbol.upper()] = (
                earnings.expected_report_date if earnings else None,
                bool(earnings.estimated) if earnings else False,
            )
    except Exception as exc:  # noqa: BLE001
        logger.info("reconcile: broker metrics unavailable: %s", exc)

    yahoo_dates: dict[str, date | None] = {}
    try:
        from advisor.data.yahoo import fetch_earnings_dates

        for symbol in symbols:
            dates = fetch_earnings_dates(symbol)
            yahoo_dates[symbol] = dates[0] if dates else None
    except Exception as exc:  # noqa: BLE001
        logger.info("reconcile: yfinance earnings unavailable: %s", exc)

    if broker_dates or yahoo_dates:
        report.findings.extend(check_earnings(broker_dates, yahoo_dates, today=report.asof))

    return report
