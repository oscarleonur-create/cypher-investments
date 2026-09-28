"""The figures a valuation is computed from, read from the SEC and the broker.

One loader for both the daemon and the research workstation, so they cannot
disagree about what a company earned. The workstation used to take price,
shares, revenue and free cash flow from ``yfinance.info``; on 2026-09-27 Yahoo
answered every request with a rate-limit error and the DCF published a $0.00
fair value for JBL off a default share count of 1.0. Nothing here comes from
Yahoo unless the SEC has nothing (a foreign private issuer filing IFRS), and
then it says so.

**Trailing twelve months, from the filings.** A 10-Q's cash-flow statement is
year-to-date, so the SEC's quarterly frames do not exist for it. The trailing
figure is rebuilt the way an analyst does it by hand:

    TTM = last fiscal year + this year to date − the same span a year earlier

and taken directly where the filing states a twelve-month period (AMZN's
10-Q does). Revenue, operating cash flow, capex and operating income are all
read this way, at the *same* period end, so a margin never divides one
quarter's cash by another quarter's revenue.

**The balance sheet** comes from ``fundamentals.latest_fundamentals``: dated
XBRL instants, cover-page shares summed across classes, and a proved 6-K
balance for foreign issuers. None of that is repeated here.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import date, timedelta

from pydantic import BaseModel, Field

from advisor.valuation.dcf import NOPAT_TAX, Margin
from advisor.valuation.models import OwnMargin

logger = logging.getLogger(__name__)

REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
OCF_CONCEPTS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
CAPEX_CONCEPTS = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
)
OPERATING_CONCEPTS = ("OperatingIncomeLoss",)

# A fiscal year of 52 or 53 weeks is 364 or 371 days; a calendar year 365–366.
_YEAR_DAYS = (350, 380)
# A comparable period a year earlier ends within a week of 365 days before.
_SLACK = timedelta(days=7)
MEDIAN_YEARS = 3


# ── Pure: trailing twelve months from reported durations ────────────────────


def _period(row: dict) -> tuple[date, date] | None:
    try:
        return date.fromisoformat(row["start"]), date.fromisoformat(row["end"])
    except (KeyError, TypeError, ValueError):
        return None


def periodic(
    rows: list[dict], *, forms: tuple[str, ...] = ("10-",)
) -> dict[tuple[date, date], float]:
    """One value per (start, end) from periodic-report facts, the latest filing's.

    A period is restated in later filings as a comparative; the newest
    statement of it is the one that stands. ``forms`` narrows the filings a
    fact may come from: "10-" is any 10-K or 10-Q, "10-K" the annual only.
    """
    best: dict[tuple[date, date], tuple[str, float]] = {}
    for row in rows:
        if not str(row.get("form", "")).startswith(forms):
            continue
        span = _period(row)
        if span is None or span[1] <= span[0]:
            continue
        try:
            value = float(row["val"])
        except (KeyError, TypeError, ValueError):
            continue
        filed = str(row.get("filed", ""))
        if span not in best or filed >= best[span][0]:
            best[span] = (filed, value)
    return {span: value for span, (_, value) in best.items()}


def _is_year(start: date, end: date) -> bool:
    return _YEAR_DAYS[0] <= (end - start).days <= _YEAR_DAYS[1]


def _days(start: date, end: date) -> int:
    """Calendar days a reported period covers, both ends included."""
    return (end - start).days + 1


def ttm_at(periods: dict[tuple[date, date], float], end: date) -> float | None:
    """The twelve months ending ``end``, or None when the filings cannot say.

    Three ways, in order, each requiring every piece to exist — a missing
    piece is a gap, never a zero:

    1. a twelve-month period stated outright (a 10-K; AMZN's 10-Q);
    2. the year to date bridged from the last fiscal year,
       ``FY + YTD − the same span a year earlier``;
    3. contiguous reported periods that add up to a year. WOLF emerged from
       bankruptcy with fresh-start accounting: its FY2026 10-K reports nine
       months as the successor and one quarter as the predecessor. Read as a
       year, the nine months understated revenue by a quarter.
    """
    for (s, e), value in periods.items():
        if e == end and _is_year(s, e):
            return value
    bridged = _bridged(periods, end)
    return bridged if bridged is not None else _chained(periods, end)


def _bridged(periods, end: date) -> float | None:
    partial = [(s, v) for (s, e), v in periods.items() if e == end and 80 <= (e - s).days < 350]
    if not partial:
        return None
    start, ytd = min(partial, key=lambda sv: sv[0])  # the longest: year to date
    length = (end - start).days
    prior_end = end - timedelta(days=365)
    prior = [
        v
        for (s, e), v in periods.items()
        if abs(e - prior_end) <= _SLACK and abs((e - s).days - length) <= 7
    ]
    fiscal = [
        v
        for (s, e), v in periods.items()
        if _is_year(s, e) and abs(e - (start - timedelta(days=1))) <= _SLACK
    ]
    if not prior or not fiscal:
        return None
    return fiscal[0] + ytd - prior[0]


def _chained(periods, end: date) -> float | None:
    covered, total, cursor = 0, 0.0, end
    while covered < _YEAR_DAYS[0]:
        fits = [
            (s, e, v)
            for (s, e), v in periods.items()
            if abs((e - cursor).days) <= 3
            and (e - s).days >= 80
            and covered + _days(s, e) <= _YEAR_DAYS[1] + 1
        ]
        if not fits:
            return None
        s, e, v = min(fits, key=lambda sev: sev[0])  # the longest that still fits
        covered += _days(s, e)
        total += v
        cursor = s - timedelta(days=1)
    return total


def latest_end(periods: dict[tuple[date, date], float]) -> date | None:
    ends = [e for (s, e) in periods if (e - s).days >= 80]
    return max(ends) if ends else None


def comparable_end(periods: dict[tuple[date, date], float], end: date) -> date | None:
    """The period end a year before ``end`` that the filings actually use."""
    target = end - timedelta(days=365)
    ends = sorted(
        {e for (_, e) in periods if abs(e - target) <= _SLACK}, key=lambda e: abs(e - target)
    )
    return ends[0] if ends else None


def spans_at(periods: dict[tuple[date, date], float], end: date) -> dict[date, float]:
    """Reported periods ending ``end`` of at least a quarter, by start date."""
    return {s: v for (s, e), v in periods.items() if e == end and (e - s).days >= 80}


def fiscal_years(rows: list[dict]) -> dict[date, float]:
    """Full fiscal years by period end, from annual reports only.

    A 10-Q can state a twelve-month period too — AMZN's gives July to June —
    and that is a trailing figure, not a fiscal year. Only a 10-K's year-long
    columns are fiscal years.
    """
    return {e: v for (s, e), v in periodic(rows, forms=("10-K",)).items() if _is_year(s, e)}


def _first(series: list[dict[tuple[date, date], float]], fn):
    for periods in series:
        value = fn(periods)
        if value is not None:
            return value
    return None


def _merge_spans(series: list[dict[tuple[date, date], float]], end: date) -> dict[date, float]:
    out: dict[date, float] = {}
    for periods in series:
        for start, value in spans_at(periods, end).items():
            out.setdefault(start, value)
    return out


@dataclass(frozen=True)
class Flows:
    """Revenue, growth and margins at one period end, each with its basis.

    Trailing twelve months wherever the filings support it. A company only
    months public has no fiscal year on file — SPCX and CBRS had two 10-Qs
    each on 2026-09-27 — so its margins are read over the longest period the
    filings do state, and labelled with it.
    """

    period_end: date
    revenue_ttm: float | None = None
    growth: float | None = None
    growth_label: str = ""
    fcf_margin: float | None = None
    fcf_label: str = ""
    nopat_margin: float | None = None
    nopat_label: str = ""
    fcf_median: float | None = None  # median FCF margin over recent fiscal years
    fcf_median_label: str = ""
    capex_intensity: float | None = None  # capex / revenue, over the FCF span
    source: str = "SEC XBRL"


def _span_label(start: date, end: date) -> str:
    months = round(_days(start, end) / 30.44)
    return f"{months} months to {end}"


def flows_from_rows(
    revenue: list[list[dict]],
    ocf: list[list[dict]],
    capex: list[list[dict]],
    operating: list[list[dict]],
    *,
    years: int = MEDIAN_YEARS,
) -> Flows | None:
    """Assemble revenue, growth and margins from raw SEC concept rows. Pure.

    Each argument is one list of rows per concept, in order of preference.
    The period end is revenue's latest; every other figure is read at that
    same end, over the same span as the revenue it is divided by.
    """
    rev = [periodic(rows) for rows in revenue]
    ends = [e for e in (latest_end(p) for p in rev) if e is not None]
    if not ends:
        return None
    end = max(ends)
    cash = [periodic(rows) for rows in ocf]
    spend = [periodic(rows) for rows in capex]
    income = [periodic(rows) for rows in operating]

    revenue_ttm = _first(rev, lambda p: ttm_at(p, end))
    if revenue_ttm is not None and revenue_ttm <= 0:
        revenue_ttm = None
    ttm_label = f"trailing 12 months to {end}"

    growth, growth_label = None, ""
    if revenue_ttm is not None:
        prior = _first(
            rev, lambda p: ttm_at(p, c) if (c := comparable_end(p, end)) is not None else None
        )
        if prior and prior > 0:
            growth, growth_label = revenue_ttm / prior - 1, f"{ttm_label} vs a year earlier"
    if growth is None:
        growth, growth_label = _span_growth(rev, end)

    # Free-cash-flow margin: TTM if all three figures have it, else the
    # longest span all three report ending at the same date.
    fcf_margin, fcf_label, capex_intensity = None, "", None
    ocf_ttm, capex_ttm = (
        _first(cash, lambda p: ttm_at(p, end)),
        _first(spend, lambda p: ttm_at(p, end)),
    )
    if revenue_ttm and ocf_ttm is not None and capex_ttm is not None:
        fcf_margin, fcf_label = (ocf_ttm - abs(capex_ttm)) / revenue_ttm, f"FCF, {ttm_label}"
        capex_intensity = abs(capex_ttm) / revenue_ttm
    else:
        r, o, c = _merge_spans(rev, end), _merge_spans(cash, end), _merge_spans(spend, end)
        common = sorted(s for s in r if s in o and s in c and r[s] > 0)
        if common:
            s = common[0]
            fcf_margin = (o[s] - abs(c[s])) / r[s]
            fcf_label = f"FCF, {_span_label(s, end)}"
            capex_intensity = abs(c[s]) / r[s]

    nopat_margin, nopat_label = None, ""
    income_ttm = _first(income, lambda p: ttm_at(p, end))
    tax = f"operating margin after {NOPAT_TAX:.0%} tax"
    if revenue_ttm and income_ttm is not None:
        nopat_margin = income_ttm * (1 - NOPAT_TAX) / revenue_ttm
        nopat_label = f"{tax}, {ttm_label}"
    else:
        r, i = _merge_spans(rev, end), _merge_spans(income, end)
        common = sorted(s for s in r if s in i and r[s] > 0)
        if common:
            s = common[0]
            nopat_margin = i[s] * (1 - NOPAT_TAX) / r[s]
            nopat_label = f"{tax}, {_span_label(s, end)}"

    median, median_label = _median_fcf_margin(revenue, ocf, capex, years)
    return Flows(
        period_end=end,
        revenue_ttm=revenue_ttm,
        growth=growth,
        growth_label=growth_label,
        fcf_margin=fcf_margin,
        fcf_label=fcf_label,
        nopat_margin=nopat_margin,
        nopat_label=nopat_label,
        fcf_median=median,
        fcf_median_label=median_label,
        capex_intensity=capex_intensity,
    )


def _span_growth(rev, end: date) -> tuple[float | None, str]:
    """Growth over the longest span ending ``end`` that has a year-earlier twin."""
    for periods in rev:
        prior_end = comparable_end(periods, end)
        if prior_end is None:
            continue
        current, earlier = spans_at(periods, end), spans_at(periods, prior_end)
        for start in sorted(current):  # longest first
            length = (end - start).days
            twin = [v for s, v in earlier.items() if abs((prior_end - s).days - length) <= 7]
            if twin and twin[0] > 0:
                return current[start] / twin[0] - 1, f"{_span_label(start, end)} vs a year earlier"
    return None, ""


def _merged_years(series: list[list[dict]]) -> dict[date, float]:
    out: dict[date, float] = {}
    for rows in series:
        for end, value in fiscal_years(rows).items():
            out.setdefault(end, value)
    return out


def _median_fcf_margin(revenue, ocf, capex, years: int) -> tuple[float | None, str]:
    r, o, c = _merged_years(revenue), _merged_years(ocf), _merged_years(capex)
    common = sorted(d for d in o if d in c and d in r and r[d] > 0)[-years:]
    if len(common) < 2:
        return None, "fewer than two fiscal years of cash-flow data"
    margins = [(o[d] - abs(c[d])) / r[d] for d in common]
    return statistics.median(margins), f"FY{common[0].year}–FY{common[-1].year} median FCF"


# ── The figures one valuation runs on ────────────────────────────────────────


class Figures(BaseModel):
    """Everything a valuation needs, each with its source. Absent is None."""

    symbol: str
    price: float | None = None
    price_source: str = ""
    shares: float | None = None
    net_cash: float | None = None
    balance_asof: date | None = None
    source_accession: str = ""

    revenue_base: float | None = None  # what the projection starts from
    revenue_base_label: str = ""
    revenue_runrate: float | None = None  # the latest period annualised
    revenue_growth: float | None = None  # year-one growth the fade starts from
    revenue_growth_label: str = ""
    start_margin: float | None = None  # today's FCF margin
    start_margin_label: str = ""
    capex_intensity: float | None = None  # capex / revenue, over the same span
    margins: list[OwnMargin] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def margin_readings(self) -> list[Margin]:
        return [Margin(m.value, m.label) for m in self.margins]

    def margin(self, kind: str) -> OwnMargin | None:
        return next((m for m in self.margins if m.kind == kind), None)

    @property
    def market_cap(self) -> float | None:
        if self.price is None or self.shares is None:
            return None
        return self.price * self.shares

    @property
    def enterprise_value(self) -> float | None:
        if self.market_cap is None or self.net_cash is None:
            return None
        return self.market_cap - self.net_cash


def build_figures(
    symbol: str,
    price: float | None,
    fundamentals=None,
    flows: Flows | None = None,
    *,
    fallback_margin: tuple[float, str] | None = None,
    price_source: str = "",
) -> Figures:
    """Combine the balance sheet, the flows and a price. Pure.

    Trailing twelve months is the base when the SEC has it; otherwise the
    latest period annualised, labelled as such — NBIS files IFRS 6-Ks and has
    no SEC series, SPCX has no fiscal year on file yet.
    """
    fig = Figures(symbol=symbol.upper(), price=price, price_source=price_source)
    if fundamentals is not None:
        fig.shares = fundamentals.shares_outstanding
        fig.net_cash = fundamentals.net_cash
        fig.balance_asof = fundamentals.period_end
        fig.source_accession = fundamentals.source_accession
        fig.revenue_runrate = fundamentals.revenue_runrate

    if flows is not None:
        if flows.revenue_ttm is not None:
            fig.revenue_base = flows.revenue_ttm
            fig.revenue_base_label = f"trailing 12 months to {flows.period_end} ({flows.source})"
        if flows.growth is not None:
            fig.revenue_growth, fig.revenue_growth_label = flows.growth, flows.growth_label
        fig.capex_intensity = flows.capex_intensity
        if flows.fcf_margin is not None:
            fig.start_margin, fig.start_margin_label = flows.fcf_margin, flows.fcf_label
            fig.margins.append(OwnMargin(kind="fcf", value=flows.fcf_margin, label=flows.fcf_label))
        if flows.fcf_median is not None:
            fig.margins.append(
                OwnMargin(kind="median", value=flows.fcf_median, label=flows.fcf_median_label)
            )
        if flows.nopat_margin is not None:
            fig.margins.append(
                OwnMargin(kind="nopat", value=flows.nopat_margin, label=flows.nopat_label)
            )
        if fundamentals is not None and abs((fundamentals.period_end - flows.period_end).days) > 45:
            fig.notes.append(
                f"flows run to {flows.period_end} but the balance sheet to "
                f"{fundamentals.period_end}"
            )

    if fig.revenue_base is None and fundamentals is not None and fundamentals.revenue_runrate:
        fig.revenue_base = fundamentals.revenue_runrate
        fig.revenue_base_label = f"{fundamentals.runrate_label()} (no trailing 12 months on file)"
        fig.notes.append(
            "no trailing twelve months on file: the base is the latest period annualised"
        )

    if fig.revenue_growth is None and fundamentals is not None:
        yoy = fundamentals.revenue_yoy
        if yoy is not None:
            fig.revenue_growth = yoy
            fig.revenue_growth_label = f"period to {fundamentals.period_end} vs a year earlier"

    if not fig.margins and fallback_margin is not None:
        value, label = fallback_margin
        fig.margins.append(OwnMargin(kind="yahoo", value=value, label=label))
        if fig.start_margin is None:
            fig.start_margin, fig.start_margin_label = value, label
    return fig


# ── Live ─────────────────────────────────────────────────────────────────────


def load_flows(symbol: str) -> Flows | None:
    """Flows from the SEC ``companyconcept`` API. None when absent; never raises."""
    from advisor.news.edgar import company_for
    from advisor.valuation.history import _concept_rows

    try:
        cik = getattr(company_for(symbol), "cik", None)
        if not cik:
            return None
        cik = int(cik)

        def rows(concepts):
            return [_concept_rows(cik, c) for c in concepts]

        return flows_from_rows(
            rows(REVENUE_CONCEPTS),
            rows(OCF_CONCEPTS),
            rows(CAPEX_CONCEPTS),
            rows(OPERATING_CONCEPTS),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("figures: SEC series unavailable for %s: %s", symbol, exc)
        return None


def broker_closes(symbols: list[str]) -> dict[str, tuple[float, str]]:
    """The broker's official close per symbol, from its market-data endpoint,
    in one request. Symbols it does not know, or with no positive close, are
    absent. Raises on a network or auth failure; callers fall back."""
    import asyncio

    from tastytrade.market_data import get_market_data_by_type

    from advisor.market.tastytrade_client import get_session

    wanted = sorted({s.upper() for s in symbols if s})
    if not wanted:
        return {}

    async def fetch():
        return await get_market_data_by_type(await get_session(), equities=wanted)

    out: dict[str, tuple[float, str]] = {}
    for data in asyncio.run(fetch()) or []:
        sym = str(data.symbol).upper()
        if sym in wanted and (close := _latest_close(data)) is not None:
            out[sym] = close
    return out


def _latest_close(data) -> tuple[float, str] | None:
    """Today's close once there is one, else the previous session's, dated.

    A new session clears ``close``: at 05:56 on Monday 2026-09-28 every symbol
    had ``close=None`` and Friday's final close sat in ``prev_close``. Reading
    only ``close`` priced nothing from the broker until the bell.
    """
    for price, kind, day in (
        (data.close, data.close_price_type, data.summary_date),
        (
            getattr(data, "prev_close", None),
            getattr(data, "prev_close_price_type", None),
            getattr(data, "prev_close_date", None),
        ),
    ):
        if price is not None and float(price) > 0:
            label = str(getattr(kind, "value", kind) or "").lower()
            return float(price), f"TastyTrade {label + ' ' if label else ''}close {day}"
    return None


def _broker_close(symbol: str) -> tuple[float, str] | None:
    return broker_closes([symbol]).get(symbol.upper())


def last_price(symbol: str) -> tuple[float, str] | None:
    """The latest close: the broker's, then Yahoo's. None if neither answers.

    Not the broker's daily *candles*: on Sunday 2026-09-27 DXLink's daily
    series ended at Wednesday 09-23 for every symbol tried, and SPCX's carried
    closes of 0.0 for Thursday and Friday. The market-data endpoint had
    Friday's final close (MSFT $516.17, CRDO $210.97 — the position's own).
    """
    from datetime import datetime

    sym = symbol.upper()
    try:
        quoted = _broker_close(sym)
        if quoted is not None:
            return quoted
    except Exception as exc:  # noqa: BLE001
        logger.info("figures: no broker price for %s: %s", sym, exc)
    try:
        import yfinance as yf

        closes = yf.Ticker(sym).history(period="5d")["Close"].dropna()
        if len(closes) and float(closes.iloc[-1]) > 0:
            when = closes.index[-1]
            day = when.date() if isinstance(when, datetime) else when
            return float(closes.iloc[-1]), f"Yahoo close {day}"
    except Exception as exc:  # noqa: BLE001
        logger.info("figures: no Yahoo price for %s: %s", sym, exc)
    return None


def load_figures(
    symbol: str,
    price: float | None = None,
    *,
    price_source: str = "given",
    fundamentals=None,
) -> Figures:
    """Everything a valuation needs, live. Absent inputs stay absent."""
    from advisor.valuation.fundamentals import latest_fundamentals

    sym = symbol.upper()
    if price is None:
        quoted = last_price(sym)
        if quoted is not None:
            price, price_source = quoted
        else:
            price_source = ""
    if fundamentals is None:
        try:
            fundamentals = latest_fundamentals(sym)
        except Exception as exc:  # noqa: BLE001
            logger.warning("figures: no fundamentals for %s: %s", sym, exc)
    flows = load_flows(sym)
    fallback = None
    if flows is None or (flows.fcf_margin is None and flows.nopat_margin is None):
        # A foreign private issuer has no SEC series. Yahoo's statements are
        # the only trailing cash flow on the free tier; labelled as Yahoo's.
        from advisor.valuation.margins import load_trailing_margin

        yahoo = load_trailing_margin(sym)
        if yahoo.margin is not None:
            fallback = (yahoo.margin, f"FCF {yahoo.label} (Yahoo)")
    return build_figures(
        sym,
        price,
        fundamentals if fundamentals is not None and fundamentals.complete else None,
        flows,
        fallback_margin=fallback,
        price_source=price_source if price is not None else "",
    )
