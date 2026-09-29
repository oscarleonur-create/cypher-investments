"""The company's health from its own filings: growth and its trend, cash, balance sheet, dilution.

MDB, 2026-09-28: its CEO left for Meta and the stock fell 18.5%. The reading
saw the position and four headlines and nothing about the business — the
facts had "no valuation stored" — so it could only react to the price. The
filings to 2026-07-31 said revenue +25.5% a year and accelerating, free cash
flow 23.7% of revenue, $2.41bn of net cash. Whether a drop is the business
weakening or only the price is the bigger picture, and it needs these numbers
beside the news.

The numbers are the SEC's (``valuation.history`` and ``valuation.figures``),
dated by the period they cover: they predate any news after that period and
say so. This module describes; it changes no action (user decision,
2026-09-28: "si es add es add").

Computed with network for names with an action on the table, once per name
per day, and stored; the reading takes it from the store like every other
fact.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Trailing growth that moved by more than this over two quarters is a trend.
TREND_POINTS = 0.02
# Diluted shares growing faster than this a year is a concern.
DILUTION_CONCERN = 0.05
# Net debt above this many years of free cash flow is a concern.
NET_DEBT_FCF_YEARS = 3.0
# Free cash flow margin this far under the company's own median is a concern:
# META at 18.0% against 32.7% (2026-06-30), its capex build.
MARGIN_DROP = 0.10
# A median margin beyond ±100% is from a year with almost no revenue (TE's
# -2610%): not a level the company could return to, so not shown.
MARGIN_MEANINGFUL = 1.0
# SIC range of banks, lenders, insurers and REITs: deposits and loans run
# through their cash flow, so FCF and net cash say nothing about their health
# (NU "burned" 12.7% of revenue by that measure).
FINANCIAL_SIC = (6000, 6799)
# A stored health older than this is not read: a new 10-Q has likely landed.
MAX_AGE_DAYS = 100


class Health(BaseModel):
    symbol: str
    computed_on: date
    period_end: date | None = None
    source: str = ""

    revenue_ttm: float | None = None
    growth: float | None = None  # trailing 12 months vs a year earlier
    growth_two_q: float | None = None  # the same measure two quarters before
    growth_year: float | None = None  # the same measure a year before
    # When no trailing comparison exists (IFRS filers: Yahoo has five quarters),
    # the latest quarter against the same quarter a year earlier.
    quarter_growth: float | None = None
    quarter_end: date | None = None

    fcf_margin: float | None = None
    fcf_label: str = ""
    fcf_median: float | None = None
    fcf_median_label: str = ""
    operating_margin: float | None = None
    operating_label: str = ""

    net_cash: float | None = None
    balance_asof: date | None = None
    market_cap: float | None = None

    dilution: float | None = None  # diluted shares vs a year earlier
    dilution_end: date | None = None

    financial: bool = False  # a bank, lender, insurer or REIT (FINANCIAL_SIC)
    notes: list[str] = []

    def trend(self) -> str | None:
        if self.growth is None:
            if self.quarter_growth is not None and self.quarter_growth < 0:
                return "shrinking"
            return None
        if self.growth < 0:
            return "shrinking"
        if self.growth_two_q is None:
            return None
        change = self.growth - self.growth_two_q
        if change > TREND_POINTS:
            return "accelerating"
        if change < -TREND_POINTS:
            return "decelerating"
        return "steady"

    def concerns(self) -> list[str]:
        out = []
        trend = self.trend()
        if trend in ("shrinking", "decelerating"):
            out.append(f"revenue {trend}")
        if self.financial:
            pass  # cash flow and net cash are not its health; see lines()
        elif self.fcf_margin is not None and self.fcf_margin < 0:
            out.append("burning cash")
        elif (
            self.fcf_margin is not None
            and self._median() is not None
            and self._median() - self.fcf_margin > MARGIN_DROP
        ):
            out.append(
                f"free cash flow margin {(self._median() - self.fcf_margin) * 100:.1f} points "
                f"under its own median"
            )
        if not self.financial and self.net_cash is not None and self.net_cash < 0:
            fcf = (self.fcf_margin or 0) * (self.revenue_ttm or 0)
            if fcf <= 0 or -self.net_cash / fcf > NET_DEBT_FCF_YEARS:
                out.append(f"net debt above {NET_DEBT_FCF_YEARS:g} years of free cash flow")
        if self.dilution is not None and self.dilution > DILUTION_CONCERN:
            out.append(f"diluting more than {DILUTION_CONCERN * 100:g}% a year")
        return out

    def lines(self) -> list[str]:
        """The health as sentences with their numbers, each dated."""
        out = []
        if self.revenue_ttm is not None:
            text = f"Revenue {_money(self.revenue_ttm)} over the 12 months to {self.period_end}"
            if self.growth is not None:
                text += f", {self.growth * 100:+.1f}% on a year earlier"
            elif self.quarter_growth is not None:
                text += (
                    f"; the quarter to {self.quarter_end} was {self.quarter_growth * 100:+.1f}% "
                    f"on the same quarter a year earlier (Yahoo)"
                )
            else:
                text += " (no comparison a year earlier on file)"
            before = []
            if self.growth_two_q is not None:
                before.append(f"{self.growth_two_q * 100:+.1f}% two quarters before")
            if self.growth_year is not None:
                before.append(f"{self.growth_year * 100:+.1f}% a year before")
            if before:
                text += "; the same measure was " + " and ".join(before)
            trend = self.trend()
            out.append(text + (f": {trend}." if trend else "."))
        cash = []
        if self.financial:
            out.append(
                "A financial company (bank, lender, insurer or REIT): deposits and loans run "
                "through its cash flow, so free cash flow and net cash are not read as its health."
            )
        else:
            if self.fcf_margin is not None:
                cash.append(
                    f"free cash flow {self.fcf_margin * 100:.1f}% of revenue ({self.fcf_label})"
                )
            if self._median() is not None:
                cash.append(f"{self.fcf_median * 100:.1f}% at the {self.fcf_median_label}")
            if self.operating_margin is not None:
                cash.append(f"{self.operating_label} {self.operating_margin * 100:.1f}%")
        if cash:
            out.append("Cash: " + "; ".join(cash) + ".")
        if self.net_cash is not None and not self.financial:
            kind = "Net cash" if self.net_cash >= 0 else "Net debt"
            text = f"{kind} {_money(abs(self.net_cash))} as of {self.balance_asof}"
            if self.market_cap:
                text += f", {abs(self.net_cash) / self.market_cap * 100:.1f}% of its market cap"
            out.append(text + ".")
        if self.dilution is not None:
            out.append(
                f"Diluted shares {self.dilution * 100:+.1f}% on a year earlier "
                f"(to {self.dilution_end})."
            )
        out.extend(self.notes)
        if out:
            concerns = self.concerns()
            out.append(
                f"Health concerns in the filings to {self.period_end}: "
                + (", ".join(concerns) if concerns else "none flagged")
                + "; news since then is not in these numbers."
            )
        return out

    def _median(self) -> float | None:
        if self.fcf_median is None or abs(self.fcf_median) > MARGIN_MEANINGFUL:
            return None
        return self.fcf_median

    def source_label(self) -> str:
        return f"SEC filings to {self.period_end} ({self.source})"


def _money(value: float) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}bn"
    return f"${value / 1e6:,.1f}M"


# ── Pure assembly ────────────────────────────────────────────────────────


def _growth_at(points, index: int) -> float | None:
    """TTM growth at ``points[index]`` against the point four quarters before it."""
    from advisor.valuation.history import quarter_key

    if index < 0 or index >= len(points):
        return None
    year, q = quarter_key(points[index].end)
    prior = {quarter_key(p.end): p for p in points}.get((year - 1, q))
    if prior is None or prior.value <= 0:
        return None
    return points[index].value / prior.value - 1


def _year_change(points) -> tuple[float | None, date | None]:
    from advisor.valuation.history import quarter_key

    if not points:
        return None, None
    by_key = {quarter_key(p.end): p for p in points}
    last = max(points, key=lambda p: p.end)
    year, q = quarter_key(last.end)
    prior = by_key.get((year - 1, q))
    if prior is None or prior.value <= 0:
        return None, last.end
    return last.value / prior.value - 1, last.end


def assess(
    symbol: str,
    today: date,
    *,
    series=None,
    figures=None,
    sic: int | None = None,
    quarter: tuple[float, date] | None = None,
) -> Health | None:
    """Health from a revenue/share ``Series`` and ``Figures``. Pure. None when both are absent."""
    if series is None and figures is None:
        return None
    h = Health(
        symbol=symbol.upper(),
        computed_on=today,
        financial=sic is not None and FINANCIAL_SIC[0] <= sic <= FINANCIAL_SIC[1],
    )
    sources = []
    if series is not None and series.revenue_ttm:
        points = sorted(series.revenue_ttm, key=lambda p: p.end)
        last = len(points) - 1
        h.period_end = points[last].end
        h.revenue_ttm = points[last].value
        h.growth = _growth_at(points, last)
        h.growth_two_q = _growth_at(points, last - 2)
        h.growth_year = _growth_at(points, last - 4)
        year_ago = h.period_end - timedelta(days=366)
        if series.broken_after is None or series.broken_after < year_ago:
            h.dilution, h.dilution_end = _year_change(series.shares)
        sources.append(series.source)
    if figures is not None:
        if h.revenue_ttm is None and figures.revenue_base is not None:
            h.revenue_ttm, h.growth = figures.revenue_base, figures.revenue_growth
        if figures.start_margin is not None:
            h.fcf_margin, h.fcf_label = figures.start_margin, figures.start_margin_label
        median = figures.margin("median")
        if median is not None:
            h.fcf_median, h.fcf_median_label = median.value, median.label
        nopat = figures.margin("nopat")
        if nopat is not None:
            h.operating_margin, h.operating_label = nopat.value, nopat.label
        h.net_cash, h.balance_asof = figures.net_cash, figures.balance_asof
        h.market_cap = figures.market_cap
        h.period_end = h.period_end or figures.balance_asof
        if "SEC XBRL" not in sources:
            sources.append("SEC XBRL")
    if h.growth is None and quarter is not None:
        h.quarter_growth, h.quarter_end = quarter
    if h.revenue_ttm is None:
        h.notes.append("No trailing revenue on file (a foreign filer's IFRS figures, or too new).")
    h.source = ", ".join(dict.fromkeys(s for s in sources if s))
    return h if h.lines() else None


# ── Store ────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS company_health (
    symbol TEXT NOT NULL,
    computed_on TEXT NOT NULL,
    period_end TEXT,
    body TEXT NOT NULL,
    PRIMARY KEY (symbol, computed_on)
)
"""


def save_health(db_path: Path, health: Health) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(_SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO company_health VALUES (?, ?, ?, ?)",
            (
                health.symbol,
                health.computed_on.isoformat(),
                health.period_end.isoformat() if health.period_end else None,
                health.model_dump_json(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def latest_health(db_path: Path, symbol: str, today: date | None = None) -> Health | None:
    """The newest stored health for ``symbol``, if computed within MAX_AGE_DAYS. No network."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT body FROM company_health WHERE symbol = ? ORDER BY computed_on DESC LIMIT 1",
            (symbol.upper(),),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    health = Health.model_validate_json(row[0])
    if today is not None and (today - health.computed_on).days > MAX_AGE_DAYS:
        return None
    return health


# ── Live ─────────────────────────────────────────────────────────────────

# (symbol, day) computed this process: the hourly job does not repeat SEC calls.
_COMPUTED: set[tuple[str, date]] = set()


def load_health(symbol: str, today: date) -> Health | None:
    """Health from the SEC, live. None when neither series nor figures load."""
    from advisor.valuation.figures import load_figures
    from advisor.valuation.history import load_series

    series = figures = None
    try:
        series = load_series(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning("health: no series for %s: %s", symbol, exc)
    try:
        figures = load_figures(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning("health: no figures for %s: %s", symbol, exc)
    sic = None
    try:
        from advisor.news.edgar import company_for

        raw = getattr(company_for(symbol), "sic", None)
        sic = int(raw) if raw not in (None, "") else None
    except Exception as exc:  # noqa: BLE001
        logger.info("health: no SIC for %s: %s", symbol, exc)
    health = assess(symbol, today, series=series, figures=figures, sic=sic)
    if health is not None and health.growth is None:
        quarter = yahoo_quarter_growth(symbol)
        if quarter is not None:
            health = assess(symbol, today, series=series, figures=figures, sic=sic, quarter=quarter)
    return health


def quarter_growth(quarters: dict[date, float]) -> tuple[float, date] | None:
    """The latest quarter against the one ending about a year before it. Pure."""
    if not quarters:
        return None
    last = max(quarters)
    prior = [d for d in quarters if 350 <= (last - d).days <= 380]
    if not prior or quarters[prior[0]] <= 0:
        return None
    return quarters[last] / quarters[prior[0]] - 1, last


def yahoo_quarter_growth(symbol: str) -> tuple[float, date] | None:
    try:
        import yfinance as yf

        frame = yf.Ticker(symbol).quarterly_income_stmt
        if frame is None or "Total Revenue" not in frame.index:
            return None
        row = frame.loc["Total Revenue"].dropna()
        return quarter_growth({ts.date(): float(v) for ts, v in row.items() if v > 0})
    except Exception as exc:  # noqa: BLE001
        logger.info("health: no Yahoo quarters for %s: %s", symbol, exc)
        return None


def refresh_health(db_path: Path, symbol: str, now: datetime, *, loader=None) -> Health | None:
    """Today's health for ``symbol``: stored if already computed today, else computed and stored.

    Never raises. A failed load leaves the last stored health in place.
    """
    today = now.date()
    stored = latest_health(db_path, symbol, today)
    key = (symbol.upper(), today)
    if (stored is not None and stored.computed_on == today) or key in _COMPUTED:
        return stored
    _COMPUTED.add(key)
    try:
        health = (loader or load_health)(symbol, today)
    except Exception as exc:  # noqa: BLE001
        logger.warning("health failed for %s: %s", symbol, exc)
        return stored
    if health is None:
        return stored
    save_health(db_path, health)
    return health
