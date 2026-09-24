"""Reading consolidated figures out of a filing's XBRL.

The whole difficulty is in one word: *consolidated*. XBRL reports the same
concept many times over — once per segment, per instrument, per share class.
SPCX's Q2 revenue appears sixteen times inside the same period, and fifteen of
those are slices. Only the undimensioned fact means $7,814M; a naive read that
took the first row would have reported $461M of products revenue as the
company's total and every number downstream would have been wrong by 17x.

So every extraction here filters on ``is_dimensioned == False``, and anything
that cannot be established that way is recorded in ``missing`` rather than
defaulted. A valuation built on a guessed input is worse than no valuation.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from advisor.valuation.models import Fundamentals, ValuationSnapshot

logger = logging.getLogger(__name__)

# Companies tag revenue under different concepts; the first that yields an
# undimensioned fact wins. Order matters — the ASC 606 concepts are the
# modern ones and the legacy names are fallbacks.
REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)
CASH_CONCEPTS = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "CashAndCashEquivalentsFairValueDisclosure",
)
SECURITIES_CONCEPTS = (
    "MarketableSecuritiesCurrent",
    "ShortTermInvestments",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
)
# Share counts are per class on the cover page; they must be summed, never
# taken singly. SPCX has 7.7bn Class A and 5.5bn Class B — either alone
# understates the company by billions of shares.
# Only the cover-page concept. `CommonStockSharesOutstanding` reports every
# class for both the current and the prior balance sheet — 38 rows for SPCX —
# and summing it would double count the company.
SHARE_CONCEPT = "EntityCommonStockSharesOutstanding"
OPERATING_CONCEPTS = ("OperatingIncomeLoss",)
NET_INCOME_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")


def _frame(xbrl, concept: str):
    try:
        df = xbrl.query().by_concept(concept).to_dataframe()
    except Exception as exc:  # noqa: BLE001
        logger.debug("xbrl: %s query failed: %s", concept, exc)
        return None
    if df is None or df.empty or "is_dimensioned" not in df.columns:
        return None
    if "numeric_value" not in df.columns:
        return None
    plain = df[~df["is_dimensioned"].astype(bool)]
    return plain if not plain.empty else None


def _duration_value(xbrl, concepts, start: date | None, end: date) -> float | None:
    """A flow figure (revenue, income) for exactly one period."""
    for concept in concepts:
        df = _frame(xbrl, concept)
        if df is None or "period_end" not in df.columns:
            continue
        rows = df[df["period_end"].astype(str) == end.isoformat()]
        if start is not None and "period_start" in rows.columns:
            rows = rows[rows["period_start"].astype(str) == start.isoformat()]
        rows = rows.dropna(subset=["numeric_value"])
        if not rows.empty:
            return float(rows["numeric_value"].iloc[0])
    return None


def _balance_rows(xbrl, concept: str, asof: date):
    """Undimensioned balance-sheet facts for one instant.

    Every balance fact carries ``period_key`` of the form ``instant_<date>``,
    which is the difference between reading a balance and guessing at one. A
    first version of this took the largest undimensioned value and picked up
    "Proceeds from debt and other financing obligations" — a cash-flow line
    of $51.8bn — as SPCX's total debt of $39.4bn. Dating the fact removes the
    guess entirely.
    """
    df = _frame(xbrl, concept)
    if df is None or "period_key" not in df.columns or "numeric_value" not in df.columns:
        return None
    dated = df[df["period_key"].astype(str) == f"instant_{asof.isoformat()}"].dropna(
        subset=["numeric_value"]
    )
    if dated.empty:
        return None
    # Dating the fact is what removes the guesswork; the statement filter is a
    # refinement that not every issuer supports, so it narrows when it can and
    # is dropped when it would leave nothing.
    if "statement_type" in dated.columns:
        on_balance = dated[dated["statement_type"].astype(str) == "BalanceSheet"]
        if not on_balance.empty:
            return on_balance
    return dated


def _instant_value(xbrl, concepts, asof: date) -> float | None:
    """A single balance figure (cash, securities) at ``asof``."""
    for concept in concepts:
        rows = _balance_rows(xbrl, concept, asof)
        if rows is not None and not rows.empty:
            return float(rows["numeric_value"].iloc[0])
    return None


def _share_total(xbrl) -> float | None:
    """Shares outstanding, summed across every class on the cover page.

    The one place the undimensioned rule is deliberately not applied: share
    *classes are* the dimension. SPCX reports 7.70bn Class A and 5.49bn Class
    B, both dimensioned, and filtering them out leaves nothing — while taking
    either alone understates the company by billions of shares.

    The cover-page concept states each class exactly once, as of the filing
    date, so summing every row is correct and a single-class issuer simply
    has one row.
    """
    try:
        df = xbrl.query().by_concept(SHARE_CONCEPT).to_dataframe()
    except Exception as exc:  # noqa: BLE001
        logger.debug("xbrl: share query failed: %s", exc)
        return None
    if df is None or df.empty or "numeric_value" not in df.columns:
        return None
    values = df["numeric_value"].dropna()
    return float(values.sum()) if not values.empty else None


DEBT_CONCEPTS = (
    "DebtLongtermAndShorttermCombinedAmount",
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "DebtCurrent",
    "NotesPayable",
    "ConvertibleDebtNoncurrent",
)


def _total_debt(xbrl, asof: date) -> tuple[float | None, bool]:
    """Total debt at ``asof``, and whether the filing mentions debt at all.

    Returns ``(amount, filing_mentions_debt)``. The second value is what
    separates a genuinely debt-free company from one whose debt could not be
    parsed — a distinction that matters because treating "unparseable" as zero
    overstates enterprise value in the flattering direction, while treating
    "debt-free" as missing makes an entire class of company unvaluable.

    CRDO is the case: no undimensioned balance-sheet debt reading, because
    there is no debt to report.
    """
    mentioned = False
    for concept in DEBT_CONCEPTS:
        try:
            df = xbrl.query().by_concept(concept).to_dataframe()
        except Exception:  # noqa: BLE001
            continue
        if df is not None and not df.empty:
            mentioned = True
        rows = _balance_rows(xbrl, concept, asof)
        if rows is not None and not rows.empty:
            # The balance sheet splits debt into current and non-current
            # lines; both are needed and each appears once at this instant.
            return float(rows["numeric_value"].sum()), True
    return None, mentioned


# A 52/53-week fiscal year moves the comparable quarter by a day or two: AMD's
# Q2 2026 ended 2026-06-27 and its Q2 2025 on 2025-06-28.
_COMPARABLE_SLACK_DAYS = 7


def _prior_year_value(xbrl, concepts, start: date | None, end: date) -> float | None:
    """The same-length period a year before ``start``..``end``, from the
    filing's comparative column. Undimensioned facts only."""
    if start is None:
        return None
    length = (end - start).days
    target_end = end - timedelta(days=365)
    for concept in concepts:
        df = _frame(xbrl, concept)
        if df is None or "period_start" not in df.columns or "period_end" not in df.columns:
            continue
        for _, row in df.dropna(subset=["numeric_value", "period_start"]).iterrows():
            try:
                row_end = date.fromisoformat(str(row["period_end"])[:10])
                row_start = date.fromisoformat(str(row["period_start"])[:10])
            except ValueError:
                continue
            if (
                abs((row_end - target_end).days) <= _COMPARABLE_SLACK_DAYS
                and abs((row_end - row_start).days - length) <= _COMPARABLE_SLACK_DAYS
            ):
                return float(row["numeric_value"])
    return None


def fundamentals_from_filing(symbol: str, filing) -> Fundamentals | None:
    """Extract the figures a valuation needs from one filing, or None."""
    try:
        xbrl = filing.xbrl()
    except Exception as exc:  # noqa: BLE001
        logger.info("valuation: no XBRL on %s for %s: %s", filing.accession_no, symbol, exc)
        return None
    if xbrl is None:
        return None

    period_end = getattr(filing, "period_of_report", None) or filing.filing_date
    if isinstance(period_end, str):
        period_end = date.fromisoformat(period_end[:10])

    # The quarter's own revenue, not the year to date: find the shortest
    # duration ending at the period end.
    start = None
    df = None
    for concept in REVENUE_CONCEPTS:
        df = _frame(xbrl, concept)
        if df is not None and "period_start" in df.columns:
            rows = df[df["period_end"].astype(str) == period_end.isoformat()].dropna(
                subset=["numeric_value", "period_start"]
            )
            if not rows.empty:
                start = date.fromisoformat(str(rows["period_start"].max())[:10])
                break

    revenue = _duration_value(xbrl, REVENUE_CONCEPTS, start, period_end)
    debt, debt_mentioned = _total_debt(xbrl, period_end)
    if debt is None and not debt_mentioned:
        # Nothing in the filing mentions debt in any form: the company has
        # none, and zero is the correct reading rather than a guess.
        debt = 0.0
    fundamentals = Fundamentals(
        symbol=symbol.upper(),
        source_accession=str(filing.accession_no),
        period_end=period_end,
        period_start=start,
        fiscal_period="Q" if str(filing.form).startswith("10-Q") else "FY",
        revenue=revenue,
        prior_revenue=_prior_year_value(xbrl, REVENUE_CONCEPTS, start, period_end),
        operating_income=_duration_value(xbrl, OPERATING_CONCEPTS, start, period_end),
        net_income=_duration_value(xbrl, NET_INCOME_CONCEPTS, start, period_end),
        cash=_instant_value(xbrl, CASH_CONCEPTS, period_end),
        marketable_securities=_instant_value(xbrl, SECURITIES_CONCEPTS, period_end),
        total_debt=debt,
        shares_outstanding=_share_total(xbrl),
    )

    # Only the inputs a valuation cannot proceed without are "missing";
    # securities and debt merely make the answer conservative.
    for name in ("revenue", "cash", "shares_outstanding", "total_debt"):
        if getattr(fundamentals, name) is None:
            fundamentals.missing.append(name)
    if fundamentals.missing:
        logger.info(
            "valuation: %s filing %s is missing %s",
            symbol,
            fundamentals.source_accession,
            ", ".join(fundamentals.missing),
        )
    return fundamentals


def latest_fundamentals(symbol: str) -> Fundamentals | None:
    """Newest periodic filing for ``symbol``, parsed. None on any failure."""
    from advisor.news.edgar import company_for

    company = company_for(symbol)
    if company is None:
        return None
    try:
        # Foreign private issuers (NBIS is a Dutch N.V.) never file a 10-Q;
        # they file 20-F annually. Excluding them silently would make a
        # position permanently unvaluable for no stated reason.
        filings = company.get_filings(form=["10-Q", "10-K", "20-F", "40-F"]).head(6)
    except Exception as exc:  # noqa: BLE001
        logger.warning("valuation: filing lookup failed for %s: %s", symbol, exc)
        return None

    for filing in filings:
        fundamentals = fundamentals_from_filing(symbol, filing)
        if fundamentals is not None and fundamentals.complete:
            return _prefer_interim(symbol, fundamentals)
    return None


def _prefer_interim(symbol: str, periodic: Fundamentals) -> Fundamentals:
    """Use a newer 6-K when it proves a complete set of figures.

    A foreign private issuer files a 20-F once a year, so the newest periodic
    filing can be two quarters behind the business. Nebius's was 263 days old
    and priced the position at 107x revenue against a real 25x — a run-rate of
    $0.53bn where the last reported quarter annualises to $2.33bn.

    The periodic filing is not discarded: it is what the interim balance sheet
    is proved against, and it is what gets returned when the proof fails.
    """
    if not periodic.complete:
        return periodic
    # A filer whose last periodic report is recent has nothing newer to find,
    # and looking costs an EDGAR round trip per symbol. The threshold is the
    # one the snapshot already uses to decide a run-rate may have stopped
    # describing the business.
    age = (date.today() - periodic.period_end).days
    if age <= ValuationSnapshot.STALE_AFTER_DAYS:
        return periodic
    try:
        interim = interim_fundamentals(symbol, periodic)
    except Exception as exc:  # noqa: BLE001 — a valuation must survive this
        logger.warning("interim: rebuild failed for %s: %s", symbol, exc)
        return periodic
    if interim is None or not interim.complete:
        return periodic
    if interim.period_end <= periodic.period_end:
        return periodic
    return interim


# How far back to look for an interim report. A quarter is filed within weeks
# of its close, so a handful of 6-Ks always reaches the newest one.
_INTERIM_LOOKBACK = 8

_RESULTS_EXHIBITS = ("EX-99.1", "EX-99.2", "EX-99")


def _exhibit_texts(filing) -> list[str]:
    texts = []
    for attachment in getattr(filing, "attachments", []) or []:
        if str(getattr(attachment, "document_type", "")).upper() not in _RESULTS_EXHIBITS:
            continue
        try:
            text = attachment.text()
        except Exception as exc:  # noqa: BLE001
            logger.info("interim: an exhibit would not render: %s", exc)
            continue
        if text:
            texts.append(text)
    return texts


def interim_fundamentals(symbol: str, known: Fundamentals) -> Fundamentals | None:
    """A complete set of figures from a 6-K newer than ``known``, or None.

    The income statement comes from `interim.parse_interim`, which takes only
    sentences whose own arithmetic reconciles. The balance sheet comes from
    `balance.confirm_balance`, which takes the current column only where the
    prior column reproduces ``known``. Both halves must land, from the same
    filing, or nothing is returned — a valuation is not improved by mixing a
    fresh numerator with a stale denominator.

    That mixing is not hypothetical here. Nebius raised $5.75bn of convertible
    notes in August; its cash went from $3.7bn to $8.0bn and its debt from
    $4.1bn to $8.5bn between the two columns. A June revenue over a December
    balance sheet would have moved enterprise value by billions in the
    flattering direction.
    """
    from advisor.news.edgar import company_for
    from advisor.valuation.balance import confirm_balance
    from advisor.valuation.interim import Period, parse_interim

    company = company_for(symbol)
    if company is None:
        return None
    try:
        filings = company.get_filings(form=["6-K", "6-K/A"]).head(_INTERIM_LOOKBACK)
    except Exception as exc:  # noqa: BLE001
        logger.warning("interim: filing lookup failed for %s: %s", symbol, exc)
        return None

    for filing in filings:
        texts = _exhibit_texts(filing)
        if not texts:
            continue
        revenue = None
        for text in texts:
            figure = parse_interim(text).find(metric="total revenue", period=Period.QUARTER)
            if figure is not None:
                revenue = figure
                break
        if revenue is None:
            continue
        balance = next(
            (b for b in (confirm_balance(t, known) for t in texts) if b.complete),
            None,
        )
        if balance is None or balance.period_end is None:
            logger.info("interim: %s reports revenue but its balance sheet is unproved", symbol)
            continue
        logger.info(
            "interim: %s rebuilt on %s (%s), revenue %.1fm",
            symbol,
            filing.accession_no,
            balance.period_end,
            revenue.current / 1e6,
        )
        return Fundamentals(
            symbol=symbol,
            source_accession=str(filing.accession_no),
            period_end=balance.period_end,
            fiscal_period="Q",
            revenue=revenue.current,
            cash=balance.cash,
            total_debt=balance.total_debt,
            shares_outstanding=balance.shares_outstanding,
        )
    return None
