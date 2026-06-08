"""Portfolio research review — health-check the companies you actually hold.

Pulls the underlying equities held across one or more TastyTrade accounts, then
for each name asks two questions:

  1. Has the investment **thesis** changed?  → `kpi_tracker.re_check_kpis`
     (ON_TRACK / CAUTION / INVALIDATED + KPI breach alerts).
  2. Are there any important **events** coming up?  → `catalysts.build_catalysts`
     (earnings dates + near-term catalysts).

Tickers with no cached research report can be auto-built first (full 7-layer
`report.build_report`) so KPI monitoring has a thesis to derive from.

The assembled `PortfolioReview` is a plain Pydantic model so it serialises to
JSON for export and is reused directly by the Streamlit dashboard page. The
latest review is cached in the research store's artifact table under the
synthetic symbol ``_PORTFOLIO``.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# The two brokerage accounts on file. Used as the default scope.
DEFAULT_ACCOUNTS: tuple[str, ...] = ("5WI30382", "5WI47366")

# Synthetic key under which the latest review is cached in research_artifacts.
_REVIEW_SYMBOL = "_PORTFOLIO"
_REVIEW_KIND = "review"
_REVIEW_KEY = "latest"


# ── Models ─────────────────────────────────────────────────────────────────────


class PositionReview(BaseModel):
    """Per-holding research health snapshot."""

    symbol: str
    company_name: str = ""
    accounts: list[str] = Field(default_factory=list)
    instrument_types: list[str] = Field(default_factory=list)

    # Thesis health (from kpi_tracker.re_check_kpis)
    thesis_status: str = "UNKNOWN"  # ON_TRACK | CAUTION | INVALIDATED | UNKNOWN
    kpi_alerts: list[str] = Field(default_factory=list)

    # Cached-report context
    conviction: str | None = None  # HIGH | MEDIUM | LOW
    base_target: float | None = None
    base_upside: float | None = None  # decimal, e.g. 0.18 = +18%
    report_was_built: bool = False
    has_report: bool = False

    # Events (from catalysts.build_catalysts)
    near_term_catalysts: list[str] = Field(default_factory=list)
    next_earnings_date: str | None = None

    # Derived
    attention: str = "LOW"  # HIGH | MEDIUM | LOW
    error: str | None = None


class PortfolioReview(BaseModel):
    """Aggregated review across all held companies."""

    generated_at: datetime = Field(default_factory=datetime.now)
    account_numbers: list[str] = Field(default_factory=list)
    positions: list[PositionReview] = Field(default_factory=list)

    @property
    def n_invalidated(self) -> int:
        return sum(1 for p in self.positions if p.thesis_status == "INVALIDATED")

    @property
    def n_caution(self) -> int:
        return sum(1 for p in self.positions if p.thesis_status == "CAUTION")

    @property
    def n_near_term_events(self) -> int:
        return sum(1 for p in self.positions if p.near_term_catalysts or p.next_earnings_date)

    @property
    def n_high_attention(self) -> int:
        return sum(1 for p in self.positions if p.attention == "HIGH")


# ── Holdings ───────────────────────────────────────────────────────────────────


class _HeldInfo(BaseModel):
    accounts: set[str] = Field(default_factory=set)
    instrument_types: set[str] = Field(default_factory=set)

    class Config:
        arbitrary_types_allowed = True


def get_held_symbols(account_numbers: list[str] | None = None) -> dict[str, _HeldInfo]:
    """Return {underlying_symbol: _HeldInfo} across the given accounts (deduped).

    Keeps equity and equity-option positions (both resolve to a stock underlying);
    futures/crypto/etc. are skipped. ``account_numbers`` defaults to both accounts
    on file.
    """
    from advisor.market.tastytrade_client import get_positions

    accounts = account_numbers or list(DEFAULT_ACCOUNTS)
    held: dict[str, _HeldInfo] = {}

    for acct in accounts:
        try:
            positions = asyncio.run(get_positions(account_number=acct))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch positions for account %s: %s", acct, exc)
            continue

        for p in positions:
            itype = str(p.get("instrument_type", "")).upper()
            if "EQUITY" not in itype:  # covers EQUITY and EQUITY OPTION
                continue
            underlying = (p.get("underlying_symbol") or p.get("symbol") or "").strip().upper()
            if not underlying:
                continue
            info = held.setdefault(underlying, _HeldInfo())
            info.accounts.add(acct)
            info.instrument_types.add("OPTION" if "OPTION" in itype else "EQUITY")

    return held


# ── Per-ticker review ───────────────────────────────────────────────────────────


def _review_symbol(
    symbol: str,
    held: _HeldInfo,
    *,
    rebuild_uncovered: bool,
    with_catalysts: bool,
) -> PositionReview:
    """Build the research health snapshot for a single holding (never raises)."""
    from advisor.research.config import get_settings
    from advisor.research.store import ResearchStore

    review = PositionReview(
        symbol=symbol,
        accounts=sorted(held.accounts),
        instrument_types=sorted(held.instrument_types),
    )

    store = ResearchStore(get_settings().db_path)
    try:
        report = store.load_latest_report(symbol)

        if report is None and rebuild_uncovered:
            try:
                from advisor.research.report import build_report

                report = build_report(symbol)
                review.report_was_built = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Auto-build failed for %s: %s", symbol, exc)
                review.error = f"report build failed: {exc}"

        review.has_report = report is not None

        if report is not None:
            review.company_name = report.business_model or symbol
            if report.thesis is not None:
                review.conviction = report.thesis.conviction
            if report.dcf is not None and report.dcf.base is not None:
                review.base_target = report.dcf.base.implied_price
                review.base_upside = report.dcf.base.upside_pct

        # Thesis health
        try:
            from advisor.research.kpi_tracker import re_check_kpis

            monitor = re_check_kpis(symbol)
            review.thesis_status = monitor.thesis_status
            review.kpi_alerts = monitor.alerts
        except Exception as exc:  # noqa: BLE001
            logger.warning("KPI re-check failed for %s: %s", symbol, exc)
            if review.error is None:
                review.error = f"kpi check failed: {exc}"

        # Events
        if with_catalysts:
            try:
                from advisor.research.catalysts import build_catalysts

                cat = build_catalysts(symbol, review.company_name)
                review.near_term_catalysts = [
                    c.description for c in cat.catalysts if c.is_near_term and c.description
                ]
                review.next_earnings_date = _next_earnings_date(cat)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Catalyst build failed for %s: %s", symbol, exc)

        review.attention = _attention(review)
        return review
    finally:
        store.close()


def _next_earnings_date(cat) -> str | None:  # type: ignore[no-untyped-def]
    """Nearest future earnings date (ISO string) from a CatalystRiskResult."""
    from advisor.research.models import CatalystType

    today = date.today()
    best: date | None = None
    for c in cat.catalysts:
        if c.catalyst_type != CatalystType.EARNINGS:
            continue
        try:
            d = date.fromisoformat((c.expected_date or "")[:10])
        except ValueError:
            continue
        if d >= today and (best is None or d < best):
            best = d
    return best.isoformat() if best else None


def _attention(review: PositionReview) -> str:
    """HIGH for invalidated thesis or imminent earnings; MEDIUM for caution/events."""
    earnings_soon = False
    if review.next_earnings_date:
        try:
            d = date.fromisoformat(review.next_earnings_date)
            earnings_soon = d <= date.today() + timedelta(days=7)
        except ValueError:
            earnings_soon = False

    if review.thesis_status == "INVALIDATED" or earnings_soon:
        return "HIGH"
    if review.thesis_status == "CAUTION" or review.near_term_catalysts or review.next_earnings_date:
        return "MEDIUM"
    return "LOW"


# ── Orchestration ───────────────────────────────────────────────────────────────


def build_portfolio_review(
    account_numbers: list[str] | None = None,
    *,
    rebuild_uncovered: bool = True,
    with_catalysts: bool = True,
    concurrency: int = 3,
) -> PortfolioReview:
    """Pull holdings and assemble a per-ticker research health review.

    Each ticker is processed independently; a failure on one is captured in its
    ``PositionReview.error`` rather than aborting the run.
    """
    accounts = account_numbers or list(DEFAULT_ACCOUNTS)
    held = get_held_symbols(accounts)
    symbols = sorted(held)

    review = PortfolioReview(account_numbers=accounts)
    if not symbols:
        return review

    def _worker(sym: str) -> PositionReview:
        return _review_symbol(
            sym,
            held[sym],
            rebuild_uncovered=rebuild_uncovered,
            with_catalysts=with_catalysts,
        )

    workers = max(1, min(concurrency, len(symbols)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_worker, symbols))

    # Sort by attention (HIGH first), then symbol, for a useful default order.
    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    results.sort(key=lambda r: (rank.get(r.attention, 3), r.symbol))
    review.positions = results
    return review


# ── Persistence (artifact table — no schema change) ─────────────────────────────


def save_review(review: PortfolioReview) -> None:
    from advisor.research.config import get_settings
    from advisor.research.store import ResearchStore

    store = ResearchStore(get_settings().db_path)
    try:
        store.save_artifact(_REVIEW_SYMBOL, _REVIEW_KIND, _REVIEW_KEY, review.model_dump_json())
    finally:
        store.close()


def load_latest_review() -> tuple[PortfolioReview, datetime] | None:
    """Return (review, fetched_at) for the last saved review, or None."""
    from advisor.research.config import get_settings
    from advisor.research.store import ResearchStore

    store = ResearchStore(get_settings().db_path)
    try:
        loaded = store.load_artifact(_REVIEW_SYMBOL, _REVIEW_KIND, _REVIEW_KEY)
    finally:
        store.close()
    if loaded is None:
        return None
    payload_json, fetched_at = loaded
    return PortfolioReview.model_validate_json(payload_json), fetched_at


# ── Rendering ───────────────────────────────────────────────────────────────────


def render_portfolio_review(review: PortfolioReview) -> str:
    """Render the review as Markdown (for CLI export and dashboard download)."""
    lines: list[str] = []
    lines.append("# Portfolio Research Review")
    lines.append("")
    lines.append(f"_Generated {review.generated_at:%Y-%m-%d %H:%M}_  ")
    lines.append(f"_Accounts: {', '.join(review.account_numbers) or '—'}_")
    lines.append("")
    lines.append(
        f"**{len(review.positions)} holdings** · "
        f"{review.n_invalidated} invalidated · "
        f"{review.n_caution} caution · "
        f"{review.n_near_term_events} with near-term events"
    )
    lines.append("")

    # Summary table
    lines.append(
        "| Ticker | Accounts | Thesis | Conviction | Base upside | Next earnings | Attention |"
    )
    lines.append(
        "|--------|----------|--------|-----------|------------|---------------|-----------|"
    )
    for p in review.positions:
        upside = f"{p.base_upside:+.0%}" if p.base_upside is not None else "—"
        lines.append(
            f"| {p.symbol} "
            f"| {', '.join(p.accounts) or '—'} "
            f"| {p.thesis_status.replace('_', ' ')} "
            f"| {p.conviction or '—'} "
            f"| {upside} "
            f"| {p.next_earnings_date or '—'} "
            f"| {p.attention} |"
        )
    lines.append("")

    # Per-ticker detail
    for p in review.positions:
        lines.append(f"## {p.symbol}" + (f" — {p.company_name}" if p.company_name else ""))
        meta = [f"Thesis: **{p.thesis_status.replace('_', ' ')}**"]
        if p.conviction:
            meta.append(f"Conviction: {p.conviction}")
        if p.base_target is not None:
            up = f" ({p.base_upside:+.0%})" if p.base_upside is not None else ""
            meta.append(f"DCF base target: ${p.base_target:,.2f}{up}")
        lines.append(" · ".join(meta))
        lines.append("")

        if p.report_was_built:
            lines.append("> _New research report built for this run._")
            lines.append("")
        elif not p.has_report:
            lines.append("> _No research report on file — thesis monitoring limited._")
            lines.append("")

        if p.kpi_alerts:
            lines.append("**KPI alerts**")
            for a in p.kpi_alerts:
                lines.append(f"- {a}")
            lines.append("")

        if p.next_earnings_date:
            lines.append(f"**Next earnings:** {p.next_earnings_date}")
            lines.append("")

        if p.near_term_catalysts:
            lines.append("**Near-term catalysts**")
            for c in p.near_term_catalysts:
                lines.append(f"- {c}")
            lines.append("")

        if p.error:
            lines.append(f"> ⚠ {p.error}")
            lines.append("")

    return "\n".join(lines)
