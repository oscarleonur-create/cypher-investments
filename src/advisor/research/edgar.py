"""SEC EDGAR client — thin wrapper over `edgartools`.

Responsibilities (Phase 1):
- Set SEC-required identity once per process.
- Fetch filings by form (10-K / 10-Q / 8-K / DEF 14A / 13F-HR / Form 4).
- Pull 5-yr financial statement frames as raw dicts (parsed in statements.py).
- Disk-cache filing lists; statement frames are cached one layer up.

`edgartools` is imported lazily so tests can mock without the package
installed.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime
from typing import Any

from advisor.data.cache import DiskCache
from advisor.research.config import ResearchSettings, get_settings
from advisor.research.models import FilingRef, FormType

logger = logging.getLogger(__name__)


_IDENTITY_SET: bool = False
_IDENTITY_LOCK = threading.Lock()


def _ensure_identity(user_agent: str) -> None:
    """Set the SEC User-Agent once per process. Idempotent and thread-safe."""
    global _IDENTITY_SET
    if _IDENTITY_SET:
        return
    with _IDENTITY_LOCK:
        if _IDENTITY_SET:
            return
        from edgar import set_identity

        set_identity(user_agent)
        _IDENTITY_SET = True


class RateLimiter:
    """Simple sliding-window rate limiter — SEC asks for ≤10 req/s."""

    def __init__(self, max_per_second: int) -> None:
        self._interval = 1.0 / max(1, max_per_second)
        self._last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last_call
            if delta < self._interval:
                time.sleep(self._interval - delta)
            self._last_call = time.monotonic()


class EdgarClient:
    """Wraps `edgartools` with caching and rate-limiting."""

    def __init__(
        self,
        settings: ResearchSettings | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or DiskCache(
            cache_dir=self.settings.cache_dir,
            ttl_hours=self.settings.filing_ttl_days * 24,
        )
        self.rate_limiter = RateLimiter(self.settings.edgar_rate_limit_per_sec)

    # ── Filings ───────────────────────────────────────────────────────────

    def list_filings(
        self,
        symbol: str,
        form: FormType,
        lookback_years: int = 5,
    ) -> list[FilingRef]:
        """Return filings of a given form for the trailing `lookback_years`."""
        cache_key = ("edgar", "filings", symbol.upper(), form.value, str(lookback_years))
        cached = self.cache.get_json(*cache_key)
        if cached is not None:
            return [FilingRef.model_validate(row) for row in cached["filings"]]

        _ensure_identity(self.settings.edgar_user_agent)
        self.rate_limiter.acquire()

        from edgar import Company

        start = date(date.today().year - lookback_years, 1, 1).isoformat()
        end = date.today().isoformat()

        company = Company(symbol.upper())
        raw = company.get_filings(form=form.value, date=f"{start}:{end}")

        refs: list[FilingRef] = []
        for f in raw:
            try:
                refs.append(_filing_to_ref(f, form))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse filing for %s: %s", symbol, exc)

        self.cache.set_json({"filings": [r.model_dump(mode="json") for r in refs]}, *cache_key)
        return refs

    # ── Statements (raw frames) ───────────────────────────────────────────

    def get_financials(self, symbol: str, n_years: int = 5) -> dict[str, Any]:
        """Pull the latest 10-K's multi-period financial frames as raw dicts.

        Returns dict with keys: 'income', 'balance', 'cashflow', each a list
        of period rows. Parsing into typed models is done in statements.py.
        """
        cache_key = ("edgar", "financials", symbol.upper(), str(n_years))
        cached = self.cache.get_json(*cache_key)
        if cached is not None:
            return cached

        _ensure_identity(self.settings.edgar_user_agent)
        self.rate_limiter.acquire()

        from edgar import Company

        company = Company(symbol.upper())
        financials = company.financials  # edgartools Financials helper

        income = _frame_to_records(getattr(financials, "income_statement", None))
        balance = _frame_to_records(getattr(financials, "balance_sheet", None))
        cashflow = _frame_to_records(getattr(financials, "cashflow_statement", None))

        # Trim to requested year count (most-recent first)
        payload = {
            "income": income[:n_years],
            "balance": balance[:n_years],
            "cashflow": cashflow[:n_years],
        }
        self.cache.set_json(payload, *cache_key)
        return payload


# ── Helpers ────────────────────────────────────────────────────────────────


def _filing_to_ref(f: Any, form: FormType) -> FilingRef:
    """Convert an edgartools Filing into the typed FilingRef shape."""
    filing_date = _coerce_date(getattr(f, "filing_date", None))
    period = _coerce_date(getattr(f, "period_of_report", None))
    return FilingRef(
        accession_number=str(getattr(f, "accession_no", getattr(f, "accession_number", ""))),
        form=form,
        filing_date=filing_date or date.today(),
        period_of_report=period,
        url=str(getattr(f, "filing_url", getattr(f, "url", ""))),
    )


def _frame_to_records(frame: Any) -> list[dict[str, Any]]:
    """Convert an edgartools statement frame (DataFrame-like) to list[dict]."""
    if frame is None:
        return []
    # edgartools returns a wrapper with `.data` (DataFrame) or behaves DataFrame-like
    df = getattr(frame, "data", frame)
    if df is None or not hasattr(df, "to_dict"):
        return []
    # Periods are columns; transpose so each row is a period
    try:
        return df.T.reset_index().to_dict(orient="records")
    except Exception:  # noqa: BLE001
        return df.to_dict(orient="records") if hasattr(df, "to_dict") else []


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None
