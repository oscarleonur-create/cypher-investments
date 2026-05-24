"""Insider transaction summary (Form 4 proxy via yfinance)."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from advisor.research.models import InsiderSummary, InsiderTransaction

logger = logging.getLogger(__name__)

_C_SUITE_TITLES = {"ceo", "cfo", "coo", "cto", "president", "chairman", "director"}


def get_insiders(symbol: str, lookback_days: int = 365) -> InsiderSummary:
    import yfinance as yf

    ticker = yf.Ticker(symbol.upper())
    transactions: list[InsiderTransaction] = []
    cutoff = date.today() - timedelta(days=lookback_days)

    try:
        df = ticker.insider_transactions
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                tx_date = _safe_date(row.get("Start Date") or row.get("Date"))
                if tx_date and tx_date < cutoff:
                    continue
                shares = _safe_float(row.get("Shares"))
                value = _safe_float(row.get("Value"))
                price = value / shares if (shares and value is not None) else None
                transactions.append(
                    InsiderTransaction(
                        insider_name=str(row.get("Insider", "")),
                        title=str(row.get("Position", row.get("Relation", ""))),
                        transaction_type=_classify(row),
                        shares=shares,
                        price_per_share=price,
                        value_usd=_safe_float(row.get("Value")),
                        transaction_date=tx_date,
                    )
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("insider_transactions failed for %s: %s", symbol, exc)

    net_buying = sum(
        (t.value_usd or 0) * (1 if "purchase" in t.transaction_type.lower() else -1)
        for t in transactions
        if t.transaction_type.lower() not in ("option exercise", "other")
    )

    c_suite_buying = any(
        "purchase" in t.transaction_type.lower()
        and any(title in t.title.lower() for title in _C_SUITE_TITLES)
        for t in transactions
    )

    return InsiderSummary(
        symbol=symbol.upper(),
        transactions=transactions,
        net_buying_usd=net_buying,
        c_suite_buying=c_suite_buying,
        lookback_days=lookback_days,
    )


def _classify(row: object) -> str:
    tx = str(row.get("Transaction", "")).strip().lower()  # type: ignore[union-attr]
    if not tx:
        shares = _safe_float(row.get("Shares"))  # type: ignore[union-attr]
        return "Purchase" if (shares and shares > 0) else "Sale"
    if "purchase" in tx or "buy" in tx:
        return "Purchase"
    if "sale" in tx or "sell" in tx:
        return "Sale"
    if "option" in tx or "exercise" in tx:
        return "Option Exercise"
    return tx.title()


def _safe_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _safe_date(v: object) -> date | None:
    if v is None:
        return None
    try:
        import pandas as pd

        return pd.Timestamp(v).date()
    except Exception:  # noqa: BLE001
        return None
