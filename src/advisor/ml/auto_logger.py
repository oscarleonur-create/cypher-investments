"""Auto-logger — batch-log scanner output with price lookup."""

from __future__ import annotations

from typing import Any, Dict, List

import yfinance as yf

from advisor.ml.outcome_tracker import log_signal


def auto_log_from_scanner_output(
    scanner_name: str,
    results_list: List[Dict[str, Any]],
) -> int:
    """Log signals from a scanner's output list. Each item needs: ticker, score, verdict.

    Fetches current prices via batch yfinance download.
    Returns number of signals logged.
    """
    if not results_list:
        return 0

    tickers = list({r["ticker"].upper() for r in results_list})
    prices: Dict[str, float] = {}

    try:
        data = yf.download(tickers, period="1d", progress=False)
        if len(tickers) == 1:
            last = data["Close"].dropna()
            if len(last) > 0:
                prices[tickers[0]] = float(last.iloc[-1])
        else:
            for tk in tickers:
                try:
                    last = data[tk]["Close"].dropna()
                    if len(last) > 0:
                        prices[tk] = float(last.iloc[-1])
                except Exception:
                    pass
    except Exception:
        pass

    count = 0
    for r in results_list:
        tk = r["ticker"].upper()
        metadata = {k: v for k, v in r.items() if k not in ("ticker", "score", "verdict")}
        log_signal(
            ticker=tk,
            scanner=scanner_name,
            score=r.get("score", 0),
            verdict=r.get("verdict", ""),
            price=prices.get(tk, 0.0),
            metadata=metadata,
        )
        count += 1
    return count
