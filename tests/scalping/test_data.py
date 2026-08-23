"""Tests for candle-frame assembly and serialization (no network)."""

from __future__ import annotations

import pandas as pd
from advisor.scalping.data import _frame_from_rows, candles_to_records


def test_frame_from_rows_sorted_and_columned():
    rows = {
        2_000: {"open": 2, "high": 3, "low": 1, "close": 2.5, "volume": 10, "vwap": 2.2},
        1_000: {"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 8, "vwap": 1.3},
    }
    df = _frame_from_rows(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "vwap"]
    assert df.index.is_monotonic_increasing  # earlier epoch first
    assert df["close"].tolist() == [1.5, 2.5]


def test_candles_to_records_shape():
    idx = pd.to_datetime([1_000, 2_000], unit="ms")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.0],
            "close": [1.5, 2.5],
            "volume": [8.0, 10.0],
            "vwap": [1.3, 2.2],
        },
        index=idx,
    )
    recs = candles_to_records(df)
    assert recs[0] == {
        "t": 1_000,
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "volume": 8.0,
        "vwap": 1.3,
    }
    assert recs[1]["t"] == 2_000
