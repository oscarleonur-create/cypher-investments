"""The bar store as a panel: one row per session, one column per symbol.

Every family, the universe cut and the outcomes read the same panel, so a
replay over two years and the nightly live run compute from the same numbers
with the same code. Sessions are the union of the days any stored name
traded; a name with no bar on a session is NaN there, never zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from advisor.breadth.store import BreadthStore


@dataclass
class Panel:
    close: pd.DataFrame
    high: pd.DataFrame
    volume: pd.DataFrame

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    def row_of(self, day: date) -> int | None:
        """Index of the last session at or before ``day``."""
        pos = self.sessions.searchsorted(pd.Timestamp(day), side="right") - 1
        return int(pos) if pos >= 0 else None


def build_panel(rows: list[tuple[str, str, float, float | None, float | None]]) -> Panel:
    """From ``(symbol, day, close, high, volume)`` rows. Pure."""
    if not rows:
        empty = pd.DataFrame(index=pd.DatetimeIndex([]))
        return Panel(empty, empty.copy(), empty.copy())
    df = pd.DataFrame(rows, columns=["symbol", "day", "close", "high", "volume"])
    df["day"] = pd.to_datetime(df["day"])
    close = df.pivot(index="day", columns="symbol", values="close").sort_index()
    high = df.pivot(index="day", columns="symbol", values="high").reindex_like(close)
    volume = df.pivot(index="day", columns="symbol", values="volume").reindex_like(close)
    # A bar without a high is still a bar; its close stands in.
    high = high.where(high.notna() | close.isna(), close)
    return Panel(close.astype(float), high.astype(float), volume.astype(float))


def load_panel(
    store: BreadthStore,
    *,
    start: date | None = None,
    end: date | None = None,
    symbols: list[str] | None = None,
) -> Panel:
    """Read symbol by symbol (the table's key leads with it) into arrays.

    The full store is ~4.6M bars; one ``fetchall`` of it as Python tuples
    costs about a gigabyte. Per symbol, each series is a few KB.
    """
    if symbols is None:
        symbols = [r[0] for r in store.conn.execute("SELECT DISTINCT symbol FROM breadth_bars")]
    lo = start.isoformat() if start else "0000"
    hi = end.isoformat() if end else "9999"
    close, high, volume = {}, {}, {}
    for s in symbols:
        rows = store.conn.execute(
            "SELECT day, close, high, volume FROM breadth_bars "
            "WHERE symbol = ? AND day >= ? AND day <= ? ORDER BY day",
            (s, lo, hi),
        ).fetchall()
        if not rows:
            continue
        idx = pd.to_datetime([r[0] for r in rows])
        arr = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
        close[s] = pd.Series(arr[:, 0], index=idx)
        high[s] = pd.Series(np.where(np.isnan(arr[:, 1]), arr[:, 0], arr[:, 1]), index=idx)
        volume[s] = pd.Series(arr[:, 2], index=idx)
    if not close:
        return build_panel([])
    c = pd.DataFrame(close).sort_index()
    return Panel(c, pd.DataFrame(high).reindex_like(c), pd.DataFrame(volume).reindex_like(c))
