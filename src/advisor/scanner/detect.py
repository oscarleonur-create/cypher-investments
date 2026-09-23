"""The two setup rules, as pure functions over a ``Mover``.

Thresholds come from the user's own trades (March–September 2026), not from
a backtest optimiser:

- Setup A winners opened with gaps of +4% to +23% (AMPG, WOLF, ASTS, CCXI,
  META, MU, DAL, ARM). A gap is "held" only if the price is still at least
  that far above yesterday's close when the scan sees it.
- Setup C winners were large caps down 4–6% on the day (AAPL, INTC, META,
  NVDA, NFLX). The losers were small names on news that did change the
  business (POET, PENG) or a macro shock (EWY). The size floor keeps the
  first group and drops most of the second; telling news that changes the
  business from news that does not is left to the user.

Every threshold is inclusive (``>=``), so a move of exactly 4.00% qualifies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time

from advisor.daemon import market_calendar as mc
from advisor.scanner.models import Mover, Setup

# Share of a day's volume normally traded by N minutes after the open. US
# intraday volume is U-shaped, so linear pacing would call almost every stock
# "heavy volume" in the first half hour. Approximate, and deliberately so:
# the rule needs to tell 1x from 3x, not 1.4x from 1.5x.
_VOLUME_CURVE: tuple[tuple[float, float], ...] = (
    (0, 0.0),
    (5, 0.05),
    (30, 0.18),
    (60, 0.28),
    (120, 0.43),
    (240, 0.68),
    (360, 0.90),
    (390, 1.0),
)
_FULL_SESSION_MINUTES = 390


@dataclass(frozen=True)
class Thresholds:
    gap_min: float = 0.04  # A: open at least 4% above prev close
    hold_min: float = 0.04  # A: still at least 4% above when seen
    rvol_min: float = 1.5  # A: volume at 1.5x the usual pace
    a_min_cap: float = 300e6
    a_min_price: float = 2.0
    drop_min: float = 0.04  # C: down at least 4% on the day
    sigma_min: float = 2.0  # C: and at least 2 sigma for this name
    c_min_cap: float = 10e9
    c_min_price: float = 5.0
    min_dollar_volume: float = 1e6  # both: traded at least $1M so far


DEFAULT = Thresholds()


def _valid(x: float | None) -> bool:
    return x is not None and math.isfinite(x) and x > 0


def change(m: Mover) -> float | None:
    if not (_valid(m.price) and _valid(m.prev_close)):
        return None
    return m.price / m.prev_close - 1


def gap(m: Mover) -> float | None:
    if not (_valid(m.open) and _valid(m.prev_close)):
        return None
    return m.open / m.prev_close - 1


def expected_volume_share(now: datetime) -> float | None:
    """Fraction of a normal day's volume expected by ``now``, or None outside session."""
    et = mc.to_et(now)
    if not mc.is_market_open(et):
        return None
    close = mc.session_close(et.date())
    session_minutes = (close.hour * 60 + close.minute) - (9 * 60 + 30)
    elapsed = (et.hour * 60 + et.minute + et.second / 60) - (9 * 60 + 30)
    # Early closes compress the same curve into the shorter session.
    scaled = elapsed * _FULL_SESSION_MINUTES / session_minutes
    for (m0, s0), (m1, s1) in zip(_VOLUME_CURVE, _VOLUME_CURVE[1:]):
        if scaled <= m1:
            return s0 + (s1 - s0) * (scaled - m0) / (m1 - m0)
    return 1.0


def relative_volume(m: Mover, now: datetime) -> float | None:
    """Volume so far against what an ordinary day has traded by now."""
    share = expected_volume_share(now)
    if not share or not _valid(m.avg_volume) or m.volume is None or m.volume < 0:
        return None
    return m.volume / (m.avg_volume * share)


def _liquid(m: Mover, t: Thresholds) -> bool:
    return _valid(m.price) and m.volume is not None and m.volume * m.price >= t.min_dollar_volume


def is_catalyst_gap(m: Mover, now: datetime, t: Thresholds = DEFAULT) -> bool:
    """Setup A, before any news check: a held gap up on heavy volume."""
    g, c, rv = gap(m), change(m), relative_volume(m, now)
    if g is None or c is None or rv is None:
        return False
    if not _valid(m.market_cap) or m.market_cap < t.a_min_cap:
        return False
    return (
        m.price >= t.a_min_price
        and g >= t.gap_min
        and c >= t.hold_min
        and rv >= t.rvol_min
        and _liquid(m, t)
    )


def is_news_dip(
    m: Mover, sigma_daily: float | None, t: Thresholds = DEFAULT
) -> tuple[bool, float | None]:
    """Setup C, before any news check. Returns (qualifies, move in sigmas).

    Without a volatility estimate the 2-sigma test cannot be run. The rule
    then falls back to the absolute floor alone and reports sigma as None, so
    the record shows the weaker test was used rather than hiding it.
    """
    c = change(m)
    if c is None or not _valid(m.market_cap) or m.market_cap < t.c_min_cap:
        return False, None
    if m.price < t.c_min_price or not _liquid(m, t) or c > -t.drop_min:
        return False, None
    if sigma_daily is None or not math.isfinite(sigma_daily) or sigma_daily <= 0:
        return True, None
    z = abs(c) / sigma_daily
    return z >= t.sigma_min, z


def detect(
    m: Mover, now: datetime, sigma_daily: float | None, t: Thresholds = DEFAULT
) -> list[tuple[Setup, float | None]]:
    """Which setups ``m`` qualifies for at ``now``, each with its sigma move."""
    out: list[tuple[Setup, float | None]] = []
    if is_catalyst_gap(m, now, t):
        c = change(m)
        z = abs(c) / sigma_daily if c is not None and sigma_daily and sigma_daily > 0 else None
        out.append((Setup.CATALYST_GAP, z))
    ok, z = is_news_dip(m, sigma_daily, t)
    if ok:
        out.append((Setup.NEWS_DIP, z))
    return out


# Before this, the opening print is still settling and "the gap" is a guess.
FIRST_SCAN = time(9, 35)
