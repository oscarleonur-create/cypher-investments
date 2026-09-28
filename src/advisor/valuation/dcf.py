"""One discounted-cash-flow engine, run in both directions.

Everything in this repo that prices a company goes through here: the daemon's
weekly valuation, the research workstation's DCF, its reverse DCF, and the
Bayesian Monte Carlo (a vectorised mirror of ``project``, pinned by a test).
There used to be two engines, and they were wrong in different ways. Measured
on 2026-09-27:

- The daemon's implied growth did not discount. It asked what revenue would
  make the business worth today's enterprise value *in ten years* — a 0%
  return — and ignored the cash in between. At a 25% margin that had AMZN
  "requiring" revenue to shrink 6% a year for a decade, JBL 17%.
- The research DCF valued its exit multiple on an EBITDA of 25% of revenue for
  every company, and took price, shares and net debt from ``yfinance.info``.
  Rate-limited, that returned nothing, shares defaulted to 1.0 and the price
  to 0, and JBL was reported at a fair value of $0.00 without a warning.
- Its reverse DCF rebuilt base revenue as year-one FCF over the *target*
  margin, which is neither the base year nor the margin that FCF was made at.

The model is deliberately plain. Revenue grows at today's rate for a number
of years and then fades in a straight line to terminal growth by year ten;
the free-cash-flow margin moves in a straight line from today's to a steady
state reached in year ten; cash flows are discounted at a stated rate; the
tail is a Gordon perpetuity. The workstation's sliders move a two-step shape
(years 1–3, years 4–10), which is the same engine with a flat path in each
step. Every input is visible on every
answer, because a valuation whose assumptions are hidden cannot be argued with.

Forward, it turns stated assumptions into a value per share. Backward, it
turns a price into the growth — or the margin — that price requires. The
backward reading is arithmetic and falsifiable; the forward one is an opinion,
which is why it is only ever published as a range built from the company's
own filed margins, and refused when the filings offer no positive margin to
build it from.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

YEARS = 10
EARLY_YEARS = 3  # years 1–3 grow at ``growth_early``; the rest at ``growth_late``

# A stated hurdle, fixed so a thesis tests the same quantity from week to week.
# On 2026-09-24 the 10-year Treasury closed at 5.18% (FRED DGS10); 10% is that
# plus a 4.8% equity premium for an average-risk stock. Per-name betas are
# noise at this precision; the rate is shown beside every answer instead, and
# it is the investor's requirement, not a scenario of the business — so it is
# the same in the bear, base and bull cases.
DISCOUNT_RATE = 0.10
TERMINAL_GROWTH = 0.03

# A range needs at least this many positive readings of the steady-state
# margin. With one, bear, base and bull share the assumption that moves the
# answer most, and the range is falsely narrow: COHR's lone 3.3% (a three-year
# median, against -14.4% today) valued it at $0.00 in every case on
# 2026-09-27, INTC's lone 5.0% at $5–$7 against $123.
MIN_MARGIN_READINGS = 2

# How many years each case holds today's growth before it fades to terminal.
# The scenarios differ in how long the business keeps growing as it is, not
# by multiplying today's rate. A first version faded every case from year one
# and moved growth ±25%; the bull then sat below the price for MSFT, AMZN,
# AMD and PENG (2026-09-28), because it assumed 18% growth was already down
# to 14% by year three. Holding growth for a stated period and then fading it
# is the standard two-stage shape.
HELD_YEARS = {"bear": 0, "base": 3, "bull": 5}

# Today's margin is where the fade starts. Cash-burning names report margins of
# -400%; faded over ten years that sinks every projected year, so the start is
# bounded. The Bayesian mirror applies the same bounds.
START_MARGIN_BOUNDS = (-0.30, 0.40)

# Year-one growth is where the fade starts, bounded the same way: NBIS grew
# over 300% in 2026 and a decade faded from there values it on arithmetic, not
# on the business. The bound is shown with the answer when it binds.
GROWTH_BOUNDS = (-0.30, 0.60)

# Brackets for the backward solvers. A price outside them is reported as
# beyond the bracket, never as the bracket's edge.
GROWTH_BRACKET = (-0.50, 1.50)
MARGIN_BRACKET = (0.0, 1.0)

# The tax rate applied to operating income to read a steady-state cash margin
# off the income statement, where the cash-flow statement is distorted by a
# capex cycle.
NOPAT_TAX = 0.21


@dataclass(frozen=True)
class Path:
    """The assumptions one projection runs on. Rates are fractions."""

    growth_early: float  # revenue growth, years 1–3
    growth_late: float  # revenue growth, years 4–10
    target_margin: float  # steady-state FCF margin, reached in year ten
    discount_rate: float = DISCOUNT_RATE
    terminal_growth: float = TERMINAL_GROWTH
    # Year-by-year growth. When given it is the path, and the two steps above
    # are only its averages, for display and for the sliders' starting point.
    growth: tuple[float, ...] = ()

    def rates(self, years: int = YEARS) -> tuple[float, ...]:
        if self.growth:
            return tuple(self.growth)
        return tuple(
            self.growth_early if y <= EARLY_YEARS else self.growth_late for y in range(1, years + 1)
        )


@dataclass(frozen=True)
class Projection:
    """Ten years of revenue and free cash flow, discounted."""

    revenue: tuple[float, ...]
    fcf: tuple[float, ...]
    pv_fcf: float
    pv_terminal: float

    @property
    def enterprise_value(self) -> float:
        return self.pv_fcf + self.pv_terminal

    @property
    def terminal_share(self) -> float | None:
        """How much of the value sits in the perpetuity — usually most of it."""
        ev = self.enterprise_value
        return self.pv_terminal / ev if ev > 0 else None


def clamp(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, value))


def gordon_multiple(discount_rate: float, terminal_growth: float) -> float | None:
    """Terminal value over the final year's FCF: what the perpetuity implies."""
    if discount_rate <= terminal_growth:
        return None
    return (1 + terminal_growth) / (discount_rate - terminal_growth)


def project(
    base_revenue: float, start_margin: float, path: Path, *, years: int = YEARS
) -> Projection | None:
    """Project and discount. None when the inputs cannot describe a business.

    A perpetuity needs a discount rate above its growth; a projection needs
    revenue to start from. A negative terminal cash flow carries no terminal
    value — a business that burns cash forever is not worth an infinite amount
    of anything.
    """
    rates = path.rates(years)
    if len(rates) != years or not _finite(base_revenue, start_margin, *rates):
        return None
    if not _finite(path.target_margin, path.discount_rate, path.terminal_growth):
        return None
    if base_revenue <= 0 or years <= 0 or path.discount_rate <= path.terminal_growth:
        return None
    if any(g <= -1 for g in rates):
        return None

    start = clamp(start_margin, START_MARGIN_BOUNDS)
    r = path.discount_rate
    revenue, revenues, fcfs, pv = base_revenue, [], [], 0.0
    for year, growth in enumerate(rates, 1):
        revenue *= 1 + growth
        margin = start + (path.target_margin - start) * year / years
        fcf = revenue * margin
        revenues.append(revenue)
        fcfs.append(fcf)
        pv += fcf / (1 + r) ** year

    terminal_fcf = fcfs[-1] * (1 + path.terminal_growth)
    terminal = terminal_fcf / (r - path.terminal_growth) if terminal_fcf > 0 else 0.0
    return Projection(
        revenue=tuple(revenues),
        fcf=tuple(fcfs),
        pv_fcf=pv,
        pv_terminal=terminal / (1 + r) ** years,
    )


def per_share(enterprise_value: float, net_cash: float, shares: float) -> float | None:
    """Equity value per share. Limited liability floors equity at zero."""
    if not _finite(enterprise_value, net_cash, shares) or shares <= 0:
        return None
    return max(enterprise_value + net_cash, 0.0) / shares


def market_ev(price: float, shares: float, net_cash: float) -> float | None:
    """What the market charges for the operating business."""
    if not _finite(price, shares, net_cash) or price <= 0 or shares <= 0:
        return None
    return price * shares - net_cash


def growth_path(
    current: float, held: int, terminal: float = TERMINAL_GROWTH, *, years: int = YEARS
) -> tuple[float, ...]:
    """Today's growth for ``held`` years, then a straight line to ``terminal``
    in the final year. ``held=0`` fades from year one."""
    held = max(0, min(held, years - 1))
    if held == 0:
        return tuple(
            current + (terminal - current) * (y - 1) / (years - 1) for y in range(1, years + 1)
        )
    return tuple(
        current if y <= held else current + (terminal - current) * (y - held) / (years - held)
        for y in range(1, years + 1)
    )


def step_averages(rates: tuple[float, ...]) -> tuple[float, float]:
    """(years 1–3, years 4–10) averages of a yearly path: the two-step shape
    the workstation's sliders move."""
    early, late = rates[:EARLY_YEARS], rates[EARLY_YEARS:]
    return sum(early) / len(early), sum(late) / len(late)


def path_for(
    current: float,
    held: int,
    target_margin: float,
    *,
    discount_rate: float = DISCOUNT_RATE,
    terminal_growth: float = TERMINAL_GROWTH,
) -> Path:
    """A full path: today's growth held, faded, and the steady-state margin."""
    rates = growth_path(current, held, terminal_growth)
    early, late = step_averages(rates)
    return Path(early, late, target_margin, discount_rate, terminal_growth, growth=rates)


# ── Backward: what a price requires ──────────────────────────────────────────


@dataclass(frozen=True)
class Solved:
    """A backward solve. ``value`` is None when the answer is off the bracket,
    and ``beyond`` says which side — "above" means the price needs more than
    the bracket's top."""

    value: float | None
    beyond: str | None = None

    def describe(self, fmt: str = "{:.1%}") -> str:
        if self.value is not None:
            return fmt.format(self.value)
        return f"beyond the {self.beyond} bound" if self.beyond else "unsolvable"


def _bisect(fn, target: float, lo: float, hi: float, *, iters: int = 100) -> Solved:
    """Root of ``fn(x) = target`` for ``fn`` increasing on [lo, hi]."""
    f_lo, f_hi = fn(lo), fn(hi)
    if f_lo is None or f_hi is None:
        return Solved(None)
    if target < f_lo:
        return Solved(None, "below")
    if target > f_hi:
        return Solved(None, "above")
    for _ in range(iters):
        mid = (lo + hi) / 2
        value = fn(mid)
        if value is None:
            return Solved(None)
        if value < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return Solved((lo + hi) / 2)


def implied_growth(
    enterprise_value: float,
    base_revenue: float,
    start_margin: float,
    target_margin: float,
    *,
    discount_rate: float = DISCOUNT_RATE,
    terminal_growth: float = TERMINAL_GROWTH,
) -> Solved:
    """The constant revenue growth for ten years that the price requires.

    "The price requires 18% a year for a decade" — at the stated margin,
    discount rate and terminal growth, and with today's margin fading to the
    stated one. Constant, not faded, because that is the number a thesis can
    hold a business to year by year.
    """
    if not _finite(enterprise_value) or enterprise_value <= 0 or target_margin <= 0:
        return Solved(None)

    def value(g: float) -> float | None:
        p = project(
            base_revenue,
            start_margin,
            Path(g, g, target_margin, discount_rate, terminal_growth),
        )
        return p.enterprise_value if p else None

    return _bisect(value, enterprise_value, *GROWTH_BRACKET)


def implied_margin(
    enterprise_value: float,
    base_revenue: float,
    start_margin: float | None,
    current_growth: float,
    *,
    held: int = HELD_YEARS["base"],
    discount_rate: float = DISCOUNT_RATE,
    terminal_growth: float = TERMINAL_GROWTH,
) -> Solved:
    """The steady-state FCF margin the price requires at today's growth, held
    for the base case's years and then faded (bounded, as the value range).

    The other axis of the same question. AMZN's filed free-cash-flow margins
    run 1–8% through a capex cycle; asking what margin the price needs at the
    growth it is actually delivering says more than any growth figure could.
    Without a known margin today, the steady state applies from year one.
    """
    if not _finite(enterprise_value) or enterprise_value <= 0:
        return Solved(None)

    growth = clamp(current_growth, GROWTH_BOUNDS)

    def value(m: float) -> float | None:
        p = project(
            base_revenue,
            m if start_margin is None else start_margin,
            path_for(growth, held, m, discount_rate=discount_rate, terminal_growth=terminal_growth),
        )
        return p.enterprise_value if p else None

    return _bisect(value, enterprise_value, *MARGIN_BRACKET)


def required_path(
    enterprise_value: float,
    base_revenue: float,
    start_margin: float,
    target_margin: float,
    *,
    discount_rate: float = DISCOUNT_RATE,
    terminal_growth: float = TERMINAL_GROWTH,
) -> tuple[Solved, Projection | None]:
    """``implied_growth`` plus the projection it solves to (year-ten revenue)."""
    solved = implied_growth(
        enterprise_value,
        base_revenue,
        start_margin,
        target_margin,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
    )
    if solved.value is None:
        return solved, None
    g = solved.value
    return solved, project(
        base_revenue, start_margin, Path(g, g, target_margin, discount_rate, terminal_growth)
    )


# ── Forward: a value range from the company's own figures ────────────────────


@dataclass(frozen=True)
class Margin:
    """One steady-state margin reading and where it came from."""

    value: float
    label: str


@dataclass(frozen=True)
class Scenario:
    name: str  # "bear" | "base" | "bull"
    path: Path
    current_growth: float  # today's rate, held and then faded
    held_years: int  # years today's rate is held before the fade
    margin_label: str
    projection: Projection
    value_per_share: float
    upside: float  # value / price − 1


@dataclass(frozen=True)
class ValueRange:
    """Bear, base and bull values per share, or the reason there are none."""

    scenarios: tuple[Scenario, ...] = ()
    refused: str | None = None
    notes: tuple[str, ...] = field(default=())

    def get(self, name: str) -> Scenario | None:
        return next((s for s in self.scenarios if s.name == name), None)


def steady_state_margins(margins: list[Margin]) -> tuple[Margin, Margin, Margin] | None:
    """(bear, base, bull) from the company's own positive margins.

    Each reading is wrong in a known direction. Free cash flow in a capex
    cycle understates what the business converts at steady state — META's fell
    from 33% to 18% as it built data centres — while operating margin after
    tax ignores the capex entirely. So the lowest is the bear case, the highest
    the bull, and the median the base. A company that has shown no positive
    margin has no steady state to read, and gets None.
    """
    positive = sorted((m for m in margins if m.value > 0), key=lambda m: m.value)
    if not positive:
        return None
    mid = statistics.median(m.value for m in positive)
    base = min(positive, key=lambda m: abs(m.value - mid))
    if len(positive) % 2 == 0:
        labels = " / ".join(
            m.label for m in positive[len(positive) // 2 - 1 : len(positive) // 2 + 1]
        )
        base = Margin(mid, f"median of {labels}")
    return positive[0], base, positive[-1]


def held_years(current: float, terminal: float = TERMINAL_GROWTH) -> dict[str, int]:
    """Years each case holds today's growth. Holding a rate longer is the bull
    case only when that rate is above terminal: for a shrinking business the
    bear holds the decline longest and the bull fades it from year one."""
    if current >= terminal:
        return dict(HELD_YEARS)
    return {"bear": HELD_YEARS["bull"], "base": HELD_YEARS["base"], "bull": HELD_YEARS["bear"]}


def value_range(
    *,
    price: float | None,
    shares: float | None,
    net_cash: float | None,
    base_revenue: float | None,
    start_margin: float | None,
    current_growth: float | None,
    margins: list[Margin],
    discount_rate: float = DISCOUNT_RATE,
    terminal_growth: float = TERMINAL_GROWTH,
) -> ValueRange:
    """Bear/base/bull value per share, or a stated refusal. Pure.

    Refuses rather than defaults. A missing price, share count, revenue or
    growth comparison, or a company with no positive margin on file, produces
    no range and says which — the previous engine defaulted shares to 1.0 and
    price to 0 and published the result.
    """
    for name, value in (
        ("price", price),
        ("share count", shares),
        ("revenue", base_revenue),
    ):
        if value is None or not _finite(value) or value <= 0:
            return ValueRange(refused=f"no {name}")
    if net_cash is None or not _finite(net_cash):
        return ValueRange(refused="no balance sheet (cash and debt)")
    if current_growth is None or not _finite(current_growth):
        return ValueRange(refused="no year-over-year revenue comparison")
    anchors = steady_state_margins(margins)
    if anchors is None:
        return ValueRange(
            refused="no positive margin in the filings to anchor a steady state",
        )
    positive = [m for m in margins if m.value > 0]
    if len(positive) < MIN_MARGIN_READINGS:
        only = positive[0]
        return ValueRange(
            refused=(
                f"one positive margin on file ({only.value:.1%}, {only.label}); a range "
                f"needs at least {MIN_MARGIN_READINGS} readings of the steady state"
            ),
        )

    notes = []
    if current_growth != clamp(current_growth, GROWTH_BOUNDS):
        notes.append(
            f"current growth {current_growth:+.0%} bounded to "
            f"{clamp(current_growth, GROWTH_BOUNDS):+.0%} for year one"
        )
    growth = clamp(current_growth, GROWTH_BOUNDS)
    held = held_years(growth, terminal_growth)
    start = start_margin if start_margin is not None and _finite(start_margin) else None
    # Without a trailing FCF margin every scenario starts from the base case's
    # steady state, so the range differs only by what the scenarios assume.
    origin = start if start is not None else anchors[1].value
    scenarios = []
    for name, margin in zip(("bear", "base", "bull"), anchors):
        path = path_for(
            growth,
            held[name],
            margin.value,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
        )
        projection = project(base_revenue, origin, path)
        if projection is None:
            return ValueRange(refused=f"the {name} projection is undefined")
        value = per_share(projection.enterprise_value, net_cash, shares)
        if value is None:
            return ValueRange(refused="no share count")
        scenarios.append(
            Scenario(
                name=name,
                path=path,
                current_growth=growth,
                held_years=held[name],
                margin_label=margin.label,
                projection=projection,
                value_per_share=value,
                upside=value / price - 1,
            )
        )
    if start is None:
        notes.append("no trailing free cash flow: the base steady state applies from year one")
    return ValueRange(scenarios=tuple(scenarios), notes=tuple(notes))


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
