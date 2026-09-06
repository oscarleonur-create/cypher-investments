"""Book-level factor exposure.

Aggregates per-symbol loadings into one answer to "how am I actually
positioned". Weights are **signed** notional over net liq, so a short position
subtracts its exposure — which is the whole point once options are in the book:
a short put adds positive market beta, and it has to net against a hedge rather
than add to it.

Two honest limitations, both consequences of correlated factors:

- Individual loadings should not be read as clean causal betas. MKT and SIZE
  move together, so ridge splits attribution between them in a way that is
  stable but not uniquely interpretable.
- What *is* well defined is prediction: expected return under a set of observed
  factor moves, and the residual left over. Those recombine the factors exactly
  as the fit did, so the collinearity cancels.

Use this for "what does today's move mean for my book" and for ranking who
carries an exposure. Do not use a single loading as a standalone fact.
"""

from __future__ import annotations

import logging
from datetime import date

from pydantic import BaseModel, Field

from advisor.daemon.book import BookSnapshot
from advisor.macro.sensitivity import SymbolSensitivity

logger = logging.getLogger(__name__)


class FactorExposure(BaseModel):
    """The book's net loading on one factor, and who carries it."""

    factor: str
    net_loading: float
    contributors: list[tuple[str, float]] = Field(default_factory=list)  # (symbol, share)

    @property
    def top_contributors(self) -> list[tuple[str, float]]:
        return self.contributors[:3]

    def concentration(self) -> float:
        """Share of the absolute exposure carried by the top three names."""
        total = sum(abs(share) for _, share in self.contributors)
        if total <= 0:
            return 0.0
        return sum(abs(share) for _, share in self.top_contributors) / total


class BookExposure(BaseModel):
    """The whole book's factor profile at a point in time."""

    asof: date
    net_liq: float
    covered_weight: float  # fraction of gross notional with a usable estimate
    uncovered: list[str] = Field(default_factory=list)  # symbols with no estimate
    factors: dict[str, FactorExposure] = Field(default_factory=dict)

    def loading(self, factor: str) -> float:
        entry = self.factors.get(factor)
        return entry.net_loading if entry else 0.0

    def ranked(self) -> list[FactorExposure]:
        """Factors ordered by absolute size — the book's biggest bets first."""
        return sorted(self.factors.values(), key=lambda f: -abs(f.net_loading))

    def expected_move(self, factor_moves: dict[str, float]) -> float:
        """Expected book return (as a fraction of net liq) for given factor moves."""
        return sum(self.loading(f) * move for f, move in factor_moves.items())


def build_book_exposure(
    book: BookSnapshot,
    sensitivities: dict[str, SymbolSensitivity],
    *,
    asof: date | None = None,
) -> BookExposure:
    """Weight per-symbol loadings by signed notional to get the book profile.

    Positions without a sensitivity estimate (too little history, no data) are
    reported in ``uncovered`` and excluded from the aggregate rather than
    treated as zero exposure — "unknown" and "neutral" are different answers,
    and conflating them understates the book's real bets.
    """
    net_liq = book.net_liq
    exposure = BookExposure(asof=asof or book.as_of.date(), net_liq=net_liq, covered_weight=0.0)
    if net_liq <= 0 or not book.positions:
        exposure.uncovered = book.symbols
        return exposure

    gross = book.gross_notional
    covered_notional = 0.0
    uncovered: set[str] = set()
    accumulator: dict[str, list[tuple[str, float]]] = {}

    for position in book.positions:
        symbol = position.underlying.upper()
        sensitivity = sensitivities.get(symbol)
        if sensitivity is None:
            uncovered.add(symbol)
            continue
        covered_notional += position.notional
        weight = position.signed_notional / net_liq
        for entry in sensitivity.loadings:
            accumulator.setdefault(entry.factor, []).append((symbol, weight * entry.loading))

    for factor, parts in accumulator.items():
        merged: dict[str, float] = {}
        for symbol, share in parts:
            merged[symbol] = merged.get(symbol, 0.0) + share
        ordered = sorted(merged.items(), key=lambda kv: -abs(kv[1]))
        exposure.factors[factor] = FactorExposure(
            factor=factor,
            net_loading=round(sum(merged.values()), 4),
            contributors=[(s, round(v, 4)) for s, v in ordered],
        )

    exposure.uncovered = sorted(uncovered)
    exposure.covered_weight = round(covered_notional / gross, 4) if gross > 0 else 0.0
    return exposure
