"""The book: a normalized snapshot of what is actually held.

Everything downstream — position mechanics, factor exposure, the relevance
gate — reads the book rather than the broker. That keeps one normalization
point for the broker's quirks, and makes the whole system testable without a
live session.

Two quirks this module absorbs:

- ``quantity`` arrives positive with a separate ``quantity_direction``. Short
  positions are stored here as **negative** quantity, because a short put is a
  long-delta, short-vega position and every later calculation needs the sign.
- ``mark_price`` is 0 outside market hours. ``Position.price`` falls back to
  the prior close, so the daemon's overnight and pre-open jobs still see a
  valid book instead of a portfolio marked to zero.
"""

from __future__ import annotations

import logging
from datetime import datetime

from pydantic import BaseModel, Field

from advisor.daemon.market_calendar import now_et

logger = logging.getLogger(__name__)

EQUITY = "EQUITY"
EQUITY_OPTION = "EQUITY_OPTION"


def normalize_instrument(raw: object) -> str:
    """Map the broker's instrument type onto EQUITY / EQUITY_OPTION / other.

    Arrives as an enum (``InstrumentType.EQUITY``) rather than a plain string,
    so this matches on the stringified form.
    """
    text = str(raw).upper()
    if "OPTION" in text:
        return EQUITY_OPTION
    if "EQUITY" in text:
        return EQUITY
    return text.rsplit(".", 1)[-1] or "UNKNOWN"


class Position(BaseModel):
    """One open position, sign-normalized."""

    account: str
    symbol: str  # OCC symbol for options, ticker for equity
    underlying: str
    instrument: str
    quantity: float  # negative when short
    multiplier: int = 1
    avg_open_price: float = 0.0
    close_price: float = 0.0
    mark_price: float = 0.0

    @property
    def is_option(self) -> bool:
        return self.instrument == EQUITY_OPTION

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def price(self) -> float:
        """Current price, falling back to the prior close when unmarked.

        The broker reports ``mark_price`` 0 outside the session; without this
        fallback every overnight job would see a book worth nothing.
        """
        return self.mark_price or self.close_price

    @property
    def notional(self) -> float:
        """Absolute dollar exposure."""
        return abs(self.quantity) * self.price * self.multiplier

    @property
    def signed_notional(self) -> float:
        """Dollar exposure carrying the direction of the position."""
        return self.quantity * self.price * self.multiplier

    @property
    def cost_basis(self) -> float:
        return abs(self.quantity) * self.avg_open_price * self.multiplier

    @property
    def unrealized_pnl(self) -> float:
        if not self.price or not self.avg_open_price:
            return 0.0
        return (self.price - self.avg_open_price) * self.quantity * self.multiplier

    @property
    def unrealized_pct(self) -> float:
        """Return vs entry, as a fraction. Sign follows the position."""
        if not self.avg_open_price or not self.price:
            return 0.0
        move = (self.price - self.avg_open_price) / self.avg_open_price
        return move if self.quantity >= 0 else -move

    def key(self) -> str:
        """Identity across snapshots: one account, one instrument."""
        return f"{self.account}:{self.symbol}"


class BookSnapshot(BaseModel):
    """The whole book at a moment, across every account."""

    as_of: datetime = Field(default_factory=now_et)
    accounts: list[str] = Field(default_factory=list)  # requested
    loaded_accounts: list[str] = Field(default_factory=list)  # actually returned
    positions: list[Position] = Field(default_factory=list)
    net_liq: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    partial: bool = False  # True when an account failed to load

    @property
    def symbols(self) -> list[str]:
        """Distinct underlyings held, sorted."""
        return sorted({p.underlying.upper() for p in self.positions})

    @property
    def equities(self) -> list[Position]:
        return [p for p in self.positions if p.instrument == EQUITY]

    @property
    def options(self) -> list[Position]:
        return [p for p in self.positions if p.is_option]

    @property
    def gross_notional(self) -> float:
        return sum(p.notional for p in self.positions)

    def by_key(self) -> dict[str, Position]:
        return {p.key(): p for p in self.positions}

    def restricted_to_loaded(self, loaded: list[str]) -> BookSnapshot:
        """A view of this snapshot covering only ``loaded`` accounts.

        Used to diff a partial snapshot safely: comparing a full baseline
        against a book that is missing an account would report every position
        in that account as closed.
        """
        keep = set(loaded)
        return self.model_copy(
            update={"positions": [p for p in self.positions if p.account in keep]}
        )

    def weight(self, position: Position) -> float:
        """Position notional as a fraction of net liq (0.0 when net liq is 0)."""
        return position.notional / self.net_liq if self.net_liq > 0 else 0.0

    def exposure_by_underlying(self) -> dict[str, float]:
        """Summed absolute notional per underlying — equity and options together."""
        out: dict[str, float] = {}
        for p in self.positions:
            out[p.underlying.upper()] = out.get(p.underlying.upper(), 0.0) + p.notional
        return out


def position_from_broker(raw: dict, account: str) -> Position:
    """Build a Position from one `tastytrade_client.get_positions` row."""
    direction = str(raw.get("quantity_direction", "Long")).upper()
    quantity = abs(float(raw.get("quantity") or 0))
    if "SHORT" in direction:
        quantity = -quantity
    return Position(
        account=account,
        symbol=str(raw.get("symbol") or ""),
        underlying=str(raw.get("underlying_symbol") or raw.get("symbol") or ""),
        instrument=normalize_instrument(raw.get("instrument_type")),
        quantity=quantity,
        multiplier=int(raw.get("multiplier") or 1),
        avg_open_price=float(raw.get("average_open_price") or 0),
        close_price=float(raw.get("close_price") or 0),
        mark_price=float(raw.get("mark_price") or raw.get("mark") or 0),
    )


async def fetch_book(accounts: list[str] | None = None) -> BookSnapshot:
    """Pull positions and balances for every account into one snapshot.

    An account that fails to load is skipped and the snapshot is flagged
    ``partial`` rather than raising: a broker hiccup on one account must not
    blind the daemon to the other.
    """
    from advisor.api import deps
    from advisor.market.tastytrade_client import get_balances, get_positions
    from advisor.research.portfolio_review import DEFAULT_ACCOUNTS

    account_list = accounts or list(DEFAULT_ACCOUNTS)
    session = await deps.get_tt_session()

    snapshot = BookSnapshot(accounts=account_list)
    for account in account_list:
        try:
            rows = await get_positions(session, account)
            balances = await get_balances(session, account)
        except Exception as exc:  # noqa: BLE001
            logger.warning("book: account %s unavailable: %s", account, exc)
            snapshot.partial = True
            continue
        snapshot.loaded_accounts.append(account)
        snapshot.positions.extend(position_from_broker(r, account) for r in rows)
        snapshot.net_liq += float(balances.get("net_liq") or 0)
        snapshot.cash += float(balances.get("cash") or 0)
        snapshot.buying_power += float(balances.get("buying_power") or 0)
    return snapshot
