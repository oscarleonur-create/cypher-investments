"""Pydantic domain models for options advisor."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from advisor.core.enums import OptionType, OrderSide


class Greeks(BaseModel):
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


class OptionContract(BaseModel):
    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiration: date
    premium: float = 0.0
    implied_volatility: float = 0.0
    greeks: Greeks = Field(default_factory=Greeks)
    open_interest: int = 0
    volume: int = 0

    @property
    def is_expired(self) -> bool:
        return date.today() >= self.expiration

    @property
    def days_to_expiry(self) -> int:
        return max(0, (self.expiration - date.today()).days)


class Position(BaseModel):
    symbol: str
    quantity: float
    avg_price: float
    side: OrderSide
    option_contract: OptionContract | None = None
    opened_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_option(self) -> bool:
        return self.option_contract is not None

    @property
    def market_value(self) -> float:
        multiplier = 100 if self.is_option else 1
        return self.quantity * self.avg_price * multiplier


class Trade(BaseModel):
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.now)
    option_contract: OptionContract | None = None
    commission: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_option(self) -> bool:
        return self.option_contract is not None

    @property
    def total_cost(self) -> float:
        multiplier = 100 if self.is_option else 1
        return self.quantity * self.price * multiplier + self.commission
