"""Trade record models and persistence for options tracking."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, computed_field

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
TRADES_FILE = DATA_DIR / "trades.json"


TradeType = Literal["naked_put", "put_credit_spread", "covered_call", "wheel"]


class TradeRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    trade_type: TradeType
    symbol: str
    strike: float
    long_strike: float | None = None
    expiry: str
    premium: float
    contracts: int = 1
    opened_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    closed_at: str | None = None
    close_price: float | None = None
    close_reason: str | None = None
    status: str = "open"

    @computed_field
    @property
    def pnl(self) -> float | None:
        if self.close_price is None:
            return None
        return (self.premium - self.close_price) * 100 * self.contracts


def load_trades() -> list[TradeRecord]:
    if not TRADES_FILE.exists():
        return []
    data = json.loads(TRADES_FILE.read_text())
    return [TradeRecord(**t) for t in data]


def save_trades(trades: list[TradeRecord]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_FILE.write_text(json.dumps([t.model_dump() for t in trades], indent=2))
