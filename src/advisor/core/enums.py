"""Domain enumerations for the options advisor."""

from enum import StrEnum


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class StrategyType(StrEnum):
    EQUITY = "equity"
    OPTIONS = "options"
    MIXED = "mixed"
