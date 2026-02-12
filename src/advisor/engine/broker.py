"""Options-aware broker extensions for Backtrader."""

from __future__ import annotations

import logging

import backtrader as bt

logger = logging.getLogger(__name__)


class OptionsBroker(bt.brokers.BackBroker):
    """Extended broker with options assignment and expiry handling.

    For Phase 1, this is a thin wrapper. Full options logic (assignment,
    exercise, expiry) will be added as the synthetic options data feed
    is implemented.
    """

    params = (
        ("options_commission", 0.65),  # Per-contract commission
        ("assignment_fee", 0.0),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug("OptionsBroker initialized")
