"""Intraday setup scanner — a measurement bench, not an alert system.

Two setups, each named after how the user already trades (see
``docs/swing-entry-plan.md`` and the September 2026 review of 45 short trades):

- **A — catalyst gap.** A stock opens well above yesterday's close on fresh
  news and trades on heavy volume. Traded long, with the move.
- **C — news dip.** A large company falls hard on news the user judges does
  not change the business. Traded long, against the move.

The scanner records every candidate and, later, what the price did next. It
never alerts, never ranks, and never says "buy": until the recorded outcomes
say a setup works, a notification would be an opinion formatted as data.
"""
