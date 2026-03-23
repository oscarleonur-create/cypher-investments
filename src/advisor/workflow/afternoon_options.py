"""Afternoon workflow — Options mode (4 phases)."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

from advisor.workflow.models import (
    AfternoonPositionStatus,
    AfternoonResult,
    BrierScoreSummary,
    EarningsAlert,
    PositionAction,
    WatchCandidate,
    WorkflowState,
)
from advisor.workflow.shared import run_account_snapshot

logger = logging.getLogger(__name__)


class OptionsAfternoonWorkflow:
    """4-phase afternoon review for options positions."""

    def __init__(
        self,
        skip_account: bool = False,
        skip_validation: bool = False,
        account_number: str | None = None,
    ):
        self.skip_account = skip_account
        self.skip_validation = skip_validation
        self.account_number = account_number

    def run(self) -> AfternoonResult:
        start = time.time()
        errors: list[str] = []
        state = WorkflowState.load()
        result = AfternoonResult(mode="options")

        # Phase 1: Account Update
        if not self.skip_account:
            result.account = run_account_snapshot("options", account_number=self.account_number)
        else:
            errors.append("Account snapshot skipped (offline mode)")

        # Phase 2: Position Management
        result.position_statuses = self._check_positions(result.account.positions, errors)

        # Phase 3: Prediction Validation
        if not self.skip_validation:
            result.brier_summary = self._validate_predictions(errors)

        # Phase 4: Next-Day Prep
        all_symbols = self._get_all_symbols(result, state)
        result.watch_updates = self._recheck_watch(
            state.watch_candidates, state.morning_alpha_scores, errors
        )
        result.earnings_alerts = self._check_earnings(all_symbols, result.account.positions, errors)
        result.tomorrow_watchlist = self._build_tomorrow_watchlist(
            result.watch_updates, state.watchlist
        )

        result.errors = errors
        result.elapsed_seconds = round(time.time() - start, 2)
        return result

    # ── Phase 2: Position Management ─────────────────────────────────────

    def _check_positions(
        self,
        positions: list[dict],
        errors: list[str],
    ) -> list[AfternoonPositionStatus]:
        results: list[AfternoonPositionStatus] = []

        for pos in positions:
            try:
                sym = pos.get("symbol", "")
                underlying = pos.get("underlying_symbol") or sym
                avg_price = pos.get("average_open_price", 0)
                current = (
                    pos.get("close_price", 0) or pos.get("mark_price", 0) or pos.get("mark", 0)
                )
                direction = pos.get("quantity_direction", "Long")

                # P&L for credit trades
                if direction == "Short" and avg_price > 0:
                    pnl_pct = (avg_price - current) / avg_price * 100
                elif avg_price > 0 and current > 0:
                    pnl_pct = (current - avg_price) / avg_price * 100
                else:
                    pnl_pct = 0

                action = PositionAction.HOLD
                reason = f"P&L: {pnl_pct:+.1f}%"
                limit_price = None

                # Profit target for credit trades
                if direction == "Short" and pnl_pct >= 50:
                    action = PositionAction.CLOSE
                    reason = f"Profit target hit ({pnl_pct:.0f}% of credit)"
                    # BTC at ~50% of credit received
                    limit_price = round(avg_price * 0.50, 2)

                # DTE check
                dte = self._parse_dte(sym)
                if dte is not None and dte <= 7:
                    if action == PositionAction.HOLD:
                        action = PositionAction.ROLL
                        reason = f"DTE={dte} — roll to next month or close"

                # Earnings proximity
                earnings_days = self._days_to_earnings(underlying)
                if earnings_days is not None and earnings_days <= 3:
                    if action == PositionAction.HOLD:
                        action = PositionAction.CLOSE
                        reason = f"Close before earnings in {earnings_days} days"

                results.append(
                    AfternoonPositionStatus(
                        symbol=sym,
                        action=action,
                        reason=reason,
                        pnl_pct=round(pnl_pct, 2),
                        dte=dte,
                        limit_price=limit_price,
                    )
                )
            except Exception as e:
                errors.append(f"Position check failed for {pos.get('symbol', '?')}: {e}")

        return results

    def _parse_dte(self, symbol: str) -> int | None:
        from datetime import datetime

        try:
            parts = symbol.split()
            if len(parts) >= 2:
                date_part = parts[1][:6]
                exp_date = datetime.strptime(date_part, "%y%m%d").date()
                return (exp_date - date.today()).days
        except Exception:
            pass
        return None

    # ── Phase 3: Prediction Validation ───────────────────────────────────

    def _validate_predictions(self, errors: list[str]) -> BrierScoreSummary:
        summary = BrierScoreSummary()

        # Resolve outcomes
        n_resolved = 0
        try:
            from advisor.simulator.db import SimulatorStore
            from advisor.simulator.validation import resolve_outcomes

            store = SimulatorStore()
            outcomes = resolve_outcomes(store)
            n_resolved = len(outcomes)
            summary.n_resolved = n_resolved
        except Exception as e:
            errors.append(f"Outcome resolution failed: {e}")
            return summary

        # Compute Brier scores
        try:
            brier = store.compute_brier_scores(lookback_days=90)
            summary.pop_brier = brier.get("pop_brier")
            summary.touch_brier = brier.get("touch_brier")
            summary.stop_brier = brier.get("stop_brier")
            summary.n_samples = brier.get("n_samples", 0)

            if summary.pop_brier is not None and summary.pop_brier > 0.25:
                summary.warning = (
                    f"MODEL CALIBRATION WARNING — POP Brier {summary.pop_brier:.3f} > 0.25, "
                    "consider retraining"
                )
        except Exception as e:
            errors.append(f"Brier score computation failed: {e}")

        return summary

    # ── Phase 4: Next-Day Prep ───────────────────────────────────────────

    def _recheck_watch(
        self,
        watch_candidates: list[str],
        morning_scores: dict[str, float],
        errors: list[str],
    ) -> list[WatchCandidate]:
        results: list[WatchCandidate] = []

        for sym in watch_candidates:
            try:
                wc = WatchCandidate(symbol=sym)

                # Re-check alpha score
                try:
                    from advisor.confluence.alpha_scorer import compute_alpha

                    alpha = compute_alpha(sym, skip_layers={"sentiment"})
                    wc.alpha_score = round(alpha.alpha_score, 2)
                    morning_score = morning_scores.get(sym)
                    if morning_score is not None:
                        wc.score_change = round(wc.alpha_score - morning_score, 2)
                except Exception as e:
                    logger.debug("Alpha re-check failed for %s: %s", sym, e)

                # Check IV rank for sell candidates
                try:
                    from advisor.market.iv_analysis import compute_iv_percentile

                    iv = compute_iv_percentile(sym)
                    if iv.iv_rank is not None and iv.iv_rank >= 70:
                        wc.status = f"IV rank {iv.iv_rank:.0f} — good sell candidate"
                    elif wc.score_change and wc.score_change > 5:
                        wc.status = f"Score improved +{wc.score_change:.0f}"
                    else:
                        wc.status = "No change"
                except Exception:
                    if wc.score_change and wc.score_change > 5:
                        wc.status = f"Score improved +{wc.score_change:.0f}"
                    else:
                        wc.status = "No change"

                results.append(wc)
            except Exception as e:
                errors.append(f"Watch re-check failed for {sym}: {e}")

        return results

    def _check_earnings(
        self,
        symbols: list[str],
        positions: list[dict],
        errors: list[str],
    ) -> list[EarningsAlert]:
        alerts: list[EarningsAlert] = []
        position_symbols = {
            (pos.get("underlying_symbol") or pos.get("symbol", "")).upper() for pos in positions
        }

        for sym in symbols:
            try:
                days = self._days_to_earnings(sym)
                if days is not None and days <= 7:
                    alerts.append(
                        EarningsAlert(
                            symbol=sym,
                            earnings_date=str(date.today() + timedelta(days=days)),
                            days_until=days,
                            has_position=sym.upper() in position_symbols,
                        )
                    )
            except Exception as e:
                logger.debug("Earnings check failed for %s: %s", sym, e)

        alerts.sort(key=lambda a: a.days_until)
        return alerts

    def _build_tomorrow_watchlist(
        self,
        watch_updates: list[WatchCandidate],
        current_watchlist: list[str],
    ) -> list[str]:
        # Promote high IV rank candidates
        promoted = [w.symbol for w in watch_updates if "good sell" in (w.status or "")]
        improved = [
            w.symbol for w in watch_updates if w.score_change is not None and w.score_change > 5
        ]
        result = (
            promoted
            + improved
            + [s for s in current_watchlist if s not in promoted and s not in improved]
        )
        return result[:20]

    def _get_all_symbols(self, result: AfternoonResult, state: WorkflowState) -> list[str]:
        symbols = set()
        for pos in result.account.positions:
            sym = pos.get("underlying_symbol") or pos.get("symbol", "")
            if sym:
                symbols.add(sym.upper())
        symbols.update(s.upper() for s in state.watchlist)
        symbols.update(s.upper() for s in state.watch_candidates)
        return list(symbols)

    def _days_to_earnings(self, symbol: str) -> int | None:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is not None:
                if hasattr(cal, "get"):
                    earnings_date = cal.get("Earnings Date")
                    if earnings_date is not None:
                        if isinstance(earnings_date, list) and earnings_date:
                            ed = earnings_date[0]
                        else:
                            ed = earnings_date
                        if hasattr(ed, "date"):
                            ed = ed.date()
                        return (ed - date.today()).days
        except Exception:
            pass
        return None
