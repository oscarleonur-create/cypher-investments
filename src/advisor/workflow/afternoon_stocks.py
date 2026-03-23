"""Afternoon workflow — Stocks mode (4 phases)."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

from advisor.workflow.models import (
    AfternoonPositionStatus,
    AfternoonResult,
    EarningsAlert,
    PositionAction,
    WatchCandidate,
    WorkflowState,
)
from advisor.workflow.shared import run_account_snapshot

logger = logging.getLogger(__name__)


class StocksAfternoonWorkflow:
    """4-phase afternoon review for equity positions."""

    def __init__(
        self,
        skip_account: bool = False,
        account_number: str | None = None,
    ):
        self.skip_account = skip_account
        self.account_number = account_number

    def run(self) -> AfternoonResult:
        start = time.time()
        errors: list[str] = []
        state = WorkflowState.load()
        result = AfternoonResult(mode="stocks")

        # Phase 1: Account Update
        if not self.skip_account:
            result.account = run_account_snapshot("stocks", account_number=self.account_number)
        else:
            errors.append("Account snapshot skipped (offline mode)")

        # Phase 2: Position Management
        result.position_statuses = self._check_positions(result.account.positions, errors)

        # Phase 3: Signal Re-Check
        result.watch_updates = self._recheck_watch(
            state.watch_candidates, state.morning_alpha_scores, errors
        )

        # Phase 4: Next-Day Prep
        all_symbols = self._get_all_symbols(result, state)
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
                sym = pos.get("underlying_symbol") or pos.get("symbol", "")
                avg_price = pos.get("average_open_price", 0)
                current = pos.get("close_price", 0) or pos.get("mark_price", 0)
                pnl_pct = (current - avg_price) / avg_price * 100 if avg_price > 0 else 0

                action = PositionAction.HOLD
                reason = f"P&L: {pnl_pct:+.1f}%"

                # Check if hit new high today → suggest tightening stop
                try:
                    import yfinance as yf

                    hist = yf.Ticker(sym).history(period="5d")
                    if len(hist) >= 2:
                        today_high = hist["High"].iloc[-1]
                        prev_high = hist["High"].iloc[:-1].max()
                        if today_high > prev_high:
                            action = PositionAction.TRAIL_STOP
                            reason = f"New HOD ${today_high:.2f} — tighten trailing stop"

                        # Check SMA-50 break
                        hist_long = yf.Ticker(sym).history(period="3mo")
                        if len(hist_long) >= 50:
                            sma50 = hist_long["Close"].rolling(50).mean().iloc[-1]
                            close = hist_long["Close"].iloc[-1]
                            prev_close = hist_long["Close"].iloc[-2]
                            if close < sma50 and prev_close >= sma50:
                                action = PositionAction.TREND_BREAK
                                reason = f"Broke below SMA-50 (${sma50:.2f}) today"
                except Exception:
                    pass

                # Earnings check
                earnings_days = self._days_to_earnings(sym)
                if earnings_days is not None and earnings_days <= 7:
                    if action == PositionAction.HOLD:
                        action = PositionAction.EARNINGS_WARNING
                        reason = f"Earnings in {earnings_days} days"

                results.append(
                    AfternoonPositionStatus(
                        symbol=sym,
                        action=action,
                        reason=reason,
                        pnl_pct=round(pnl_pct, 2),
                    )
                )
            except Exception as e:
                errors.append(f"Position check failed for {pos.get('symbol', '?')}: {e}")

        return results

    # ── Phase 3: Signal Re-Check ─────────────────────────────────────────

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

                # Re-scan signals
                try:
                    from advisor.engine.scanner import SignalScanner

                    scanner = SignalScanner()
                    scan = scanner.scan(sym)
                    wc.new_buy_signals = [
                        s.strategy_name for s in scan.signals if s.action.value == "BUY"
                    ]
                except Exception as e:
                    logger.debug("Signal re-check failed for %s: %s", sym, e)

                # Re-check alpha
                try:
                    from advisor.confluence.alpha_scorer import compute_alpha

                    alpha = compute_alpha(sym, skip_layers={"sentiment"})
                    wc.alpha_score = round(alpha.alpha_score, 2)
                    morning_score = morning_scores.get(sym)
                    if morning_score is not None:
                        wc.score_change = round(wc.alpha_score - morning_score, 2)
                except Exception as e:
                    logger.debug("Alpha re-check failed for %s: %s", sym, e)

                # Status summary
                if wc.new_buy_signals:
                    wc.status = f"NEW BUY on {', '.join(wc.new_buy_signals)}"
                elif wc.score_change and wc.score_change > 5:
                    wc.status = f"Score improved +{wc.score_change:.0f}"
                else:
                    wc.status = "No change"

                results.append(wc)
            except Exception as e:
                errors.append(f"Watch re-check failed for {sym}: {e}")

        return results

    # ── Phase 4: Next-Day Prep ───────────────────────────────────────────

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
        # Promote candidates that got new BUY signals
        promoted = [w.symbol for w in watch_updates if w.new_buy_signals]
        # Keep existing watchlist
        result = promoted + [s for s in current_watchlist if s not in promoted]
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
                # yfinance returns different formats depending on version
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
