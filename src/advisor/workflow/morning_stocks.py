"""Morning workflow — Stocks mode (6 phases)."""

from __future__ import annotations

import logging
import time

from advisor.workflow.models import (
    CandidateVerdict,
    DeepDiveResult,
    MorningResult,
    PositionAction,
    PositionHealth,
    ValidatedCandidate,
    WorkflowState,
)
from advisor.workflow.shared import (
    run_account_snapshot,
    run_discovery,
    run_lean_discovery,
    run_regime_check,
)

logger = logging.getLogger(__name__)


class StocksMorningWorkflow:
    """6-phase morning scan for equity positions and opportunities."""

    def __init__(
        self,
        account_size: float = 5_000,
        universes: list[str] | None = None,
        watchlist: list[str] | None = None,
        quick: bool = False,
        skip_account: bool = False,
        account_number: str | None = None,
        lean: bool = False,
    ):
        self.account_size = account_size
        self.universes = universes or ["sp500"]
        self.watchlist = watchlist or []
        self.quick = quick
        self.skip_account = skip_account
        self.account_number = account_number
        self.lean = lean

    def run(self) -> MorningResult:
        start = time.time()
        errors: list[str] = []
        result = MorningResult(mode="stocks")

        # Phase 1: Regime
        result.regime = run_regime_check()

        # Phase 2: Account
        if not self.skip_account:
            result.account = run_account_snapshot("stocks", account_number=self.account_number)
        else:
            errors.append("Account snapshot skipped (offline mode)")

        # Phase 3: Position Health
        result.position_health = self._check_position_health(result.account.positions, errors)

        # Phase 4: Discovery
        if self.lean:
            result.scanner_hits = run_lean_discovery(self.universes, self.watchlist, errors)
        else:
            result.scanner_hits = run_discovery(self.universes, self.watchlist, errors)

        # Phase 5: Validation
        if self.lean:
            result.validated = self._validate_candidates_lean(result.scanner_hits, errors)
        else:
            result.validated = self._validate_candidates(result.scanner_hits, errors)

        # Phase 6: Deep Dive (skipped in quick mode)
        if not self.quick:
            deep_dive_symbols = [
                c.symbol for c in result.validated if c.verdict == CandidateVerdict.DEEP_DIVE
            ]
            result.deep_dives = self._deep_dive(deep_dive_symbols, errors)

        # Collect WATCH candidates for afternoon
        result.watch_candidates = [
            c.symbol for c in result.validated if c.verdict == CandidateVerdict.WATCH
        ]

        result.errors = errors
        result.elapsed_seconds = round(time.time() - start, 2)

        # Save state for afternoon
        self._save_state(result)

        return result

    # ── Phase 3: Position Health Check ───────────────────────────────────

    def _check_position_health(
        self,
        positions: list[dict],
        errors: list[str],
    ) -> list[PositionHealth]:
        results: list[PositionHealth] = []
        for pos in positions:
            try:
                sym = pos.get("underlying_symbol") or pos.get("symbol", "")
                qty = pos.get("quantity", 0)
                avg_price = pos.get("average_open_price", 0)
                current = pos.get("close_price", 0) or pos.get("mark_price", 0)

                if avg_price <= 0 or current <= 0:
                    results.append(
                        PositionHealth(
                            symbol=pos.get("symbol", ""),
                            underlying=sym,
                            quantity=qty,
                            avg_open_price=avg_price,
                            current_price=current,
                            action=PositionAction.HOLD,
                            reason="Insufficient price data",
                        )
                    )
                    continue

                pnl_pct = (current - avg_price) / avg_price * 100

                action = PositionAction.HOLD
                reason = ""

                if pnl_pct >= 15:
                    action = PositionAction.CONSIDER_PROFIT
                    reason = f"+{pnl_pct:.1f}% — consider trimming or tightening stop"
                elif pnl_pct <= -10:
                    action = PositionAction.REVIEW
                    reason = f"{pnl_pct:.1f}% — check if thesis still intact"

                # Check trend (SMA-50)
                if action == PositionAction.HOLD:
                    try:
                        import yfinance as yf

                        hist = yf.Ticker(sym).history(period="3mo")
                        if len(hist) >= 50:
                            sma50 = hist["Close"].rolling(50).mean().iloc[-1]
                            if current < sma50:
                                action = PositionAction.TREND_WARNING
                                reason = f"Below SMA-50 (${sma50:.2f})"
                    except Exception:
                        pass

                if not reason:
                    reason = f"P&L: {pnl_pct:+.1f}%"

                results.append(
                    PositionHealth(
                        symbol=pos.get("symbol", ""),
                        underlying=sym,
                        quantity=qty,
                        avg_open_price=avg_price,
                        current_price=current,
                        pnl_pct=round(pnl_pct, 2),
                        action=action,
                        reason=reason,
                    )
                )
            except Exception as e:
                errors.append(f"Position health check failed for {pos.get('symbol', '?')}: {e}")

        return results

    # ── Phase 5: Candidate Validation ────────────────────────────────────

    def _validate_candidates(
        self,
        hits: list,
        errors: list[str],
    ) -> list[ValidatedCandidate]:
        results: list[ValidatedCandidate] = []

        for hit in hits:
            try:
                # Confluence
                confluence_verdict = "PASS"
                try:
                    from advisor.confluence.orchestrator import run_confluence

                    conf = run_confluence(hit.symbol, force_all=True)
                    confluence_verdict = conf.verdict.value
                except Exception as e:
                    logger.warning("Confluence failed for %s: %s", hit.symbol, e)

                # ML Signal
                ml_signal = None
                ml_win_prob = None
                try:
                    from advisor.ml.signal_generator import MLSignalGenerator

                    gen = MLSignalGenerator()
                    sig = gen.generate_signal(hit.symbol)
                    if sig:
                        ml_signal = sig.action.value
                        # Get win probability from explain
                        explanation = gen.explain_prediction(hit.symbol)
                        ml_win_prob = explanation.get("win_probability")
                except Exception as e:
                    logger.debug("ML signal failed for %s: %s", hit.symbol, e)

                # Strategy signals (stocks mode)
                strategy_signals: list[str] = []
                try:
                    from advisor.engine.scanner import SignalScanner

                    scanner = SignalScanner()
                    scan = scanner.scan(hit.symbol)
                    for s in scan.signals:
                        if s.action.value == "BUY":
                            strategy_signals.append(f"{s.strategy_name}:BUY")
                except Exception as e:
                    logger.debug("Signal scan failed for %s: %s", hit.symbol, e)

                # Decision gate
                is_enter = confluence_verdict == "ENTER"
                is_caution = confluence_verdict == "CAUTION"
                ml_buy = ml_signal == "BUY" and ml_win_prob is not None and ml_win_prob >= 0.60
                ml_not_sell = ml_signal != "SELL"
                has_strategy_buy = len(strategy_signals) > 0

                if is_enter and ml_buy and has_strategy_buy:
                    verdict = CandidateVerdict.DEEP_DIVE
                    reason = "ENTER + ML BUY + strategy BUY → deep dive"
                elif (is_enter or is_caution) and ml_not_sell:
                    verdict = CandidateVerdict.WATCH
                    reason = f"{confluence_verdict} + ML not SELL → watch"
                else:
                    verdict = CandidateVerdict.SKIP
                    reason = f"{confluence_verdict} — does not meet criteria"

                results.append(
                    ValidatedCandidate(
                        symbol=hit.symbol,
                        verdict=verdict,
                        confluence_verdict=confluence_verdict,
                        ml_signal=ml_signal,
                        ml_win_prob=ml_win_prob,
                        strategy_signals=strategy_signals,
                        alpha_score=hit.alpha_score,
                        reason=reason,
                    )
                )
            except Exception as e:
                errors.append(f"Validation failed for {hit.symbol}: {e}")

        return [r for r in results if r.verdict != CandidateVerdict.SKIP]

    # ── Phase 5 (lean): Lightweight Validation ──────────────────────────

    def _validate_candidates_lean(
        self,
        hits: list,
        errors: list[str],
    ) -> list[ValidatedCandidate]:
        """Lean validation: ML signal + strategy signals (no confluence API calls)."""
        results: list[ValidatedCandidate] = []

        for hit in hits:
            try:
                alpha = hit.alpha_score or 0

                # ML Signal (local inference, no API)
                ml_signal = None
                ml_win_prob = None
                try:
                    from advisor.ml.signal_generator import MLSignalGenerator

                    gen = MLSignalGenerator()
                    sig = gen.generate_signal(hit.symbol)
                    if sig:
                        ml_signal = sig.action.value
                        explanation = gen.explain_prediction(hit.symbol)
                        ml_win_prob = explanation.get("win_probability")
                except Exception as e:
                    logger.debug("ML signal failed for %s: %s", hit.symbol, e)

                # Strategy signals (yfinance-based, no API)
                strategy_signals: list[str] = []
                try:
                    from advisor.engine.scanner import SignalScanner

                    scanner = SignalScanner()
                    scan = scanner.scan(hit.symbol)
                    for s in scan.signals:
                        if s.action.value == "BUY":
                            strategy_signals.append(f"{s.strategy_name}:BUY")
                except Exception as e:
                    logger.debug("Signal scan failed for %s: %s", hit.symbol, e)

                # Lean decision gate
                ml_buy = ml_signal == "BUY" and ml_win_prob is not None and ml_win_prob >= 0.60
                ml_not_sell = ml_signal != "SELL"
                has_strategy_buy = len(strategy_signals) > 0

                if alpha >= 65 and (ml_buy or (has_strategy_buy and alpha >= 75)):
                    verdict = CandidateVerdict.DEEP_DIVE
                    reason = (
                        f"alpha {alpha:.0f} + {'ML BUY' if ml_buy else 'strategy BUY'} → deep dive"
                    )
                elif alpha >= 50 and ml_not_sell:
                    verdict = CandidateVerdict.WATCH
                    reason = f"alpha {alpha:.0f} + ML not SELL → watch"
                else:
                    verdict = CandidateVerdict.SKIP
                    reason = f"alpha {alpha:.0f} — does not meet lean criteria"

                results.append(
                    ValidatedCandidate(
                        symbol=hit.symbol,
                        verdict=verdict,
                        confluence_verdict="lean",
                        ml_signal=ml_signal,
                        ml_win_prob=ml_win_prob,
                        strategy_signals=strategy_signals,
                        alpha_score=hit.alpha_score,
                        reason=reason,
                    )
                )
            except Exception as e:
                errors.append(f"Lean validation failed for {hit.symbol}: {e}")

        return [r for r in results if r.verdict != CandidateVerdict.SKIP]

    # ── Phase 6: Deep Dive ───────────────────────────────────────────────

    def _deep_dive(
        self,
        symbols: list[str],
        errors: list[str],
    ) -> list[DeepDiveResult]:
        results: list[DeepDiveResult] = []

        for sym in symbols:
            dd = DeepDiveResult(symbol=sym)

            # Scenario simulation
            try:
                from advisor.scenario.models import ScenarioConfig
                from advisor.scenario.pipeline import run_scenario_simulation

                sim_result = run_scenario_simulation(
                    symbol=sym,
                    config=ScenarioConfig(dte=30, n_paths=500),
                    include_signals=True,
                )
                if sim_result.composites:
                    best = sim_result.composites[0]
                    dd.scenario_expected_return = best.expected_return
                    dd.scenario_max_dd = best.expected_max_dd
                    dd.scenario_prob_profit = best.prob_positive
                    dd.best_strategy = sim_result.best_strategy
            except Exception as e:
                errors.append(f"Scenario sim failed for {sym}: {e}")

            # Research agent
            try:
                from research_agent.config import ResearchConfig
                from research_agent.models import InputMode, ResearchInput
                from research_agent.pipeline import run

                card = run(
                    ResearchInput(mode=InputMode.TICKER, value=sym),
                    ResearchConfig(),
                )
                dd.research_verdict = card.verdict.value
                dd.research_dip_type = card.dip_type.value
                dd.bull_case = card.bull_case[:2]
                dd.bear_case = card.bear_case[:2]
                # Map verdict to conviction: BUY_THE_DIP=80, WATCH=50, AVOID=20
                _verdict_conviction = {"BUY_THE_DIP": 80, "WATCH": 50, "AVOID": 20}
                dd.research_conviction = _verdict_conviction.get(card.verdict.value, 50)
            except Exception as e:
                errors.append(f"Research agent failed for {sym}: {e}")

            results.append(dd)

        return results

    # ── State Persistence ────────────────────────────────────────────────

    def _save_state(self, result: MorningResult) -> None:
        state = WorkflowState(
            mode="stocks",
            morning_run_at=result.run_at,
            morning_account=result.account.model_dump() if result.account else None,
            watchlist=result.watch_candidates,
            watch_candidates=result.watch_candidates,
            morning_alpha_scores={
                h.symbol: h.alpha_score for h in result.scanner_hits if h.alpha_score is not None
            },
            lean_mode=self.lean,
        )
        try:
            state.save()
        except Exception as e:
            logger.warning("Failed to save workflow state: %s", e)
