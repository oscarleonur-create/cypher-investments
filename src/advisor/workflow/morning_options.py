"""Morning workflow — Options mode (6 phases)."""

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
from advisor.workflow.shared import run_account_snapshot, run_discovery, run_regime_check

logger = logging.getLogger(__name__)


class OptionsMorningWorkflow:
    """6-phase morning scan for options positions and opportunities."""

    def __init__(
        self,
        account_size: float = 5_000,
        universes: list[str] | None = None,
        watchlist: list[str] | None = None,
        quick: bool = False,
        skip_account: bool = False,
        account_number: str | None = None,
    ):
        self.account_size = account_size
        self.universes = universes or ["wheel", "leveraged"]
        self.watchlist = watchlist or []
        self.quick = quick
        self.skip_account = skip_account
        self.account_number = account_number

    def run(self) -> MorningResult:
        start = time.time()
        errors: list[str] = []
        result = MorningResult(mode="options")

        # Phase 1: Regime
        result.regime = run_regime_check()

        # Phase 2: Account
        if not self.skip_account:
            result.account = run_account_snapshot("options", account_number=self.account_number)
        else:
            errors.append("Account snapshot skipped (offline mode)")

        # Phase 3: Options Position Health
        result.position_health = self._check_position_health(result.account.positions, errors)

        # Phase 4: Discovery
        result.scanner_hits = run_discovery(self.universes, self.watchlist, errors)

        # Phase 5: Validation
        result.validated = self._validate_candidates(result.scanner_hits, errors)

        # Phase 6: Deep Dive (skipped in quick mode)
        if not self.quick:
            deep_dive_symbols = [
                c.symbol for c in result.validated if c.verdict == CandidateVerdict.DEEP_DIVE
            ]
            result.deep_dives = self._deep_dive(deep_dive_symbols, result.regime, errors)

        # Collect WATCH candidates for afternoon
        result.watch_candidates = [
            c.symbol for c in result.validated if c.verdict == CandidateVerdict.WATCH
        ]

        result.errors = errors
        result.elapsed_seconds = round(time.time() - start, 2)

        # Save state for afternoon
        self._save_state(result)

        return result

    # ── Phase 3: Options Position Health Check ───────────────────────────

    def _check_position_health(
        self,
        positions: list[dict],
        errors: list[str],
    ) -> list[PositionHealth]:
        results: list[PositionHealth] = []

        for pos in positions:
            try:
                sym = pos.get("symbol", "")
                underlying = pos.get("underlying_symbol") or sym
                qty = pos.get("quantity", 0)
                avg_price = pos.get("average_open_price", 0)
                current = (
                    pos.get("close_price", 0) or pos.get("mark_price", 0) or pos.get("mark", 0)
                )

                actions: list[tuple[PositionAction, str]] = []

                # P&L check (for credit trades: profit when price drops)
                if avg_price > 0 and current > 0:
                    # For short options, average_open_price is the credit received
                    direction = pos.get("quantity_direction", "Long")
                    if direction == "Short":
                        # Credit trade: profit = credit - current
                        pnl_pct = (avg_price - current) / avg_price * 100
                        if pnl_pct >= 50:
                            actions.append(
                                (
                                    PositionAction.TAKE_PROFIT,
                                    f"P&L {pnl_pct:.0f}% of credit (target 50%)",
                                )
                            )
                        elif pnl_pct <= -100:
                            actions.append(
                                (
                                    PositionAction.CONSIDER_CLOSING,
                                    f"Loss {abs(pnl_pct):.0f}% of credit (>2x)",
                                )
                            )
                    else:
                        pnl_pct = (current - avg_price) / avg_price * 100 if avg_price > 0 else 0
                else:
                    pnl_pct = 0

                # DTE check — parse from symbol if available
                dte = self._parse_dte(sym)
                if dte is not None and dte <= 14:
                    actions.append(
                        (
                            PositionAction.APPROACHING_EXPIRY,
                            f"DTE={dte} — consider rolling or closing",
                        )
                    )

                # Pick most urgent action
                if actions:
                    action, reason = actions[0]
                else:
                    action = PositionAction.HOLD
                    reason = f"P&L: {pnl_pct:+.1f}%"

                results.append(
                    PositionHealth(
                        symbol=sym,
                        underlying=underlying,
                        quantity=qty,
                        avg_open_price=avg_price,
                        current_price=current,
                        pnl_pct=round(pnl_pct, 2),
                        action=action,
                        reason=reason,
                        dte=dte,
                    )
                )
            except Exception as e:
                errors.append(f"Position health check failed for {pos.get('symbol', '?')}: {e}")

        return results

    def _parse_dte(self, symbol: str) -> int | None:
        """Attempt to parse DTE from an options symbol (e.g., PLTR 250418P00023000)."""
        from datetime import date, datetime

        try:
            # Standard OCC format: SYMBOL YYMMDD[C/P]SSSSSSSS
            parts = symbol.split()
            if len(parts) >= 2:
                date_part = parts[1][:6]
                exp_date = datetime.strptime(date_part, "%y%m%d").date()
                return (exp_date - date.today()).days
        except Exception:
            pass
        return None

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
                        explanation = gen.explain_prediction(hit.symbol)
                        ml_win_prob = explanation.get("win_probability")
                except Exception as e:
                    logger.debug("ML signal failed for %s: %s", hit.symbol, e)

                # Decision gate (options mode: no strategy scan needed)
                is_enter = confluence_verdict == "ENTER"
                is_caution = confluence_verdict == "CAUTION"
                ml_buy = ml_signal == "BUY" and ml_win_prob is not None and ml_win_prob >= 0.60
                ml_not_sell = ml_signal != "SELL"

                if is_enter and ml_buy:
                    verdict = CandidateVerdict.DEEP_DIVE
                    reason = "ENTER + ML BUY → pipeline deep dive"
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
                        alpha_score=hit.alpha_score,
                        reason=reason,
                    )
                )
            except Exception as e:
                errors.append(f"Validation failed for {hit.symbol}: {e}")

        return [r for r in results if r.verdict != CandidateVerdict.SKIP]

    # ── Phase 6: Pipeline Deep Dive + Research ───────────────────────────

    def _deep_dive(
        self,
        symbols: list[str],
        regime: object,
        errors: list[str],
    ) -> list[DeepDiveResult]:
        results: list[DeepDiveResult] = []

        for sym in symbols:
            dd = DeepDiveResult(symbol=sym)

            # Options Pipeline (PipelineOrchestrator)
            try:
                from advisor.pipeline.models import PipelineConfig
                from advisor.pipeline.orchestrator import PipelineOrchestrator

                config = PipelineConfig(
                    account_size=self.account_size,
                    min_conviction=50.0,
                )
                orchestrator = PipelineOrchestrator(config=config)
                pipe_result = orchestrator.run([sym], top_n=1)

                if pipe_result.recommendations:
                    rec = pipe_result.recommendations[0]
                    dd.strikes = f"${rec.short_strike}/{rec.long_strike}"
                    dd.dte = rec.dte
                    dd.credit = rec.credit
                    dd.conviction = rec.conviction_score
                    dd.pop = rec.mc_pop
                    dd.ev = rec.ev
                    dd.suggested_contracts = rec.suggested_contracts
            except Exception as e:
                errors.append(f"Pipeline failed for {sym}: {e}")

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

            # Gate: only include if conviction >= 50
            if dd.conviction is not None and dd.conviction < 50:
                continue

            results.append(dd)

        return results

    # ── State Persistence ────────────────────────────────────────────────

    def _save_state(self, result: MorningResult) -> None:
        state = WorkflowState(
            mode="options",
            morning_run_at=result.run_at,
            morning_account=result.account.model_dump() if result.account else None,
            watchlist=result.watch_candidates,
            watch_candidates=result.watch_candidates,
            morning_alpha_scores={
                h.symbol: h.alpha_score for h in result.scanner_hits if h.alpha_score is not None
            },
        )
        try:
            state.save()
        except Exception as e:
            logger.warning("Failed to save workflow state: %s", e)
