"""The Actions tab: verbs only, one or two bullets, answered once.

User, 2026-10-04: "los accionables son sobre nuestras acciones o sobre el
tracking que estamos haciendo, los accionables deben tener uno o dos bullets.
Y son solo acciones".
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from advisor.action.decisions import Decision, Direction, SubjectKind, Verdict
from advisor.daemon import market_calendar as mc
from advisor.daemon.book import EQUITY, EQUITY_OPTION, BookSnapshot, Position
from advisor.entry import actionables as A
from advisor.entry.proposal import Action, Leg, Proposal, Reason, position_stop_pct

ET = mc.MARKET_TZ
FRI = date(2026, 10, 2)
THU = date(2026, 10, 1)
NOW = datetime(2026, 10, 2, 14, 0, tzinfo=ET)


def prop(symbol, action, *, session=FRI, hour=9, price=100.0, exits=(), legs=(), blockers=(),
         triggers=(), reasons=None, features=None, net_liq=10_000.0):  # fmt: skip
    return Proposal(
        symbol=symbol,
        session=session,
        built_at=datetime.combine(session, datetime.min.time(), tzinfo=ET).replace(
            hour=hour, minute=45
        ),
        action=action,
        price=price,
        exits=list(exits),
        legs=list(legs),
        blockers=list(blockers),
        triggers=list(triggers),
        reasons=reasons if reasons is not None else [Reason(text="r", source="s")],
        features={"sigma": 0.03, **(features or {})},
        net_liq=net_liq,
    )


def pos(symbol, qty=10.0, cost=100.0, price=100.0, instrument=EQUITY, account="A"):
    return Position(account=account, symbol=symbol, underlying=symbol, instrument=instrument,
                    quantity=qty, avg_open_price=cost, mark_price=price, close_price=price,
                    multiplier=1.0 if instrument == EQUITY else 100.0)  # fmt: skip


def book(*positions, net_liq=10_000.0):
    return BookSnapshot(positions=list(positions), net_liq=net_liq)


def call(action, rule, why="why", shares=None, evidence=()):
    return {"action": action, "rule": rule, "why": why, "shares": shares,
            "evidence": [{"text": t, "source": "your thesis claims"} for t in evidence],
            "would_change": "x"}  # fmt: skip


def leg(horizon="position", entry=100.0, stop=90.0, shares=5, target=130.0):
    return Leg(horizon=horizon, entry=entry, stop=stop, stop_basis="b", risk_pct=0.02,
               shares=shares, notional=entry * shares, target=target)  # fmt: skip


def build(proposals, b=None, picks=None, claims=None, now=NOW):
    return A.build(proposals, b if b is not None else book(), picks, claims or {}, now)


class TestHeld:
    def test_an_exit_on_a_held_name_is_a_sell_of_everything(self):
        b = book(pos("CCXI", qty=35, cost=17.16, price=12.01))
        [a] = build([prop("CCXI", Action.EXIT, price=11.90, exits=[call("EXIT", "stop")],
                          features={"sigma": 0.0427})], b)  # fmt: skip
        assert a.verb == "SELL" and a.section == "book" and a.shares == 35
        assert a.title == "SELL CCXI — all 35 shares (~$420.35)"
        stop = 17.16 * (1 - position_stop_pct(0.0427))
        assert a.bullets == [f"$11.90 is past its stop ${stop:,.2f}: -30.7% from your cost $17.16"]
        assert a.answers == ["DONE"]

    def test_accounts_are_summed(self):
        b = book(pos("CCXI", qty=20, account="A"), pos("CCXI", qty=15, account="B"))
        [a] = build([prop("CCXI", Action.EXIT, exits=[call("EXIT", "filing", "went bankrupt")])], b)
        assert a.shares == 35 and a.bullets == ["went bankrupt"]

    def test_a_call_on_a_name_no_longer_held_is_dropped(self):
        # TE: EXIT on 09-29, sold since — the call is not an action any more.
        assert build([prop("TE", Action.EXIT, exits=[call("EXIT", "stop")])]) == []

    def test_options_on_the_name_are_not_a_holding(self):
        b = book(pos("CCXI", qty=1, instrument=EQUITY_OPTION))
        assert build([prop("CCXI", Action.EXIT, exits=[call("EXIT", "stop")])], b) == []

    def test_only_the_newest_session_counts(self):
        b = book(pos("AAOI"))
        old = prop("AAOI", Action.EXIT, session=THU, exits=[call("EXIT", "stop")])
        new = prop("AMZN", Action.IN_ZONE, reasons=[])
        assert build([old, new], b) == []

    def test_the_newest_call_of_the_session_wins(self):
        b = book(pos("AAOI"))
        morning = prop("AAOI", Action.EXIT, hour=9, exits=[call("EXIT", "stop")])
        later = prop("AAOI", Action.HOLD, hour=11)
        assert build([morning, later], b) == []

    def test_a_sell_says_two_reasons_at_most_and_nothing_else(self):
        b = book(pos("X"))
        calls = [call("EXIT", "filing", "a"), call("EXIT", "halt", "b"), call("EXIT", "news", "c"),
                 call("REVIEW", "thesis", "d", evidence=["r"])]  # fmt: skip
        [a] = build([prop("X", Action.EXIT, exits=calls)], b)
        assert a.verb == "SELL" and a.bullets == ["a", "b"]  # the REVIEW is moot

    def test_hold_is_not_an_action(self):
        assert build([prop("CRDO", Action.HOLD)], book(pos("CRDO"))) == []


class TestNoTrim:
    """The user removed the 20% trim (2026-10-04); proposals recorded before still say TRIM."""

    def test_a_recorded_trim_is_not_an_action(self):
        # SPCX on 10-02: TRIM at 21.4% of the book, recorded before the rule went.
        b = book(pos("SPCX", qty=11, cost=128.16, price=158.78), net_liq=8_148.0)
        p = prop("SPCX", Action.TRIM, exits=[call("TRIM", "concentration", shares=1)])
        assert build([p], b) == []


class TestDecide:
    def test_a_broken_thesis_rule_is_answered_on_the_claims(self):
        b = book(pos("AAOI", qty=4, cost=133.71, price=114.84))
        rules = ["Any equity raise above 5% breaks it", "Una emisión sobre 5% la rompe"]
        p = prop("AAOI", Action.REVIEW, exits=[call("REVIEW", "thesis", evidence=rules)],
                 features={"sigma": 0.0776})  # fmt: skip
        [a] = build([p], b, claims={"AAOI": [("c1", rules[0]), ("c2", rules[1] + " ")]})
        assert a.verb == "DECIDE" and a.answers == ["KEEP"]
        assert a.title == "DECIDE AAOI — keep or sell: a rule of your thesis broke"
        assert a.bullets[0] == f'Your rule: "{rules[0]}" (+1 more)'
        assert a.bullets[1].startswith("-14.1% from your cost $133.71; its stop is $")
        assert a.subject == {"kind": "CLAIM", "ids": ["c1", "c2"]}

    def test_a_rule_rewritten_since_is_answered_as_the_situation(self):
        p = prop("AAOI", Action.REVIEW, exits=[call("REVIEW", "thesis", evidence=["old text"])])
        [a] = build([p], book(pos("AAOI")), claims={"AAOI": [("c1", "new text")]})
        assert a.subject["kind"] == "ACTIONABLE" and a.subject["id"].startswith("act:DECIDE:thesis")

    def test_rich_is_kept_at_its_percentile(self):
        why = ("P/S 9.0x is at percentile 83% of its own two years (median 3.3x): it has been this "
               "expensive on 17% of days; the 80th-percentile price is $308.89")  # fmt: skip
        p = prop("COHR", Action.REVIEW, exits=[call("REVIEW", "rich", why)],
                 features={"ps_percentile": 0.83})  # fmt: skip
        [a] = build([p], book(pos("COHR")))
        assert a.title == "DECIDE COHR — trim or keep: priced rich against its own two years"
        assert a.bullets[0] == (
            "P/S 9.0x is at percentile 83% of its own two years (median 3.3x); "
            "the 80th-percentile price is $308.89"
        )
        assert a.subject["observed"] == 0.83 and a.subject["worse_is"] == "UP"

    def test_each_review_is_its_own_decision_and_a_recorded_trim_is_not(self):
        b = book(pos("SPCX", qty=11, price=158.78), net_liq=8_148.0)
        calls = [call("TRIM", "concentration"), call("REVIEW", "rich", "r"),
                 call("REVIEW", "news (unconfirmed)", "one outlet says so")]  # fmt: skip
        got = build([prop("SPCX", Action.TRIM, exits=calls)], b)
        assert [a.verb for a in got] == ["DECIDE", "DECIDE"]
        assert got[1].title == "DECIDE SPCX — keep or sell: a news report to answer"

    def test_no_cost_says_nothing_about_the_position(self):
        b = book(pos("X", cost=0.0))
        [a] = build([prop("X", Action.REVIEW, exits=[call("REVIEW", "filing", "auditor")])], b)
        assert a.bullets == ["auditor"]


class TestResults:
    """A held name's results within the entry guard's sessions: hold through, or cut before.

    Figures are SPCX's of 2026-09-24 (as in the scorecard's tests): 25.4%
    required at $147.60, +91.9% delivered, consensus FY2027 $108.3bn — met,
    the years to 2036 need 12.8% a year.
    """

    @staticmethod
    def spcx(**kw):
        from advisor.valuation.implied import undiscounted_expectations
        from advisor.valuation.models import ValuationSnapshot

        base = undiscounted_expectations(1.885e12, 31.256e9, terminal_multiple=25, fcf_margin=0.25)
        return ValuationSnapshot(**{
            "symbol": "SPCX", "asof": date(2026, 9, 24), "price": 147.60,
            # Shares such that the price reproduces the stored EV: the item re-prices.
            "shares_outstanding": 1.885e12 / 147.60, "market_cap": 1.885e12, "net_cash": None,
            "enterprise_value": 1.885e12, "revenue_runrate": 31.256e9, "ev_to_revenue": 60.3,
            "source_accession": "0001628280-26-052535", "period_end": date(2026, 6, 30),
            "revenue_yoy": 0.9194, "scenarios": [base], **kw})  # fmt: skip

    @staticmethod
    def consensus(fy2027=108.3e9):
        from advisor.valuation.consensus import Consensus, RevenueEstimate

        return Consensus(symbol="SPCX", asof=NOW, years=[
            RevenueEstimate(label="FY2026", fiscal_year_end=date(2026, 12, 31), avg=44.8e9),
            RevenueEstimate(label="FY2027", fiscal_year_end=date(2027, 12, 31), avg=fy2027,
                            analysts=19),
        ])  # fmt: skip

    FEATURES = {"required_low": 0.254, "required_high": 0.254, "delivered_growth": 0.9194,
                "consensus_growth": 0.832}  # fmt: skip

    def results(self, day, *, features=None, action=Action.HOLD, exits=(), held=True,
                expectations="spcx", now=NOW):  # fmt: skip
        f = {**self.FEATURES, "next_earnings": day, **(features or {})}
        b = book(pos("SPCX", qty=11, cost=128.16, price=147.60)) if held else book()
        exp = {"SPCX": (self.spcx(), self.consensus())} if expectations == "spcx" else expectations
        p = prop("SPCX", action, price=147.60, features=f, exits=exits)
        return A.build([p], b, None, {}, now, exp)

    def test_the_decision_states_the_bar_and_what_is_left_after_the_consensus(self):
        [a] = self.results("2026-10-08")
        assert a.verb == "DECIDE" and a.section == "book" and a.answers == ["KEEP"]
        assert a.id == "SPCX:DECIDE:results:2026-10-08"
        assert a.title == "DECIDE SPCX — hold through results on 10-08, or cut before"
        assert a.bullets == [
            "Results Thu 10-08, in 4 sessions: at $147.60 the price requires 25.4% a year "
            "for 10 years",
            "Revenue grew +91.9% last reported; consensus +83.2% a year to FY2027; "
            "if FY2027 is met, the years to 2036 need 12.8% a year",
        ]
        assert a.subject == {"kind": "ACTIONABLE", "id": "act:DECIDE:results:2026-10-08",
                             "observed": 0.254, "worse_is": "UP"}  # fmt: skip
        assert "results date from the yfinance calendar" in a.source

    def test_the_remaining_growth_is_the_scorecards_own_arithmetic(self, tmp_path):
        from advisor.daemon.store import DaemonStore
        from advisor.story.scorecard import build_scorecard

        store = DaemonStore(tmp_path / "research.db")
        store.save_valuation(self.spcx())
        card = build_scorecard(store, "SPCX", consensus_loader=lambda *_: self.consensus())
        store.close()
        row = next(r for r in card.expectations if r.label == "If FY2027 holds")
        _, end, lo, hi = A.remaining_after(self.spcx(), self.consensus(), 147.60)
        assert row.value == f"{lo:.1%}/yr" and lo == hi and end == 2036

    @pytest.mark.parametrize("day,shown", [
        ("2026-10-09", True),    # exactly the guard's 5 sessions: Mon..Fri
        ("2026-10-12", False),   # the sixth
    ])  # fmt: skip
    def test_the_window_is_the_entry_guards(self, day, shown):
        assert bool(self.results(day)) is shown

    def test_results_today_still_ask(self):
        [a] = self.results("2026-10-02")
        assert a.bullets[0].startswith("Results Fri 10-02, today:")

    def test_over_a_weekend_monday_is_the_next_session(self):
        sat = datetime(2026, 10, 3, 11, 0, tzinfo=ET)
        [a] = self.results("2026-10-05", now=sat)
        assert a.bullets[0].startswith("Results Mon 10-05, next session:")

    def test_a_date_already_past_is_not_due(self):
        # A proposal built Thursday for Thursday's results, read on Friday.
        assert self.results("2026-10-01") == []

    @pytest.mark.parametrize("raw", [None, "", "soon", "2026-13-01"])
    def test_no_date_or_an_unreadable_one_is_nothing(self, raw):
        assert self.results(raw) == []

    def test_a_proposal_built_before_the_date_was_recorded_is_nothing(self):
        p = prop("SPCX", Action.HOLD, features=self.FEATURES)
        assert build([p], book(pos("SPCX"))) == []

    def test_only_held_names(self):
        assert self.results("2026-10-08", held=False, action=Action.IN_ZONE) == []

    def test_a_sell_makes_it_moot(self):
        got = self.results("2026-10-08", action=Action.EXIT, exits=[call("EXIT", "stop")])
        assert [a.verb for a in got] == ["SELL"]

    def test_it_stands_beside_a_review(self):
        got = self.results("2026-10-08", action=Action.REVIEW,
                           exits=[call("REVIEW", "rich", "r")])  # fmt: skip
        assert [a.id for a in got] == ["SPCX:DECIDE:rich", "SPCX:DECIDE:results:2026-10-08"]

    def test_no_valuation_says_so_and_keeps_what_is_known(self):
        f = {"required_low": None, "required_high": None, "consensus_growth": None}
        [a] = self.results("2026-10-08", features=f, expectations={})
        assert a.bullets == [
            "Results Thu 10-08, in 4 sessions: no valuation on file, so what the price "
            "requires cannot be stated",
            "Revenue grew +91.9% last reported",
        ]
        assert a.subject["observed"] is None and a.subject["worse_is"] == "NEITHER"

    def test_nothing_but_the_date_is_still_a_decision(self):
        f = dict.fromkeys(self.FEATURES)
        [a] = self.results("2026-10-08", features=f, expectations={})
        assert len(a.bullets) == 1

    def test_a_consensus_that_pays_for_the_price_says_revenue_could_shrink(self):
        # $301.6bn needed by mid-2036 from $400bn at the end of 2027: (301.6/400)^(1/8.5) - 1.
        exp = {"SPCX": (self.spcx(), self.consensus(fy2027=400e9))}
        [a] = self.results("2026-10-08", expectations=exp)
        assert a.bullets[1].endswith("if FY2027 is met, revenue could shrink 3.3% a year to 2036")

    def test_kept_it_returns_only_on_a_materially_higher_bar(self):
        [a] = self.results("2026-10-08")
        sid = a.subject["id"]
        [kept] = A.decision_for(a, "KEEP", "AI segment carries it")
        assert A.answered(a, {sid: kept}, NOW.date()).endswith(": AI segment carries it")
        [ran] = self.results("2026-10-08", features={"required_high": 0.254 * 1.11})
        assert A.answered(ran, {sid: kept}, NOW.date()) is None

    def test_next_quarters_results_ask_again(self):
        [a] = self.results("2026-10-08")
        [kept] = A.decision_for(a, "KEEP", "fine")
        [later] = self.results("2026-10-08", features={})
        later = later.model_copy(
            update={"subject": {**later.subject, "id": "act:DECIDE:results:2027-01-07"}}
        )
        assert A.answered(later, {kept.subject_id: kept}, NOW.date()) is None


class TestReported:
    """After the results: keep or sell, with the bar before and after them.

    SPCX reports Tue 11-03. The last call before is Mon 11-02 at $171.92
    (46.1% required, consensus +136.1%); the first after, Wed 11-04.
    """

    BEFORE = {"next_earnings": "2026-11-03", "required_low": 0.461, "required_high": 0.461,
              "consensus_growth": 1.361, "delivered_growth": 0.537}  # fmt: skip
    AFTER = {"next_earnings": "2027-02-02", "required_low": 0.50, "required_high": 0.50,
             "consensus_growth": 1.40, "delivered_growth": 0.537,
             "consensus_asof": "2026-11-04T09:30:00-05:00"}  # fmt: skip

    def calls(self, *, before=None, after=None, after_day=date(2026, 11, 4), extra=()):
        return [
            prop("SPCX", Action.HOLD, session=date(2026, 11, 2), price=171.92,
                 features={**self.BEFORE, **(before or {})}),
            *extra,
            prop("SPCX", Action.HOLD, session=after_day, price=190.0,
                 features={**self.AFTER, **(after or {})}),
        ]  # fmt: skip

    def run(self, calls, now=datetime(2026, 11, 4, 14, 0, tzinfo=ET), held=True):
        b = book(pos("SPCX", qty=11, cost=128.16, price=190.0)) if held else book()
        return A.build(calls, b, None, {}, now)

    def test_the_bar_and_the_consensus_before_and_after(self):
        [a] = self.run(self.calls())
        assert a.verb == "DECIDE" and a.answers == ["KEEP"]
        assert a.id == "SPCX:DECIDE:reported:2026-11-03"
        assert a.title == "DECIDE SPCX — keep or sell after its results of 11-03"
        assert a.bullets == [
            "Results of Tue 11-03: $171.92 → $190.00 (+10.5%) since 11-02; the price requires "
            "50.0% a year (was 46.1%)",
            "Consensus +136.1% → +140.0% a year; the new quarter is not in the filings read yet",
        ]
        assert a.source.endswith("against the call of 11-02")
        assert a.subject["observed"] == 0.50 and a.subject["worse_is"] == "UP"

    def test_a_new_filing_says_what_it_delivered(self):
        [a] = self.run(self.calls(after={"delivered_growth": 0.70}))
        assert a.bullets[1].endswith("revenue grew +70.0% in the new filing (was +53.7%)")

    @pytest.mark.parametrize("day,now,shown", [
        (date(2026, 11, 6), datetime(2026, 11, 6, 10, 0, tzinfo=ET), True),   # 3 sessions on
        (date(2026, 11, 9), datetime(2026, 11, 9, 10, 0, tzinfo=ET), False),  # the 4th
    ])  # fmt: skip
    def test_it_is_asked_for_three_sessions(self, day, now, shown):
        got = self.run(self.calls(after_day=day), now=now)
        assert bool(got) is shown

    def test_on_the_day_itself_it_is_still_before(self):
        calls = [prop("SPCX", Action.HOLD, session=date(2026, 11, 3), price=171.0,
                      features=self.BEFORE)]  # fmt: skip
        [a] = self.run(calls, now=datetime(2026, 11, 3, 10, 0, tzinfo=ET))
        assert a.id == "SPCX:DECIDE:results:2026-11-03"  # hold through or cut, not "after"

    def test_a_date_moved_before_it_came_was_never_a_report(self):
        # 10-30 listed 11-03; by 11-02 the calendar said 11-10.
        early = prop("SPCX", Action.HOLD, session=date(2026, 10, 30), features=self.BEFORE)
        calls = self.calls(before={"next_earnings": "2026-11-10"}, extra=[early])
        calls[-1].features["next_earnings"] = "2026-11-10"
        # Only the moved date's own "before" decision: 11-10 is four sessions out.
        assert [a.id for a in self.run(calls)] == ["SPCX:DECIDE:results:2026-11-10"]

    def test_no_call_before_the_day_is_nothing(self):
        assert self.run(self.calls()[1:]) == []

    def test_no_call_since_the_day_is_nothing(self):
        # The daemon was down: the newest call is still Monday's.
        assert self.run(self.calls()[:1], now=datetime(2026, 11, 4, 14, 0, tzinfo=ET)) == []

    def test_only_held_names_and_a_sell_makes_it_moot(self):
        assert self.run(self.calls(), held=False) == []
        calls = self.calls()
        calls[-1] = calls[-1].model_copy(
            update={"action": Action.EXIT, "exits": [call("EXIT", "stop")]}
        )
        assert [a.verb for a in self.run(calls)] == ["SELL"]

    def test_what_is_missing_is_left_out_not_invented(self):
        gone = {"required_low": None, "required_high": None, "consensus_growth": None,
                "delivered_growth": None}  # fmt: skip
        [a] = self.run(self.calls(before=gone, after={"required_low": None,
                                                      "required_high": None}))  # fmt: skip
        assert a.bullets == [
            "Results of Tue 11-03: $171.92 → $190.00 (+10.5%) since 11-02",
            "Consensus +140.0% a year; the new quarter is not in the filings read yet",
        ]
        assert a.subject["observed"] is None and a.subject["worse_is"] == "NEITHER"

    def test_a_consensus_read_before_the_results_is_not_a_revision(self):
        # PENG, 2026-10-07: the 09:45 call read a cached estimate from before the 10-06 report.
        [a] = self.run(self.calls(after={"consensus_asof": "2026-11-03T08:00:00-05:00"}))
        assert a.bullets[1].startswith(
            "Consensus +140.0% a year, read 11-03: not re-read since the results"
        )

    def test_a_consensus_with_no_read_time_says_so(self):
        [a] = self.run(self.calls(after={"consensus_asof": None}))
        assert a.bullets[1].startswith("Consensus +140.0% a year (when it was read is not")

    def test_the_read_time_is_taken_in_new_york(self):
        # 01:00 UTC on 11-04 is still 11-03 in New York: before the results.
        [a] = self.run(self.calls(after={"consensus_asof": "2026-11-04T01:00:00+00:00"}))
        assert "not re-read since the results" in a.bullets[1]

    @pytest.mark.parametrize("raw", [None, "", "soon"])
    def test_an_unreadable_date_in_the_history_is_ignored(self, raw):
        noise = prop("SPCX", Action.HOLD, session=date(2026, 10, 29),
                     features={"next_earnings": raw})  # fmt: skip
        [a] = self.run(self.calls(extra=[noise]))
        assert a.id == "SPCX:DECIDE:reported:2026-11-03"

    def test_kept_it_stays_quiet_unless_the_bar_rises_materially(self):
        [a] = self.run(self.calls())
        [kept] = A.decision_for(a, "KEEP", "the AI segment beat")
        sid = a.subject["id"]
        assert A.answered(a, {sid: kept}, date(2026, 11, 5))
        [higher] = self.run(self.calls(after={"required_high": 0.56}))
        assert A.answered(higher, {sid: kept}, date(2026, 11, 5)) is None


class TestBuyAndRead:
    def test_an_enter_is_a_buy_with_size_and_stop(self):
        p = prop("AMZN", Action.ENTER, legs=[leg()], triggers=["crossed into its zone today"])
        [a] = build([p])
        assert a.verb == "BUY" and a.section == "tracking"
        assert a.title == "BUY AMZN — buy 5 shares near $100.00, stop $90.00"
        assert a.bullets == [
            "Crossed into its zone today",
            "risks $50.00 (2.0% of net liq) to the stop; a trim is reviewed at $130.00",
        ]
        assert a.answers == ["DONE", "SKIP"]

    def test_a_trade_leg_says_when_it_ends(self):
        [a] = build([prop("X", Action.ENTER, legs=[leg(horizon="trade", target=None)])])
        assert a.bullets[-1].endswith("a trade: out by the next session's close")

    def test_unsized_is_not_a_buy(self):
        assert build([prop("X", Action.ENTER, legs=[leg(shares=0)])]) == []
        assert build([prop("X", Action.ENTER, legs=[])]) == []

    def test_an_add_on_a_held_name_until_the_book_grows(self):
        p = prop("CRDO", Action.ADD, legs=[leg()], features={"weight": 0.10}, price=100.0)
        assert [a.title for a in build([p], book(pos("CRDO", qty=10)))] == [
            "BUY CRDO — add 5 shares near $100.00, stop $90.00"
        ]
        assert build([p], book(pos("CRDO", qty=15))) == []  # bought since

    def test_an_enter_on_a_name_bought_since_is_dropped(self):
        # Bought after the call: the book says held, and an ENTER is not an ADD.
        assert build([prop("AMZN", Action.ENTER, legs=[leg()])], book(pos("AMZN"))) == []

    def test_a_wait_on_a_tier_a_event_is_a_read(self):
        p = prop("NBIS", Action.WAIT, legs=[leg()],
                 blockers=["a tier-A event on this name today: read it before entering"],
                 reasons=[Reason(text="10-02 08:15 ET: distress reading",
                                 source="event stream (news distress, tier A)")])  # fmt: skip
        [a] = build([p], book(pos("NBIS", qty=1)))
        assert a.verb == "READ" and a.section == "book"
        assert a.title == "READ NBIS before adding — today's tier-A event"
        assert a.bullets == ["10-02 08:15 ET: distress reading",
                             "then: buy 5 shares near $100.00, stop $90.00"]  # fmt: skip

    @pytest.mark.parametrize("blockers", [
        ["results due 2026-10-07 (in 3 sessions): wait for the number, not a bet on it"],
        ["a tier-A event on this name today: read it before entering",
         "the reading of the recent facts is AT_RISK"],
        ["price history is 4 days old: refresh it before entering"],
    ])  # fmt: skip
    def test_a_wait_on_anything_but_reading_is_no_action(self, blockers):
        assert build([prop("X", Action.WAIT, legs=[leg()], blockers=blockers)]) == []

    def test_a_wait_with_nothing_sized_is_no_action(self):
        # NBIS on 10-02: in zone, a tier-A event, but no leg qualified.
        p = prop("NBIS", Action.WAIT, blockers=["a tier-A event on this name today: read it"])
        assert build([p]) == []

    @pytest.mark.parametrize("action", [Action.IN_ZONE, Action.NONE, Action.CANNOT_SAY])
    def test_quiet_calls_are_not_actions(self, action):
        assert build([prop("AMZN", action, reasons=[])]) == []


def picks(*rows, day="2026-10-02", provisional=True):
    return {"day": day, "built_at": "2026-10-02T13:30:00-04:00", "provisional": provisional,
            "picks": list(rows)}  # fmt: skip


def pick(symbol="TSM", action="ENTER", stage="fresh", shares=2, caveat=None, held=False):
    verdict = {
        "ENTER": {
            "action": "ENTER",
            "entry": 500.0,
            "target": 700.0,
            "caveat": caveat,
            "headline": "ENTER up to $500.00 …",
            "evidence": "Measured: 402 past picks like this beat peers by +2.3% over 20 "
            "sessions (95% interval +0.6% … +4.6%) — EDGE, 3-year replay.",
        },
        "TRADE ONLY": {
            "action": "TRADE ONLY",
            "entry": None,
            "target": None,
            "caveat": caveat,
            "headline": "TRADE ONLY — at $86.03 the price asks a 93.5% free-cash-flow "
            "margin; the best the company has filed is 2.0%. Nothing in "
            "the filings supports the price: no value to enter at.",
            "evidence": None,
        },
    }.get(action, {"action": action, "headline": action})
    return {"symbol": symbol, "held": held, "provisional": True,
            "asof": "2026-10-02T13:30:00-04:00", "verdict": verdict,
            "plan": {"ok": True, "entry": 472.78, "stop": 410.71, "stage": stage,
                     "size": {"shares": shares}}}  # fmt: skip


class TestPicks:
    def test_a_first_day_enter_during_its_session_is_a_buy(self):
        [a] = build([], picks=picks(pick(caveat="a contractor: cash flow swings")))
        assert a.verb == "BUY" and a.section == "tracking" and a.shares == 2
        assert a.title == "BUY TSM — 2 shares near $472.78, stop $410.71 · by the 16:00 ET close"
        assert a.bullets[0] == (
            "Under its base value $500.00; target $700.00 (bull) — a contractor: check the business"
        )
        assert a.bullets[1] == (
            "402 past picks like this beat peers by +2.3% over 20 sessions — EDGE, 3-year replay."
        )
        assert a.source == "picks 2026-10-02 (provisional, live prices)"

    def test_trade_only_is_a_buy_that_says_nothing_supports_the_price(self):
        [a] = build([], picks=picks(pick("FEIM", "TRADE ONLY")))
        assert a.title == (
            "BUY FEIM — trade only: 2 shares near $472.78, stop $410.71 · by the 16:00 ET close"
        )
        assert a.bullets == [
            "The price asks a 93.5% free-cash-flow margin; the best it has filed is 2.0%"
        ]

    def test_trade_only_says_survivorship_flatters_the_record_and_names_a_misread(self):
        p = pick("SBLK", "TRADE ONLY", caveat="a shipper: earnings follow charter rates")
        p["verdict"]["headline"] = (
            "TRADE ONLY — at $30.75 the price asks a 87.3% free-cash-flow margin; the company has "
            "never filed a positive one. Nothing in the filings supports the price: no value."
        )
        p["verdict"]["evidence"] = (
            "Measured: 427 past picks like this beat peers by +2.1% over 20 sessions (95% interval "
            "+0.4% … +3.8%) — EDGE, 3-year replay. Names like these drop out of the history most "
            "often when they fail, so this record flatters them most."
        )
        [a] = build([], picks=picks(p))
        assert a.bullets == [
            "The price asks a 87.3% free-cash-flow margin; it has never filed a positive one"
            " — a shipper: check the business",
            "427 past picks like this beat peers by +2.1% over 20 sessions — EDGE, 3-year "
            "replay; survivorship flatters it.",
        ]

    @pytest.mark.parametrize("action", ["WAIT", "UNPROVEN", "CAN'T VALUE"])
    def test_other_verdicts_are_not_buys(self, action):
        assert build([], picks=picks(pick(action=action))) == []

    @pytest.mark.parametrize("stage", ["late", "past"])
    def test_past_its_first_day_is_not_a_buy(self, stage):
        # The plan replay measured entering 5–15 sessions in at about zero.
        assert build([], picks=picks(pick(stage=stage))) == []

    @pytest.mark.parametrize("now", [
        datetime(2026, 10, 2, 16, 0, tzinfo=ET),     # the bell: day 0's close has printed
        datetime(2026, 10, 2, 20, 45, tzinfo=ET),    # the final list, after the close
        datetime(2026, 10, 4, 10, 0, tzinfo=ET),     # the weekend
        datetime(2026, 10, 5, 10, 0, tzinfo=ET),     # Monday: Friday's pick is a day late
    ])  # fmt: skip
    def test_only_before_its_own_close(self, now):
        assert build([], picks=picks(pick()), now=now) == []

    def test_an_early_close_ends_it_early(self):
        # 2026-11-27, the day after Thanksgiving, closes at 13:00.
        p = picks(pick(), day="2026-11-27")
        assert build([], picks=p, now=datetime(2026, 11, 27, 13, 30, tzinfo=ET)) == []
        [a] = build([], picks=p, now=datetime(2026, 11, 27, 12, 0, tzinfo=ET))
        assert a.title.endswith("· by the 13:00 ET close")

    def test_held_or_unsized_is_not_a_buy(self):
        assert build([], book(pos("TSM")), picks=picks(pick())) == []
        assert build([], picks=picks(pick(shares=0))) == []

    def test_no_picks_on_file(self):
        assert build([], picks={"day": None, "picks": []}) == []
        assert build([], picks=None) == []


def test_order_is_sell_decide_read_buy_and_book_first():
    b = book(pos("CCXI"), pos("SPCX", qty=30, price=100.0), pos("COHR"), pos("NBIS"),
             net_liq=10_000.0)  # fmt: skip
    proposals = [
        prop("AMZN", Action.ENTER, legs=[leg()]),
        prop("NBIS", Action.WAIT, legs=[leg()], blockers=["a tier-A event on this name today"]),
        prop("COHR", Action.REVIEW, exits=[call("REVIEW", "rich", "r")]),
        prop("SPCX", Action.TRIM, exits=[call("TRIM", "concentration")]),
        prop("CCXI", Action.EXIT, exits=[call("EXIT", "stop")]),
        prop("CRDO", Action.ADD, legs=[leg()]),
    ]
    got = build(proposals, b)
    assert [(a.verb, a.symbol) for a in got] == [
        ("SELL", "CCXI"), ("DECIDE", "COHR"), ("READ", "NBIS"),
        ("BUY", "AMZN"),
    ]  # fmt: skip
    for a in got:
        assert 1 <= len(a.bullets) <= 2 and all(a.bullets)


def test_nothing_on_file_is_nothing_to_do():
    assert A.build([], None, None, {}, NOW) == []


def decision(subject_id, verdict, *, days_ago=0, observed=None, worse_is=Direction.NEITHER,
             kind=SubjectKind.ACTIONABLE, note=""):  # fmt: skip
    return Decision(symbol="X", subject_kind=kind, subject_id=subject_id, verdict=verdict,
                    observed=observed, worse_is=worse_is, note=note,
                    decided_at=NOW - timedelta(days=days_ago))  # fmt: skip


def item(subject, answers=("KEEP",)):
    return A.Actionable(id="X:1", verb="DECIDE", symbol="X", section="book", title="t",
                        bullets=["b"], answers=list(answers), asof=NOW, source="s",
                        subject=subject)  # fmt: skip


class TestAnswers:
    def test_a_thesis_kept_on_every_claim_is_quiet(self):
        it = item({"kind": "CLAIM", "ids": ["c1", "c2"]})
        one = {"c1": decision("c1", Verdict.ACKNOWLEDGED, kind=SubjectKind.CLAIM)}
        assert A.answered(it, one, NOW.date()) is None  # c2 still asks
        both = {**one, "c2": decision("c2", Verdict.ACKNOWLEDGED, kind=SubjectKind.CLAIM,
                                      note="dilution funds the capacity")}  # fmt: skip
        assert A.answered(it, both, NOW.date()) == (
            "you kept it on 2026-10-02: dilution funds the capacity"
        )

    def test_rich_kept_returns_when_it_gets_materially_richer(self):
        kept = decision("act:DECIDE:rich", Verdict.ACKNOWLEDGED, observed=0.83,
                        worse_is=Direction.UP)  # fmt: skip
        quiet = item({"kind": "ACTIONABLE", "id": "act:DECIDE:rich", "observed": 0.86})
        assert A.answered(quiet, {"act:DECIDE:rich": kept}, NOW.date())
        back = item({"kind": "ACTIONABLE", "id": "act:DECIDE:rich", "observed": 0.92})
        assert A.answered(back, {"act:DECIDE:rich": kept}, NOW.date()) is None

    def test_done_is_quiet_for_the_grace_then_asks_again(self):
        sid = "act:SELL:stop"
        it = item({"kind": "ACTIONABLE", "id": sid}, answers=["DONE"])
        assert A.answered(it, {sid: decision(sid, Verdict.ACTED, days_ago=1)}, NOW.date())
        late = {sid: decision(sid, Verdict.ACTED, days_ago=4)}
        assert A.answered(it, late, (NOW + timedelta(days=0)).date()) is None

    def test_skip_is_quiet_for_good(self):
        sid = "act:BUY:2026-10-02"
        it = item({"kind": "ACTIONABLE", "id": sid}, answers=["DONE", "SKIP"])
        assert "dismissed" in A.answered(it, {sid: decision(sid, Verdict.DISMISSED, days_ago=30)},
                                         NOW.date())  # fmt: skip

    def test_a_decision_on_another_subject_does_not_answer(self):
        it = item({"kind": "ACTIONABLE", "id": "act:SELL:stop"}, answers=["DONE"])
        other = {"act:DECIDE:rich": decision("act:DECIDE:rich", Verdict.DISMISSED)}
        assert A.answered(it, other, NOW.date()) is None

    def test_keep_needs_a_reason(self):
        with pytest.raises(ValueError, match="reason"):
            A.decision_for(item({"kind": "ACTIONABLE", "id": "a"}), "KEEP", "  ")

    def test_an_answer_the_item_does_not_take_is_refused(self):
        with pytest.raises(ValueError, match="takes"):
            A.decision_for(item({"kind": "ACTIONABLE", "id": "a"}), "SKIP")

    def test_keep_on_a_thesis_records_one_decision_per_claim(self):
        ds = A.decision_for(item({"kind": "CLAIM", "ids": ["c1", "c2"]}), "keep", " funded ")
        assert [(d.subject_kind, d.subject_id, d.verdict, d.note) for d in ds] == [
            (SubjectKind.CLAIM, "c1", Verdict.ACKNOWLEDGED, "funded"),
            (SubjectKind.CLAIM, "c2", Verdict.ACKNOWLEDGED, "funded"),
        ]

    def test_done_records_the_reading_and_its_direction(self):
        subject = {"kind": "ACTIONABLE", "id": "act:SELL:stop", "observed": 0.214, "worse_is": "UP"}
        it = item(subject, answers=["DONE"])
        [d] = A.decision_for(it, "DONE")
        assert d.subject_kind is SubjectKind.ACTIONABLE and d.verdict is Verdict.ACTED
        assert d.observed == 0.214 and d.worse_is is Direction.UP


class TestStoredAndServed:
    """Over real stores: the book, the proposals, the decisions, and the endpoint."""

    @pytest.fixture
    def db(self, tmp_path):
        from advisor.daemon.store import DaemonStore
        from advisor.entry.store import EntryStore

        path = tmp_path / "research.db"
        daemon, entries = DaemonStore(path), EntryStore(path)
        daemon.save_book(book(pos("COHR", cost=382.92, price=336.92), pos("CCXI", qty=35)))
        entries.add(prop("COHR", Action.REVIEW, exits=[call("REVIEW", "rich", "rich")],
                         features={"ps_percentile": 0.83}))  # fmt: skip
        entries.add(prop("CCXI", Action.EXIT, exits=[call("EXIT", "stop")]))
        entries.add(prop("TE", Action.EXIT, session=THU, exits=[call("EXIT", "stop")]))
        daemon.close()
        entries.close()
        return path

    def test_load_reads_the_stores_without_a_picks_file(self, db):
        out = A.load(db, NOW)
        assert [a["id"] for a in out["items"]] == ["CCXI:SELL:stop", "COHR:DECIDE:rich"]
        assert out["answered"] == [] and out["book_asof"]

    def test_the_endpoint_answers_and_the_item_moves_to_answered(self, db, monkeypatch):
        from advisor.api.app import create_app
        from advisor.api.routers import actionables as router
        from fastapi.testclient import TestClient

        monkeypatch.setattr(router, "_db", lambda: db)
        monkeypatch.setattr("advisor.daemon.market_calendar.now_et", lambda: NOW)
        url = "/api/actionables/answer"
        with TestClient(create_app()) as c:
            assert len(c.get("/api/actionables").json()["items"]) == 2
            r = c.post(url, json={"id": "COHR:DECIDE:rich", "answer": "KEEP"})
            assert r.status_code == 400  # no reason given
            r = c.post(url, json={"id": "COHR:DECIDE:rich", "answer": "KEEP", "note": "AI optics"})
            assert r.status_code == 200
            out = c.get("/api/actionables").json()
            assert [a["id"] for a in out["items"]] == ["CCXI:SELL:stop"]
            assert out["answered"][0]["why"].endswith(": AI optics")
            # Twice is not found: it is no longer open.
            r = c.post(url, json={"id": "COHR:DECIDE:rich", "answer": "KEEP", "note": "again"})
            assert r.status_code == 404
            assert c.post(url, json={"id": "nope", "answer": "DONE"}).status_code == 404

    def test_load_reads_valuation_and_consensus_from_the_store_never_the_network(self, tmp_path):
        from advisor.daemon.store import DaemonStore
        from advisor.entry.store import EntryStore

        path = tmp_path / "research.db"
        daemon, entries = DaemonStore(path), EntryStore(path)
        daemon.save_book(book(pos("SPCX", qty=11, cost=128.16, price=147.60)))
        daemon.save_valuation(TestResults.spcx())
        daemon.save_consensus("SPCX", TestResults.consensus().model_dump_json())
        f = {**TestResults.FEATURES, "next_earnings": "2026-10-08"}
        entries.add(prop("SPCX", Action.HOLD, price=147.60, features=f))
        daemon.close()
        entries.close()
        [item] = A.load(path, NOW)["items"]
        assert item["id"] == "SPCX:DECIDE:results:2026-10-08"
        assert item["bullets"][1].endswith("the years to 2036 need 12.8% a year")
