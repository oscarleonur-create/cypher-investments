"""The value test: a pick record valued with only what was public on its day.

User, 2026-09-30: picks need "el valor de la empresa ... debería estar en X
valor". Before a pick says it, this measures whether value adds anything.
"""

from __future__ import annotations

from datetime import date

import pytest
from advisor.breadth import value_replay as V


def row(val, end, filed, start=None, accn="a1", form="10-Q"):
    r = {"val": val, "end": end, "filed": filed, "form": form, "accn": accn}
    if start:
        r["start"] = start
    return r


class TestAsOf:
    def test_only_what_was_filed_by_the_day(self):
        rows = [row(1, "2024-03-31", "2024-05-01"), row(2, "2024-06-30", "2024-08-01")]
        assert [r["val"] for r in V.as_of(rows, date(2024, 7, 1))] == [1]
        assert len(V.as_of(rows, date(2024, 8, 1))) == 2  # the filing day itself counts
        assert V.as_of([{"val": 3, "end": "2024-06-30"}], date(2030, 1, 1)) == []  # no date

    def test_slim_keeps_the_valuation_concepts_only(self):
        facts = {"facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [{**row(5, "2024-06-30", "2024-08-01",
                                                  start="2024-04-01"), "extra": 1}]}},
            "Goodwill": {"units": {"USD": [row(9, "2024-06-30", "2024-08-01")]}},
        }}}  # fmt: skip
        out = V.slim(facts)
        assert set(out) == {"Revenues"} and "extra" not in out["Revenues"][0]
        assert out["Revenues"][0]["accn"] == "a1"
        assert V.slim({}) == {}


class TestBalance:
    CASH = "CashAndCashEquivalentsAtCarryingValue"

    def test_cash_plus_securities_less_debt_at_the_latest_instant(self):
        facts = {
            self.CASH: [row(100, "2024-03-31", "2024-05-01", accn="q1"),
                        row(150, "2024-06-30", "2024-08-01", accn="q2")],
            "ShortTermInvestments": [row(20, "2024-06-30", "2024-08-01", accn="q2")],
            "LongTermDebt": [row(60, "2024-06-30", "2024-08-01", accn="q2")],
        }  # fmt: skip
        assert V.balance(facts, date(2024, 9, 1)) == (110.0, date(2024, 6, 30), "")
        # A month earlier, only Q1 was public; debt not in that filing: debt-free.
        assert V.balance(facts, date(2024, 7, 1))[0] == 100.0

    def test_debt_in_the_filing_but_not_at_the_instant_is_not_zero(self):
        facts = {
            self.CASH: [row(150, "2024-06-30", "2024-08-01", accn="q2")],
            "LongTermDebt": [row(60, "2023-12-31", "2024-08-01", accn="q2")],  # comparative only
        }
        net, _, why = V.balance(facts, date(2024, 9, 1))
        assert net is None and "debt" in why

    def test_debt_repaid_long_ago_is_debt_free_now(self):
        facts = {
            self.CASH: [row(150, "2024-06-30", "2024-08-01", accn="q2")],
            "LongTermDebt": [row(60, "2021-12-31", "2022-02-01", accn="k21")],
        }
        assert V.balance(facts, date(2024, 9, 1))[0] == 150.0

    def test_no_cash_is_no_balance(self):
        assert V.balance({}, date(2024, 9, 1)) == (None, None, "no cash reported")

    def test_a_later_filing_of_the_same_instant_wins(self):
        facts = {self.CASH: [row(150, "2024-06-30", "2024-08-01", accn="q2"),
                             row(155, "2024-06-30", "2024-11-01", accn="q3")]}  # fmt: skip
        assert V.balance(facts, date(2024, 9, 1))[0] == 150.0  # the restatement not yet public
        assert V.balance(facts, date(2024, 12, 1))[0] == 155.0


class TestShares:
    C = "WeightedAverageNumberOfDilutedSharesOutstanding"

    def test_latest_period_public_by_the_day(self):
        facts = {self.C: [row(100, "2024-03-31", "2024-05-01", start="2024-01-01"),
                          row(110, "2024-06-30", "2024-08-01", start="2024-04-01")]}  # fmt: skip
        assert V.shares(facts, date(2024, 7, 1), []) == (100.0, date(2024, 3, 31))
        assert V.shares(facts, date(2024, 9, 1), [])[0] == 110.0

    def test_splits_after_the_filing_put_it_on_todays_basis(self):
        facts = {self.C: [row(100, "2024-06-30", "2024-08-01", start="2024-04-01")]}
        splits = [(date(2024, 7, 15), 3.0), (date(2025, 1, 10), 2.0)]  # the first is before filing
        assert V.shares(facts, date(2024, 9, 1), splits)[0] == 200.0

    def test_no_duration_rows_is_no_count(self):
        facts = {self.C: [row(100, "2024-06-30", "2024-08-01")]}  # an instant, not an average
        assert V.shares(facts, date(2024, 9, 1), []) == (None, None)


class TestBucket:
    @pytest.mark.parametrize(
        "v,b",
        [
            ({"verdict": "below_bear"}, "below_base"),
            ({"verdict": "above_bear"}, "below_base"),
            ({"verdict": "above_base"}, "above_base"),
            ({"verdict": "above_bull"}, "above_base"),
            ({"verdict": None, "market_margin": 0.9, "best_margin": 0.02}, "no_support"),
            ({"verdict": None, "market_margin": 0.05, "best_margin": -0.1}, "no_support"),
            ({"verdict": None, "market_margin": 0.05, "best_margin": None}, "no_support"),
            ({"verdict": None, "market_margin": 0.09, "best_margin": 0.168}, "one_reading"),
            ({"verdict": None, "market_margin": 0.168, "best_margin": 0.168}, "one_reading"),
            ({"verdict": None, "market_margin": None}, "no_data"),
        ],
    )
    def test_five_plain_groups(self, v, b):
        assert V.bucket(v) == b


def test_value_on_refuses_a_stale_or_empty_company():
    assert V.value_on("X", date(2024, 9, 1), 10.0, {}, [])["bucket"] == "no_data"
    rev = "Revenues"
    old = {rev: [row(100, "2022-12-31", "2023-02-01", start="2022-01-01", form="10-K")]}
    out = V.value_on("X", date(2024, 9, 1), 10.0, old, [])
    assert out["bucket"] == "no_data" and "stale" in out["why"]


def test_evaluate_lists_every_group_bucket_and_horizon():
    rows = [{"grp": "F+P", "day": date(2024, 1, 2), "value": {"bucket": "below_base"},
             "outcomes": {"d20": {"ret": 0.05, "excess": 0.02}, "d60": None}}]  # fmt: skip
    cells = V.evaluate(rows)
    assert len(cells) == len(V.GROUPS) * len(V.BUCKETS) * len(V.HORIZONS)
    got = next(c for c in cells if (c["group"], c["bucket"], c["horizon"])
               == ("F+P", "below_base", "d20"))  # fmt: skip
    assert got["n"] == 1 and got["excess"] == pytest.approx(0.02)
    assert got["verdict"] == "UNDETERMINED"  # one record proves nothing
