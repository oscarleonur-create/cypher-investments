"""A pick's verdict: what to do, at what price, from the company's own value.

User, 2026-09-30: "Posible entry, por esta tesis, el valor de la empresa está
por debajo de lo que esperamos, debería estar en X valor".
"""

from __future__ import annotations

import pytest
from advisor.breadth import verdict as V


def card(verdict="above_base", bear=168.39, base=197.45, bull=225.74, need=0.24,
         own=(0.223, 0.244, 0.198), refused=None):  # fmt: skip
    cases = [{"name": "market", "value_per_share": 209.18, "margin": need}]
    if base is not None:
        cases += [{"name": "bear", "value_per_share": bear},
                  {"name": "base", "value_per_share": base},
                  {"name": "bull", "value_per_share": bull}]  # fmt: skip
    return {
        "verdict": verdict,
        "cases": cases,
        "own_margins": [{"value": m} for m in own],
        "refused": refused,
        "assumptions": "discounted at 10%, 3% terminal growth",
        "rationale": [
            "At $209.18 the market values the business at $40.0bn.",
            "That price earns a 10% return if growth of +12.2% holds 3 years.",
            "Its own filed margins run 19.8% to 24.4%.",
            "The price sits between the base and bull cases.",
        ],
    }


EVIDENCE = {
    "below_base": {"n": 402, "excess": 0.0231, "ci": [0.0058, 0.0455], "verdict": "EDGE",
                   "years": 3},
    "above_base": {"n": 655, "excess": -0.0041, "ci": [-0.0158, 0.0086],
                   "verdict": "UNDETERMINED", "years": 3},
}  # fmt: skip


class TestVerdict:
    def test_above_its_value_waits_for_the_base(self):
        v = V.pick_verdict(209.18, card(), evidence=EVIDENCE)
        assert v["action"] == "WAIT" and v["entry"] == pytest.approx(197.45)
        assert v["target"] == pytest.approx(225.74)
        assert v["headline"].startswith("WAIT — entry at $197.45 (its base value); at $209.18")
        assert "+5.9% above it" in v["headline"]
        assert "beat peers by -0.4%" in v["evidence"] and "UNDETERMINED" in v["evidence"]
        assert any(x.startswith("Value from its own filings: bear $168.39") for x in v["lines"])
        assert any(x.startswith("That price earns") for x in v["lines"])
        assert not any(x.startswith("At $209.18") for x in v["lines"])  # repeats the price

    def test_below_its_value_enters_up_to_the_base(self):
        v = V.pick_verdict(190.0, card(verdict="above_bear"), evidence=EVIDENCE)
        assert v["action"] == "ENTER" and v["entry"] == pytest.approx(197.45)
        assert v["headline"].startswith("ENTER up to $197.45 (its base value) — at $190.00")
        assert "+2.3%" in v["evidence"] and "EDGE" in v["evidence"]
        assert not any("Below even the bear" in x for x in v["lines"])

    def test_below_the_bear_says_check_the_business(self):
        v = V.pick_verdict(378.39, card(verdict="below_bear", bear=621.1, base=1000.0,
                                        bull=2300.0, need=0.023, own=(0.459, 0.199, 0.121)),
                           sic=1623)  # fmt: skip
        assert v["action"] == "ENTER"
        assert any("Below even the bear case ($621.10)" in x for x in v["lines"])
        assert "contractor" in v["caveat"]

    def test_a_price_no_filing_supports_is_trade_only(self):
        c = card(verdict=None, base=None, need=0.914, own=(-0.003, -0.025, 0.020),
                 refused="one positive margin on file")  # fmt: skip
        record = {"no_support": {"n": 427, "excess": 0.0206, "ci": None, "verdict": "EDGE"}}
        v = V.pick_verdict(84.15, c, evidence=record)
        assert v["action"] == "TRADE ONLY" and v["entry"] is None and v["target"] is None
        assert "record flatters them most" in v["evidence"]  # survivorship, said
        assert (
            "asks a 91.4% free-cash-flow margin; the best the company has filed is 2.0%"
            in (v["headline"])
        )

    def test_never_a_positive_margin(self):
        c = card(verdict=None, base=None, need=0.053, own=(-0.044, -0.052))
        assert "has never filed a positive one" in V.pick_verdict(7.82, c)["headline"]

    def test_one_reading_is_unproven(self):
        c = card(verdict=None, base=None, need=0.091, own=(0.168,))
        v = V.pick_verdict(74.88, c, sic=6531)
        assert v["action"] == "UNPROVEN" and "one margin on file, 16.8%" in v["headline"]
        assert "real-estate" in v["caveat"]

    def test_no_card_is_cant_value_with_the_reason(self):
        why = "no price, share count, balance sheet or revenue"
        v = V.pick_verdict(12.6, None, why_missing=why)
        assert v["action"] == "CAN'T VALUE" and v["headline"].startswith("CAN'T VALUE: no price")
        assert v["evidence"] is None

    def test_no_evidence_on_file_is_left_out(self):
        assert V.pick_verdict(209.18, card())["evidence"] is None
        assert V.evidence_line({"n": 0}) is None

    def test_exactly_at_the_base_is_below_base(self):
        # The card's own verdict decides: at the base value it is "above_bear".
        v = V.pick_verdict(197.45, card(verdict="above_bear"))
        assert v["action"] == "ENTER" and "+0.0% against it" in v["headline"]


@pytest.mark.parametrize("sic,kind", [(1623, "contractor"), (4412, "shipper"), (6022, "bank"),
                                      (6798, "REIT"), (6512, "real-estate"), (3672, None),
                                      (None, None)])  # fmt: skip
def test_misread_businesses(sic, kind):
    got = V.misread(sic)
    assert (got is None) if kind is None else (kind in got)


def test_filings_never_raise_and_are_read_once_per_symbol_and_day(monkeypatch):
    from datetime import date

    from advisor.valuation import figures

    calls = []

    def boom(symbol, price, **kw):
        calls.append(symbol)
        raise RuntimeError("SEC down")

    monkeypatch.setattr(figures, "load_figures", boom)
    monkeypatch.setattr(V, "_CARDS", {})
    got, why = V.figures_for("XYZ", 10.0, date(2026, 10, 2))
    again, _ = V.figures_for("xyz", 11.0, date(2026, 10, 2))
    assert got is None and again is None and "RuntimeError" in why
    assert calls == ["XYZ"]  # once per symbol and day
    V.figures_for("XYZ", 10.0, date(2026, 10, 3))
    assert calls == ["XYZ", "XYZ"]  # a new day reads again
