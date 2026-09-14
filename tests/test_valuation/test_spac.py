"""Valuing a blank-cheque company, where EV/revenue is a category error.

CCXI is Churchill Capital Corp XI — a SPAC merging with Agility Robotics, not
the biotech its ticker once belonged to. It has no revenue and never will
under that name. The revenue model returned "no usable filing", which is true
and useless; its trust is a real valuation with a real floor.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from advisor.valuation.spac import TrustValue, _public_shares, is_spac, trust_value


class FakeQuery:
    def __init__(self, frames, concept=None):
        self._frames, self._concept = frames, concept

    def by_concept(self, concept):
        return FakeQuery(self._frames, concept)

    def to_dataframe(self):
        return self._frames.get(self._concept)


class FakeXbrl:
    def __init__(self, frames):
        self._frames = frames

    def query(self):
        return FakeQuery(self._frames)


class FakeFiling:
    accession_no = "0001213900-26-089245"
    period_of_report = "2026-06-30"

    def __init__(self, frames):
        self._frames = frames

    def xbrl(self):
        return FakeXbrl(self._frames)


# CCXI's real figures.
TRUST = pd.DataFrame(
    {"numeric_value": [420_879_831.0, 414_549_783.0], "is_dimensioned": [False, False]}
)
SHARES = pd.DataFrame(
    {
        "numeric_value": [41_900_000.0, 13_800_000.0],
        "is_dimensioned": [True, True],
        "label": ["Class A", "Class B"],
    }
)
FRAMES = {"AssetsHeldInTrustNoncurrent": TRUST, "EntityCommonStockSharesOutstanding": SHARES}


class TestTheRealSpac:
    def test_it_is_recognised_as_a_blank_cheque_company(self):
        assert is_spac(FakeXbrl(FRAMES)) is True

    def test_an_operating_company_is_not(self):
        assert is_spac(FakeXbrl({"Revenues": TRUST})) is False

    def test_the_trust_floor_is_per_public_share(self):
        value = trust_value("CCXI", FakeFiling(FRAMES), 13.40)
        assert value.per_share == pytest.approx(10.04, abs=0.01)

    def test_the_premium_is_measured_against_that_floor(self):
        value = trust_value("CCXI", FakeFiling(FRAMES), 13.40)
        assert value.premium == pytest.approx(0.334, abs=0.01)

    def test_the_description_names_the_floor(self):
        value = trust_value("CCXI", FakeFiling(FRAMES), 13.40)
        assert "redemption floor" in value.describe()
        assert "premium to" in value.describe()


class TestShareClasses:
    def test_founder_shares_do_not_dilute_the_trust(self):
        """Counting Class B would understate the floor by a third."""
        assert _public_shares(FakeXbrl(FRAMES)) == pytest.approx(41_900_000.0)

    def test_including_the_founder_class_would_give_a_lower_floor(self):
        correct = trust_value("CCXI", FakeFiling(FRAMES), 13.40).per_share
        naive = 420_879_831.0 / 55_700_000.0
        assert correct > naive
        assert naive == pytest.approx(7.56, abs=0.01)

    def test_a_single_class_filer_counts_every_share(self):
        one = pd.DataFrame(
            {"numeric_value": [30_000_000.0], "is_dimensioned": [False], "label": ["Ordinary"]}
        )
        frames = {**FRAMES, "EntityCommonStockSharesOutstanding": one}
        assert _public_shares(FakeXbrl(frames)) == pytest.approx(30_000_000.0)

    def test_classes_present_but_no_public_one_identified_yields_nothing(self):
        """A floor from the wrong share count would be quoted as the downside."""
        odd = pd.DataFrame(
            {
                "numeric_value": [10.0, 20.0],
                "is_dimensioned": [True, True],
                "label": ["Founder Series", "Class F"],
            }
        )
        frames = {**FRAMES, "EntityCommonStockSharesOutstanding": odd}
        assert _public_shares(FakeXbrl(frames)) is None


class TestRefusals:
    def test_a_non_spac_returns_nothing(self):
        assert trust_value("AMD", FakeFiling({"Revenues": TRUST}), 516.13) is None

    def test_a_missing_share_count_returns_nothing(self):
        assert (
            trust_value("CCXI", FakeFiling({"AssetsHeldInTrustNoncurrent": TRUST}), 13.40) is None
        )

    @pytest.mark.parametrize("price", [0, -5])
    def test_a_nonsense_price_returns_nothing(self, price):
        assert trust_value("CCXI", FakeFiling(FRAMES), price) is None

    def test_a_filing_without_xbrl_does_not_raise(self):
        class NoXbrl:
            accession_no = "x"

            def xbrl(self):
                raise RuntimeError("none")

        assert trust_value("CCXI", NoXbrl(), 13.40) is None


class TestPremiumDirection:
    def value(self, price: float) -> TrustValue:
        return TrustValue(
            symbol="CCXI",
            asof=date(2026, 6, 30),
            trust_total=420_879_831.0,
            public_shares=41_900_000.0,
            price=price,
            source_accession="x",
        )

    def test_trading_below_trust_is_a_discount(self):
        value = self.value(9.50)
        assert value.premium < 0
        assert "discount to" in value.describe()

    def test_trading_at_trust_is_neither(self):
        assert self.value(10.0449).premium == pytest.approx(0.0, abs=0.001)

    def test_the_floor_does_not_move_with_the_price(self):
        """The redemption right is fixed; only the premium changes."""
        assert self.value(9.0).per_share == self.value(20.0).per_share
