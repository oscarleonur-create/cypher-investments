"""What a prospectus is offering, and whether the deal is done.

Both rules here come from an end-to-end run against AMD, and both were
producing a false statement about a real filing:

- AMD's August 424B5 offered $4.75bn of 4.600% Senior Notes due 2029 and three
  further tranches. Not one share was issued. Classifying every 424B as
  DILUTION reported a leverage decision as shareholder dilution.
- AMD filed the preliminary on 13 August with no amount, because the deal had
  not priced, and the final on the 14th carrying $4,750,000,000. The system
  raised Tier A on the unpriced preliminary and filed the priced final as a
  digest item — exactly backwards.
"""

from __future__ import annotations

import pytest
from advisor.news.offering import SecurityType, classify_offering

# The real cover pages, verbatim from EDGAR.
AMD_FINAL = (
    "Table of Contents Filed pursuant to Rule 424(b)(5) PROSPECTUS SUPPLEMENT "
    "(To prospectus dated August 13, 2026) $4,750,000,000 $1,250,000,000 4.600% "
    "Senior Notes due 2029 $1,500,000,000 5.000% Senior Notes due 2031 "
    "$1,000,000,000 5.250% Senior Notes due 2033 $1,000,000,000 5.500% Senior "
    "Notes due 2036 Advanced Micro Devices, Inc."
)
AMD_PRELIMINARY = (
    "Table of Contents The information in this preliminary prospectus supplement "
    "is not complete and may be changed. This preliminary prospectus supplement "
    "and the accompanying prospectus are not an offer to sell these securities "
    "and we are not soliciting offers to buy these securities in any jurisdiction "
    "where the offer or sale is not permitted. Senior Notes due 2029"
)
AAOI_EQUITY = (
    "TABLE OF CONTENTS Filed pursuant to Rule 424(b)(5) PROSPECTUS SUPPLEMENT "
    "$600,000,000 Applied Optoelectronics, Inc. Common Stock We have entered into "
    "an Equity Distribution Agreement relating to shares of our common stock "
    "offered by this prospectus supplement."
)
TE_CONVERTIBLE = (
    "Item 3.02. Unregistered Sale of Securities. The Company agreed to sell "
    "$120.0 million aggregate principal amount of the Convertible Notes to the "
    "Purchasers in a private placement."
)


class TestTheRealFilings:
    def test_amds_senior_notes_are_debt_not_dilution(self):
        shape = classify_offering(AMD_FINAL)
        assert shape.security is SecurityType.DEBT
        assert shape.dilutive is False

    def test_aaois_at_the_market_is_equity_and_dilutive(self):
        shape = classify_offering(AAOI_EQUITY)
        assert shape.security is SecurityType.EQUITY
        assert shape.dilutive is True

    def test_t1_energys_convertible_notes_are_dilutive_debt(self):
        """Debt that becomes equity dilutes if it converts."""
        shape = classify_offering(TE_CONVERTIBLE)
        assert shape.security is SecurityType.CONVERTIBLE
        assert shape.dilutive is True

    def test_the_preliminary_is_recognised_as_unpriced(self):
        assert classify_offering(AMD_PRELIMINARY).preliminary is True

    def test_the_final_supplement_is_not_preliminary(self):
        assert classify_offering(AMD_FINAL).preliminary is False


class TestDisambiguation:
    def test_a_convertible_is_not_swallowed_by_the_debt_pattern(self):
        """'Convertible senior notes' matches both; the convertible wins."""
        text = "$500,000,000 3.250% Convertible Senior Notes due 2030"
        assert classify_offering(text).security is SecurityType.CONVERTIBLE

    def test_when_both_appear_the_first_on_the_cover_is_the_offering(self):
        debt_first = "PROSPECTUS SUPPLEMENT 5.000% Senior Notes due 2031. Proceeds may "
        debt_first += "be used to repurchase shares of our common stock."
        assert classify_offering(debt_first).security is SecurityType.DEBT

        equity_first = "PROSPECTUS SUPPLEMENT shares of our common stock. Proceeds will "
        equity_first += "repay our 5.000% Senior Notes due 2031."
        assert classify_offering(equity_first).security is SecurityType.EQUITY

    def test_only_the_cover_is_read(self):
        """A debt prospectus discusses common stock in its risk factors."""
        text = "5.000% Senior Notes due 2031. " + ("filler " * 2000)
        text += "shares of our common stock may be issued later."
        assert classify_offering(text, cover_chars=200).security is SecurityType.DEBT

    def test_depositary_shares_count_as_equity(self):
        assert classify_offering("10,000,000 Depositary Shares").security is SecurityType.EQUITY


class TestFailsClosed:
    @pytest.mark.parametrize("text", ["", None, "   ", "PROSPECTUS SUPPLEMENT"])
    def test_an_unreadable_cover_is_unknown_not_assumed(self, text):
        shape = classify_offering(text)
        assert shape.security is SecurityType.UNKNOWN

    def test_unknown_is_not_treated_as_dilutive(self):
        """Guessing dilution on an unidentifiable offering invents a fact."""
        assert classify_offering("PROSPECTUS SUPPLEMENT").dilutive is False


class TestEventGeneration:
    def _event(self, monkeypatch, shape, size=None, cap=842_569_343_427.0):
        import advisor.news.ingest as mod
        from advisor.news.enrich import OfferingSize

        from tests.test_news.test_ingest import item

        monkeypatch.setattr(mod, "offering_shape_for", lambda _: shape)
        monkeypatch.setattr(
            mod,
            "offering_size_for",
            lambda _: OfferingSize(amount_usd=size, quote="q") if size else None,
        )
        return mod._event_for_filing(item(), market_caps={"AAOI": cap})

    def test_a_debt_offering_does_not_become_a_dilution_event(self, monkeypatch):
        from advisor.daemon.models import EventTier
        from advisor.news.offering import OfferingShape

        event = self._event(monkeypatch, OfferingShape(SecurityType.DEBT, False), size=4.75e9)
        assert event.kind == "FILING_DEBT_ISSUANCE"
        assert event.tier is EventTier.B
        assert "dilution_pct" not in event.payload
        assert event.payload["offering_pct_of_cap"] == pytest.approx(0.0056, abs=0.0005)

    def test_a_debt_offering_is_still_sized(self, monkeypatch):
        """New leverage is material even when it dilutes nobody."""
        from advisor.news.offering import OfferingShape

        event = self._event(monkeypatch, OfferingShape(SecurityType.DEBT, False), size=4.75e9)
        assert event.payload["offering_usd"] == pytest.approx(4.75e9)

    def test_an_equity_offering_still_interrupts(self, monkeypatch):
        from advisor.daemon.models import EventTier
        from advisor.news.offering import OfferingShape

        event = self._event(
            monkeypatch,
            OfferingShape(SecurityType.EQUITY, False),
            size=6e8,
            cap=8_960_160_678.0,
        )
        assert event.kind == "FILING_DILUTION"
        assert event.tier is EventTier.A
        assert event.payload["dilution_pct"] == pytest.approx(0.067, abs=0.001)

    def test_a_preliminary_equity_offering_is_demoted(self, monkeypatch):
        """An unpriced intention is not a completed deal."""
        from advisor.daemon.models import EventTier
        from advisor.news.offering import OfferingShape

        event = self._event(
            monkeypatch,
            OfferingShape(SecurityType.EQUITY, True),
            size=6e8,
            cap=8_960_160_678.0,
        )
        assert event.tier is EventTier.B
        assert "preliminary" in event.payload["label"]

    def test_the_security_type_travels_in_the_payload(self, monkeypatch):
        from advisor.news.offering import OfferingShape

        event = self._event(monkeypatch, OfferingShape(SecurityType.DEBT, True))
        assert event.payload["security_type"] == "DEBT"
        assert event.payload["preliminary"] is True

    def test_an_unreadable_cover_leaves_the_classification_alone(self, monkeypatch):
        """Failing to read the cover must not silence a real offering."""
        import advisor.news.ingest as mod
        from advisor.daemon.models import EventTier

        from tests.test_news.test_ingest import item

        monkeypatch.setattr(mod, "offering_shape_for", lambda _: None)
        monkeypatch.setattr(mod, "offering_size_for", lambda _: None)
        event = mod._event_for_filing(item(), market_caps={})
        assert event.kind == "FILING_DILUTION"
        assert event.tier is EventTier.A
