"""Tests for the source verification / grounding engine."""

from __future__ import annotations

from advisor.verification.grounding import (
    verify_extraction,
    verify_headline,
)

# ── Sample source text (realistic SEC filing excerpt) ────────────────────────

SEC_FILING_TEXT = """\
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Form 4 - Statement of Changes in Beneficial Ownership

Issuer: Apple Inc. (AAPL)
Filing Date: 2024-08-15

Reporting Person: Timothy D. Cook
Title: Chief Executive Officer (CEO)
Transaction Type: Purchase
Shares: 50,000
Price Per Share: $182.50
Total Value: $9,125,000

---

Reporting Person: Luca Maestri
Title: Chief Financial Officer (CFO)
Transaction Type: Sale
Shares: 25,000
Price Per Share: $185.00
Total Value: $4,625,000

---

Reporting Person: Katherine Adams
Title: Senior Vice President, General Counsel
Transaction Type: Purchase
Shares: 10,000
Price Per Share: $181.75
Total Value: $1,817,500
"""

NEWS_ARTICLES_TEXT = """\
[s1] Apple Reports Record Q3 Revenue
URL: https://reuters.com/apple-q3
Apple Inc reported record third-quarter revenue of $85.8 billion, up 5% year
over year. CEO Tim Cook said the company saw strong performance across all
product categories. iPhone revenue grew 8% to $42.3 billion.

---

[s2] Analysts Upgrade Apple Stock After Earnings Beat
URL: https://cnbc.com/apple-upgrade
Several Wall Street analysts upgraded Apple stock following the earnings beat.
Morgan Stanley raised its price target to $220 from $200. The consensus
rating moved to Overweight.

---

[s3] Tech Sector Rally Continues Amid Fed Rate Cut Hopes
URL: https://wsj.com/tech-rally
The technology sector continued its rally on Thursday as investors bet on
upcoming Federal Reserve interest rate cuts. Apple shares rose 2.3% to close
at $188.50.
"""


# ── Name matching tests ──────────────────────────────────────────────────────


class TestNameMatching:
    def test_exact_match(self):
        result = verify_extraction(
            extracted={"insider_name": "Timothy D. Cook"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["insider_name"],
        )
        assert result.fields[0].status == "GROUNDED"
        assert "Timothy D. Cook" in result.fields[0].matched_snippet

    def test_case_insensitive(self):
        result = verify_extraction(
            extracted={"insider_name": "timothy d. cook"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["insider_name"],
        )
        assert result.fields[0].status == "GROUNDED"

    def test_partial_match_last_name(self):
        result = verify_extraction(
            extracted={"insider_name": "John Cook"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["insider_name"],
        )
        assert result.fields[0].status == "PARTIAL"

    def test_no_match(self):
        result = verify_extraction(
            extracted={"insider_name": "Warren Buffett"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["insider_name"],
        )
        assert result.fields[0].status == "UNGROUNDED"

    def test_empty_name(self):
        result = verify_extraction(
            extracted={"insider_name": ""},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["insider_name"],
        )
        assert result.fields[0].status == "UNGROUNDED"


# ── Date matching tests ──────────────────────────────────────────────────────


class TestDateMatching:
    def test_exact_date_match(self):
        result = verify_extraction(
            extracted={"filing_date": "2024-08-15"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["filing_date"],
        )
        assert result.fields[0].status == "GROUNDED"

    def test_adjacent_day_match(self):
        result = verify_extraction(
            extracted={"filing_date": "2024-08-14"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["filing_date"],
        )
        # Should be PARTIAL since ±1 day
        assert result.fields[0].status == "PARTIAL"

    def test_no_date_match(self):
        result = verify_extraction(
            extracted={"filing_date": "2024-01-01"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["filing_date"],
        )
        assert result.fields[0].status == "UNGROUNDED"

    def test_invalid_date_format(self):
        result = verify_extraction(
            extracted={"filing_date": "not-a-date"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["filing_date"],
        )
        assert result.fields[0].status == "UNGROUNDED"


# ── Number matching tests ────────────────────────────────────────────────────


class TestNumberMatching:
    def test_exact_price_match(self):
        result = verify_extraction(
            extracted={"price": "182.50"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["price"],
        )
        assert result.fields[0].status == "GROUNDED"

    def test_dollar_sign_match(self):
        result = verify_extraction(
            extracted={"price": 182.50},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["price"],
        )
        assert result.fields[0].status == "GROUNDED"
        assert "$182.50" in result.fields[0].matched_snippet

    def test_qty_match(self):
        result = verify_extraction(
            extracted={"qty": 50000},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["qty"],
        )
        assert result.fields[0].status == "GROUNDED"

    def test_within_tolerance(self):
        # 182.50 with 1% tolerance = 180.675 to 184.325
        result = verify_extraction(
            extracted={"price": 183.0},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["price"],
        )
        assert result.fields[0].status == "GROUNDED"

    def test_outside_tolerance(self):
        result = verify_extraction(
            extracted={"price": 200.00},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["price"],
        )
        assert result.fields[0].status == "UNGROUNDED"

    def test_zero_value(self):
        result = verify_extraction(
            extracted={"price": 0.0},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["price"],
        )
        assert result.fields[0].status == "UNGROUNDED"

    def test_comma_formatted_number(self):
        source = "The total value of the trade was $9,125,000."
        result = verify_extraction(
            extracted={"value": 9125000},
            source_text=source,
            fields_to_check=["value"],
        )
        assert result.fields[0].status == "GROUNDED"


# ── Title/role matching tests ────────────────────────────────────────────────


class TestTitleMatching:
    def test_ceo_near_name(self):
        result = verify_extraction(
            extracted={"insider_name": "Timothy D. Cook", "title": "CEO"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["title"],
        )
        assert result.fields[0].status == "GROUNDED"

    def test_cfo_keyword(self):
        result = verify_extraction(
            extracted={"insider_name": "Luca Maestri", "title": "CFO"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["title"],
        )
        assert result.fields[0].status == "GROUNDED"

    def test_svp_keyword(self):
        result = verify_extraction(
            extracted={"insider_name": "Katherine Adams", "title": "SVP"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["title"],
        )
        assert result.fields[0].status == "GROUNDED"

    def test_unmatched_title(self):
        result = verify_extraction(
            extracted={"insider_name": "Timothy D. Cook", "title": "Janitor"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["title"],
        )
        assert result.fields[0].status in ("PARTIAL", "UNGROUNDED")


# ── Overall grounding score tests ────────────────────────────────────────────


class TestGroundingScore:
    def test_fully_grounded(self):
        result = verify_extraction(
            extracted={
                "insider_name": "Timothy D. Cook",
                "filing_date": "2024-08-15",
                "trade_type": "Purchase",
                "price": 182.50,
                "qty": 50000,
            },
            source_text=SEC_FILING_TEXT,
            fields_to_check=["insider_name", "filing_date", "trade_type", "price", "qty"],
        )
        assert result.grounding_score == 1.0
        assert result.ungrounded_fields == []

    def test_partially_grounded(self):
        result = verify_extraction(
            extracted={
                "insider_name": "Timothy D. Cook",  # GROUNDED
                "filing_date": "2024-08-15",  # GROUNDED
                "trade_type": "Purchase",  # GROUNDED
                "price": 999.99,  # UNGROUNDED
                "qty": 1,  # UNGROUNDED
            },
            source_text=SEC_FILING_TEXT,
            fields_to_check=["insider_name", "filing_date", "trade_type", "price", "qty"],
        )
        assert 0.4 <= result.grounding_score <= 0.7
        assert "price" in result.ungrounded_fields
        assert "qty" in result.ungrounded_fields

    def test_fully_ungrounded(self):
        result = verify_extraction(
            extracted={
                "insider_name": "Xyz Zzyzx",
                "filing_date": "2020-01-01",
                "price": 999.99,
            },
            source_text=SEC_FILING_TEXT,
            fields_to_check=["insider_name", "filing_date", "price"],
        )
        assert result.grounding_score == 0.0
        assert len(result.ungrounded_fields) == 3

    def test_empty_fields(self):
        result = verify_extraction(
            extracted={},
            source_text=SEC_FILING_TEXT,
            fields_to_check=[],
        )
        assert result.grounding_score == 0.0

    def test_missing_field_counted_as_ungrounded(self):
        result = verify_extraction(
            extracted={"insider_name": "Timothy D. Cook"},
            source_text=SEC_FILING_TEXT,
            fields_to_check=["insider_name", "nonexistent_field"],
        )
        assert result.grounding_score == 0.5
        assert "nonexistent_field" in result.ungrounded_fields

    def test_threshold_filtering(self):
        """Trades below 0.5 grounding should be rejected."""
        result = verify_extraction(
            extracted={
                "insider_name": "Hallucinated Person",
                "filing_date": "2020-03-01",
                "trade_type": "Purchase",
                "price": 999.99,
                "qty": 1,
            },
            source_text=SEC_FILING_TEXT,
            fields_to_check=["insider_name", "filing_date", "trade_type", "price", "qty"],
        )
        # Only "Purchase" might match, so score should be low
        assert result.grounding_score < 0.5


# ── Headline verification tests ─────────────────────────────────────────────


class TestHeadlineVerification:
    def test_grounded_headline(self):
        sources = [NEWS_ARTICLES_TEXT]
        assert verify_headline("Apple Reports Record Q3 Revenue", sources) is True

    def test_partially_grounded_headline(self):
        sources = [NEWS_ARTICLES_TEXT]
        # Contains enough matching words from the source
        assert verify_headline("Apple stock rallies after strong Q3 revenue", sources) is True

    def test_ungrounded_headline(self):
        sources = [NEWS_ARTICLES_TEXT]
        assert (
            verify_headline("Tesla announces new battery technology breakthrough", sources) is False
        )

    def test_empty_headline(self):
        sources = [NEWS_ARTICLES_TEXT]
        assert verify_headline("", sources) is False

    def test_empty_sources(self):
        assert verify_headline("Apple Reports Record Q3 Revenue", []) is False

    def test_custom_threshold(self):
        sources = [NEWS_ARTICLES_TEXT]
        # With a very high threshold, fewer headlines should pass
        result = verify_headline("Something about Apple maybe", sources, threshold=0.9)
        assert result is False


# ── Integration-style tests with realistic data ─────────────────────────────


class TestRealisticScenarios:
    def test_real_insider_trade_grounded(self):
        """A real insider trade extracted correctly from SEC filing."""
        result = verify_extraction(
            extracted={
                "insider_name": "Timothy D. Cook",
                "filing_date": "2024-08-15",
                "title": "CEO",
                "trade_type": "Purchase",
                "price": 182.50,
                "qty": 50000,
            },
            source_text=SEC_FILING_TEXT,
            fields_to_check=["insider_name", "filing_date", "trade_type", "price", "qty"],
        )
        assert result.grounding_score >= 0.8
        assert result.ungrounded_fields == []

    def test_hallucinated_insider_trade_rejected(self):
        """A hallucinated trade should score below the 0.5 threshold."""
        result = verify_extraction(
            extracted={
                "insider_name": "Elon Musk",
                "filing_date": "2024-09-01",
                "title": "Director",
                "trade_type": "Purchase",
                "price": 250.00,
                "qty": 100000,
            },
            source_text=SEC_FILING_TEXT,
            fields_to_check=["insider_name", "filing_date", "trade_type", "price", "qty"],
        )
        assert result.grounding_score < 0.5

    def test_congress_trade_verification(self):
        """Verify congressional trade extraction."""
        congress_source = """\
        Congressional Stock Trading Disclosures
        Senator Nancy Pelosi disclosed a purchase of Apple (AAPL) stock
        on 2024-07-15. The transaction was valued between $1M and $5M.
        """
        result = verify_extraction(
            extracted={
                "politician": "Nancy Pelosi",
                "transaction_date": "2024-07-15",
                "transaction_type": "Purchase",
            },
            source_text=congress_source,
            fields_to_check=["politician", "transaction_date", "transaction_type"],
        )
        assert result.grounding_score >= 0.8
