"""The one line that makes an event actionable.

`advisor daemon events --tier A` is the command you run in the morning, and
it listed six rows without a single number in them — a Tier A event is
defined by being actionable, and "STOP_BREACHED" alone is not.

Every branch reads keys the emitter is known to write. Anything absent is left
out rather than guessed at, printed as None, or crashed on.
"""

from __future__ import annotations

import pytest
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.summarize import summarize


def event(kind="STOP_BREACHED", tier=EventTier.A, symbol="CBRS", **payload) -> Event:
    return Event(
        source=EventSource.COMPUTED,
        kind=kind,
        tier=tier,
        symbol=symbol,
        dedup_key=f"{kind}:{symbol}",
        payload=payload,
    )


class TestTheRealEvents:
    """The six that were actually in the stream when this was written."""

    def test_a_stop_says_where_it_came_from_and_how_far_past(self):
        line = summarize(
            event(
                entry=217.09,
                price=199.66,
                unrealized_pct=-0.0803,
                threshold=-0.08,
                unrealized_usd=-34.86,
            )
        )
        assert "217.09 → 199.66" in line
        assert "-8.0%" in line
        assert "vs -8.0%" in line
        assert "-$34.86" in line

    def test_a_sized_offering_reports_both_the_amount_and_the_share(self):
        line = summarize(
            event(kind="FILING_DILUTION", symbol="AAOI", offering_usd=6e8, dilution_pct=0.067)
        )
        assert "$600M" in line
        assert "6.7% dilution" in line

    def test_a_debt_offering_is_not_called_dilution(self):
        line = summarize(
            event(
                kind="FILING_DEBT_ISSUANCE",
                symbol="AMD",
                offering_usd=4.75e9,
                offering_pct_of_cap=0.0056,
            )
        )
        assert "$4.75bn" in line
        assert "of market cap" in line
        assert "dilution" not in line

    def test_a_data_quality_failure_names_the_check_and_the_symbols(self):
        line = summarize(
            event(
                kind="DATA_QUALITY_FAILURE",
                symbol=None,
                check="price_agreement",
                failed=9,
                symbols=["BE", "CBRS", "COHR", "TE", "SPCX"],
            )
        )
        assert "price_agreement" in line
        assert "9 symbol(s)" in line
        assert "BE" in line

    def test_a_filing_without_a_figure_falls_back_to_its_form_and_items(self):
        line = summarize(
            event(kind="FILING_RESULTS", symbol="CRDO", form="8-K", items=["2.02", "9.01"])
        )
        assert "8-K" in line
        assert "2.02" in line


class TestOtherKinds:
    def test_a_factor_shock_reports_the_book_impact(self):
        line = summarize(
            event(
                kind="FACTOR_SHOCK_HITTING_BOOK",
                symbol=None,
                factor="BREADTH",
                z=-2.8,
                expected_book_move=-0.014,
            )
        )
        assert "BREADTH" in line and "z -2.80" in line and "-1.4%" in line

    def test_a_residual_divergence_reports_actual_against_expected(self):
        line = summarize(
            event(
                kind="RESIDUAL_DIVERGENCE",
                symbol="CRDO",
                actual_return=-0.2237,
                expected_return=0.0105,
                residual_z=-4.9,
            )
        )
        assert "-22.4%" in line and "+1.1% expected" in line and "z -4.90" in line

    def test_an_insider_cluster_reports_people_and_money(self):
        line = summarize(
            event(
                kind="INSIDER_SELLING_CLUSTER",
                symbol="CBRS",
                insider_count=6,
                side="SELLING",
                total_value=274_917_557.0,
            )
        )
        assert "6 insiders selling" in line and "$275M" in line

    def test_concentration_reports_the_weight_against_the_limit(self):
        line = summarize(
            event(kind="CONCENTRATION_WARNING", symbol="SPCX", weight=0.223, threshold=0.20)
        )
        assert "22.3% of the book" in line and "vs 20.0%" in line

    def test_an_expectations_shift_reports_both_readings(self):
        line = summarize(
            event(
                kind="IMPLIED_EXPECTATIONS_SHIFT",
                symbol="SPCX",
                implied_cagr=0.283,
                previous_implied_cagr=0.258,
            )
        )
        assert "25.8%" in line and "28.3%" in line

    def test_a_preliminary_offering_says_so(self):
        line = summarize(
            event(kind="FILING_DILUTION", symbol="AMD", offering_usd=4.75e9, preliminary=True)
        )
        assert "preliminary" in line


class TestMissingData:
    """Absent is left out, never guessed at or printed as None."""

    def test_an_empty_payload_produces_an_empty_line(self):
        assert summarize(event()) == ""

    def test_a_stop_without_an_entry_still_reports_the_move(self):
        line = summarize(event(unrealized_pct=-0.0803, threshold=-0.08))
        assert "-8.0%" in line
        assert "→" not in line
        assert "None" not in line

    def test_a_none_value_is_not_rendered(self):
        line = summarize(event(unrealized_pct=-0.08, entry=None, price=None, unrealized_usd=None))
        assert "None" not in line

    def test_a_string_where_a_number_belongs_does_not_crash(self):
        assert "None" not in summarize(event(unrealized_pct="a lot", entry="x"))

    def test_an_unsized_offering_still_names_itself(self):
        line = summarize(event(kind="FILING_DILUTION", offering_usd=None, label="share offering"))
        assert line == "share offering"

    def test_an_unknown_kind_with_a_label_uses_it(self):
        assert summarize(event(kind="SOMETHING_NEW", label="a new thing")) == "a new thing"

    @pytest.mark.parametrize("payload", [{}, {"url": "https://x"}, {"provider": "SEC EDGAR"}])
    def test_payloads_with_nothing_numeric_are_empty_not_noisy(self, payload):
        assert summarize(event(**payload)) == ""


class TestFormatting:
    def test_billions_and_millions_are_scaled(self):
        assert "$4.75bn" in summarize(event(kind="X", offering_usd=4.75e9))
        assert "$600M" in summarize(event(kind="X", offering_usd=6e8))

    def test_a_small_amount_is_not_scaled(self):
        line = summarize(event(unrealized_pct=-0.08, unrealized_usd=-34.86))
        assert "-$34.86" in line
        assert "$-34.86" not in line, "the sign belongs outside the currency mark"

    def test_a_gain_carries_its_sign(self):
        assert "+25.0%" in summarize(event(kind="PROFIT_TARGET_HIT", unrealized_pct=0.25))

    def test_a_weight_does_not_carry_a_plus(self):
        """A position size is not a change; a leading + would misread."""
        line = summarize(event(kind="CONCENTRATION_WARNING", weight=0.223))
        assert "22.3%" in line and "+22.3%" not in line


class TestNewsContext:
    """A headline is the entire content of a context event.

    Without it, the eight items the review pulled to explain CBRS's stop
    rendered as eight identical blank rows.
    """

    def test_the_headline_is_the_detail(self):
        line = summarize(
            event(
                kind="NEWS_CONTEXT",
                tier=EventTier.C,
                title="Cerebras Systems (CBRS) Stock Looks Fairly Valued After Its 42% Fall",
                url="https://example.com/a",
                provider="Simply Wall St.",
                explains="STOP_BREACHED",
            )
        )
        assert "Cerebras Systems" in line
        assert "42% Fall" in line

    def test_it_says_what_it_was_pulled_to_explain(self):
        """News here is pulled to explain an event, never polled."""
        line = summarize(
            event(
                kind="NEWS_CONTEXT", tier=EventTier.C, title="A headline", explains="STOP_BREACHED"
            )
        )
        assert "stop breached" in line

    def test_a_context_item_without_a_trigger_still_renders(self):
        assert summarize(event(kind="NEWS_CONTEXT", title="A headline")) == "A headline"


class TestPositionLifecycle:
    """The realised return is the point of a close, and it was the one thing
    not rendered — "BE position closed" said nothing about whether it was
    closed at a gain or a loss."""

    def test_a_close_reports_the_realised_return(self):
        line = summarize(
            event(kind="POSITION_CLOSED", symbol="BE", quantity=2, realized_pct=0.0702)
        )
        assert "2 shares" in line
        assert "closed at +7.0%" in line

    def test_a_close_at_a_loss_carries_the_sign(self):
        line = summarize(event(kind="POSITION_CLOSED", quantity=-5, realized_pct=-0.12))
        assert "-12.0%" in line

    def test_an_open_reports_size_price_and_notional(self):
        line = summarize(
            event(
                kind="POSITION_OPENED", symbol="GOOGL", quantity=3, price=344.04, notional=1032.12
            )
        )
        assert "+3 @ 344.04" in line
        assert "$1,032.12" in line

    def test_a_resize_reports_both_ends_and_the_direction(self):
        line = summarize(
            event(
                kind="POSITION_SIZE_CHANGED",
                symbol="SPCX",
                from_quantity=9,
                to_quantity=11,
                direction="increased",
            )
        )
        assert "+9 → +11 shares" in line
        assert "increased" in line

    def test_a_lifecycle_event_missing_its_numbers_does_not_crash(self):
        assert "None" not in summarize(event(kind="POSITION_CLOSED", realized_pct=None))


class TestEveryStoredKindRenders:
    """A kind that renders empty is a row the reader cannot act on.

    Three position-lifecycle kinds were silently blank until a live stream
    was read row by row; this fails if a fourth appears.
    """

    KINDS_WITH_NO_FIGURE = {"HEARTBEAT"}

    def test_the_known_payload_shapes_all_produce_a_line(self):
        shapes = [
            ("STOP_BREACHED", {"unrealized_pct": -0.08, "entry": 1.0, "price": 0.92}),
            ("PROFIT_TARGET_HIT", {"unrealized_pct": 0.25, "entry": 1.0, "price": 1.25}),
            ("DEEP_DRAWDOWN", {"unrealized_pct": -0.30}),
            ("CONCENTRATION_WARNING", {"weight": 0.22, "threshold": 0.20}),
            ("POSITION_OPENED", {"quantity": 3, "price": 344.04, "notional": 1032.12}),
            ("POSITION_CLOSED", {"quantity": 2, "realized_pct": 0.07}),
            ("POSITION_SIZE_CHANGED", {"from_quantity": 9, "to_quantity": 11}),
            ("FILING_DILUTION", {"offering_usd": 6e8, "dilution_pct": 0.067}),
            ("FILING_RESULTS", {"form": "8-K", "items": ["2.02"]}),
            (
                "FACTOR_SHOCK_HITTING_BOOK",
                {"factor": "MKT", "z": -2.8, "expected_book_move": -0.014},
            ),
            (
                "RESIDUAL_DIVERGENCE",
                {"actual_return": -0.22, "expected_return": 0.01, "residual_z": -4.9},
            ),
            (
                "INSIDER_SELLING_CLUSTER",
                {"insider_count": 6, "side": "SELLING", "total_value": 2.7e8},
            ),
            ("IMPLIED_EXPECTATIONS_SHIFT", {"implied_cagr": 0.28, "previous_implied_cagr": 0.25}),
            ("DATA_QUALITY_FAILURE", {"check": "price_agreement", "failed": 9, "symbols": ["BE"]}),
            ("NEWS_CONTEXT", {"title": "A headline"}),
        ]
        blank = [kind for kind, payload in shapes if not summarize(event(kind=kind, **payload))]
        assert blank == [], f"these render as an empty row: {blank}"
