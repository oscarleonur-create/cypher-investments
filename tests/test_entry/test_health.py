"""The company's health beside every reading and proposal.

User, 2026-09-28: MDB's ADD the day its CEO left for Meta showed the price and
the headlines, never the business — "el agente no está viendo el big picture"
— and "esto lo tienes que hacer para todos los análisis".
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from advisor.daemon.store import DaemonStore
from advisor.entry import health as H
from advisor.entry import run
from advisor.entry.health import Health, assess, latest_health, quarter_growth, save_health
from advisor.valuation.figures import Figures, OwnMargin
from advisor.valuation.history import Point, Series

from tests.test_entry.test_proposal import ENTERED, IN, NET_LIQ, NOW, OUT, mk

TODAY = date(2026, 9, 28)
QUARTER_ENDS = [date(2024, 7, 31), date(2024, 10, 31), date(2025, 1, 31), date(2025, 4, 30),
                date(2025, 7, 31), date(2025, 10, 31), date(2026, 1, 31), date(2026, 4, 30),
                date(2026, 7, 31)]  # fmt: skip
# MDB's trailing revenue, $M, as the SEC series gave it on 2026-09-28.
MDB_TTM = [1819.6, 1916.0, 2006.4, 2104.9, 2218.2, 2317.1, 2463.8, 2602.4, 2782.8]


def series(ttm=MDB_TTM, shares=(81.1, 82.0), broken_after=None):
    return Series(
        symbol="MDB",
        revenue_ttm=[Point(end=e, value=v * 1e6, known=e) for e, v in zip(QUARTER_ENDS, ttm)],
        shares=[
            Point(end=date(2025, 7, 31), value=shares[0] * 1e6, known=TODAY),
            Point(end=date(2026, 7, 31), value=shares[1] * 1e6, known=TODAY),
        ],  # fmt: skip
        broken_after=broken_after,
    )


def figures(fcf=0.237, median=0.069, nopat=-0.004, net_cash=2.41e9):
    margins = [OwnMargin(kind="fcf", value=fcf, label="FCF, trailing 12 months to 2026-07-31")]
    if median is not None:
        margins.append(OwnMargin(kind="median", value=median, label="FY2024–FY2026 median FCF"))
    margins.append(OwnMargin(kind="nopat", value=nopat, label="operating margin after 21% tax"))
    return Figures(
        symbol="MDB", price=334.68, shares=80.56e6, net_cash=net_cash,
        balance_asof=date(2026, 7, 31), revenue_base=2.78e9, revenue_growth=0.255,
        start_margin=fcf, start_margin_label="FCF, trailing 12 months to 2026-07-31",
        margins=margins,
    )  # fmt: skip


# ── assessment ───────────────────────────────────────────────────────────


def test_mdb_healthy_and_accelerating_from_its_real_filings():
    h = assess("MDB", TODAY, series=series(), figures=figures(), sic=7372)
    assert h.trend() == "accelerating" and h.concerns() == []
    assert round(h.growth * 100, 1) == 25.5 and round(h.growth_two_q * 100, 1) == 22.8
    text = "\n".join(h.lines())
    assert "+25.5% on a year earlier" in text and "Net cash $2.41bn" in text
    assert "Diluted shares +1.1%" in text
    assert "none flagged; news since then is not in these numbers" in text


@pytest.mark.parametrize(
    "last_two,trend",
    [((2602.4, 2782.8), "accelerating"), ((2602.4, 2620.0), "decelerating"),
     ((2560.0, 2680.0), "steady")],
)  # fmt: skip
def test_trend_is_measured_over_two_quarters(last_two, trend):
    ttm = MDB_TTM[:-2] + list(last_two)
    assert assess("MDB", TODAY, series=series(ttm=ttm)).trend() == trend


def test_exactly_at_the_trend_threshold_is_steady():
    h = Health(symbol="X", computed_on=TODAY, growth=0.20 + H.TREND_POINTS, growth_two_q=0.20)
    assert h.trend() == "steady"


def test_shrinking_revenue_is_a_concern():
    h = assess("X", TODAY, series=series(ttm=list(reversed(MDB_TTM))))
    assert h.trend() == "shrinking" and "revenue shrinking" in h.concerns()


def test_cash_burn_heavy_dilution_and_net_debt_are_concerns():
    h = assess("TE", TODAY, series=series(shares=(50.0, 90.0)),
               figures=figures(fcf=-0.185, median=None, net_cash=-5e9))  # fmt: skip
    concerns = h.concerns()
    assert "burning cash" in concerns
    assert f"net debt above {H.NET_DEBT_FCF_YEARS:g} years of free cash flow" in concerns
    assert f"diluting more than {H.DILUTION_CONCERN * 100:g}% a year" in concerns
    assert any(line.startswith("Net debt $5.00bn") for line in h.lines())


def test_a_margin_far_under_its_own_median_is_a_concern():
    # META, 2026-06-30: FCF 18.0% against a 32.7% median (its capex build).
    h = assess("META", TODAY, series=series(), figures=figures(fcf=0.180, median=0.327))
    assert h.concerns() == ["free cash flow margin 14.7 points under its own median"]


def test_a_meaningless_median_is_not_shown_or_compared():
    # TE's FY2024–FY2025 median was -2610%: a year with almost no revenue.
    h = assess("TE", TODAY, series=series(), figures=figures(fcf=0.05, median=-26.1))
    assert "-2610" not in "\n".join(h.lines()) and h.concerns() == []


def test_a_bank_is_not_judged_by_free_cash_flow_or_net_cash():
    h = assess("NU", TODAY, series=series(), figures=figures(fcf=-0.127, net_cash=-9e9), sic=6199)
    assert h.financial and h.concerns() == []
    text = "\n".join(h.lines())
    assert "financial company" in text and "-12.7%" not in text and "Net debt" not in text


def test_no_trailing_comparison_falls_back_to_the_latest_quarter():
    fig = figures().model_copy(update={"revenue_growth": None})
    h = assess("NU", TODAY, figures=fig, sic=6199, quarter=(0.502, date(2026, 6, 30)))
    assert "the quarter to 2026-06-30 was +50.2%" in h.lines()[0]


def test_quarter_growth_needs_the_same_quarter_a_year_before():
    q = {date(2025, 6, 30): 100.0, date(2025, 9, 30): 110.0, date(2026, 6, 30): 150.0}
    assert quarter_growth(q) == (0.5, date(2026, 6, 30))
    assert quarter_growth({date(2026, 6, 30): 150.0, date(2026, 3, 31): 140.0}) is None
    assert quarter_growth({}) is None
    assert quarter_growth({date(2025, 6, 30): 0.0, date(2026, 6, 30): 1.0}) is None


def test_a_share_count_break_within_the_year_hides_dilution():
    h = assess("X", TODAY, series=series(broken_after=date(2026, 1, 31)))
    assert h.dilution is None


def test_nothing_on_file_is_none_not_an_empty_block():
    assert assess("X", TODAY) is None


def test_no_revenue_on_file_is_said():
    h = assess("X", TODAY, figures=figures().model_copy(update={"revenue_base": None}))
    assert any("No trailing revenue on file" in line for line in h.lines())


# ── store and refresh ────────────────────────────────────────────────────


def test_store_round_trip_and_age_limit(tmp_path):
    db = tmp_path / "research.db"
    assert latest_health(db, "MDB") is None  # no table yet
    h = assess("MDB", TODAY, series=series(), figures=figures())
    save_health(db, h)
    save_health(db, h)  # the same day twice: one row
    assert latest_health(db, "mdb", TODAY) == h
    too_old = TODAY + timedelta(days=H.MAX_AGE_DAYS + 1)
    assert latest_health(db, "MDB", too_old) is None


def test_refresh_computes_once_a_day_and_a_failure_keeps_the_last(tmp_path, monkeypatch):
    db = tmp_path / "research.db"
    monkeypatch.setattr(H, "_COMPUTED", set())
    calls = []

    def loader(sym, today):
        calls.append(today)
        return assess(sym, today, series=series(), figures=figures())

    now = datetime(2026, 9, 28, 6, 0)
    first = H.refresh_health(db, "MDB", now, loader=loader)
    again = H.refresh_health(db, "MDB", now, loader=loader)
    assert calls == [TODAY] and again == first

    def boom(sym, today):
        raise TimeoutError("sec")

    kept = H.refresh_health(db, "MDB", now + timedelta(days=1), loader=boom)
    assert kept == first  # yesterday's, not nothing


# ── every analysis sees it ───────────────────────────────────────────────


def test_every_reading_carries_the_health_facts(tmp_path):
    from advisor.story.reading import gather_facts

    db = tmp_path / "research.db"
    save_health(db, assess("MDB", NOW.date(), series=series(), figures=figures()))
    store = DaemonStore(db)
    facts = gather_facts(store, "MDB")
    store.close()
    health = [f for f in facts if f.kind == "HEALTH"]
    assert health and "+25.5% on a year earlier" in health[0].text
    assert health[0].date == "2026-07-31"


def test_the_reading_prompt_asks_for_the_bigger_picture():
    from advisor.story.reading import SYSTEM_PROMPT

    assert "HEALTH facts" in SYSTEM_PROMPT and "bigger picture" in SYSTEM_PROMPT


def _run(tmp_path, sheets, *, read=True, health_source=None):
    from advisor.daemon.book import BookSnapshot

    daemon = DaemonStore(tmp_path / "research.db")
    daemon.save_book(BookSnapshot(net_liq=NET_LIQ))

    def reader(sym):
        return SimpleNamespace(status=SimpleNamespace(value="OK"),
                               stance=SimpleNamespace(value="NEUTRAL"), sentences=[])  # fmt: skip

    out = run.propose_all(
        daemon, NOW, symbols=list(sheets), reader=reader, read=read,
        news=lambda *a: [], health_source=health_source,
        sheet_builder=lambda st, sym, n, scanner_store=None: sheets[sym].model_copy(
            update={"symbol": sym}
        ),
    )  # fmt: skip
    daemon.close()
    return out


def test_every_proposal_quiet_or_acting_carries_the_health(tmp_path):
    h = assess("MDB", NOW.date(), series=series(), figures=figures())
    asked = []

    def source(store, sym, now):
        asked.append(sym)
        return h

    proposals, errors = _run(tmp_path, {"QUIET": mk(IN, IN), "GO": mk(ENTERED, OUT)},
                             health_source=source)  # fmt: skip
    assert errors == [] and asked == ["QUIET", "GO"]
    for p in proposals:
        texts = [r.text for r in p.reasons if r.text.startswith("Health: ")]
        assert texts and "+25.5%" in texts[0]
        assert all(r.source == "SEC filings to 2026-07-31 (SEC companyconcept, SEC XBRL)"
                   for r in p.reasons if r.text.startswith("Health: "))  # fmt: skip
    go = next(p for p in proposals if p.symbol == "GO")
    assert go.action.value == "ENTER"  # unchanged: the health describes, never decides


def test_a_failed_health_is_an_error_not_a_lost_proposal(tmp_path):
    def boom(*a):
        raise TimeoutError("sec")

    proposals, errors = _run(tmp_path, {"GO": mk(ENTERED, OUT)}, health_source=boom)
    assert len(proposals) == 1 and errors == ["GO: health failed: sec"]


def test_without_reading_the_stored_health_is_used_and_nothing_is_fetched(tmp_path):
    save_health(tmp_path / "research.db",
                assess("GO", NOW.date(), series=series(), figures=figures()))  # fmt: skip

    def never(*a):
        raise AssertionError("no network when read=False")

    proposals, errors = _run(tmp_path, {"GO": mk(IN, IN)}, read=False, health_source=never)
    assert errors == [] and any(r.text.startswith("Health: ") for r in proposals[0].reasons)


# ── deep research ────────────────────────────────────────────────────────


def test_deep_research_numbers_the_health_as_its_own_source():
    from advisor.research.deep_research import _build_context
    from advisor.research.models import Reference, SourceType

    ref = Reference(id=1, title="MDB financial health", url="https://sec/x",
                    source_type=SourceType.OTHER, published_date="2026-07-31",
                    detail="SEC XBRL figures")  # fmt: skip
    context = _build_context("", {"https://sec/x": "Revenue $2.78bn ..."}, [ref])
    assert context.startswith("[1] (FINANCIAL HEALTH, SEC XBRL figures, period to 2026-07-31)")
    assert "Revenue $2.78bn" in context


def test_deep_research_keeps_the_health_and_renumbers_the_abstracts_citations():
    from advisor.research.deep_research import _assemble
    from advisor.research.models import Reference, SourceType

    def ref(i, kind):
        return Reference(id=i, title=f"s{i}", url=f"https://x/{i}", source_type=kind)

    sources = [ref(1, SourceType.NEWS), ref(2, SourceType.OTHER), ref(3, SourceType.NEWS),
               ref(4, SourceType.NEWS)]  # fmt: skip
    out = SimpleNamespace(
        abstract="Healthy [2]; the CEO left [4]; unknown [9].", what_they_do="db",
        customers=[], supply_chain=None, recent_developments=[], management_quotes=[],
        second_order_thesis=None,
    )  # fmt: skip
    brief = _assemble("MDB", out, sources, None, "", None)
    # Kept: the health (always) and source 4 (cited inline); 1 and 3 never cited.
    assert [(r.id, r.title) for r in brief.references] == [(1, "s2"), (2, "s4")]
    assert brief.abstract == "Healthy [1]; the CEO left [2]; unknown ."
