"""Family I: parsing both sources, which purchases count, clusters, and the I / 2+ groups."""

from __future__ import annotations

import io
import zipfile
from dataclasses import replace
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from advisor.breadth import insiders as I
from advisor.breadth import signals as S
from advisor.breadth.panel import Panel
from advisor.breadth.store import BreadthStore
from advisor.daemon.market_calendar import MARKET_TZ

NOW = datetime(2026, 9, 28, 21, 0, tzinfo=MARKET_TZ)


def txn(owner=1, issuer=100, day="2026-05-04", filed=None, code="P", shares=1000.0, price=50.0,
        officer=True, director=False, ten_pct=False, plan=False, acq="A"):  # fmt: skip
    d = date.fromisoformat(day)
    return I.Txn(
        accession=f"acc-{owner}-{day}", issuer_cik=issuer, owner_cik=owner, officer=officer,
        director=director, ten_pct=ten_pct, title=None, code=code, acq_disp=acq, shares=shares,
        price=price, trans_date=d, filed=date.fromisoformat(filed) if filed else d,
        direct="D", plan=plan, source="dataset",
    )  # fmt: skip


# ── parsing ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,flag",
    [("1", True), ("true", True), ("0", False), ("false", False), ("", None), (None, None)],
)
def test_the_four_spellings_of_the_10b5_1_box(raw, flag):
    assert I.plan_flag(raw) is flag


def test_dataset_links_are_read_not_built():
    html = (
        '<a href="/files/datastandardsinnovation/data/insider-transactions-data-sets/'
        '2026q2_form345.zip">x</a><a href="/files/structureddata/data/'
        'insider-transactions-data-sets/2026q1_form345.zip">y</a>'
    )
    links = I.dataset_links(html)
    assert links["2026q2"].startswith("https://www.sec.gov/files/datastandardsinnovation/")
    assert "structureddata" in links["2026q1"]


def _zip(subs, owners, trans) -> bytes:
    def tsv(header, rows):
        return "\t".join(header) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "SUBMISSION.tsv",
            tsv(
                ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "AFF10B5ONE"],
                subs,
            ),
        )
        z.writestr(
            "REPORTINGOWNER.tsv",
            tsv(
                ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"],
                owners,
            ),
        )
        z.writestr(
            "NONDERIV_TRANS.tsv",
            tsv(
                [
                    "ACCESSION_NUMBER",
                    "TRANS_DATE",
                    "TRANS_CODE",
                    "TRANS_SHARES",
                    "TRANS_PRICEPERSHARE",
                    "TRANS_ACQUIRED_DISP_CD",
                    "DIRECT_INDIRECT_OWNERSHIP",
                ],
                trans,
            ),
        )
    return buf.getvalue()  # fmt: skip


def test_parse_dataset():
    blob = _zip(
        subs=[
            ["A1", "30-JUN-2026", "4", "0000000100", "0"],
            ["A2", "30-JUN-2026", "4/A", "0000000100", "0"],  # amendment: dropped
            ["A3", "01-JUL-2026", "4", "0000000100", "true"],
            ["A4", "01-JUL-2026", "3", "0000000100", ""],  # Form 3: dropped
        ],
        owners=[
            ["A1", "0000000009", "Director,Officer", "CFO"],
            ["A1", "0000000005", "TenPercentOwner", ""],  # joint filing: one buyer, lowest CIK
            ["A2", "0000000009", "Officer", "CFO"],
            ["A3", "0000000007", "Officer", "CEO"],
        ],
        trans=[
            ["A1", "29-JUN-2026", "P", "1000", "20.5", "A", "D"],
            ["A1", "29-JUN-2026", "M", "500", "0", "A", "D"],  # exercise: not kept
            ["A2", "29-JUN-2026", "P", "1000", "20.5", "A", "D"],
            ["A3", "30-JUN-2026", "S", "200", "21", "D", "I"],
        ],
    )
    got = I.parse_dataset(blob)
    assert [(t.accession, t.code) for t in got] == [("A1", "P"), ("A3", "S")]
    a1 = got[0]
    assert a1.owner_cik == 5 and a1.officer and a1.director and a1.ten_pct
    assert a1.filed == date(2026, 6, 30) and a1.trans_date == date(2026, 6, 29)
    assert a1.value == pytest.approx(20500.0) and a1.plan is False
    assert got[1].plan is True


FORM4 = """<SEC-DOCUMENT>
<XML>
<ownershipDocument>
  <documentType>4</documentType>
  <issuer><issuerCik>0000000100</issuerCik></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0000000042</rptOwnerCik></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>0</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner></reportingOwnerRelationship>
  </reportingOwner>
  <aff10b5One>0</aff10b5One>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-09-24</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>3000</value></transactionShares>
        <transactionPricePerShare><value>12.5</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-09-24</value></transactionDate>
      <transactionCoding><transactionCode>F</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>10</value></transactionShares>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
</XML>
</SEC-DOCUMENT>"""


def test_parse_form4():
    (t,) = I.parse_form4(FORM4, "acc-1", date(2026, 9, 25))
    assert (t.issuer_cik, t.owner_cik, t.code, t.shares, t.price) == (100, 42, "P", 3000, 12.5)
    assert t.director and not t.officer and t.plan is False and t.source == "daily"
    assert t.filed == date(2026, 9, 25)


@pytest.mark.parametrize(
    "text", ["", "<ownershipDocument><broken", FORM4.replace(">4<", ">4/A<", 1)]
)
def test_unreadable_or_amended_form4_yields_nothing(text):
    assert I.parse_form4(text, "x", date(2026, 9, 25)) == []


def test_form4_rows_keep_the_issuers_line_once():
    idx = (
        "CIK|Company Name|Form Type|Date Filed|Filename\n"
        "100|ACME|4|20260925|edgar/data/100/0000000042-26-000001.txt\n"
        "42|DOE JOHN|4|20260925|edgar/data/42/0000000042-26-000001.txt\n"
        "100|ACME|4|20260925|edgar/data/100/0000000042-26-000001.txt\n"
        "200|OTHER|4|20260925|edgar/data/200/0000000043-26-000001.txt\n"
        "100|ACME|8-K|20260925|edgar/data/100/0000000100-26-000009.txt\n"
    )
    assert I.form4_rows(idx, {100}) == [
        ("0000000042-26-000001", "edgar/data/100/0000000042-26-000001.txt")
    ]


# ── syncing ───────────────────────────────────────────────────────────────


HOLIDAYS = {date(2026, 7, 3), date(2026, 9, 7)}  # no EDGAR index on these


def listing(start: date, end: date, skip=HOLIDAYS) -> str:
    """A daily-index ``index.json`` with one master file per weekday."""
    import json

    items, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in skip:
            items.append({"name": f"master.{d:%Y%m%d}.idx"})
        d += timedelta(days=1)
    items.append({"name": "form.20260701.idx"})  # other files in the directory
    return json.dumps({"directory": {"item": items}})


class Sec:
    def __init__(self, fail_day=None, published_until=date(2026, 9, 25)):
        self.fail_day = fail_day
        self.published_until = published_until
        self.urls = []

    def text(self, url):
        self.urls.append(url)
        if url == I.DATASETS_PAGE:
            return '<a href="/files/x/2026q2_form345.zip">'
        if url.endswith("index.json"):
            return listing(date(2026, 7, 1), self.published_until)
        return FORM4

    def bytes(self, url):
        self.urls.append(url)
        if url.endswith("2026q2_form345.zip"):
            return _zip([["A1", "30-JUN-2026", "4", "100", "0"]],
                        [["A1", "9", "Officer", "CFO"]],
                        [["A1", "29-JUN-2026", "P", "1000", "20", "A", "D"]])  # fmt: skip
        if any(f"master.{h:%Y%m%d}" in url for h in HOLIDAYS):
            raise Refusal(403)  # what the SEC really answers for a missing index
        if self.fail_day and self.fail_day in url:
            raise TimeoutError("sec.gov")
        return b"100|ACME|4|20260925|edgar/data/100/0000000042-26-000001.txt\n"


def test_sync_loads_data_sets_then_days_and_is_idempotent(tmp_path):
    sec = Sec(fail_day="master.20260910")
    with BreadthStore(tmp_path / "b.db") as store:
        r = I.sync_insiders(
            store, NOW, {100}, get_text=sec.text, get_bytes=sec.bytes, backfill_days=120
        )
        assert r["datasets"] == 1
        # Weekdays 2026-07-01 .. 09-25 read, less two holidays and the one that failed.
        weekdays = sum(1 for k in range(87) if (date(2026, 7, 1) + timedelta(k)).weekday() < 5)
        assert r["days"] == weekdays - 2 - 1 and r["errors"] == ["2026-09-10: sec.gov"]
        assert not r.get("refused")
        # Holidays are never asked for: the SEC's 403 for them looks like a block.
        assert not any("master.20260703" in u or "master.20260907" in u for u in sec.urls)
        done = {k for (k,) in store.conn.execute("SELECT key FROM breadth_insider_state")}
        assert {"2026-07-03", "2026-09-07"} <= done
        n = store.conn.execute("SELECT COUNT(*) FROM breadth_insider_txns").fetchone()[0]
        # The same Form 4 served every day is one transaction, not sixty.
        assert n == 2
        sec2 = Sec()
        r2 = I.sync_insiders(
            store, NOW, {100}, get_text=sec2.text, get_bytes=sec2.bytes, backfill_days=120
        )
        assert r2["datasets"] == 0 and r2["days"] == 1  # only the day that failed
        assert not any("2026q2" in u for u in sec2.urls)


class Refusal(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = type("R", (), {"status_code": status})()


@pytest.mark.parametrize("status", [429, 403])
def test_a_refusal_stops_the_days_and_leaves_them_for_next_run(tmp_path, status):
    calls = []

    def get_bytes(url):
        calls.append(url)
        if "master.20260706" in url:
            raise Refusal(status)
        return None  # no Form 4 index content needed

    def get_text(url):
        return listing(date(2026, 6, 1), date(2026, 9, 25)) if url.endswith("index.json") else ""

    with BreadthStore(tmp_path / "b.db") as store:
        r = I.sync_insiders(
            store, NOW, {100}, get_text=get_text, get_bytes=get_bytes, backfill_days=120
        )
        # No data set: days start 90 back (Jun 30); Jun 30, Jul 1, 2 read, Jul 3 a
        # holiday, stopped on the 6th.
        assert r["refused"] and r["days"] == 3
        assert not any("master.20260707" in u for u in calls)  # no hammering
        done = {k for (k,) in store.conn.execute("SELECT key FROM breadth_insider_state")}
        assert "2026-07-06" not in done


def test_a_day_not_yet_published_stops_the_read_without_being_marked(tmp_path):
    sec = Sec(published_until=date(2026, 9, 24))
    with BreadthStore(tmp_path / "b.db") as store:
        I.sync_insiders(store, NOW, {100}, get_text=sec.text, get_bytes=sec.bytes)
        done = {k for (k,) in store.conn.execute("SELECT key FROM breadth_insider_state")}
        assert "2026-09-24" in done and "2026-09-25" not in done
        assert not any("master.20260925" in u for u in sec.urls)


def test_the_daily_path_reaches_back_only_so_far(tmp_path):
    sec = Sec()
    with BreadthStore(tmp_path / "b.db") as store:
        I.sync_insiders(store, NOW, {100}, get_text=sec.text, get_bytes=sec.bytes)
        days = [u for u in sec.urls if "master." in u]
        assert days and all(u.split("master.")[1][:8] >= "20260824" for u in days)


def test_daily_days_reads_only_master_files():
    days = I.daily_days(listing(date(2026, 7, 1), date(2026, 7, 7)))
    assert days == {date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 6), date(2026, 7, 7)}
    assert I.daily_days("not json") == set()


def test_a_linked_data_set_that_is_missing_is_retried(tmp_path):
    with BreadthStore(tmp_path / "b.db") as store:
        r = I.sync_insiders(
            store,
            NOW,
            {100},
            get_text=lambda u: '<a href="/f/2026q2_form345.zip">',
            get_bytes=lambda u: None,
        )
        assert r["datasets"] == 0 and "2026q2" in r["errors"][0]


def test_the_sec_limiter_is_shared_and_waits(monkeypatch):
    import time

    from advisor.news import edgar

    monkeypatch.setattr(edgar, "_LIMITER", None)
    monkeypatch.setattr("advisor.research.edgar._ensure_identity", lambda ua: None)
    start = time.monotonic()
    for _ in range(5):
        edgar._client_ready()
    # Five calls at the default 8/s take at least four intervals.
    assert time.monotonic() - start >= 4 / 8 - 0.01


def test_sync_survives_an_unreachable_index_page(tmp_path):
    def down(url):
        raise TimeoutError("sec.gov")

    with BreadthStore(tmp_path / "b.db") as store:
        r = I.sync_insiders(store, NOW, {100}, get_text=down, get_bytes=lambda u: None)
        assert r["errors"][0].startswith("data set index")


# ── which purchases count ─────────────────────────────────────────────────


def test_opportunistic_filters():
    keep = txn()
    dropped = [
        txn(code="S"),
        txn(plan=True),
        txn(officer=False, director=False, ten_pct=True),  # a fund, not an insider
        txn(price=0.0),
        txn(shares=None),
        txn(acq="D"),
    ]
    assert S.opportunistic([keep, *dropped]) == [keep]
    assert S.opportunistic([txn(plan=None)])  # the box not stated is not a plan


def test_a_routine_buyer_is_not_information():
    history = [txn(day=f"{y}-05-10") for y in (2023, 2024, 2025)]
    now = txn(day="2026-05-04")
    assert now not in S.opportunistic([*history, now])
    # Two of the three years is not a habit.
    assert now in S.opportunistic([*history[1:], now])
    # The same month for another company is another habit.
    assert now in S.opportunistic([replace(h, issuer_cik=999) for h in history] + [now])


# ── clusters ──────────────────────────────────────────────────────────────

SESSIONS = pd.bdate_range("2026-01-01", "2026-12-31")


def events(purchases, t=S.DEFAULT):
    return S.insider_events(purchases, {100: ["ACME"]}, SESSIONS, t)


def test_two_buyers_make_a_cluster_on_the_session_after_the_filing():
    ev = events([txn(owner=1, day="2026-05-04"), txn(owner=2, day="2026-05-08")])  # a Friday
    assert [(SESSIONS[e.row].date(), e.buyers) for e in ev] == [(date(2026, 5, 11), 2)]


def test_one_buyer_twice_is_not_a_cluster():
    assert events([txn(owner=1, day="2026-05-04"), txn(owner=1, day="2026-05-08")]) == []


def test_the_window_is_thirty_days_inclusive_of_its_end():
    inside = [txn(owner=1, day="2026-05-01"), txn(owner=2, day="2026-05-30")]
    outside = [txn(owner=1, day="2026-05-01"), txn(owner=2, day="2026-05-31")]
    assert events(inside) and not events(outside)


def test_value_floor_is_inclusive():
    half = S.DEFAULT.insider_min_value / 2
    at = [txn(owner=1, shares=half / 50), txn(owner=2, shares=half / 50)]
    under = [txn(owner=1, shares=half / 50), txn(owner=2, shares=half / 50 - 1)]
    assert events(at) and not events(under)


def test_an_issuer_without_a_listed_symbol_or_a_filing_outside_the_panel():
    assert S.insider_events([txn(), txn(owner=2)], {}, SESSIONS) == []
    late = [txn(owner=1, day="2026-12-31"), txn(owner=2, day="2026-12-31")]
    assert events(late) == []
    # Before the panel: not piled onto its first session (found live: GBDC had
    # 101 "events" on 2023-01-02 from 2020-22 filings).
    early = [txn(owner=1, day="2025-06-02"), txn(owner=2, day="2025-06-03")]
    assert events(early) == []


# ── the I and 2+ groups ───────────────────────────────────────────────────

N = 400
DAYS = pd.bdate_range("2025-01-01", periods=N)


def _panel():
    close = pd.DataFrame(50.0, index=DAYS, columns=["ACME"])
    high = close.copy() + 10  # never near its high: P stays off unless forced
    return Panel(close, high, pd.DataFrame(1e6, index=DAYS, columns=["ACME"]))


def _elig(p):
    return pd.DataFrame(True, index=p.close.index, columns=p.close.columns)


def _ie(row):
    return S.IEvent("ACME", row, 2, 250_000.0, DAYS[row - 1].date())


def test_i_is_recorded_and_cooled():
    p = _panel()
    recs = S.records(p, _elig(p), [], 260, N - 1, ievents=[_ie(300), _ie(310), _ie(370)])
    assert [r["row"] for r in recs if r["grp"] == "I"] == [300, 370]
    assert next(r for r in recs if r["grp"] == "I")["detail"]["buyers"] == 2


def test_i_with_f_is_a_two_family_candidate():
    p = _panel()
    f = S.FEvent("ACME", 310, "2026Q1", 1e8, 0.3, 0.1)
    recs = S.records(p, _elig(p), [f], 260, N - 1, ievents=[_ie(300)])
    multi = [r for r in recs if r["grp"] == "2+"]
    assert [(r["row"], r["detail"]["families"]) for r in multi] == [(310, ["F", "I"])]


def test_one_family_alone_is_never_a_candidate():
    p = _panel()
    recs = S.records(p, _elig(p), [], 260, N - 1, ievents=[_ie(300)])
    assert not [r for r in recs if r["grp"] == "2+"]


def test_i_far_from_f_does_not_converge():
    p = _panel()
    f = S.FEvent("ACME", 300 + S.CONVERGE_SESSIONS + 1, "2026Q1", 1e8, 0.3, 0.1)
    recs = S.records(p, _elig(p), [f], 260, N - 1, ievents=[_ie(300)])
    assert not [r for r in recs if r["grp"] == "2+"]


def test_ineligible_names_record_no_i():
    p = _panel()
    e = _elig(p)
    e.iloc[:, :] = False
    assert S.records(p, e, [], 260, N - 1, ievents=[_ie(300)]) == []


def test_live_equals_replay_with_insiders():
    rng = np.random.default_rng(5)
    symbols = [f"S{i:02d}" for i in range(20)]
    close = pd.DataFrame(50 * np.cumprod(1 + rng.normal(0, 0.02, (N, 20)), axis=0),
                         index=DAYS, columns=symbols)  # fmt: skip
    p = Panel(close, close * 1.01, pd.DataFrame(1e6, index=DAYS, columns=symbols))
    e = _elig(p)
    ie = [S.IEvent("S03", 320, 3, 5e5, DAYS[319].date())]
    fe = [S.FEvent("S03", 330, "2026Q1", 1e8, 0.3, 0.1)]
    full = S.records(p, e, fe, 260, N - 1, ievents=ie)
    for row in range(300, 360, 3):
        one = S.records(p, e, fe, row, row, ievents=ie)
        assert sorted((r["grp"], r["symbol"]) for r in one) == sorted(
            (r["grp"], r["symbol"]) for r in full if r["row"] == row
        )
