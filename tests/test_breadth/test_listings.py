"""The symbol directory: parsing both exchange files, cutting to common stock, joining CIKs."""

from __future__ import annotations

import pytest
from advisor.breadth.listings import (
    Listing,
    build_directory,
    exclusion,
    fetch_directory,
    parse_nasdaq,
    parse_other,
    parse_sec_tickers,
)

NASDAQ = """\
Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAOI|Applied Optoelectronics, Inc. - Common Stock|G|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ZXZZT|NASDAQ TEST STOCK|G|Y|N|100|N|N
ABCDW|Some Acquisition Corp - Warrant|S|N|N|100|N|N
SPCQ|Rocket Acquisition Corp - Class A Ordinary Shares|S|N|N|100|N|N
DLNQ|Late Filer Inc - Common Stock|S|N|E|100|N|N
BKRP|Broke Co - Common Stock|S|N|Q|100|N|N
DEFC|Penny Co - Common Stock|S|N|D|100|N|N
NOSEC|Foreign Only Ltd - Ordinary Shares|S|N|N|100|N|N
File Creation Time: 0927202618:01|||||||
"""

OTHER = """\
ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
BRK.B|Berkshire Hathaway Inc. Class B|N|BRK.B|N|100|N|BRK.B
ABR$D|Arbor Realty Trust 6.375% Series D Cumulative Redeemable Preferred Stock|N|ABRpD|N|100|N|ABR$D
XYZ.U|XYZ Holdings Units, each consisting of one share and one warrant|N|XYZ.U|N|100|N|XYZ=
GUT|Gabelli Utility Trust Closed-End Fund|N|GUT|N|100|N|GUT
AAOI|Applied Optoelectronics duplicate row|N|AAOI|N|100|N|AAOI
File Creation Time: 0927202618:01|||||||
"""

SEC = {
    "0": {"cik_str": 1158114, "ticker": "AAOI", "title": "Applied Optoelectronics"},
    "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire"},
    "2": {"cik_str": 1067983, "ticker": "BRK-A", "title": "Berkshire"},
    "3": {"cik_str": 1, "ticker": "DLNQ", "title": "x"},
    "4": {"cik_str": 2, "ticker": "DEFC", "title": "x"},
    "5": {"cik_str": "bad", "ticker": "BAD", "title": "x"},
}


def test_parse_drops_the_trailer_line():
    rows = parse_nasdaq(NASDAQ)
    assert [r.symbol for r in rows][-1] == "NOSEC"
    assert all("File Creation" not in r.symbol for r in rows)


def test_parse_other_maps_exchange_codes():
    rows = {r.symbol: r for r in parse_other(OTHER)}
    assert rows["BRK.B"].exchange == "NYSE"


def test_class_share_spelling_for_yahoo_and_sec():
    assert Listing(symbol="BRK.B", name="", exchange="NYSE").yahoo == "BRK-B"


@pytest.mark.parametrize(
    "symbol,name,etf,test,status,reason",
    [
        ("QQQ", "Invesco QQQ Trust", True, False, "N", "etf"),
        ("ZXZZT", "NASDAQ TEST STOCK", False, True, "N", "test issue"),
        ("ABCDW", "Some Co - Warrant", False, False, "N", "warrant"),
        ("XYZ.U", "XYZ Units, each consisting of", False, False, "", "unit"),
        ("ABCR", "Some Co - Rights", False, False, "N", "right"),
        ("ABR$D", "Arbor 6.375% Series D", False, False, "", "preferred"),
        ("PFX", "Foo Corp 7% Preferred Stock", False, False, "", "preferred"),
        ("NTE", "Foo Corp 6.5% Senior Notes due 2031", False, False, "", "debt"),
        ("GUT", "Gabelli Utility Trust Closed-End Fund", False, False, "", "fund"),
        ("SPCQ", "Rocket Acquisition Corp - Class A", False, False, "N", "probable SPAC"),
        ("DLNQ", "Late Filer Inc", False, False, "E", "delinquent in filings"),
        ("BKRP", "Broke Co", False, False, "Q", "bankrupt"),
        ("BOTH", "Broke and late", False, False, "J", "bankrupt"),
        # Deficient alone (usually a sub-$1 bid) is left to the price floor.
        ("DEFC", "Penny Co - Common Stock", False, False, "D", None),
        # Words that merely contain a marker are not the marker.
        ("UNTY", "Unity Bancorp, Inc. - Common Stock", False, False, "N", None),
        ("UAL", "United Airlines Holdings", False, False, "", None),
        ("BTSG", "BrightSpring Health Services", False, False, "N", None),
        ("BRK.B", "Berkshire Hathaway Inc. Class B", False, False, "", None),
        ("TSM", "Taiwan Semiconductor American Depositary Shares", False, False, "", None),
    ],
)
def test_exclusion(symbol, name, etf, test, status, reason):
    x = Listing(symbol=symbol, name=name, exchange="X", etf=etf, test=test, status=status)
    assert exclusion(x) == reason


def test_sec_tickers_skip_malformed_rows():
    m = parse_sec_tickers(SEC)
    assert m["AAOI"] == 1158114
    assert m["BRK-B"] == 1067983
    assert "BAD" not in m
    assert parse_sec_tickers({}) == {}
    assert parse_sec_tickers(None) == {}


def test_directory_joins_ciks_dedups_and_requires_a_filer():
    d = build_directory(NASDAQ, OTHER, SEC)
    symbols = [x.yahoo for x in d.listings]
    assert symbols.count("AAOI") == 1  # Nasdaq's row kept, the duplicate dropped
    assert {x.yahoo for x in d.common()} == {"AAOI", "BRK-B", "DEFC"}
    assert d.ciks["BRK-B"] == 1067983
    assert d.reasons["NOSEC"] == "no SEC filer"
    # An ETF keeps its structural reason even without a CIK.
    assert d.reasons["QQQ"] == "etf"


def test_fetch_refuses_a_malformed_file():
    with pytest.raises(ValueError):
        fetch_directory(get_text=lambda url: "<html>maintenance</html>", get_json=lambda url: {})


def test_fetch_propagates_an_unreachable_directory():
    def down(url):
        raise TimeoutError("nasdaqtrader.com timed out")

    with pytest.raises(TimeoutError):
        fetch_directory(get_text=down, get_json=lambda url: {})
