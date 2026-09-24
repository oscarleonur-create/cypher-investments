"""Universe fetchers with disk caching."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import pandas as pd
import requests

from advisor.data.cache import DiskCache

logger = logging.getLogger(__name__)

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass(frozen=True)
class StockInfo:
    symbol: str
    name: str
    sector: str
    sub_industry: str


# ── Built-in universes ────────────────────────────────────────────────

SEMICONDUCTORS = [
    # Mega/Large cap
    ("NVDA", "NVIDIA Corporation"),
    ("TSM", "Taiwan Semiconductor"),
    ("AVGO", "Broadcom Inc"),
    ("AMD", "Advanced Micro Devices"),
    ("QCOM", "Qualcomm Inc"),
    ("TXN", "Texas Instruments"),
    ("INTC", "Intel Corporation"),
    ("MU", "Micron Technology"),
    ("AMAT", "Applied Materials"),
    ("LRCX", "Lam Research"),
    ("KLAC", "KLA Corporation"),
    ("ADI", "Analog Devices"),
    ("MRVL", "Marvell Technology"),
    ("NXPI", "NXP Semiconductors"),
    ("ON", "ON Semiconductor"),
    ("MCHP", "Microchip Technology"),
    ("MPWR", "Monolithic Power Systems"),
    ("SWKS", "Skyworks Solutions"),
    ("QRVO", "Qorvo Inc"),
    ("TER", "Teradyne Inc"),
    ("ENTG", "Entegris Inc"),
    ("WOLF", "Wolfspeed Inc"),
    # Mid/Small cap & equipment
    ("SMCI", "Super Micro Computer"),
    ("ARM", "Arm Holdings"),
    ("ASML", "ASML Holding"),
    ("GFS", "GlobalFoundries"),
    ("CRUS", "Cirrus Logic"),
    ("SYNA", "Synaptics Inc"),
    ("DIOD", "Diodes Inc"),
    ("MKSI", "MKS Instruments"),
    ("ACLS", "Axcelis Technologies"),
    ("FORM", "FormFactor Inc"),
    ("RMBS", "Rambus Inc"),
    ("SITM", "SiTime Corporation"),
    ("COHR", "Coherent Corp"),
    ("MTSI", "MACOM Technology"),
    ("POWI", "Power Integrations"),
    ("ALGM", "Allegro MicroSystems"),
    ("LSCC", "Lattice Semiconductor"),
    ("AOSL", "Alpha and Omega Semi"),
    ("UMC", "United Microelectronics"),
    # Leveraged ETFs (for options)
    ("SOXL", "Direxion Semis Bull 3X"),
    ("SOXS", "Direxion Semis Bear 3X"),
    ("SOXX", "iShares Semi ETF"),
    ("SMH", "VanEck Semi ETF"),
]


def fetch_universe(name: str, cache: DiskCache | None = None) -> list[StockInfo]:
    """Fetch a named universe. Supported: 'sp500', 'semiconductors'."""
    if name == "sp500":
        return fetch_sp500(cache=cache)
    elif name == "semiconductors":
        return fetch_semiconductors()
    else:
        raise ValueError(f"Unknown universe: {name}. Use 'sp500' or 'semiconductors'.")


def fetch_semiconductors() -> list[StockInfo]:
    """Return the built-in semiconductor universe."""
    return [
        StockInfo(
            symbol=sym,
            name=name,
            sector="Information Technology",
            sub_industry="Semiconductors",
        )
        for sym, name in SEMICONDUCTORS
    ]


def fetch_sp500(cache: DiskCache | None = None) -> list[StockInfo]:
    """Fetch the current S&P 500 constituents from Wikipedia.

    Results are cached as JSON for 24 hours via DiskCache.
    Symbols with dots are converted to dashes for yfinance compatibility
    (e.g. BRK.B -> BRK-B).
    """
    if cache is not None:
        cached = cache.get_json("universe", "sp500")
        if cached is not None:
            logger.debug("Using cached S&P 500 universe (%d tickers)", len(cached))
            return [StockInfo(**item) for item in cached]

    logger.info("Fetching S&P 500 list from Wikipedia...")
    resp = requests.get(_WIKI_URL, headers={"User-Agent": "advisor/1.0"}, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]

    stocks: list[StockInfo] = []
    for _, row in df.iterrows():
        raw_symbol = str(row["Symbol"]).strip()
        symbol = raw_symbol.replace(".", "-")
        stocks.append(
            StockInfo(
                symbol=symbol,
                name=str(row.get("Security", "")),
                sector=str(row.get("GICS Sector", "")),
                sub_industry=str(row.get("GICS Sub-Industry", "")),
            )
        )

    logger.info("Fetched %d S&P 500 constituents", len(stocks))

    if cache is not None:
        cache.set_json(
            [
                {
                    "symbol": s.symbol,
                    "name": s.name,
                    "sector": s.sector,
                    "sub_industry": s.sub_industry,
                }
                for s in stocks
            ],
            "universe",
            "sp500",
        )

    return stocks
