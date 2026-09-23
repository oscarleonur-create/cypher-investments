"""Peer choice by correlation — the part of the peer lookup with no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
from advisor.scanner.sources import choose_peers


def _returns(n=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.02, n)
    return pd.DataFrame(
        {
            "EXPE": base,
            "BKNG": base + rng.normal(0, 0.005, n),  # close twin
            "ABNB": base + rng.normal(0, 0.015, n),  # looser
            "CCL": rng.normal(0, 0.02, n),  # unrelated
            "NEG": -base,  # inverse
        }
    )


def test_most_correlated_first_and_unrelated_excluded():
    assert choose_peers(_returns(), "EXPE") == ["BKNG", "ABNB"]


def test_max_peers():
    assert choose_peers(_returns(), "EXPE", max_peers=1) == ["BKNG"]


def test_symbol_missing():
    assert choose_peers(_returns(), "ZZZZ") == []


def test_too_little_history_gives_no_peers():
    assert choose_peers(_returns(n=10), "EXPE") == []


def test_peer_with_gaps_uses_overlapping_rows_only():
    df = _returns()
    df.loc[:45, "BKNG"] = np.nan  # only 14 overlapping sessions: below min_obs
    assert choose_peers(df, "EXPE") == ["ABNB"]
