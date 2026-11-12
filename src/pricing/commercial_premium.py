"""
commercial_premium.py
=======================
Phase 11: turns the Phase 7 technical premium (expected claims cost) into a
**commercial premium** (what would be charged), loaded for expenses. This is
the premium the Phase 11 scenario engine reprices.

WHERE THE LOADING COMES FROM
------------------------------
The technical premium reconciles with paid claims on the training and test
sets (Phase 7 and Phase 8), so it is used as the expected claims cost, an
expected Loss Ratio of 100%. With the documented 27% expense ratio, the
loading that makes expected claims plus expenses equal the premium is:

    COR = LR / (1 + loading) + expense_ratio = 1
    =>  loading = LR / (1 - expense_ratio) - 1

With LR = 100% and expenses at 27%: loading = 37.0%. The realized Loss Ratio
of the book is not used, because one year's large losses would then be priced
into every policy (Phase 8 Section 3). The Phase 11 scenario engine then asks
what an additional, segment-specific adjustment does to profitability.
"""

from __future__ import annotations

import pandas as pd


def calibrate_breakeven_loading(loss_ratio: float, expense_ratio: float) -> float:
    """
    The commercial loading `L` such that scaling every policy's premium by
    `(1 + L)` brings the portfolio's Combined Operating Ratio to exactly
    100%, given the current (technical-premium-based) Loss Ratio and the
    assumed expense ratio. See module docstring for the one-line derivation.
    """
    return loss_ratio / (1 - expense_ratio) - 1


def add_commercial_premium(df: pd.DataFrame, loading: float) -> pd.DataFrame:
    """
    Adds a `commercial_premium` column: `technical_premium * (1 + loading)`.
    Does not mutate `df`. `loading` is required (not defaulted) so every
    caller states explicitly which loading it's using: this project's
    running discipline of never leaving an assumption implicit.
    """
    out = df.copy()
    out["commercial_premium"] = out["technical_premium"] * (1 + loading)
    return out
