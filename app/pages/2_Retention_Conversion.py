"""
2_Retention_Conversion.py
===============================
Phase 12: test how a price change moves (a) an existing customer's renewal
probability, or (b) a new prospect's quote-to-bind (conversion) probability.
Both use the Phase 9 / Phase 10 literature-calibrated simulators (see their
module docstrings in `src/retention/renewal_model.py` and
`src/prospect/conversion_model.py`): clearly labeled here as *simulated*
scenario tools, not fitted predictions, exactly as this project's disclosure
discipline requires.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Two directories need to be on sys.path: app/ (for `utils`, a sibling module) and the repo root (for
# `src`, imported directly below): relying on just one, or on another page having already run first
# in this process and added the other as a side effect, is what caused a real bug here: this page threw
# ModuleNotFoundError when it was the *first* page loaded in a fresh process (e.g. a user landing on it
# directly via URL, bypassing the home page's own sys.path setup).
_APP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _APP_DIR.parent
for _p in (_APP_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.prospect.conversion_model import BASELINE_CONVERSION_RATE, conversion_probability
from src.retention.renewal_model import BASELINE_RENEWAL_RATE, renewal_probability
from utils import format_pct, slider_with_input

st.set_page_config(page_title="Retention & Conversion", layout="wide")

st.title("Retention & Conversion")
st.info(
    "**These are simulated scenarios, not fitted predictions.** This dataset "
    "has no historical renewal or quote-to-bind outcomes to learn from (a single-year "
    "cross-section: see `docs/data_dictionary.md`). The curves below come from "
    "literature-calibrated formulas (Phase 9 / Phase 10), documented in full in "
    "`src/retention/renewal_model.py` and `src/prospect/conversion_model.py`.",
)

tab_retain, tab_convert = st.tabs(["Existing customer: renewal", "New prospect: conversion"])

# --------------------------------------------------------------------------- #
# Tab 1: Retention
# --------------------------------------------------------------------------- #
with tab_retain:
    st.subheader("Will this customer renew?")
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        tenure = slider_with_input(
            "Customer tenure (years)", min_value=0.0, max_value=15.0, value=3.0, step=0.5, key="tenure",
            help="Derived from the French bonus-malus formula (Phase 9): long-tenured "
                 "customers are more loyal and less price-sensitive (the 'loyalty dampening' effect).",
        )
        price_change = slider_with_input(
            "Price change at renewal (%)", min_value=-20, max_value=50, value=10, step=1,
            key="retain_price", is_percent=True,
        )

        p_renew = float(renewal_probability(price_change, pd.Series([tenure])).iloc[0])
        p_base = float(renewal_probability(0.0, pd.Series([tenure])).iloc[0])

        st.metric(
            "Renewal probability",
            format_pct(p_renew),
            delta=format_pct(p_renew - p_base, decimals=1) + " vs. no price change",
        )
        st.caption(f"Portfolio baseline renewal rate (no price change, average tenure): {format_pct(BASELINE_RENEWAL_RATE)}")

    with col_out:
        st.markdown("**Renewal probability across a range of price changes, at this tenure**")
        grid = np.linspace(-0.20, 0.50, 60)
        curve = renewal_probability(grid, pd.Series([tenure] * len(grid)))
        chart_df = pd.DataFrame({"Price change": grid, "Renewal probability": curve}).set_index("Price change")
        st.line_chart(chart_df)
        st.caption(
            "Longer-tenured customers are dampened toward the baseline rate (flatter curve); "
            "a brand-new customer's renewal probability reacts to price almost at full elasticity."
        )

# --------------------------------------------------------------------------- #
# Tab 2: Conversion
# --------------------------------------------------------------------------- #
with tab_convert:
    st.subheader("Will this prospect accept the quote?")
    col_in2, col_out2 = st.columns([1, 1], gap="large")

    with col_in2:
        bonus_malus = slider_with_input(
            "Prospect's Bonus-Malus coefficient", min_value=50, max_value=200, value=100, step=5,
            key="convert_bm",
            help="A claims-history signal the prospect carries even as a new customer. High "
                 "Bonus-Malus (higher risk) prospects have structurally fewer competing insurers "
                 "(Phase 10's 'risk access' effect), making them less price-sensitive at the margin.",
        )
        price_vs_market = slider_with_input(
            "Quoted price vs. assumed market parity (%)", min_value=-20, max_value=50, value=10, step=1,
            key="convert_price", is_percent=True,
        )

        p_convert = float(conversion_probability(price_vs_market, pd.Series([bonus_malus])).iloc[0])
        p_base2 = float(conversion_probability(0.0, pd.Series([bonus_malus])).iloc[0])

        st.metric(
            "Conversion probability",
            format_pct(p_convert),
            delta=format_pct(p_convert - p_base2, decimals=1) + " vs. quoting at market parity",
        )
        st.caption(f"Portfolio baseline conversion rate (quoted at market parity, neutral risk): {format_pct(BASELINE_CONVERSION_RATE)}")

    with col_out2:
        st.markdown("**Conversion probability across a range of prices, at this risk profile**")
        grid2 = np.linspace(-0.20, 0.50, 60)
        curve2 = conversion_probability(grid2, pd.Series([bonus_malus] * len(grid2)))
        chart_df2 = pd.DataFrame({"Price vs. market": grid2, "Conversion probability": curve2}).set_index("Price vs. market")
        st.line_chart(chart_df2)
        st.caption(
            "Conversion is more price-elastic than renewal (new business has no switching cost): "
            "this curve is steeper than the retention curve for a comparable price move."
        )
