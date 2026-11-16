"""
2_Retention_Conversion.py
=========================
Phase 12: how a price change moves (a) an existing customer's probability of
renewing, (b) a prospect's probability of buying. Both come from the
simulators of notebooks 09 and 10 (stated assumptions, not fitted on data):

    logit(P renew)   = logit(88%) - 4.5 x price change x (1 - loyalty)
    logit(P convert) = logit(15%) - 6.0 x price vs market x (1 - risk access)
"""

import sys  # access to Python's module search path
from pathlib import Path  # file paths

import numpy as np  # maths
import pandas as pd  # tables
import streamlit as st  # the app framework

_APP_DIR = Path(__file__).resolve().parents[1]  # app/ (for utils)
_PROJECT_ROOT = _APP_DIR.parent  # project root (for src)
for _p in (_APP_DIR, _PROJECT_ROOT):  # both are needed even when this page is opened first
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.prospect.conversion_model import (BASELINE_CONVERSION_RATE, CONVERSION_PRICE_ELASTICITY,
                                           conversion_probability, risk_access_dampening)
from src.retention.renewal_model import (BASELINE_RENEWAL_RATE, PRICE_ELASTICITY, loyalty_dampening,
                                         renewal_probability)
from utils import format_pct, slider_with_input

st.set_page_config(page_title="Retention & Conversion", layout="wide")

st.title("Retention & Conversion")
st.info(
    "**These are simulated scenarios, not fitted predictions.** The dataset has no renewal or "
    "quote outcome to learn from (a one-year photograph, see `docs/data_dictionary.md`). The "
    "formulas and their assumptions are in `src/retention/renewal_model.py` and "
    "`src/prospect/conversion_model.py` (notebooks 09 and 10).",
)

tab_retain, tab_convert = st.tabs(["Existing customer: renewal", "New prospect: conversion"])

with tab_retain:  # ---------- renewal ----------
    st.subheader("Will this customer renew?")
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        tenure = slider_with_input(
            "Customer tenure (years)", min_value=0.0, max_value=15.0, value=3.0, step=0.5, key="tenure",
            help="Claim-free years, computed from the bonus-malus (notebook 09). Long-standing customers "
                 "react less to price.",
        )
        price_change = slider_with_input("Price change at renewal (%)", min_value=-20, max_value=50, value=10,
                                         step=1, key="retain_price", is_percent=True)

        p_renew = float(renewal_probability(price_change, pd.Series([tenure])).iloc[0])  # after the change
        p_base = float(renewal_probability(0.0, pd.Series([tenure])).iloc[0])  # without change (88%)
        st.metric("Renewal probability", format_pct(p_renew),
                  delta=format_pct(p_renew - p_base, decimals=1) + " vs. no price change")

        with st.expander("How it is calculated"):
            loyalty = float(loyalty_dampening(pd.Series([tenure])).iloc[0])  # 0.6 x min(tenure / 10, 1)
            logit_base = np.log(BASELINE_RENEWAL_RATE / (1 - BASELINE_RENEWAL_RATE))  # logit(88%)
            logit_new = logit_base + PRICE_ELASTICITY * price_change * (1 - loyalty)
            st.markdown(
                f"- loyalty = 0.6 × min({tenure:g} / 10, 1) = **{loyalty:.2f}** (assumption, notebook 09)\n"
                f"- logit(88%) = ln(0.88 / 0.12) = **{logit_base:.4f}** (88% = assumed renewal rate without change)\n"
                f"- logit = {logit_base:.4f} + ({PRICE_ELASTICITY}) × {price_change:+.2f} × (1 - {loyalty:.2f}) = **{logit_new:.4f}**\n"
                f"- P(renew) = 1 / (1 + exp(-{logit_new:.4f})) = **{1 / (1 + np.exp(-logit_new)):.2%}**"
            )

    with col_out:
        st.markdown("**Renewal probability across price changes, at this tenure**")
        grid = np.linspace(-0.20, 0.50, 60)  # from -20% to +50%
        curve = renewal_probability(grid, pd.Series([tenure] * len(grid)))
        st.line_chart(pd.DataFrame({"Price change": grid, "Renewal probability": curve}).set_index("Price change"))
        st.caption("The longer the tenure, the flatter the curve: loyal customers react less to price.")

with tab_convert:  # ---------- conversion ----------
    st.subheader("Will this prospect accept the quote?")
    col_in2, col_out2 = st.columns([1, 1], gap="large")

    with col_in2:
        bonus_malus = slider_with_input(
            "Prospect's Bonus-Malus coefficient", min_value=50, max_value=200, value=100, step=5, key="convert_bm",
            help="Above 100, fewer insurers compete for the prospect, who is then less sensitive to price (notebook 10).",
        )
        price_vs_market = slider_with_input("Quoted price vs. market (%)", min_value=-20, max_value=50, value=10,
                                            step=1, key="convert_price", is_percent=True)

        p_convert = float(conversion_probability(price_vs_market, pd.Series([bonus_malus])).iloc[0])  # at this price
        p_base2 = float(conversion_probability(0.0, pd.Series([bonus_malus])).iloc[0])  # at the market price (15%)
        st.metric("Conversion probability", format_pct(p_convert),
                  delta=format_pct(p_convert - p_base2, decimals=1) + " vs. quoting at the market price")

        with st.expander("How it is calculated"):
            access = float(risk_access_dampening(pd.Series([bonus_malus])).iloc[0])  # 0 up to BM 100, 0.5 from BM 200
            logit_base = np.log(BASELINE_CONVERSION_RATE / (1 - BASELINE_CONVERSION_RATE))  # logit(15%)
            logit_new = logit_base + CONVERSION_PRICE_ELASTICITY * price_vs_market * (1 - access)
            st.markdown(
                f"- risk access = 0.5 × min(max({bonus_malus} - 100, 0) / 100, 1) = **{access:.2f}** (assumption, notebook 10)\n"
                f"- logit(15%) = ln(0.15 / 0.85) = **{logit_base:.4f}** (15% = assumed conversion at the market price)\n"
                f"- logit = {logit_base:.4f} + ({CONVERSION_PRICE_ELASTICITY}) × {price_vs_market:+.2f} × (1 - {access:.2f}) = **{logit_new:.4f}**\n"
                f"- P(convert) = 1 / (1 + exp(-{logit_new:.4f})) = **{1 / (1 + np.exp(-logit_new)):.2%}**"
            )

    with col_out2:
        st.markdown("**Conversion probability across prices, at this risk level**")
        grid2 = np.linspace(-0.20, 0.50, 60)
        curve2 = conversion_probability(grid2, pd.Series([bonus_malus] * len(grid2)))
        st.line_chart(pd.DataFrame({"Price vs. market": grid2, "Conversion probability": curve2}).set_index("Price vs. market"))
        st.caption("Prospects react much more to price than existing customers (no switching cost).")
