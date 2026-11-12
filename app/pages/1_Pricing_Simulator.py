"""
1_Pricing_Simulator.py
============================
Phase 12: enter a hypothetical new client's characteristics, get an
instant technical and commercial premium, with a breakdown showing how much
of the price comes from expected claim frequency vs. expected claim
severity. Built for a non-technical user: every input is a plain-language
field (a dropdown, a slider), not a raw model feature name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# app/ (for `utils`, a sibling module): see the same setup in the other pages for why this is two
# directories there (this page only needs `utils`, which itself adds the repo root for its own imports).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import AREA_TO_DENSITY, format_eur, load_category_options, score_one_policy, slider_with_input

st.set_page_config(page_title="Pricing Simulator", layout="wide")

st.title("Pricing Simulator")
st.markdown(
    "Enter a new client's profile below to get an instant premium quote, "
    "priced by the same frequency × severity model used across this project."
)

categories = load_category_options()

col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.subheader("Client & vehicle profile")
    st.caption("Drag a slider for a quick pick, or type an exact value in the box next to it.")

    driv_age = slider_with_input("Driver age", min_value=18, max_value=90, value=40, step=1, key="driv_age")
    bonus_malus = slider_with_input(
        "Bonus-Malus coefficient", min_value=50, max_value=200, value=100, step=5, key="bonus_malus",
        help="French no-claims coefficient: 50 = maximum bonus (13 claim-free years), "
             "100 = new driver / no history, above 100 = malus (past claims).",
    )
    veh_power = slider_with_input("Vehicle power (fiscal hp)", min_value=4, max_value=15, value=6, step=1, key="veh_power")
    veh_age = slider_with_input("Vehicle age (years)", min_value=0, max_value=30, value=5, step=1, key="veh_age")

    c1, c2 = st.columns(2)
    with c1:
        area = st.selectbox("Area type", list(AREA_TO_DENSITY.keys()), index=2)
        veh_gas = st.selectbox("Fuel type", categories["VehGas"])
    with c2:
        veh_brand = st.selectbox("Vehicle brand code", categories["VehBrand"])
        region = st.selectbox("Region (INSEE code)", categories["Region"])

    submitted = st.button("Get quote", type="primary", use_container_width=True)

with col_result:
    st.subheader("Quote")
    if submitted:
        density = AREA_TO_DENSITY[area]
        result = score_one_policy(
            drivage=driv_age, bonus_malus=bonus_malus, vehpower=veh_power, vehage=veh_age,
            density=density, vehbrand=veh_brand, vehgas=veh_gas, region=region,
        )

        loading_pct = (result["commercial_premium"] / result["technical_premium"] - 1) * 100
        st.metric("Commercial premium (annual)", format_eur(result["commercial_premium"]))
        st.caption(
            f"= technical premium ({format_eur(result['technical_premium'])}) "
            f"+ {loading_pct:.1f}% commercial loading"
        )

        st.divider()
        st.markdown("**Breakdown**")
        b1, b2, b3 = st.columns(3)
        b1.metric("Expected claim frequency", f"{result['predicted_frequency']:.3f} / year")
        b2.metric("Expected claim severity", format_eur(result["predicted_severity"]))
        b3.metric("Large-loss loading", format_eur(result["large_loss_loading"]))

        st.caption(
            "Technical premium = expected frequency × expected severity + a flat large-loss "
            "loading (Phase 7). The commercial premium adds the portfolio's breakeven margin "
            "(Phase 11) on top: it is not a segment-specific price, so two clients with the "
            "same technical premium always get the same commercial premium."
        )
    else:
        st.info("Fill in the profile on the left and click **Get quote**.")
