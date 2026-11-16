"""
1_Pricing_Simulator.py
======================
Phase 12: type a client's profile, get the technical and commercial
premium, and see how it is calculated, term by term.

    technical premium  = frequency x average cost + EUR 33.54
    commercial premium = technical premium x 1.37
"""

import sys  # access to Python's module search path
from pathlib import Path  # file paths

import numpy as np  # maths
import streamlit as st  # the app framework

_APP_DIR = Path(__file__).resolve().parents[1]  # app/ (for utils)
_PROJECT_ROOT = _APP_DIR.parent  # project root (for src)
for _p in (_APP_DIR, _PROJECT_ROOT):  # both are needed even when this page is opened first
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.utils.hand_calc import glm_terms_table  # term-by-term table: x, its origin, coefficient, product
from utils import (AREA_TO_DENSITY, format_eur, load_category_options, load_model, make_policy_table,
                   score_one_policy, slider_with_input)

st.set_page_config(page_title="Pricing Simulator", layout="wide")

st.title("Pricing Simulator")
st.markdown(
    "Enter a client's profile to get a premium, priced by the same frequency × severity model "
    "used across this project."
)

categories = load_category_options()  # brands, fuels, regions known by the model

col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.subheader("Client and vehicle profile")
    st.caption("Drag a slider for a quick pick, or type an exact value in the box next to it.")

    driv_age = slider_with_input("Driver age", min_value=18, max_value=90, value=40, step=1, key="driv_age")
    bonus_malus = slider_with_input(
        "Bonus-Malus coefficient", min_value=50, max_value=200, value=100, step=5, key="bonus_malus",
        help="French no-claims coefficient: 50 = maximum bonus (13 claim-free years), "
             "100 = starting value, above 100 = malus (recent claims).",
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
        density = AREA_TO_DENSITY[area]  # typical density of the chosen area type
        inputs = dict(drivage=driv_age, bonus_malus=bonus_malus, vehpower=veh_power, vehage=veh_age,
                      density=density, vehbrand=veh_brand, vehgas=veh_gas, region=region)
        result = score_one_policy(**inputs)  # technical and commercial premium

        loading_pct = (result["commercial_premium"] / result["technical_premium"] - 1) * 100  # 37%
        st.metric("Commercial premium (per year)", format_eur(result["commercial_premium"]))
        st.caption(f"= technical premium ({format_eur(result['technical_premium'])}) + {loading_pct:.1f}% for expenses")

        st.divider()
        st.markdown("**Breakdown**")
        b1, b2, b3 = st.columns(3)
        b1.metric("Expected claim frequency", f"{result['predicted_frequency']:.3f} / year")
        b2.metric("Expected cost per claim", format_eur(result["predicted_severity"]))
        b3.metric("Large-loss loading", format_eur(result["large_loss_loading"]))

        with st.expander("How this premium is calculated, term by term"):
            model = load_model()  # the small pricing models
            policy = make_policy_table(**inputs)  # the profile typed above, as one row of the dataset
            raw = policy.iloc[0]

            freq_terms = glm_terms_table(model.freq_results, model.freq_encoder.transform(policy).iloc[0], raw)  # frequency terms
            st.markdown("**1. Frequency** (refined Poisson GLM, coefficients b from notebook 07)")
            st.dataframe(freq_terms, use_container_width=True, hide_index=True)
            eta_f = freq_terms["b x x"].sum()
            st.markdown(f"eta = sum of the 'b x x' column = {eta_f:.4f} -> frequency = exp({eta_f:.4f}) = **{np.exp(eta_f):.4f}** claims per year")

            sev_X = model.sev_encoder.transform(policy)  # Phase 3 columns
            sev_terms = glm_terms_table(model.sev_results, sev_X.iloc[0], raw)  # average-cost terms
            st.markdown("**2. Average cost per claim** (Gamma GLM, coefficients c from notebook 06)")
            st.dataframe(sev_terms, use_container_width=True, hide_index=True)
            eta_s = sev_terms["b x x"].sum()
            st.markdown(f"eta = {eta_s:.4f} -> average cost = exp({eta_s:.4f}) = **EUR {np.exp(eta_s):,.2f}**")

            pure = np.exp(eta_f) * np.exp(eta_s)  # pure premium
            st.markdown(
                f"**3. Technical premium** = {np.exp(eta_f):.4f} × {np.exp(eta_s):,.2f} + {result['large_loss_loading']:.2f} "
                f"(large-loss loading, notebook 06) = **EUR {pure + result['large_loss_loading']:,.2f}**\n\n"
                f"**4. Commercial premium** = {result['technical_premium']:,.2f} × {1 + loading_pct / 100:.4f} "
                f"(27% expenses, notebook 11) = **EUR {result['commercial_premium']:,.2f}**"
            )
            st.caption(f"Density used: {density:,} inhabitants per km² (typical value of '{area}'). "
                       "Reference levels and bands (e.g. driver age 18-24) do not appear: their multiplier is 1.")

        st.caption(
            "The commercial premium adds the same 37% to every policy (expenses): two clients with "
            "the same technical premium get the same commercial premium."
        )
    else:
        st.info("Fill in the profile on the left and click **Get quote**.")
