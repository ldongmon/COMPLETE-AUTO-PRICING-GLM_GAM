"""
3_Portfolio_Scenario.py
=============================
Phase 12: the dashboard's headline tool: pick a portfolio segment and a
price change, and project Loss Ratio, Combined Operating Ratio (COR),
Average Technical Premium (ATP), policy count, written premium and
profitability contribution over the next 1, 2 or 3 years, with one click.
Wraps Phase 11's `run_repricing_scenario` (Year 1) and Phase 12's
`project_multi_year_scenario` (Year 2+): see that module's docstring for
exactly what is exact vs. simplified in the multi-year projection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Two directories need to be on sys.path: app/ (for `utils`, a sibling module) and the repo root (for
# `scoring`/`src`, imported directly below): relying on just one, or on another page having already
# run first in this process and added the other as a side effect, is what caused a real bug here: this
# page threw ModuleNotFoundError when it was the *first* page loaded in a fresh process (e.g. a user
# landing on it directly via URL, bypassing the home page's own sys.path setup).
_APP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _APP_DIR.parent
for _p in (_APP_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scoring.scenario_engine import DEFAULT_EXPENSE_RATIO, project_multi_year_scenario, run_repricing_scenario
from src.prospect.conversion_model import BASELINE_CONVERSION_RATE
from src.retention.renewal_model import BASELINE_RENEWAL_RATE
from src.pricing.portfolio_kpis import experience_cost_factor
from src.pricing.segment_x import load_segment_x
from utils import format_eur, format_pct, get_large_loss_threshold, load_scored_portfolio, slider_with_input

st.set_page_config(page_title="Portfolio Scenario", layout="wide")

st.title("Portfolio Scenario Analysis")
st.markdown(
    "Pick a segment of the book and a price change, and project what happens to "
    "the portfolio's key indicators over the next 1, 2 or 3 years."
)

df, _ = load_scored_portfolio()

# --------------------------------------------------------------------------- #
# Segment selector
# --------------------------------------------------------------------------- #
st.subheader("1. Choose a segment")

segment_type = st.radio(
    "Segment by",
    ["Whole portfolio", "Bonus-Malus band", "Fuel type (VehGas)", "Region", "Vehicle brand (VehBrand)"],
    horizontal=True,
)

BONUS_MALUS_BANDS = {"50": (50, 50), "51-60": (51, 60), "61-80": (61, 80), "51-80 (Segment X)": (51, 80),
                     "81-100": (81, 100), ">100": (101, 1000)}

if segment_type == "Whole portfolio":
    segment_df = df
    segment_label = "Whole portfolio"
elif segment_type == "Bonus-Malus band":
    band = st.selectbox("Select a BonusMalus band", list(BONUS_MALUS_BANDS), index=3)
    low, high = BONUS_MALUS_BANDS[band]
    segment_df = df[df["BonusMalus"].between(low, high)]
    segment_label = f"BonusMalus {band}"
else:
    col_map = {"Fuel type (VehGas)": "VehGas", "Region": "Region", "Vehicle brand (VehBrand)": "VehBrand"}
    col = col_map[segment_type]
    options = sorted(df[col].unique().tolist())
    chosen = st.selectbox(f"Select {col}", options)
    segment_df = df[df[col] == chosen]
    segment_label = f"{col} = {chosen}"

st.caption(
    f"Segment **{segment_label}**: {len(segment_df):,} policies "
    f"({len(segment_df) / len(df):.1%} of the book by policy count)."
)

# Expected cost of the segment relative to the book: attritional claims from the segment's own
# experience, large losses at the flat loading (src/pricing/portfolio_kpis.py). The book itself is at
# 1.0, the basis the commercial loading assumes (technical premium = expected cost).
threshold = get_large_loss_threshold()
cost_factor = experience_cost_factor(segment_df, threshold) / experience_cost_factor(df, threshold)
st.caption(
    f"Expected cost relative to the book: **{cost_factor:.2f}** "
    "(attritional claims from the segment's own experience, large losses at the flat loading)."
)

# --------------------------------------------------------------------------- #
# Scenario controls
# --------------------------------------------------------------------------- #
st.subheader("2. Set the scenario")

col_a, col_b = st.columns(2)
with col_a:
    price_change_pct = slider_with_input(
        "Price change on this segment (%)", min_value=-20, max_value=50,
        value=round(100 * load_segment_x(_PROJECT_ROOT)["indicated_price_change"]), step=1,
        key="scenario_price", is_percent=True,
    )
with col_b:
    horizon = st.select_slider("Projection horizon", options=[1, 2, 3], value=3, format_func=lambda y: f"{y} year{'s' if y > 1 else ''}")

run = st.button("Run scenario", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if run:
    with st.spinner("Projecting..."):
        projection = project_multi_year_scenario(
            segment_df, price_change_pct=price_change_pct, n_years=horizon,
            expense_ratio=DEFAULT_EXPENSE_RATIO, cost_factor=cost_factor,
        )
        baseline = project_multi_year_scenario(
            segment_df, price_change_pct=0.0, n_years=horizon,
            expense_ratio=DEFAULT_EXPENSE_RATIO, cost_factor=cost_factor,
        )
        # Year-1 detail (renewal/conversion rates): project_multi_year_scenario's own output doesn't
        # carry these, since they're only meaningful as *rates* in Year 1 (Year 2+ uses the flat baseline
        # rate for every policy by construction: see that function's docstring).
        year1_detail = run_repricing_scenario(
            segment_df, price_change_pct=price_change_pct, expense_ratio=DEFAULT_EXPENSE_RATIO, cost_factor=cost_factor
        )

    st.subheader("3. Results")

    final_year = projection.iloc[-1]
    final_baseline = baseline.iloc[-1]
    contribution_margin = final_year["contribution"] / final_year["written_premium"]
    contribution_margin_baseline = final_baseline["contribution"] / final_baseline["written_premium"]
    is_profitable = final_year["cor"] < 1.0

    st.markdown(f"**Year {horizon} outlook vs. no price change:**")
    if abs(final_year["cor"] - 1.0) < 0.005:
        st.info(f"**About breakeven** at this price change: projected COR {format_pct(final_year['cor'])}.")
    elif is_profitable:
        st.success(f"**Profitable** at this price change: projected COR {format_pct(final_year['cor'])} (below 100%).")
    else:
        st.warning(f"**Loss-making** at this price change: projected COR {format_pct(final_year['cor'])} (above 100%).")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Loss Ratio (LR)", format_pct(final_year["loss_ratio"]), delta=format_pct(final_year["loss_ratio"] - final_baseline["loss_ratio"], 1), delta_color="inverse")
    m2.metric("Combined Operating Ratio (COR)", format_pct(final_year["cor"]), delta=format_pct(final_year["cor"] - final_baseline["cor"], 1), delta_color="inverse")
    m3.metric("Avg. Technical Premium (ATP)", format_eur(final_year["atp"]), delta=format_eur(final_year["atp"] - final_baseline["atp"]))
    m4.metric("Policies in force", f"{final_year['policies']:,.0f}", delta=f"{final_year['policies'] - final_baseline['policies']:,.0f}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Written premium", format_eur(final_year["written_premium"]), delta=format_eur(final_year["written_premium"] - final_baseline["written_premium"]))
    m6.metric("Profitability contribution", format_eur(final_year["contribution"]), delta=format_eur(final_year["contribution"] - final_baseline["contribution"]))
    m7.metric("Contribution margin", format_pct(contribution_margin), delta=format_pct(contribution_margin - contribution_margin_baseline, 1))
    m8.metric("Year 1 renewal rate", format_pct(year1_detail["renewal_rate"]))

    st.caption(
        f"Year 1 conversion rate at this price change: **{format_pct(year1_detail['conversion_rate'])}** "
        f"(portfolio baseline, at 0% price change: retention {format_pct(BASELINE_RENEWAL_RATE)}, "
        f"conversion {format_pct(BASELINE_CONVERSION_RATE)}: see the Retention & Conversion page for "
        f"the full curves)."
    )

    st.divider()
    st.markdown("**Year-by-year detail**")

    display_df = projection.copy()
    display_df["loss_ratio"] = display_df["loss_ratio"].map(format_pct)
    display_df["cor"] = display_df["cor"].map(format_pct)
    display_df["atp"] = display_df["atp"].map(format_eur)
    display_df["policies"] = display_df["policies"].map(lambda v: f"{v:,.0f}")
    display_df["written_premium"] = display_df["written_premium"].map(format_eur)
    display_df["incurred_claims"] = display_df["incurred_claims"].map(format_eur)
    display_df["contribution"] = display_df["contribution"].map(format_eur)
    display_df.columns = ["Year", "Policies", "Written Premium", "Incurred Claims", "Loss Ratio", "COR", "ATP", "Contribution"]
    st.dataframe(display_df.set_index("Year"), use_container_width=True)

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("**Loss Ratio & COR over time**")
        chart1 = projection.set_index("year")[["loss_ratio", "cor"]].rename(columns={"loss_ratio": "Loss Ratio", "cor": "COR"})
        st.line_chart(chart1)
    with col_chart2:
        st.markdown("**Profitability contribution over time**")
        chart2 = projection.set_index("year")[["contribution"]].rename(columns={"contribution": "Contribution (€)"})
        st.bar_chart(chart2)

    st.caption(
        "Year 1 uses the full row-level Phase 11 scenario engine (each policy's own tenure/risk "
        "profile drives its renewal or conversion probability). From Year 2 onward, the price change "
        "is assumed to hold flat at its new level, so the book renews/converts at portfolio baseline "
        "rates: an exact aggregate calculation (not a re-approximation), documented in "
        "`scoring/scenario_engine.py`. Treat this as a KPI outlook under stable conditions, not a "
        "long-run market-share forecast."
    )
    st.caption(
        "Expected claims use the segment's cost factor shown above, on the same basis as the "
        "commercial loading: the whole book at a 0% price change shows a contribution of zero."
    )
else:
    st.info("Choose a segment and a price change above, then click **Run scenario**.")
