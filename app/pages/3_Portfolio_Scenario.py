"""
3_Portfolio_Scenario.py
=======================
Phase 12: pick a segment and a price change, and see Loss Ratio, COR, average
premium, number of policies, premium and contribution over 1, 2 or 3 years.

    Year 1: run_repricing_scenario (notebook 11): each customer renews with
            its own P(renew), prospects buy with P(convert).
    Year 2+: no further price change, so 88% renew and 15% of quotes convert.
    contribution = premium - expected claims - 27% expenses
"""

import sys  # access to Python's module search path
from pathlib import Path  # file paths

import pandas as pd  # tables
import streamlit as st  # the app framework

_APP_DIR = Path(__file__).resolve().parents[1]  # app/ (for utils)
_PROJECT_ROOT = _APP_DIR.parent  # project root (for scoring and src)
for _p in (_APP_DIR, _PROJECT_ROOT):  # both are needed even when this page is opened first
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

# Expected cost of the segment relative to the book (notebook 08): its own capped claims plus the flat
# large-loss loading, over its premium, divided by the same ratio for the whole book (= 1.0 for the book).
threshold = get_large_loss_threshold()  # EUR 15,000 per claim
factor_segment = experience_cost_factor(segment_df, threshold)  # segment: (capped claims + 33.54 x exposure) / earned premium
factor_book = experience_cost_factor(df, threshold)  # same ratio for the whole book
cost_factor = factor_segment / factor_book  # cost relativity of the segment
st.caption(f"Expected cost relative to the book: **{cost_factor:.2f}**")
with st.expander("How the cost relativity is calculated"):
    capped = segment_df["sev_claim_amount"].clip(upper=threshold * segment_df["sev_claim_count"].clip(lower=1)).sum()  # capped claims
    mutualised = (segment_df["large_loss_loading"] * segment_df["Exposure"]).sum()  # 33.54 x exposure
    earned = segment_df["earned_premium"].sum()  # technical premium x exposure
    st.markdown(
        f"- capped claims of the segment (each claim up to EUR {threshold:,.0f}, column `sev_claim_amount`): **EUR {capped:,.0f}**\n"
        f"- + large-loss loading × exposure (33.54 × {segment_df['Exposure'].sum():,.0f} years): **EUR {mutualised:,.0f}**\n"
        f"- ÷ earned technical premium (technical premium × `Exposure`): **EUR {earned:,.0f}**\n"
        f"- = segment ratio **{factor_segment:.4f}**; whole book ratio **{factor_book:.4f}**\n"
        f"- cost relativity = {factor_segment:.4f} / {factor_book:.4f} = **{cost_factor:.3f}**"
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
        # Year 1 in detail (renewal and conversion rates, existing vs new business).
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

    with st.expander("How year 1 is calculated"):
        d = year1_detail  # the engine's breakdown
        rows = pd.DataFrame({
            "existing customers": [d["expected_renewals"], d["retained_premium"], d["retained_claims"],
                                   DEFAULT_EXPENSE_RATIO * d["retained_premium"], d["retained_contribution"]],
            "new business": [d["expected_new_policies"], d["new_premium"], d["new_claims"],
                             DEFAULT_EXPENSE_RATIO * d["new_premium"], d["new_contribution"]],
        }, index=["policies", "premium (EUR)", "claims (EUR)", "expenses 27% (EUR)", "contribution (EUR)"])
        rows["total"] = rows.sum(axis=1)  # both sources together
        st.dataframe(rows.style.format("{:,.0f}"), use_container_width=True)
        st.markdown(
            f"- **existing customers**: each policy counts for its P(renew) (its tenure, notebook 09); "
            f"premium = commercial premium × (1 {price_change_pct:+.0%}); claims = technical premium × {cost_factor:.3f}\n"
            f"- **new business**: {len(segment_df):,} quotes (assumption: as many as policies today) × "
            f"{d['conversion_rate']:.1%} conversion (notebook 10), at the segment's average premium and cost\n"
            f"- **contribution** = premium - claims - 27% × premium"
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
        "Year 1: each policy renews with its own probability (its tenure), prospects buy with the "
        "conversion probability (notebook 11). From year 2: no further price change, so 88% of "
        "policies renew and 15% of quotes convert (`scoring/scenario_engine.py`). An outlook under "
        "stable conditions, not a market-share forecast."
    )
    st.caption(
        "Expected claims = technical premium x the segment's cost relativity shown above. The whole "
        "book at a 0% price change gives a contribution of zero (check of the 37% loading)."
    )
else:
    st.info("Choose a segment and a price change above, then click **Run scenario**.")
