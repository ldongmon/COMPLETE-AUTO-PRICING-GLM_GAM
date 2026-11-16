"""
streamlit_app.py
================
Phase 12: home page of the app. Explains the business question and shows the
portfolio's headline numbers, computed live with the same models as the
pages (and as the notebooks).

    LR  = sum of claims paid / sum of earned premiums (technical premium x exposure)
    smoothed LR = (capped claims + 33.54 x exposure) / earned premium   (notebook 08)
    COR = LR + 27% expenses
"""

import streamlit as st  # the app framework
from utils import format_eur, format_pct, get_large_loss_threshold, load_scored_portfolio  # also puts the project root on sys.path

from src.pricing.portfolio_kpis import experience_cost_factor  # noqa: E402  (smoothed ratio, notebook 08)

st.set_page_config(
    page_title="Auto Insurance Pricing & Retention",
    layout="wide",
)

st.title("Auto Insurance Pricing & Retention: Dashboard")

st.markdown(
    """
This dashboard turns a full actuarial pricing project (GLM/GAM frequency-severity
modeling, portfolio profitability diagnostics, retention and conversion modeling,
and a repricing scenario engine) into three point-and-click tools for
non-technical users. Use the sidebar to navigate:

- **Pricing Simulator**: enter a new client's characteristics, get an instant
  technical and commercial premium.
- **Retention & Conversion**: see how a price change would move an existing
  customer's renewal odds, or a new prospect's odds of accepting a quote.
- **Portfolio Scenario**: pick a segment and a price change, project the
  portfolio's Loss Ratio, Combined Operating Ratio (COR), Average Technical
  Premium (ATP) and profitability contribution over the next 1, 2 or 3 years,
  with one click.

All figures are model estimates from the French Motor Third-Party Liability
dataset (`freMTPL2freq` / `freMTPL2sev`, OpenML). See the project
[README](https://github.com/ldongmon/COMPLETE-AUTO-PRICING-GLM_GAM) and `docs/` folder for full methodology,
data sources, and every assumption behind these numbers.
"""
)

st.divider()
st.subheader("Portfolio snapshot")

with st.spinner("Loading portfolio..."):
    df, loading = load_scored_portfolio()

n_policies = len(df)  # policies in the dataset
total_exposure = df["Exposure"].sum()  # years insured
total_earned_premium = df["earned_premium"].sum()  # technical premium x exposure, summed
total_claims = df["sev_claim_amount"].sum()  # claims paid (column sev_claim_amount)
loss_ratio = total_claims / total_earned_premium  # sum, then divide
smoothed_loss_ratio = experience_cost_factor(df, get_large_loss_threshold())  # large claims mutualised (notebook 08)
expense_ratio = 0.27  # assumption (notebook 08)
atp = df["technical_premium"].mean()  # average technical premium per year
commercial_atp = df["commercial_premium"].mean()  # average commercial premium per year

col1, col2, col3, col4 = st.columns(4)
col1.metric("Policies in force", f"{n_policies:,}")
col2.metric("Loss Ratio, all claims", format_pct(loss_ratio))
col3.metric("Loss Ratio, large losses smoothed", format_pct(smoothed_loss_ratio))
col4.metric("Avg. Technical Premium", format_eur(atp))

col5, col6, col7, col8 = st.columns(4)
col5.metric("Total exposure (policy-years)", f"{total_exposure:,.0f}")
col6.metric("Expense ratio (assumption)", format_pct(expense_ratio))
col7.metric("Commercial loading (expenses)", format_pct(loading))
col8.metric("Avg. Commercial Premium", format_eur(commercial_atp))

with st.expander("How these numbers are calculated"):
    st.markdown(
        f"- **Loss Ratio, all claims** = claims paid EUR {total_claims:,.0f} / earned premium EUR {total_earned_premium:,.0f} "
        f"= **{loss_ratio:.1%}**\n"
        f"- **Loss Ratio, large losses smoothed** = (claims capped at EUR 15,000 + 33.54 x {total_exposure:,.0f} years) "
        f"/ EUR {total_earned_premium:,.0f} = **{smoothed_loss_ratio:.1%}**\n"
        f"- **Commercial loading** = 1 / (1 - 27%) - 1 = **{loading:.1%}**; average commercial premium = "
        f"{atp:,.2f} x {1 + loading:.4f} = **EUR {commercial_atp:,.2f}**\n"
        "- Claims paid and exposure come from the dataset (`sev_claim_amount`, `Exposure`); the premium is "
        "the model's technical premium of each policy."
    )

st.caption(
    "Loss Ratios are computed against the model's technical premium (this dataset has no "
    "historical charged premium, see `docs/data_dictionary.md`). The all-claims figure is pushed "
    "up by a few very large claims; with the large-loss layer at its flat loading, the book is "
    "close to 100%. The commercial loading covers the 27% expense assumption (Phase 11)."
)

st.divider()
st.markdown(
    """
**The business question this project answers:** which segments of the book are
under-priced relative to their own risk, and if we correct that with a targeted
price change, what happens next: to the Loss Ratio, but also to how many
customers renew, how many prospects convert, and the portfolio's actual
profitability over time? The **Portfolio Scenario** page lets you test that
directly.
"""
)
