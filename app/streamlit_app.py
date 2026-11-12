"""
streamlit_app.py
==================
Phase 12: home page of the Auto Insurance Pricing & Retention dashboard.

This is a non-technical-facing tool: everything a business user needs
(pricing a new client, testing conversion/retention, running a portfolio
scenario) is reachable from the sidebar pages, in plain language, with no
code or statistics jargon required. This page sets the scene: the business
question the whole project answers, and the portfolio's current headline
numbers, computed live from the same models the pages below use.
"""

from __future__ import annotations

import streamlit as st
from utils import format_eur, format_pct, get_large_loss_threshold, load_scored_portfolio  # also puts the repo root on sys.path

from src.pricing.portfolio_kpis import experience_cost_factor  # noqa: E402

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

n_policies = len(df)
total_exposure = df["Exposure"].sum()
total_earned_premium = df["earned_premium"].sum()
total_claims = df["sev_claim_amount"].sum()
loss_ratio = total_claims / total_earned_premium
# Same Loss Ratio with each claim capped at the large-loss threshold and the large-loss layer at its
# flat loading (Phase 8): what the book costs without the luck of where a few very large claims fell.
smoothed_loss_ratio = experience_cost_factor(df, get_large_loss_threshold())
expense_ratio = 0.27  # Phase 8's documented assumption, reused for display consistency
atp = df["technical_premium"].mean()
commercial_atp = df["commercial_premium"].mean()

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
