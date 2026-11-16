"""
scenario_engine.py
==================
Phase 11: what happens to a segment's profit if we change its price by x%?

THE IDEA IN FOUR LINES
----------------------
For a price change x applied to every policy of the segment, next year:

  1. existing customers renew with probability P(renew)   (Phase 9)
  2. prospects buy with probability P(convert)            (Phase 10)
  3. each policy pays   commercial premium x (1 + x)
  4. each policy costs  technical premium x cost factor   (the price change does not change the risk)

    contribution = premium - claims - expenses       (expenses = 27% of the premium)

COST FACTOR
-----------
The expected cost of a policy as a multiple of its technical premium:
1.0 for the book (the technical premium is the expected cost), 1.166 for
Segment X (its cost relativity measured out of sample in Phase 8).

HAND EXAMPLE FOR ONE POLICY (IDpol 24955, BonusMalus 68, Segment X, +17%)
-----------------------------------------------------------------------
technical premium (Phase 7)  EUR 128.84
commercial premium           128.84 x 1.3699 = 176.50 -> after +17%: 176.50 x 1.17 = 206.50
expected cost                128.84 x 1.166 (Phase 8) = 150.26
expenses                     27% x 206.50 = 55.76
margin if it renews          206.50 - 150.26 - 55.76 = EUR +0.49   (today: 176.50 - 150.26 - 47.66 = EUR -21.41)
P(renew)                     tenure 7.52 years (from BonusMalus 68) -> 82.8% (Phase 9)
expected contribution        82.8% x 0.49 = EUR +0.41

TWO STATED SIMPLIFICATIONS
--------------------------
  * Number of quotes per year = number of policies in the segment (no quote data).
  * Customers who stay and new customers have the segment's average risk
    (no adverse selection modelled).
"""

import pandas as pd  # tables

from src.prospect.conversion_model import conversion_probability  # Phase 10
from src.retention.renewal_model import renewal_probability  # Phase 9

DEFAULT_EXPENSE_RATIO = 0.27  # same assumption as Phase 8
DEFAULT_QUOTE_VOLUME_MULTIPLIER = 1.0  # quotes per year = 1 x number of policies


def run_repricing_scenario(segment_df, price_change_pct, expense_ratio=DEFAULT_EXPENSE_RATIO,
                           quote_volume_multiplier=DEFAULT_QUOTE_VOLUME_MULTIPLIER, cost_factor=1.0):
    """
    One scenario for one segment (columns needed: commercial_premium,
    technical_premium, tenure_years_proxy, BonusMalus). Returns the full
    breakdown: renewals, new business, premium, claims, contribution.
    """
    # Each policy: probability of renewing, new premium, expected cost.
    p_renew = renewal_probability(price_change_pct, segment_df["tenure_years_proxy"])  # Phase 9
    repriced_commercial_premium = segment_df["commercial_premium"] * (1 + price_change_pct)  # new premium
    technical_cost = segment_df["technical_premium"] * cost_factor  # expected cost (does not move with the price)

    # 1. Existing customers: each one counts for its probability of renewing.
    expected_renewals = float(p_renew.sum())  # expected number of renewals
    retained_premium = float((p_renew * repriced_commercial_premium).sum())  # sum of P(renew) x premium
    retained_claims = float((p_renew * technical_cost).sum())  # sum of P(renew) x cost
    retained_expenses = retained_premium * expense_ratio  # 27% of the premium
    retained_contribution = retained_premium - retained_claims - retained_expenses

    # 2. New business: as many quotes as policies, at the segment's average premium and cost.
    p_convert = conversion_probability(price_change_pct, segment_df["BonusMalus"])  # Phase 10
    quote_volume = len(segment_df) * quote_volume_multiplier  # number of quotes
    avg_p_convert = float(p_convert.mean())  # average conversion probability
    avg_repriced_commercial_premium = float(repriced_commercial_premium.mean())  # average new premium
    avg_technical_cost = float(technical_cost.mean())  # average expected cost

    expected_new_policies = quote_volume * avg_p_convert  # quotes x conversion rate
    new_premium = expected_new_policies * avg_repriced_commercial_premium
    new_claims = expected_new_policies * avg_technical_cost
    new_expenses = new_premium * expense_ratio
    new_contribution = new_premium - new_claims - new_expenses

    # 3. Totals.
    total_contribution = retained_contribution + new_contribution
    total_written_premium = retained_premium + new_premium
    total_policies = expected_renewals + expected_new_policies

    return {
        "price_change_pct": price_change_pct,
        "expected_renewals": expected_renewals,
        "renewal_rate": expected_renewals / len(segment_df),
        "expected_new_policies": expected_new_policies,
        "conversion_rate": avg_p_convert,
        "retained_premium": retained_premium,
        "retained_claims": retained_claims,
        "retained_contribution": retained_contribution,
        "new_premium": new_premium,
        "new_claims": new_claims,
        "new_contribution": new_contribution,
        "total_contribution": total_contribution,
        "total_written_premium": total_written_premium,
        "total_policies": total_policies,
    }


def scenario_grid(segment_df, price_changes, **kwargs):
    """run_repricing_scenario for a list of price changes: one row per price change."""
    rows = [run_repricing_scenario(segment_df, pct, **kwargs) for pct in price_changes]
    return pd.DataFrame(rows)


def project_multi_year_scenario(segment_df, price_change_pct, n_years, expense_ratio=DEFAULT_EXPENSE_RATIO,
                                quote_volume_multiplier=DEFAULT_QUOTE_VOLUME_MULTIPLIER, cost_factor=1.0):
    """
    The segment's KPIs over the next n years (used by the app), with the price
    change applied once, at the start of year 1.

    Year 1 = run_repricing_scenario (customers react to the price change).
    Year 2 onwards: no further price change, so every customer renews at 88%
    and every prospect converts at 15% (both formulas give their base rate at
    0% change, whatever the tenure or the risk):

        policies(year n) = policies(year n-1) x 88% + quotes x 15%

    with the average premium and cost reached at the end of year 1. This is an
    outlook under stable conditions, not a market-share forecast.
    """
    from src.prospect.conversion_model import BASELINE_CONVERSION_RATE  # 15%
    from src.retention.renewal_model import BASELINE_RENEWAL_RATE  # 88%

    year1 = run_repricing_scenario(segment_df, price_change_pct, expense_ratio=expense_ratio,
                                   quote_volume_multiplier=quote_volume_multiplier, cost_factor=cost_factor)

    quote_volume = len(segment_df) * quote_volume_multiplier  # quotes per year
    avg_premium_level = year1["total_written_premium"] / year1["total_policies"]  # average premium after year 1
    avg_technical_cost_level = (year1["retained_claims"] + year1["new_claims"]) / year1["total_policies"]  # average cost after year 1

    rows = []
    policies = year1["total_policies"]
    written_premium = year1["total_written_premium"]
    incurred_claims = year1["retained_claims"] + year1["new_claims"]

    for year in range(1, n_years + 1):
        if year > 1:  # stable years: base rates only
            policies = policies * BASELINE_RENEWAL_RATE + quote_volume * BASELINE_CONVERSION_RATE
            written_premium = policies * avg_premium_level
            incurred_claims = policies * avg_technical_cost_level

        expenses = written_premium * expense_ratio  # 27% of the premium
        loss_ratio = incurred_claims / written_premium
        rows.append({
            "year": year,
            "policies": policies,
            "written_premium": written_premium,
            "incurred_claims": incurred_claims,
            "loss_ratio": loss_ratio,
            "cor": loss_ratio + expense_ratio,
            "atp": avg_premium_level,
            "contribution": written_premium - incurred_claims - expenses,
        })

    return pd.DataFrame(rows)
