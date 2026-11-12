"""
scenario_engine.py
====================
Phase 11: the direct, final answer to this project's original business
question: *"Segment X has a Loss Ratio above the rest of the book. Should we reprice it,
and how much, and what would that do to retention, conversion, and overall
profitability?"* Combines Phase 7's technical premium, Phase 9's retention
model, and Phase 10's conversion model into a single scenario evaluation.

WHAT "PROFITABILITY" MEANS HERE, PRECISELY
------------------------------------------------
For a segment repriced by `price_change_pct`, this engine projects two
sources of policies for the *next* period:

1. **Retained policies**: existing segment policyholders who renew, at the
   repriced commercial premium, with probability given by Phase 9's
   `renewal_probability`.
2. **New policies**: prospects quoted at the repriced commercial premium,
   converting with probability given by Phase 10's `conversion_probability`.

For both, the expected **contribution** (premium collected, minus expected
claims cost, minus expenses) is summed:

    contribution = written_premium - expected_claims - (written_premium * expense_ratio)

`expected_claims` is the technical premium (Phase 7) multiplied by a
`cost_factor`: the segment's expected cost as a multiple of its technical
premium, measured on experience by `experience_cost_factor`
(src/pricing/portfolio_kpis.py). For the whole book this factor is the book
Loss Ratio, the same basis the commercial loading is calibrated on, so a 0%
price change on the whole book returns a contribution of zero. Repricing
changes what is charged, not what a policy is expected to cost.

THE NEW-BUSINESS QUOTE VOLUME ASSUMPTION
------------------------------------------
This dataset (Phase 1) contains only the *existing*, in-force book: there
is no record of how many prospects were quoted but didn't convert. To
project new business at all, this engine assumes the segment's annual quote
volume equals its current policy count (`quote_volume_multiplier=1.0` by
default): a stated, adjustable simplification, not a measured figure. A
real deployment would replace this with actual quote-pipeline data.

A SECOND STATED SIMPLIFICATION: RETAINED AND NEW POLICIES SHARE THE SEGMENT'S AVERAGE RISK PROFILE
------------------------------------------------------------------------------------------------------
This engine does not model adverse selection in *who* lapses or *who*
converts (e.g. "maybe only the highest-risk policyholders stay when price
rises"): retained and new policies are assumed to carry the segment's
average technical cost. A real deployment might refine this if retention/
conversion turn out to correlate with risk level within the segment, which
this project has not tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.prospect.conversion_model import conversion_probability
from src.retention.renewal_model import renewal_probability

DEFAULT_EXPENSE_RATIO = 0.27  # Phase 8's documented assumption, reused here for consistency
DEFAULT_QUOTE_VOLUME_MULTIPLIER = 1.0  # see module docstring


def run_repricing_scenario(
    segment_df: pd.DataFrame,
    price_change_pct: float,
    expense_ratio: float = DEFAULT_EXPENSE_RATIO,
    quote_volume_multiplier: float = DEFAULT_QUOTE_VOLUME_MULTIPLIER,
    cost_factor: float = 1.0,
) -> dict:
    """
    Evaluate one repricing scenario for a segment (already filtered to it,
    e.g. `df[df["BonusMalus"].between(51, 80)]`), with columns `commercial_premium`,
    `technical_premium`, `tenure_years_proxy`, and `BonusMalus`.

    Returns a dict with the full breakdown: retained vs. new business,
    each side's premium/claims/contribution, and portfolio totals, so the
    scenario's story can be read at whatever level of detail is needed, not
    just the bottom line.
    """
    p_renew = renewal_probability(price_change_pct, segment_df["tenure_years_proxy"])
    repriced_commercial_premium = segment_df["commercial_premium"] * (1 + price_change_pct)
    # Expected cost per policy: technical premium x the segment's experience cost factor.
    # Unaffected by the price change.
    technical_cost = segment_df["technical_premium"] * cost_factor

    # --- Retained (existing) policies ---
    expected_renewals = float(p_renew.sum())
    retained_premium = float((p_renew * repriced_commercial_premium).sum())
    retained_claims = float((p_renew * technical_cost).sum())
    retained_expenses = retained_premium * expense_ratio
    retained_contribution = retained_premium - retained_claims - retained_expenses

    # --- New business ---
    p_convert = conversion_probability(price_change_pct, segment_df["BonusMalus"])
    quote_volume = len(segment_df) * quote_volume_multiplier
    avg_p_convert = float(p_convert.mean())
    avg_repriced_commercial_premium = float(repriced_commercial_premium.mean())
    avg_technical_cost = float(technical_cost.mean())

    expected_new_policies = quote_volume * avg_p_convert
    new_premium = expected_new_policies * avg_repriced_commercial_premium
    new_claims = expected_new_policies * avg_technical_cost
    new_expenses = new_premium * expense_ratio
    new_contribution = new_premium - new_claims - new_expenses

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


def scenario_grid(segment_df: pd.DataFrame, price_changes: list[float], **kwargs) -> pd.DataFrame:
    """`run_repricing_scenario` evaluated across a grid of price changes, one row per value."""
    rows = [run_repricing_scenario(segment_df, pct, **kwargs) for pct in price_changes]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Multi-year projection (Phase 12: the app's "1/2/3-year outlook" view)
# --------------------------------------------------------------------------- #


def project_multi_year_scenario(
    segment_df: pd.DataFrame,
    price_change_pct: float,
    n_years: int,
    expense_ratio: float = DEFAULT_EXPENSE_RATIO,
    quote_volume_multiplier: float = DEFAULT_QUOTE_VOLUME_MULTIPLIER,
    cost_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Projects a segment's KPIs (policy count, written premium, incurred claims,
    Loss Ratio, COR, ATP) for each of the next `n_years`, under a repricing
    action applied once, at the start of Year 1.

    THE MODELING CHOICE THAT MAKES A MULTI-YEAR PROJECTION TRACTABLE
    ---------------------------------------------------------------------
    Year 1 absorbs the price *change* itself: existing policyholders decide
    whether to renew, and prospects decide whether to convert, in reaction to
    the repricing (`run_repricing_scenario`, Phase 11, used here unmodified.
    the exact row-level, heterogeneity-aware calculation from Phase 11).

    From Year 2 onward, no *further* price change is applied: the segment
    is assumed to hold at its new, already-corrected price level. This is
    not an approximation: `renewal_probability(0, tenure)` and
    `conversion_probability(0, bonus_malus)` both collapse to their baseline
    rate *for every policy regardless of tenure or risk level* (Phase 9/10's
    own calibration: see their notebooks' Section on "the formula
    converges to baseline at zero price change"). So Year 2+ can be computed
    exactly from portfolio-level aggregates alone: no need to track
    individual policies' tenure aging or risk mix: using the average
    premium and average technical cost actually realized at the end of
    Year 1 (which already reflects the repriced level and the Year-1
    survivor/new-business mix).

    A book that keeps renewing and acquiring new business at flat baseline
    rates forever would grow without bound (`BASELINE_RENEWAL_RATE` +
    `BASELINE_CONVERSION_RATE`-driven inflow both being positive, with no
    modeled attrition beyond the renewal/conversion mechanics themselves) --
    stated here as the boundary of this projection's realism: it is a
    **KPI outlook under stable conditions**, not a long-run market-share
    forecast, and should be read as such beyond a few years.
    """
    from src.prospect.conversion_model import BASELINE_CONVERSION_RATE
    from src.retention.renewal_model import BASELINE_RENEWAL_RATE

    year1 = run_repricing_scenario(
        segment_df,
        price_change_pct,
        expense_ratio=expense_ratio,
        quote_volume_multiplier=quote_volume_multiplier,
        cost_factor=cost_factor,
    )

    quote_volume = len(segment_df) * quote_volume_multiplier
    avg_premium_level = year1["total_written_premium"] / year1["total_policies"]
    avg_technical_cost_level = (year1["retained_claims"] + year1["new_claims"]) / year1["total_policies"]

    rows = []
    policies = year1["total_policies"]
    written_premium = year1["total_written_premium"]
    incurred_claims = year1["retained_claims"] + year1["new_claims"]

    for year in range(1, n_years + 1):
        if year > 1:
            # Steady state: every existing policy renews at BASELINE_RENEWAL_RATE (price change = 0
            # from here on), and quote_volume new prospects convert at BASELINE_CONVERSION_RATE.
            policies = policies * BASELINE_RENEWAL_RATE + quote_volume * BASELINE_CONVERSION_RATE
            written_premium = policies * avg_premium_level
            incurred_claims = policies * avg_technical_cost_level

        expenses = written_premium * expense_ratio
        loss_ratio = incurred_claims / written_premium
        cor = loss_ratio + expense_ratio
        atp = avg_premium_level
        contribution = written_premium - incurred_claims - expenses

        rows.append(
            {
                "year": year,
                "policies": policies,
                "written_premium": written_premium,
                "incurred_claims": incurred_claims,
                "loss_ratio": loss_ratio,
                "cor": cor,
                "atp": atp,
                "contribution": contribution,
            }
        )

    return pd.DataFrame(rows)
