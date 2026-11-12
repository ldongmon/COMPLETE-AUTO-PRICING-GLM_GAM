"""
portfolio_kpis.py
==================
Phase 8: portfolio-level and segment-level KPIs built on top of the Phase 7
technical premium: Loss Ratio, Combined Operating Ratio (COR), Average
Technical Premium (ATP). This is where the project's original business
question, *"Segment X has a Loss Ratio above the rest of the book. Should
we reprice it?"*, becomes a computed number, cut by every rating factor.

WHERE "EARNED PREMIUM" COMES FROM (A LIMITATION)
----------------------------------------------------
freMTPL2 has no historical premium actually charged: only claims and risk
factors (Phase 1). There is no way around this; it is a property of the
dataset, not something to work around cleverly. The premium used here is the
Phase 7 **technical premium** itself, prorated to each policy's observed
exposure. This has a specific, defensible interpretation: it treats the
portfolio as if it were priced *exactly* at technical cost, with no margin.
Loss Ratio computed this way is not "how much profit did we make" (that
needs a real charged premium, which doesn't exist here): it is **"how well
does the model's own prediction match what actually happened, in aggregate
and by segment."** That is still a useful diagnostic: an LR far
from 100% for a specific segment means the model is *mis-pricing* that
segment relative to its own overall calibration, which is exactly the kind
of signal that would prompt a "should we reprice Segment X" conversation.

WHY THE EXPENSE RATIO IS A FLAT ASSUMPTION
------------------------------------------------
No expense data exists in this dataset either. `DEFAULT_EXPENSE_RATIO`
(27%) is a documented, illustrative assumption in line with commonly-cited
expense ratios for French personal motor lines (acquisition + administration
costs), not fitted from data, and stated as an assumption everywhere it's
used, per standard actuarial disclosure practice.

WHY EVERY RATIO IS EXPOSURE/PREMIUM-WEIGHTED (SUMMED), NEVER AVERAGED PER POLICY
-------------------------------------------------------------------------------------
Same discipline as Phase 2's frequency lesson: `mean(policy-level ratio)` is
dominated by tiny-premium policies with wild individual ratios (a policy
earning EUR 5 of premium with any claim at all has an ostronomical
"loss ratio" that says nothing about the segment). The correct estimator
sums numerator and denominator separately across the group, then divides.
`sum(claims) / sum(premium)`, not `mean(claims / premium)`.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_EXPENSE_RATIO = 0.27  # see module docstring: an assumption, not a fitted value

EXPOSURE_COL = "Exposure"
INCURRED_CLAIMS_COL = "sev_claim_amount"  # Phase 1: actual total paid EUR per policy (0 if no claim)
TECHNICAL_PREMIUM_COL = "technical_premium"  # Phase 7: annualized EUR / policy-year


def add_earned_premium(df: pd.DataFrame) -> pd.DataFrame:
    """
    `technical_premium` (Phase 7) is an annualized, per-policy-*year* rate.
    "Earned premium" for a specific historical policy is that rate prorated
    to however long the policy was actually observed: exactly analogous to
    how `ClaimNb_freq_target`'s offset (`log(Exposure)`) prorates expected
    claim count. Adds an `earned_premium` column; does not mutate `df`.
    """
    out = df.copy()
    out["earned_premium"] = out[TECHNICAL_PREMIUM_COL] * out[EXPOSURE_COL]
    return out


def compute_kpis(df: pd.DataFrame, expense_ratio: float = DEFAULT_EXPENSE_RATIO) -> dict:
    """
    Portfolio- (or segment-, if `df` is already filtered) level KPIs.
    `df` must already have `earned_premium` (see `add_earned_premium`).

    Returns a dict: n_policies, total_exposure, total_earned_premium,
    total_incurred_claims, loss_ratio, expense_ratio, cor, atp.
    """
    total_exposure = df[EXPOSURE_COL].sum()
    total_earned_premium = df["earned_premium"].sum()
    total_incurred_claims = df[INCURRED_CLAIMS_COL].sum()

    loss_ratio = total_incurred_claims / total_earned_premium
    cor = loss_ratio + expense_ratio
    atp = df[TECHNICAL_PREMIUM_COL].mean()  # average ANNUALIZED rate per policy, not prorated

    return {
        "n_policies": len(df),
        "total_exposure": total_exposure,
        "total_earned_premium": total_earned_premium,
        "total_incurred_claims": total_incurred_claims,
        "loss_ratio": loss_ratio,
        "expense_ratio": expense_ratio,
        "cor": cor,
        "atp": atp,
    }


def kpis_by_segment(df: pd.DataFrame, segment_col: str, expense_ratio: float = DEFAULT_EXPENSE_RATIO) -> pd.DataFrame:
    """
    `compute_kpis`, one row per level of `segment_col`, plus `exposure_share`
    (each segment's share of total portfolio exposure): essential for
    judging *materiality*: a dramatic Loss Ratio backed by 0.1% of the book
    is a very different conversation than one backed by 15% of it (the same
    caution Phase 2's EDA applied to univariate rating-factor charts).
    """
    total_exposure = df[EXPOSURE_COL].sum()
    rows = []
    for level, group in df.groupby(segment_col, observed=True):
        kpis = compute_kpis(group, expense_ratio=expense_ratio)
        kpis[segment_col] = level
        kpis["exposure_share"] = kpis["total_exposure"] / total_exposure
        rows.append(kpis)

    result = pd.DataFrame(rows)
    cols = [segment_col, "n_policies", "exposure_share"] + [
        c for c in result.columns if c not in (segment_col, "n_policies", "exposure_share")
    ]
    return result[cols].sort_values("loss_ratio", ascending=False).reset_index(drop=True)


def experience_cost_factor(group_df: pd.DataFrame, large_loss_threshold: float) -> float:
    """
    Expected claims cost of `group_df`, as a multiple of its technical premium.

    The attritional layer (each claim capped at `large_loss_threshold`) is
    taken from the group's own experience. The large-loss layer is charged at
    the flat loading already in the technical premium (`large_loss_loading`,
    Phase 6), because large losses are too sparse and too volatile to
    attribute to a segment (Phase 8):

        cost   = capped claims of the group + large_loss_loading x exposure
        factor = cost / earned technical premium of the group

    A segment's cost relativity is its factor divided by the book's factor,
    computed on the same data.

    Required columns: Exposure, sev_claim_amount, sev_claim_count,
    technical_premium, large_loss_loading.
    """
    cap = large_loss_threshold * group_df["sev_claim_count"].clip(lower=1)
    capped = group_df[INCURRED_CLAIMS_COL].clip(upper=cap).sum()
    large = (group_df["large_loss_loading"] * group_df[EXPOSURE_COL]).sum()
    earned = (group_df[TECHNICAL_PREMIUM_COL] * group_df[EXPOSURE_COL]).sum()
    return float((capped + large) / earned)
