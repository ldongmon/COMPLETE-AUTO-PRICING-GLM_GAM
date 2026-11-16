"""
portfolio_kpis.py
=================
Phase 8: the portfolio indicators, and the search for a mispriced segment.

THE INDICATORS
--------------
    earned premium   = technical premium (per year) x exposure (years insured)
    Loss Ratio (LR)  = sum of the claims paid / sum of the earned premiums
    COR              = LR + expense ratio (27%, an assumption)
    ATP              = average technical premium per policy (per year)

Example: a policy priced EUR 200 a year, insured 6 months, earns 200 x 0.5 =
EUR 100. If it had a EUR 150 claim, its own LR is 150%.

Always SUM then DIVIDE: LR of a group = sum(claims) / sum(premiums), never the
average of each policy's own LR (a policy that earned EUR 5 and had one claim
would have an LR of thousands of percent and drown everything else).

WHERE THE "PREMIUM" COMES FROM (a limit of the data)
----------------------------------------------------
freMTPL2 has no premium actually charged. The premium used here is the
Phase 7 technical premium itself. So an LR of 100% means "the claims are
exactly what the model expected", and an LR far from 100% for one segment
means the model misprices that segment compared with the rest of the book.
The expense ratio (27%) is not in the data either: it is an assumption in the
usual range for French personal motor.

TWO LAYERS OF CLAIMS (see Phase 6)
----------------------------------
For each policy, with T = EUR 15,000 per claim:

    capped claims       = min(total paid, T x number of claims)   the attritional layer
    large-loss excess   = total paid - capped claims              the large-loss layer
    attritional premium = frequency x average cost x exposure     what the model expects for the attritional layer

The large-loss layer is paid by the flat loading (EUR 33.54 per year,
mutualised). A segment must therefore be judged on its ATTRITIONAL layer:

    attritional A/E (actual / expected) = sum(capped claims) / sum(attritional premium)

and compared with the A/E of the whole book (A/E relative to the book; 1.00 =
priced like the rest).
"""

import pandas as pd  # tables

DEFAULT_EXPENSE_RATIO = 0.27  # assumption: acquisition + administration costs, 27% of the premium

EXPOSURE_COL = "Exposure"  # years insured
INCURRED_CLAIMS_COL = "sev_claim_amount"  # total paid per policy (0 if no claim)
TECHNICAL_PREMIUM_COL = "technical_premium"  # EUR per year (Phase 7)


def add_earned_premium(df):
    """Adds earned_premium = technical premium x exposure (a copy, df is not changed)."""
    out = df.copy()
    out["earned_premium"] = out[TECHNICAL_PREMIUM_COL] * out[EXPOSURE_COL]  # premium for the time actually insured
    return out


def compute_kpis(df, expense_ratio=DEFAULT_EXPENSE_RATIO):
    """LR, COR and ATP of a group of policies (the whole book or one segment)."""
    total_exposure = df[EXPOSURE_COL].sum()  # years insured
    total_earned_premium = df["earned_premium"].sum()  # premiums earned
    total_incurred_claims = df[INCURRED_CLAIMS_COL].sum()  # claims paid
    loss_ratio = total_incurred_claims / total_earned_premium  # sum of claims / sum of premiums
    return {
        "n_policies": len(df),
        "total_exposure": total_exposure,
        "total_earned_premium": total_earned_premium,
        "total_incurred_claims": total_incurred_claims,
        "loss_ratio": loss_ratio,
        "expense_ratio": expense_ratio,
        "cor": loss_ratio + expense_ratio,  # combined ratio
        "atp": df[TECHNICAL_PREMIUM_COL].mean(),  # average premium per year (not prorated)
    }


def kpis_by_segment(df, segment_col, expense_ratio=DEFAULT_EXPENSE_RATIO):
    """
    compute_kpis for each level of `segment_col`, with exposure_share = the
    level's share of the book's exposure (a high LR on 0.1% of the book matters
    much less than on 15% of it). Sorted from the worst LR to the best.
    """
    total_exposure = df[EXPOSURE_COL].sum()  # exposure of the whole book
    rows = []
    for level, group in df.groupby(segment_col, observed=True):  # one group per level
        kpis = compute_kpis(group, expense_ratio=expense_ratio)
        kpis[segment_col] = level
        kpis["exposure_share"] = kpis["total_exposure"] / total_exposure  # size of the segment
        rows.append(kpis)
    result = pd.DataFrame(rows)
    first = [segment_col, "n_policies", "exposure_share"]  # columns shown first
    result = result[first + [c for c in result.columns if c not in first]]
    return result.sort_values("loss_ratio", ascending=False).reset_index(drop=True)


def add_claim_layers(df, threshold):
    """
    Adds the two layers of each policy's claims (a copy, df is not changed):
      capped_claims       min(total paid, threshold x number of claims)
      large_loss_excess   total paid - capped_claims
      attritional_premium frequency x average cost x exposure
    """
    out = df.copy()
    cap = threshold * out["sev_claim_count"].clip(lower=1)  # the cap grows with the number of claims
    out["capped_claims"] = out[INCURRED_CLAIMS_COL].clip(upper=cap)  # attritional layer
    out["large_loss_excess"] = out[INCURRED_CLAIMS_COL] - out["capped_claims"]  # large-loss layer
    out["attritional_premium"] = out["predicted_frequency"] * out["predicted_severity"] * out[EXPOSURE_COL]  # expected attritional cost
    return out


def attritional_ae(df):
    """Actual / expected on the attritional layer: sum(capped claims) / sum(attritional premium)."""
    return df["capped_claims"].sum() / df["attritional_premium"].sum()


def attritional_ae_table(df, col):
    """
    For each level of `col`: its share of exposure, its number of paid claims,
    and its attritional A/E divided by the book's A/E, on three sets of data:
    all data, the validation set only, the test set only (column `split`).
    A real pricing gap shows on all three; noise changes direction.
    """
    book = {
        "all": attritional_ae(df),  # A/E of the whole book
        "val": attritional_ae(df[df["split"] == "val"]),  # same, validation set
        "test": attritional_ae(df[df["split"] == "test"]),  # same, test set
    }
    rows = []
    for level, group in df.groupby(col, observed=True):  # one row per level
        row = {
            col: level,
            "exposure_share": group[EXPOSURE_COL].sum() / df[EXPOSURE_COL].sum(),
            "paid_claims": int(group["sev_claim_count"].sum()),
            "AE_all": attritional_ae(group) / book["all"],
            "AE_val": attritional_ae(group[group["split"] == "val"]) / book["val"],
            "AE_test": attritional_ae(group[group["split"] == "test"]) / book["test"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def experience_cost_factor(group_df, large_loss_threshold):
    """
    What a group of policies really costs, as a multiple of its technical premium:

        cost   = its own capped claims + flat loading x its exposure
        factor = cost / its earned technical premium

    The attritional layer comes from the group's own experience; the
    large-loss layer is charged at the flat loading (mutualised), never at the
    group's own large claims. A segment's cost relativity is its factor divided
    by the book's factor (both on the same data).
    """
    cap = large_loss_threshold * group_df["sev_claim_count"].clip(lower=1)  # cap per policy
    capped = group_df[INCURRED_CLAIMS_COL].clip(upper=cap).sum()  # attritional claims of the group
    large = (group_df["large_loss_loading"] * group_df[EXPOSURE_COL]).sum()  # mutualised large-loss charge
    earned = (group_df[TECHNICAL_PREMIUM_COL] * group_df[EXPOSURE_COL]).sum()  # earned technical premium
    return float((capped + large) / earned)
