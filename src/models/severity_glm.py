"""
severity_glm.py
=================
Phase 6: fits the severity model: a Gamma GLM predicting average claim
amount, conditional on a policy having at least one paid claim.

WHY GAMMA, AND WHY NO OFFSET THIS TIME
------------------------------------------
Severity (Phase 1/2) is strictly positive and heavily right-skewed: a
handful of very large claims coexist with a mass of claims in the low
thousands of euros. A Gamma distribution with a `log` link is the standard
actuarial choice for this shape (constant coefficient of variation, i.e. the
spread scales with the mean, which matches how claim costs actually behave).

Unlike the frequency model, severity has **no exposure offset**: exposure
describes how long a policy was on risk during the year, which has no
bearing on how large a claim is *once it happens*. What severity *does* need
is a different kind of weighting: `AvgClaimAmount` (Phase 1) is the average
of `sev_claim_count` individual claim payments for policies with more than
one claim, and an average of more observations is a more precise estimate of
the "true" severity for that policy than an average of one. `statsmodels`
expresses this with `var_weights` (Var(Y_i) = phi / w_i * V(mu_i)): passing
`weights=sev_claim_count` tells the Gamma GLM to trust multi-claim policies'
averages more than single-claim ones, exactly analogous to how frequency
weights policies by `Exposure`.

WHY A SEPARATE LARGE-LOSS TREATMENT
---------------------------------------
Phase 1/2 already found this dataset's severity has an extreme tail (a
single EUR 4.08M claim against a portfolio mean of ~EUR 2,300). A Gamma
GLM's mean is estimated by (quasi-)maximum likelihood, and a handful of
gigantic claims can distort every coefficient's estimate for the entire
book, not just for the segment that happened to produce them: the standard
actuarial answer is to **cap** claims at a data-driven threshold before
fitting the "attritional" severity model, and add back the average cost of
the excess as a flat, portfolio-wide loading (mirroring how a reinsurance
excess-of-loss layer's burning cost is typically estimated: the large-loss
population is too sparse to model *whose* claim will be huge with any
precision, so its average cost is spread across the whole book instead).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# --------------------------------------------------------------------------- #
# Large-loss threshold & attritional/excess split
# --------------------------------------------------------------------------- #

DEFAULT_LARGE_LOSS_PERCENTILE = 0.99


def determine_large_loss_threshold(
    train_severity_df: pd.DataFrame,
    percentile: float = DEFAULT_LARGE_LOSS_PERCENTILE,
    amount_col: str = "AvgClaimAmount",
) -> float:
    """
    The euro threshold above which a claim is treated as "large": learned
    from the TRAINING severity data only (same discipline as Phase 3's
    `VehAge` capping threshold), never recomputed from validation/test data.
    """
    return float(train_severity_df[amount_col].quantile(percentile))


def cap_severity(severity_df: pd.DataFrame, threshold: float, amount_col: str = "AvgClaimAmount") -> pd.Series:
    """Winsorize claim severity at `threshold`: the target for the attritional Gamma GLM/GAM."""
    return severity_df[amount_col].clip(upper=threshold)


def large_loss_loading(
    train_full_df: pd.DataFrame,
    train_severity_df: pd.DataFrame,
    threshold: float,
) -> float:
    """
    The flat, per-policy-year euro amount to add on top of the attritional
    technical premium, representing the average cost of the large-loss
    "excess" spread across the *entire* portfolio (not just claiming
    policies): the same logic as a burning-cost estimate for a reinsurance
    excess-of-loss layer.

    excess_i = total claim euros for policy i above the cap (0 for
               policies whose severity never crossed the threshold)
    loading  = sum(excess_i) / total portfolio exposure (all policies)
    """
    capped_total_per_policy = (
        cap_severity(train_severity_df, threshold) * train_severity_df["sev_claim_count"]
    )
    excess_per_policy = train_severity_df["sev_claim_amount"] - capped_total_per_policy
    total_excess = excess_per_policy.clip(lower=0).sum()
    total_exposure = train_full_df["Exposure"].sum()
    return float(total_excess / total_exposure)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def fit_gamma_glm(X: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series] = None):
    """
    Fit a Gamma GLM with a log link. `weights` (typically `sev_claim_count`)
    are passed as `var_weights`: see the module docstring for why this is
    the correct weighting for an *averaged* severity target.
    """
    X_with_const = sm.add_constant(X, has_constant="add")
    model = sm.GLM(y, X_with_const, family=sm.families.Gamma(link=sm.families.links.Log()), var_weights=weights)
    return model.fit()


def _flatten(groups: dict[str, list[str]]) -> list[str]:
    return [c for cols in groups.values() for c in cols]


def backward_stepwise_selection_gamma(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    feature_groups: dict[str, list[str]],
    p_threshold: float = 0.05,
    verbose: bool = True,
):
    """
    Backward, factor-group-level stepwise elimination for the severity GLM.
    the same likelihood-ratio-test logic as
    `frequency_glm.backward_stepwise_selection` (see that function's
    docstring for the full rationale), adapted for a weighted Gamma fit
    instead of a Poisson fit with an offset. Kept as its own function rather
    than a shared abstraction: the two fitting calls differ enough (offset
    vs. var_weights) that forcing them through one generic function would
    obscure more than it would save.
    """
    from src.models.glm_diagnostics import likelihood_ratio_test  # local import avoids a cycle at module load time

    remaining_groups = dict(feature_groups)
    current_results = fit_gamma_glm(X[_flatten(remaining_groups)], y, weights)
    history = []

    while len(remaining_groups) > 1:
        candidates = []
        for name, cols in remaining_groups.items():
            trial_groups = {k: v for k, v in remaining_groups.items() if k != name}
            trial_results = fit_gamma_glm(X[_flatten(trial_groups)], y, weights)
            lr = likelihood_ratio_test(current_results, trial_results)
            candidates.append({"factor": name, "p_value": lr["p_value"], "aic_without": trial_results.aic})

        candidates.sort(key=lambda c: -c["p_value"])
        best = candidates[0]

        if best["p_value"] <= p_threshold:
            break

        history.append(
            {
                "step": len(history) + 1,
                "factor_removed": best["factor"],
                "lr_test_p_value": best["p_value"],
                "aic_before": current_results.aic,
                "aic_after": best["aic_without"],
            }
        )
        if verbose:
            print(
                f"Step {len(history)}: drop '{best['factor']}' "
                f"(LR-test p={best['p_value']:.4f}, AIC {current_results.aic:.1f} -> {best['aic_without']:.1f})"
            )

        del remaining_groups[best["factor"]]
        current_results = fit_gamma_glm(X[_flatten(remaining_groups)], y, weights)

    return current_results, remaining_groups, pd.DataFrame(history)
