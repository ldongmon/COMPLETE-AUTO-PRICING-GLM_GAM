"""
frequency_glm.py
=================
Phase 4: fits the frequency model: a Poisson GLM predicting claim count,
with `log(Exposure)` as an offset (see `src/features/pipeline.py` for why
exposure is an offset and never a feature).

WHY POISSON, AND WHY log(Exposure) AS AN OFFSET
--------------------------------------------------
Claim count is a non-negative integer with a large spike at zero (Phase 2:
~95% of policies claim zero times): a Poisson distribution is the standard
actuarial starting point for this shape. The GLM's linear predictor is:

    log(E[claims]) = log(Exposure) + beta_0 + beta_1 * x_1 + ... + beta_p * x_p

Rearranged, this says `E[claims] / Exposure = exp(beta_0 + beta_1*x_1 + ...)`
i.e. the model predicts a *rate per unit exposure*, exactly the quantity a
pricing actuary wants, with a coefficient of exactly 1.0 forced on
`log(Exposure)` rather than estimated. That forced coefficient is what
"offset" means in `statsmodels`, and it's the textbook-correct way to build
a frequency model: estimating exposure's coefficient instead (i.e. treating
it as an ordinary feature) is a common and serious modeling error.

WHY BACKWARD, FACTOR-LEVEL STEPWISE SELECTION
-------------------------------------------------
`src/features/pipeline.py` one-hot-encodes `VehBrand`, `VehGas`, and `Region`
into several dummy columns each. Eliminating one dummy column at a time based
on its individual p-value can leave a factor "half kept": e.g. 8 of 10
`VehBrand` dummies still in the model, 2 silently dropped, which is not
interpretable or defensible to an underwriter ("why does the model use brand
for some cars but not others?"). This module instead eliminates whole
factors at once, using a likelihood-ratio test on the group of columns that
belong to that factor (see `glm_diagnostics.likelihood_ratio_test`): the
standard actuarial practice for GLM factor selection.
"""

from __future__ import annotations

import gc
from typing import Optional

import pandas as pd
import statsmodels.api as sm

from src.models.glm_diagnostics import likelihood_ratio_test

# --------------------------------------------------------------------------- #
# Factor groups
# --------------------------------------------------------------------------- #
# Maps each *rating factor* (as an actuary would name it) to the design-matrix
# column(s) it expands to after Phase 3's encoding. Written explicitly rather
# than inferred by string-matching column name prefixes: explicit is easier
# to audit than clever, and this mapping changes rarely enough that hand
# maintenance is not a burden.

def build_factor_groups(feature_names: list[str]) -> dict[str, list[str]]:
    """
    Group `PolicyFeatureEncoder`'s output columns back into the rating
    factors they came from, so stepwise selection can add/drop a whole
    factor at once instead of one dummy column at a time.
    """
    groups = {
        "DrivAge": [c for c in feature_names if c == "DrivAge"],
        "BonusMalus": [c for c in feature_names if c == "BonusMalus"],
        "VehPower": [c for c in feature_names if c == "VehPower"],
        "Density": [c for c in feature_names if c == "log_Density"],
        "VehAge": [c for c in feature_names if c == "VehAge_capped"],
        "VehBrand": [c for c in feature_names if c.startswith("VehBrand_")],
        "VehGas": [c for c in feature_names if c.startswith("VehGas_")],
        "Region": [c for c in feature_names if c.startswith("Region_")],
    }
    # Defensive: catch a silent mismatch immediately (e.g. the encoder's
    # output changed but this mapping wasn't updated) instead of quietly
    # dropping a factor from every model fit downstream.
    grouped_cols = {c for cols in groups.values() for c in cols}
    missing = set(feature_names) - grouped_cols
    if missing:
        raise ValueError(f"build_factor_groups doesn't account for columns: {missing}")
    return {name: cols for name, cols in groups.items() if cols}  # drop any empty group


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def fit_poisson_glm(X: pd.DataFrame, y: pd.Series, offset: pd.Series):
    """
    Fit a Poisson GLM with a log link and the given offset. Adds an
    intercept column (`const`) automatically: every model in this project
    keeps an intercept, so relativities in Section "interpretation" are all
    read relative to the same baseline.
    """
    X_with_const = sm.add_constant(X, has_constant="add")
    model = sm.GLM(y, X_with_const, family=sm.families.Poisson(), offset=offset)
    return model.fit()


def _flatten(groups: dict[str, list[str]]) -> list[str]:
    return [c for cols in groups.values() for c in cols]


def backward_stepwise_selection(
    X: pd.DataFrame,
    y: pd.Series,
    offset: pd.Series,
    feature_groups: dict[str, list[str]],
    p_threshold: float = 0.05,
    verbose: bool = True,
):
    """
    Backward elimination at the rating-factor level: start from the full
    model (every factor in `feature_groups`), and repeatedly drop the least
    useful *whole factor*: the one whose removal a likelihood-ratio test
    says hurts the fit the least: as long as that removal's LR-test
    p-value stays above `p_threshold` (conventionally 0.05). Stops as soon
    as every remaining factor is significant, or only one factor is left.

    Returns
    -------
    final_results : the fitted statsmodels GLMResults for the selected model
    remaining_groups : dict of the factors kept
    history : pd.DataFrame, one row per elimination step, for a fully
        auditable record of what was dropped, when, and why

    A note on memory (worth keeping if you re-read this later): this dataset
    has ~407k training rows, and each fitted `GLMResults` object holds a
    reference to its own copy of the design matrix plus fit internals. Trying
    every candidate factor's removal at each step means fitting several GLMs
    back to back: if all of those fitted objects were kept alive at once
    (e.g. in a list) while a fresh one is fit, peak memory use multiplies by
    however many candidates are in flight, which is enough to exhaust RAM on
    a modest machine. So each trial model here is reduced to the two numbers
    we actually need (its p-value and AIC) and then explicitly discarded
    (`del` + `gc.collect()`) before the next one is fit: at most two full
    `GLMResults` objects (the current model and one trial) are ever alive
    simultaneously.
    """
    remaining_groups = dict(feature_groups)
    current_results = fit_poisson_glm(X[_flatten(remaining_groups)], y, offset)
    history = []

    while len(remaining_groups) > 1:
        candidates = []  # only ever holds numbers, never a fitted model object
        for name, cols in remaining_groups.items():
            trial_groups = {k: v for k, v in remaining_groups.items() if k != name}
            trial_results = fit_poisson_glm(X[_flatten(trial_groups)], y, offset)
            lr = likelihood_ratio_test(current_results, trial_results)
            candidates.append({"factor": name, "p_value": lr["p_value"], "aic_without": trial_results.aic})
            del trial_results
            gc.collect()

        # The best candidate to drop is the one whose removal is *least*
        # damaging, i.e. the highest LR-test p-value.
        candidates.sort(key=lambda c: -c["p_value"])
        best = candidates[0]

        if best["p_value"] <= p_threshold:
            # Even the least useful remaining factor is still significant --
            # nothing more to drop.
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
        del current_results
        gc.collect()
        # Refit the winning (reduced) model once more as the new baseline,
        # rather than keeping the trial object from the loop above around --
        # a small extra cost in compute, in exchange for bounded memory.
        current_results = fit_poisson_glm(X[_flatten(remaining_groups)], y, offset)

    return current_results, remaining_groups, pd.DataFrame(history)


def relativities_table(results, baseline_note: str = "") -> pd.DataFrame:
    """
    Turns raw GLM coefficients into "relativities", exp(coefficient), the
    form an underwriter or pricing committee actually reads: a relativity of
    1.30 means "30% more expected claims than the baseline, holding every
    other factor fixed," which is a directly interpretable multiplier on the
    baseline frequency, unlike the raw log-scale coefficient.
    """
    import numpy as np

    table = pd.DataFrame(
        {
            "coefficient": results.params,
            "std_error": results.bse,
            "p_value": results.pvalues,
        }
    )
    table["relativity"] = np.exp(table["coefficient"])
    table["relativity_ci_low"] = np.exp(table["coefficient"] - 1.96 * table["std_error"])
    table["relativity_ci_high"] = np.exp(table["coefficient"] + 1.96 * table["std_error"])
    return table.sort_values("relativity", ascending=False)
