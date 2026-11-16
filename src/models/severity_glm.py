"""
severity_glm.py
===============
Phase 6: the severity model, how much a claim costs once it has happened.

TWO KINDS OF CLAIMS, TWO LAYERS
-------------------------------
Most claims cost a few hundred or a few thousand euros. A few cost hundreds of
thousands or millions (serious bodily injury). Those giants are rare and
unpredictable: no rating factor can say WHICH driver will cause one. So each
claim is cut in two at a threshold (the 99th percentile of the training
claims, EUR 15,000 here):

    attritional part = min(claim, threshold)        modelled policy by policy (Gamma GLM)
    large-loss part  = max(claim - threshold, 0)    shared by everybody (flat loading)

Example: a claim of EUR 40,000 counts EUR 15,000 in the attritional layer and
EUR 25,000 in the large-loss layer.

The large-loss part is MUTUALISED: its total over the training set is divided
by the total exposure of the training set, giving one flat amount per year
insured, added to every policy's premium:

    large-loss loading = sum of the large-loss parts / sum of the exposures

This is the burning-cost logic of an excess-of-loss reinsurance layer.

THE GAMMA GLM FOR THE ATTRITIONAL LAYER
---------------------------------------
For a policy with at least one paid claim, the average capped amount per
claim is modelled with a Gamma law (positive, skewed amounts) and a log link:

    log(average amount) = c0 + c1 * x1 + c2 * x2 + ...

so  average amount = exp(c0) * exp(c1 * x1) * ...  (multiplicative, like the frequency GLM).

Weights: a policy's average built from 3 claims is more reliable than an
average of 1 claim, so each policy is weighted by its number of paid claims
(statsmodels `var_weights`).
"""

import gc  # frees memory between fits

import pandas as pd  # tables
import statsmodels.api as sm  # GLM library

DEFAULT_LARGE_LOSS_PERCENTILE = 0.99  # the threshold is the 99th percentile of training amounts


def determine_large_loss_threshold(train_severity_df, percentile=DEFAULT_LARGE_LOSS_PERCENTILE, amount_col="AvgClaimAmount"):
    """The amount above which a claim is 'large': 99th percentile of the TRAIN claims only."""
    return float(train_severity_df[amount_col].quantile(percentile))


def cap_severity(severity_df, threshold, amount_col="AvgClaimAmount"):
    """The attritional part of each average amount: min(amount, threshold)."""
    return severity_df[amount_col].clip(upper=threshold)


def large_loss_loading(train_full_df, train_severity_df, threshold):
    """
    Flat amount per year insured that pays for the large-loss layer.

        large-loss part of a policy = total paid - min(average, threshold) x number of claims
        loading = sum of the large-loss parts / total exposure of ALL train policies
    """
    capped_total = cap_severity(train_severity_df, threshold) * train_severity_df["sev_claim_count"]  # attritional amount of each policy
    large_part = (train_severity_df["sev_claim_amount"] - capped_total).clip(lower=0)  # what is above the threshold
    return float(large_part.sum() / train_full_df["Exposure"].sum())  # spread over every year insured


def fit_gamma_glm(X, y, weights=None):
    """Fit log(average capped amount) = c0 + X * c with a Gamma law, weighted by the number of claims."""
    X_with_intercept = sm.add_constant(X, has_constant="add")  # column of 1s for the intercept c0
    model = sm.GLM(
        y,  # average capped amount per claim
        X_with_intercept,
        family=sm.families.Gamma(link=sm.families.links.Log()),  # Gamma law, log link
        var_weights=weights,  # number of claims behind each average
    )
    return model.fit()


def columns_of(groups):
    """All the columns of the factors in `groups`, as one list."""
    return [c for cols in groups.values() for c in cols]


def backward_stepwise_selection_gamma(X, y, weights, feature_groups, p_threshold=0.05, verbose=True):
    """
    Same backward elimination as for frequency (see frequency_glm.py): drop
    the least useful factor while its likelihood-ratio p-value is above 5%.
    """
    from src.models.glm_diagnostics import likelihood_ratio_test  # imported here to avoid a circular import

    kept = dict(feature_groups)  # factors still in the model
    current = fit_gamma_glm(X[columns_of(kept)], y, weights)  # model with every factor
    history = []  # one row per factor removed

    while len(kept) > 1:
        trials = []
        for name in kept:  # try removing each factor
            without = {k: v for k, v in kept.items() if k != name}
            trial = fit_gamma_glm(X[columns_of(without)], y, weights)
            test = likelihood_ratio_test(current, trial)
            trials.append({"factor": name, "p_value": test["p_value"], "aic_without": trial.aic})
            del trial
            gc.collect()
        trials.sort(key=lambda t: t["p_value"], reverse=True)  # least useful factor first
        least_useful = trials[0]
        if least_useful["p_value"] <= p_threshold:  # every remaining factor is significant
            break
        history.append({
            "step": len(history) + 1,
            "factor_removed": least_useful["factor"],
            "lr_test_p_value": least_useful["p_value"],
            "aic_before": current.aic,
            "aic_after": least_useful["aic_without"],
        })
        if verbose:
            print(f"Step {len(history)}: drop '{least_useful['factor']}' (p = {least_useful['p_value']:.4f})")
        del kept[least_useful["factor"]]
        current = fit_gamma_glm(X[columns_of(kept)], y, weights)  # refit without it

    return current, kept, pd.DataFrame(history)
