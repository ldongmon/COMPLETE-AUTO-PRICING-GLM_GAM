"""
frequency_glm.py
================
Phase 4: the frequency model, a Poisson GLM.

WHAT THE MODEL SAYS
-------------------
For policy i, the number of claims follows a Poisson law with mean mu_i:

    ClaimNb_i ~ Poisson(mu_i)

    log(mu_i) = log(Exposure_i) + b0 + b1 * x1_i + b2 * x2_i + ...

  - log(Exposure_i) is the OFFSET: its coefficient is fixed to 1, so the
    expected number of claims is proportional to the time insured;
  - b0 is the intercept: the log-frequency of the "reference" policy
    (every numeric factor at 0, every category at its reference level);
  - b1, b2, ... are the coefficients of the rating factors x1, x2, ...

Taking the exponential of both sides gives the multiplicative tariff:

    mu_i = Exposure_i * exp(b0) * exp(b1 * x1_i) * exp(b2 * x2_i) * ...

  - exp(b0) is the base frequency (claims per year);
  - exp(bj) is the RELATIVITY of factor j: the multiplier applied for one
    more unit of x_j (numeric factor) or for being in level j rather than the
    reference level (category column).

Example: if the coefficient of BonusMalus is 0.026, each extra point of
bonus-malus multiplies the expected frequency by exp(0.026) = 1.026 (+2.6%),
and 50 extra points multiply it by exp(50 * 0.026) = 3.67.

The coefficients are found by maximum likelihood (statsmodels does it).
"""

import gc  # "garbage collector": frees memory we no longer need

import numpy as np  # maths
import pandas as pd  # tables
import statsmodels.api as sm  # the library that fits GLMs

from src.models.glm_diagnostics import likelihood_ratio_test  # compares a model with and without a factor


def build_factor_groups(feature_names):
    """
    Group the model's columns by rating factor. A category such as Region is
    stored as many 0/1 columns (Region_R21, Region_R22, ...); to keep or drop
    "Region" we must keep or drop all of them together.
    """
    groups = {  # factor name -> list of its columns
        "DrivAge": [c for c in feature_names if c == "DrivAge"],
        "BonusMalus": [c for c in feature_names if c == "BonusMalus"],
        "VehPower": [c for c in feature_names if c == "VehPower"],
        "Density": [c for c in feature_names if c == "log_Density"],
        "VehAge": [c for c in feature_names if c == "VehAge_capped"],
        "VehBrand": [c for c in feature_names if c.startswith("VehBrand_")],
        "VehGas": [c for c in feature_names if c.startswith("VehGas_")],
        "Region": [c for c in feature_names if c.startswith("Region_")],
    }
    used = [c for cols in groups.values() for c in cols]  # every column placed in a group
    forgotten = [c for c in feature_names if c not in used]  # columns we did not place anywhere
    if len(forgotten) > 0:  # safety check
        raise ValueError(f"build_factor_groups does not know the columns {forgotten}")
    return {name: cols for name, cols in groups.items() if len(cols) > 0}  # drop factors absent from the data


def fit_poisson_glm(X, y, offset):
    """
    Fit the Poisson GLM  log(mu) = offset + b0 + X * b  by maximum likelihood.

    X      : table of rating factors (one column per coefficient)
    y      : number of claims of each policy
    offset : log(Exposure) of each policy
    """
    X_with_intercept = sm.add_constant(X, has_constant="add")  # add a column of 1s, whose coefficient is b0
    model = sm.GLM(  # describe the model
        y,  # what we predict: claim counts
        X_with_intercept,  # the explanatory columns
        family=sm.families.Poisson(),  # Poisson law, log link by default
        offset=offset,  # log(Exposure), coefficient fixed to 1
    )
    return model.fit()  # estimate the coefficients


def columns_of(groups):
    """All the columns of the factors in `groups`, as one list."""
    return [c for cols in groups.values() for c in cols]


def backward_stepwise_selection(X, y, offset, feature_groups, p_threshold=0.05, verbose=True):
    """
    Remove useless rating factors one at a time ("backward elimination").

    1. Start with every factor.
    2. For each factor, fit the model WITHOUT it and run a likelihood-ratio
       test: does removing it make the fit significantly worse?
    3. The factor whose removal hurts least (largest p-value) is dropped if its
       p-value is above 5%, i.e. if it does not significantly improve the model.
    4. Repeat until every remaining factor is significant.

    Returns the final fitted model, the factors kept, and a table of what was
    dropped at each step.

    Memory: each fitted model keeps a copy of the data. To stay within the
    computer's memory, we keep at most two fitted models at once and delete
    each trial model right after reading its p-value.
    """
    kept = dict(feature_groups)  # the factors still in the model
    current = fit_poisson_glm(X[columns_of(kept)], y, offset)  # model with every factor
    history = []  # one row per factor removed

    while len(kept) > 1:  # stop when a single factor is left
        trials = []  # results of "the model without factor F", for each F
        for name in kept:  # try removing each factor in turn
            without = {k: v for k, v in kept.items() if k != name}  # every factor except this one
            trial = fit_poisson_glm(X[columns_of(without)], y, offset)  # fit the smaller model
            test = likelihood_ratio_test(current, trial)  # is the factor useful?
            trials.append({"factor": name, "p_value": test["p_value"], "aic_without": trial.aic})
            del trial  # forget the trial model
            gc.collect()  # and free its memory now

        trials.sort(key=lambda t: t["p_value"], reverse=True)  # largest p-value first = least useful factor
        least_useful = trials[0]
        if least_useful["p_value"] <= p_threshold:  # even the least useful factor is significant
            break  # nothing more to remove

        history.append({  # record the removal
            "step": len(history) + 1,
            "factor_removed": least_useful["factor"],
            "lr_test_p_value": least_useful["p_value"],
            "aic_before": current.aic,
            "aic_after": least_useful["aic_without"],
        })
        if verbose:
            print(f"Step {len(history)}: drop '{least_useful['factor']}' (p = {least_useful['p_value']:.4f})")

        del kept[least_useful["factor"]]  # remove the factor
        del current  # forget the previous model
        gc.collect()
        current = fit_poisson_glm(X[columns_of(kept)], y, offset)  # refit without it

    return current, kept, pd.DataFrame(history)


def relativities_table(results, baseline_note=""):
    """
    Turn the GLM coefficients into relativities, the way a tariff is read.

        relativity = exp(coefficient)

    1.30 means "+30% of expected claims compared with the reference", all other
    factors being equal. The 95% confidence interval of the relativity is
    exp(coefficient -/+ 1.96 * standard error).
    """
    table = pd.DataFrame({
        "coefficient": results.params,  # the b's
        "std_error": results.bse,  # their standard errors
        "p_value": results.pvalues,  # is each coefficient significantly different from 0?
    })
    table["relativity"] = np.exp(table["coefficient"])  # multiplier on the frequency
    table["relativity_ci_low"] = np.exp(table["coefficient"] - 1.96 * table["std_error"])  # lower bound, 95%
    table["relativity_ci_high"] = np.exp(table["coefficient"] + 1.96 * table["std_error"])  # upper bound, 95%
    return table.sort_values("relativity", ascending=False)  # highest multipliers first
