"""
glm_diagnostics.py
==================
How good is a model? The checks used for every frequency and severity model.

1. DEVIANCE AND PSEUDO-R2
   The deviance measures the distance between what happened (y) and what the
   model predicted (mu). Smaller is better. For a Poisson model:

       deviance = 2 * sum( y * log(y / mu) - (y - mu) )      (y * log(y/mu) = 0 when y = 0)

   The "null model" predicts the same average frequency for everybody. The
   pseudo-R2 says how much of the null model's deviance our model removes:

       pseudo-R2 = 1 - deviance(model) / deviance(null model)

   0% = no better than an average, 100% = perfect. For claim counts, a few
   percent is normal: whether a given driver has an accident this year is
   mostly chance. What matters is that the model RANKS risks correctly.

2. ACTUAL VS. PREDICTED BY DECILE
   Sort the policies from the lowest to the highest predicted risk, cut them
   into 10 groups of equal exposure, and compare the observed and predicted
   frequency of each group. A good model has both rising together.

3. LIKELIHOOD-RATIO TEST
   Is a rating factor worth keeping? Compare the log-likelihood of the model
   with and without it (see likelihood_ratio_test).

4. PEARSON RESIDUALS
   (observed - predicted) / sqrt(predicted): should average about 0 everywhere.
"""

import matplotlib.pyplot as plt  # charts
import numpy as np  # maths
import pandas as pd  # tables
from scipy import stats  # statistical laws (chi-square here)


def deviance_summary(results):
    """The main numbers of a fitted statsmodels GLM, including its in-sample pseudo-R2."""
    return {
        "n_obs": int(results.nobs),  # number of rows used
        "df_model": int(results.df_model),  # number of coefficients (without the intercept)
        "df_resid": int(results.df_resid),  # rows minus coefficients
        "log_likelihood": results.llf,  # the quantity maximised by the fit
        "deviance": results.deviance,  # distance to the observations
        "null_deviance": results.null_deviance,  # same for the intercept-only model
        "aic": results.aic,  # fit quality penalised by the number of coefficients
        "pseudo_r2_deviance": 1 - results.deviance / results.null_deviance,  # share of deviance explained
    }


def likelihood_ratio_test(results_full, results_reduced):
    """
    Does the bigger model fit significantly better than the smaller one?

        LR statistic = 2 * (log-likelihood of the big model - log-likelihood of the small model)

    If the extra columns were useless, LR would follow a chi-square law with as
    many degrees of freedom as extra coefficients. A small p-value (< 5%) means
    the extra columns are useful; a large one means they can be dropped.
    """
    lr_stat = 2 * (results_full.llf - results_reduced.llf)  # gain in log-likelihood, doubled
    df_diff = results_full.df_model - results_reduced.df_model  # number of extra coefficients
    if df_diff > 0:
        p_value = stats.chi2.sf(lr_stat, df_diff)  # probability of such a gain by chance
    else:
        p_value = np.nan  # the models have the same size: no test possible
    return {"lr_statistic": lr_stat, "df_diff": df_diff, "p_value": p_value}


def actual_vs_predicted_by_decile(y_true, predicted_mean, exposure, n_bins=10):
    """
    Observed vs. predicted frequency for 10 groups of policies ranked by predicted risk.

    predicted_mean is the predicted NUMBER of claims of each policy over its own
    exposure; predicted_mean / exposure is its predicted frequency per year.
    The groups have equal total exposure (not equal numbers of policies).
    """
    table = pd.DataFrame({
        "y": np.asarray(y_true),  # observed claims
        "predicted_mean": np.asarray(predicted_mean),  # predicted claims
        "exposure": np.asarray(exposure),  # years insured
    })
    table["predicted_rate"] = table["predicted_mean"] / table["exposure"]  # predicted claims per year
    table = table.sort_values("predicted_rate").reset_index(drop=True)  # lowest risk first
    share_of_exposure = table["exposure"].cumsum() / table["exposure"].sum()  # running share of exposure, 0 to 1
    table["bin"] = np.ceil(share_of_exposure * n_bins).clip(1, n_bins).astype(int)  # group 1 to 10

    result = table.groupby("bin").agg(  # one row per group
        n_policies=("y", "size"),  # number of policies
        exposure=("exposure", "sum"),  # years insured
        observed_claims=("y", "sum"),  # claims that happened
        predicted_claims=("predicted_mean", "sum"),  # claims the model expected
    )
    result["observed_frequency"] = result["observed_claims"] / result["exposure"]  # observed claims per year
    result["predicted_frequency"] = result["predicted_claims"] / result["exposure"]  # predicted claims per year
    return result.reset_index()


def plot_actual_vs_predicted(decile_df, title="Actual vs. predicted frequency, by predicted-risk decile"):
    """Bars = observed frequency, red line = predicted frequency, for each decile."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(decile_df["bin"], decile_df["observed_frequency"], color="#1f77b4", alpha=0.7, label="Observed")  # blue bars
    ax.plot(decile_df["bin"], decile_df["predicted_frequency"], color="#d62728", marker="o", linewidth=2, label="Predicted")  # red line
    ax.set_xlabel("Predicted-risk decile (1 = lowest risk, 10 = highest)")
    ax.set_ylabel("Claim frequency")
    ax.set_title(title)
    ax.set_xticks(decile_df["bin"])
    ax.legend()
    fig.tight_layout()
    return fig


def poisson_deviance(y_true, predicted_mean):
    """
    Poisson deviance = 2 * sum( y * log(y / mu) - (y - mu) ).
    The term y * log(y / mu) is taken as 0 when y = 0 (its limit).
    Works for any model's predictions (GLM or GAM).
    """
    y = np.asarray(y_true, dtype=float)  # observed claims
    mu = np.asarray(predicted_mean, dtype=float)  # predicted claims
    first_term = np.zeros_like(y)  # y * log(y / mu), 0 by default
    has_claims = y > 0  # only where y > 0 ...
    first_term[has_claims] = y[has_claims] * np.log(y[has_claims] / mu[has_claims])  # ... compute it
    return float(2 * np.sum(first_term - (y - mu)))


def out_of_sample_pseudo_r2(y_true, predicted_mean, exposure):
    """
    Pseudo-R2 on data the model has not seen (validation or test).

    Null model: every policy gets the average frequency of this data set,
    times its exposure. This is the fair way to compare two models, even when
    they were trained on different amounts of data.
    """
    y = np.asarray(y_true, dtype=float)  # observed claims
    exposure = np.asarray(exposure, dtype=float)  # years insured
    average_frequency = y.sum() / exposure.sum()  # claims per year over the whole set
    null_predicted = average_frequency * exposure  # null model: same frequency for everybody
    model_deviance = poisson_deviance(y, predicted_mean)  # our model
    null_deviance = poisson_deviance(y, null_predicted)  # the null model
    return {
        "model_deviance": model_deviance,
        "null_deviance": null_deviance,
        "pseudo_r2": 1 - model_deviance / null_deviance,  # share of the null deviance removed by the model
    }


def gamma_deviance(y_true, predicted_mean, weights=None):
    """
    Gamma deviance (for the severity models):
        2 * sum( w * ( -log(y / mu) + (y - mu) / mu ) )
    w = number of claims behind each average amount (1 if not given).
    """
    y = np.asarray(y_true, dtype=float)  # observed average amounts
    mu = np.asarray(predicted_mean, dtype=float)  # predicted average amounts
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)  # weights
    return float(2 * np.sum(w * (-np.log(y / mu) + (y - mu) / mu)))


def out_of_sample_pseudo_r2_gamma(y_true, predicted_mean, weights=None):
    """Severity version of the pseudo-R2: null model = the (weighted) average amount for everybody."""
    y = np.asarray(y_true, dtype=float)  # observed average amounts
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)  # weights
    null_predicted = np.full_like(y, np.average(y, weights=w))  # the same average for every claim
    model_deviance = gamma_deviance(y, predicted_mean, w)
    null_deviance = gamma_deviance(y, null_predicted, w)
    return {
        "model_deviance": model_deviance,
        "null_deviance": null_deviance,
        "pseudo_r2": 1 - model_deviance / null_deviance,
    }


def calibration_by_decile(y_true, predicted_mean, weights=None, n_bins=10):
    """
    Observed vs. predicted (weighted averages) for 10 groups ranked by the
    prediction, each group carrying the same total weight. Used for severity
    and for the technical premium.
    """
    y = np.asarray(y_true, dtype=float)
    table = pd.DataFrame({
        "y": y,  # observed values
        "predicted": np.asarray(predicted_mean, dtype=float),  # predicted values
        "weight": np.ones_like(y) if weights is None else np.asarray(weights, dtype=float),  # weights
    })
    table = table.sort_values("predicted").reset_index(drop=True)  # lowest prediction first
    share_of_weight = table["weight"].cumsum() / table["weight"].sum()  # running share of the weight
    table["bin"] = np.ceil(share_of_weight * n_bins).clip(1, n_bins).astype(int)  # group 1 to 10

    rows = []  # one row per group
    for bin_number, group in table.groupby("bin"):
        rows.append({
            "bin": bin_number,
            "n_rows": len(group),  # number of rows in the group
            "total_weight": group["weight"].sum(),  # total weight
            "observed": np.average(group["y"], weights=group["weight"]),  # weighted average of what happened
            "predicted": np.average(group["predicted"], weights=group["weight"]),  # weighted average of what was predicted
        })
    return pd.DataFrame(rows)


def plot_calibration_by_decile(decile_df, title="Actual vs. predicted severity, by predicted-risk decile"):
    """Bars = observed, red line = predicted, for each decile."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(decile_df["bin"], decile_df["observed"], color="#1f77b4", alpha=0.7, label="Observed")
    ax.plot(decile_df["bin"], decile_df["predicted"], color="#d62728", marker="o", linewidth=2, label="Predicted")
    ax.set_xlabel("Predicted decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Average value (EUR)")
    ax.set_title(title)
    ax.set_xticks(decile_df["bin"])
    ax.legend()
    fig.tight_layout()
    return fig


def pearson_residuals(y_true, predicted_mean):
    """Pearson residual of a Poisson model: (observed - predicted) / sqrt(predicted)."""
    y = np.asarray(y_true)
    mu = np.asarray(predicted_mean)
    return (y - mu) / np.sqrt(mu)


def plot_residuals_vs_fitted(predicted_mean, residuals, n_bins=20, title="Mean Pearson residual vs. predicted claim count (binned)"):
    """
    Average residual for 20 groups of policies with similar predictions, with a
    95% interval. A well-specified model stays around 0 everywhere.
    """
    table = pd.DataFrame({"predicted_mean": predicted_mean, "residual": residuals})
    table["bin"] = pd.qcut(table["predicted_mean"], q=n_bins, duplicates="drop")  # 20 groups of equal size
    grouped = table.groupby("bin", observed=True).agg(
        mean_predicted=("predicted_mean", "mean"),  # average prediction of the group
        mean_residual=("residual", "mean"),  # average residual of the group
        se_residual=("residual", lambda s: s.std(ddof=1) / np.sqrt(len(s))),  # standard error of that average
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(grouped["mean_predicted"], grouped["mean_residual"], yerr=1.96 * grouped["se_residual"],
                fmt="o", color="#1f77b4", ecolor="lightgray", capsize=3)  # dots with 95% bars
    ax.axhline(0, color="black", linewidth=1, linestyle="--")  # the zero line
    ax.set_xlabel("Mean predicted claim count in bin")
    ax.set_ylabel("Mean Pearson residual (+/- 95% CI)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
