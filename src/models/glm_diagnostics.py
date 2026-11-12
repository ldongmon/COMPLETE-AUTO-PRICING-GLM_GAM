"""
glm_diagnostics.py
===================
Reusable diagnostic helpers for a fitted Poisson (or Gamma) GLM. Written to
be shared across Phase 4 (frequency GLM), Phase 5 (GAM comparison), and
Phase 6 (severity GLM): the same three questions get asked of every model
in this project:

  1. How much better than "no information" does the model do? (deviance,
     AIC, pseudo-R^2)
  2. Is model A meaningfully better than a simpler model B? (likelihood
     ratio test)
  3. When we rank policies by predicted risk, does the model's *ranking*
     actually track *observed* risk? (actual-vs-predicted by decile: the
     single most-trusted diagnostic in pricing practice, because it survives
     contact with "the mean fits but the tails don't")

Kept deliberately separate from `frequency_glm.py` (which is about *how to
fit* a model): this module is only about *how to check* a model, on any
GLM-family object statsmodels can produce.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def deviance_summary(results) -> dict:
    """
    The headline numbers for judging one fitted GLM in isolation.

    - `pseudo_r2_deviance`: 1 - (residual deviance / null deviance): the
      GLM analogue of R^2. 0 means "no better than an intercept-only model",
      1 means "perfect fit". For a rare-event frequency model, values in the
      low single-digit percent are normal and not a sign of a broken model.
      claim occurrence has a large irreducible random component.
    """
    return {
        "n_obs": int(results.nobs),
        "df_model": int(results.df_model),
        "df_resid": int(results.df_resid),
        "log_likelihood": results.llf,
        "deviance": results.deviance,
        "null_deviance": results.null_deviance,
        "aic": results.aic,
        "pseudo_r2_deviance": 1 - results.deviance / results.null_deviance,
    }


def likelihood_ratio_test(results_full, results_reduced) -> dict:
    """
    Tests whether the extra parameters in `results_full` (relative to
    `results_reduced`, a model with a subset of its columns) are jointly
    worth keeping. This is the standard, principled way to decide whether to
    drop a whole rating factor from a GLM: much more defensible than
    eyeballing individual dummy-variable p-values one at a time, which can
    mislead for a factor encoded as several one-hot columns.

    A *large* `p_value` (conventionally > 0.05) means "removing these
    columns did not significantly hurt the fit": i.e. it's safe to drop
    them; a *small* p_value means they're worth keeping.
    """
    lr_stat = 2 * (results_full.llf - results_reduced.llf)
    df_diff = results_full.df_model - results_reduced.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff) if df_diff > 0 else np.nan
    return {"lr_statistic": lr_stat, "df_diff": df_diff, "p_value": p_value}


def actual_vs_predicted_by_decile(
    y_true: np.ndarray,
    predicted_mean: np.ndarray,
    exposure: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    The classic pricing-model calibration check: rank policies by predicted
    risk, split them into `n_bins` groups of roughly equal *exposure* (not
    equal policy count: a handful of long-exposure policies shouldn't be
    allowed to dominate a bin), and compare observed vs. predicted
    exposure-weighted frequency in each group.

    A well-calibrated model shows observed closely tracking predicted in
    every bin, from the lowest-risk decile to the highest, not just "on
    average across the whole book."

    Parameters
    ----------
    y_true : observed claim counts.
    predicted_mean : the GLM's predicted claim COUNT per policy (statsmodels'
        `results.predict(...)`, i.e. mu = exp(X @ beta + log(exposure)).
        already scaled by exposure, not a per-year rate).
    exposure : policy exposure (years on risk), used both to rank-normalize
        the predicted rate for bin assignment and to weight the aggregation.
    """
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true),
            "predicted_mean": np.asarray(predicted_mean),
            "exposure": np.asarray(exposure),
        }
    )
    # Rank by predicted *rate* (count / exposure), not raw predicted count --
    # otherwise a long-exposure, low-risk policy could rank above a
    # short-exposure, high-risk one purely because it has a bigger expected
    # count in absolute terms.
    df["predicted_rate"] = df["predicted_mean"] / df["exposure"]
    df = df.sort_values("predicted_rate").reset_index(drop=True)

    df["cum_exposure"] = df["exposure"].cumsum()
    total_exposure = df["exposure"].sum()
    df["bin"] = np.ceil(df["cum_exposure"] / total_exposure * n_bins).clip(1, n_bins).astype(int)

    grouped = df.groupby("bin").agg(
        n_policies=("y", "size"),
        exposure=("exposure", "sum"),
        observed_claims=("y", "sum"),
        predicted_claims=("predicted_mean", "sum"),
    )
    grouped["observed_frequency"] = grouped["observed_claims"] / grouped["exposure"]
    grouped["predicted_frequency"] = grouped["predicted_claims"] / grouped["exposure"]
    return grouped.reset_index()


def plot_actual_vs_predicted(decile_df: pd.DataFrame, title: str = "Actual vs. predicted frequency, by predicted-risk decile") -> plt.Figure:
    """Bar-and-line chart: observed frequency (bars) vs. predicted frequency (line) per decile."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(decile_df["bin"], decile_df["observed_frequency"], color="#1f77b4", alpha=0.7, label="Observed")
    ax.plot(decile_df["bin"], decile_df["predicted_frequency"], color="#d62728", marker="o", linewidth=2, label="Predicted")
    ax.set_xlabel("Predicted-risk decile (1 = lowest risk, 10 = highest)")
    ax.set_ylabel("Claim frequency")
    ax.set_title(title)
    ax.set_xticks(decile_df["bin"])
    ax.legend()
    fig.tight_layout()
    return fig


def poisson_deviance(y_true: np.ndarray, predicted_mean: np.ndarray) -> float:
    """
    Total Poisson deviance for a set of predictions, computed directly from
    observed counts and predicted means: model-agnostic (works for a
    statsmodels GLM's `.predict()` output or a pygam GAM's `.predict()`
    output identically), which is exactly what's needed to compare the two
    out-of-sample in Phase 5.

    The `y * log(y / mu)` term is mathematically defined as 0 in the limit
    y -> 0 (by convention, since x*log(x) -> 0), so it's computed only where
    y > 0 rather than via a mask-after-the-fact, which would otherwise raise
    a spurious "divide by zero" warning from evaluating log(0).
    """
    y_true = np.asarray(y_true, dtype=float)
    predicted_mean = np.asarray(predicted_mean, dtype=float)
    term = np.zeros_like(y_true)
    positive = y_true > 0
    term[positive] = y_true[positive] * np.log(y_true[positive] / predicted_mean[positive])
    return float(2 * np.sum(term - (y_true - predicted_mean)))


def out_of_sample_pseudo_r2(y_true: np.ndarray, predicted_mean: np.ndarray, exposure: np.ndarray) -> dict:
    """
    Deviance-based pseudo-R^2, computed out-of-sample: 1 - (model deviance /
    null-model deviance), both evaluated on the SAME held-out data the model
    was not fit on. The null model here is "every policy gets the portfolio's
    overall exposure-weighted frequency": i.e. no risk factors at all.

    This is the fair way to compare two models that were fit on
    different-sized training sets (as the Phase 4 GLM and Phase 5 GAM are,
    for this project's documented memory reasons): in-sample pseudo-R^2
    isn't comparable across different n, but this out-of-sample version,
    computed identically for both models on the same validation set, is.
    """
    y_true = np.asarray(y_true, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    overall_rate = y_true.sum() / exposure.sum()
    null_predicted_mean = overall_rate * exposure

    model_deviance = poisson_deviance(y_true, predicted_mean)
    null_deviance = poisson_deviance(y_true, null_predicted_mean)
    return {
        "model_deviance": model_deviance,
        "null_deviance": null_deviance,
        "pseudo_r2": 1 - model_deviance / null_deviance,
    }


def gamma_deviance(y_true: np.ndarray, predicted_mean: np.ndarray, weights: np.ndarray | None = None) -> float:
    """
    Total Gamma deviance: the severity-model analogue of `poisson_deviance`
    above. For observation i: `2 * w_i * (-log(y_i / mu_i) + (y_i - mu_i) / mu_i)`.
    `weights` (typically `sev_claim_count`, see `severity_glm.py`) scale each
    term the same way `var_weights` does when fitting: an average of several
    claims contributes more to the deviance than an average of one, matching
    how the model itself was fit.
    """
    y_true = np.asarray(y_true, dtype=float)
    predicted_mean = np.asarray(predicted_mean, dtype=float)
    weights = np.ones_like(y_true) if weights is None else np.asarray(weights, dtype=float)
    return float(2 * np.sum(weights * (-np.log(y_true / predicted_mean) + (y_true - predicted_mean) / predicted_mean)))


def out_of_sample_pseudo_r2_gamma(
    y_true: np.ndarray, predicted_mean: np.ndarray, weights: np.ndarray | None = None
) -> dict:
    """
    Gamma-deviance-based out-of-sample pseudo-R^2: the severity-model
    analogue of `out_of_sample_pseudo_r2`. The null model is "every claim
    gets the (weighted) overall average severity," i.e. no risk factors.
    """
    y_true = np.asarray(y_true, dtype=float)
    weights = np.ones_like(y_true) if weights is None else np.asarray(weights, dtype=float)
    overall_mean = np.average(y_true, weights=weights)
    null_predicted_mean = np.full_like(y_true, overall_mean)

    model_deviance = gamma_deviance(y_true, predicted_mean, weights)
    null_deviance = gamma_deviance(y_true, null_predicted_mean, weights)
    return {
        "model_deviance": model_deviance,
        "null_deviance": null_deviance,
        "pseudo_r2": 1 - model_deviance / null_deviance,
    }


def calibration_by_decile(
    y_true: np.ndarray,
    predicted_mean: np.ndarray,
    weights: np.ndarray | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    The severity-model analogue of `actual_vs_predicted_by_decile`: rank
    observations by predicted value, split into `n_bins` groups of roughly
    equal total *weight* (not equal row count: a policy whose average
    severity comes from several claims should count for more, matching how
    the model itself was fit), and compare weighted-average observed vs.
    predicted value in each group.
    """
    weights_arr = np.ones_like(np.asarray(y_true, dtype=float)) if weights is None else np.asarray(weights, dtype=float)
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(predicted_mean, dtype=float),
            "weight": weights_arr,
        }
    )
    df = df.sort_values("predicted").reset_index(drop=True)
    df["cum_weight"] = df["weight"].cumsum()
    total_weight = df["weight"].sum()
    df["bin"] = np.ceil(df["cum_weight"] / total_weight * n_bins).clip(1, n_bins).astype(int)

    grouped = df.groupby("bin").apply(
        lambda g: pd.Series(
            {
                "n_rows": len(g),
                "total_weight": g["weight"].sum(),
                "observed": np.average(g["y"], weights=g["weight"]),
                "predicted": np.average(g["predicted"], weights=g["weight"]),
            }
        ),
        include_groups=False,
    )
    return grouped.reset_index()


def plot_calibration_by_decile(
    decile_df: pd.DataFrame, title: str = "Actual vs. predicted severity, by predicted-risk decile"
) -> plt.Figure:
    """Bar-and-line chart: observed (bars) vs. predicted (line) value per decile: severity's version of `plot_actual_vs_predicted`."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(decile_df["bin"], decile_df["observed"], color="#1f77b4", alpha=0.7, label="Observed")
    ax.plot(decile_df["bin"], decile_df["predicted"], color="#d62728", marker="o", linewidth=2, label="Predicted")
    ax.set_xlabel("Predicted-severity decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Average claim severity (EUR)")
    ax.set_title(title)
    ax.set_xticks(decile_df["bin"])
    ax.legend()
    fig.tight_layout()
    return fig


def pearson_residuals(y_true: np.ndarray, predicted_mean: np.ndarray) -> np.ndarray:
    """
    Pearson residual for a Poisson model: (observed - predicted) / sqrt(predicted).
    Standardizes residuals by their expected variance (mean = variance for a
    Poisson), so residuals should be roughly unit-scale and centered at zero
    for a well-specified model, at every level of predicted risk.
    """
    y_true = np.asarray(y_true)
    predicted_mean = np.asarray(predicted_mean)
    return (y_true - predicted_mean) / np.sqrt(predicted_mean)


def plot_residuals_vs_fitted(
    predicted_mean: np.ndarray,
    residuals: np.ndarray,
    n_bins: int = 20,
    title: str = "Mean Pearson residual vs. predicted claim count (binned)",
) -> plt.Figure:
    """
    A raw scatter of residuals is unreadable with hundreds of thousands of
    mostly-zero-claim policies: it is one dense smear. Instead, bin
    predicted values into `n_bins` groups and plot the *mean* residual per
    bin with its standard error: a well-specified model shows this hovering
    around zero everywhere, with no systematic trend (e.g. consistently
    positive at high predicted values, which would mean the model
    under-predicts its highest-risk policies).
    """
    df = pd.DataFrame({"predicted_mean": predicted_mean, "residual": residuals})
    df["bin"] = pd.qcut(df["predicted_mean"], q=n_bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True).agg(
        mean_predicted=("predicted_mean", "mean"),
        mean_residual=("residual", "mean"),
        se_residual=("residual", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(
        grouped["mean_predicted"], grouped["mean_residual"], yerr=1.96 * grouped["se_residual"],
        fmt="o", color="#1f77b4", ecolor="lightgray", capsize=3,
    )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Mean predicted claim count in bin")
    ax.set_ylabel("Mean Pearson residual (± 95% CI)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
