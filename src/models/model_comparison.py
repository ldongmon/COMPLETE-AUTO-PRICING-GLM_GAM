"""
model_comparison.py
====================
Phase 5: tools for comparing two competing frequency models (here: the
Phase 4 GLM vs. a GAM) on equal footing, out-of-sample.

WHY THESE METRICS, AND WHY ON THE VALIDATION SET
---------------------------------------------------
`src/models/glm_diagnostics.py` already has `deviance_summary` (in-sample fit
quality) and `actual_vs_predicted_by_decile` (calibration). Comparing two
models' *in-sample* AIC/deviance is only valid when they were fit on the same
data: which, in this project, they are not always: compute/memory
constraints on this machine mean the GAM in Phase 5 is fit on a stratified
sub-sample while the GLM (Phase 4) is fit on the full training set (see the
Phase 5 notebook for the full account). In-sample likelihood is not
comparable across different sample sizes, so every comparison in this module
is deliberately computed **out-of-sample, on the untouched validation set**,
where sample size is identical for both models and the comparison is fair.

- `gini_lorenz`: the standard actuarial discrimination metric: does ranking
  policies by predicted risk actually separate low-risk from high-risk
  experience? Higher is better; 0 means no better than random ranking.
- `lorenz_curve_data` / `plot_lorenz_curves`: the curve behind the Gini
  number, plotted for one or more models at once for a direct visual
  comparison.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def lorenz_curve_data(y_true: np.ndarray, predicted_mean: np.ndarray, exposure: np.ndarray) -> pd.DataFrame:
    """
    Builds the ordered-Lorenz-curve points for a frequency model.

    Method: sort policies by predicted *risk* (predicted count / exposure)
    from lowest to highest, then walk through them accumulating exposure and
    observed claims. A model with real discriminatory power puts a
    disproportionately small share of observed claims in the
    low-predicted-risk end of the walk: the curve bows below the diagonal.

    Returns a DataFrame with columns `cum_exposure_share` and
    `cum_claims_share`, both running from 0 to 1, ready to plot or to
    integrate into a Gini coefficient.
    """
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true, dtype=float),
            "predicted_mean": np.asarray(predicted_mean, dtype=float),
            "exposure": np.asarray(exposure, dtype=float),
        }
    )
    df["predicted_rate"] = df["predicted_mean"] / df["exposure"]
    df = df.sort_values("predicted_rate").reset_index(drop=True)

    df["cum_exposure_share"] = df["exposure"].cumsum() / df["exposure"].sum()
    df["cum_claims_share"] = df["y"].cumsum() / df["y"].sum()

    # Prepend the (0, 0) origin point so the curve/integral are well-defined.
    origin = pd.DataFrame({"cum_exposure_share": [0.0], "cum_claims_share": [0.0]})
    return pd.concat([origin, df[["cum_exposure_share", "cum_claims_share"]]], ignore_index=True)


def gini_coefficient(lorenz_df: pd.DataFrame) -> float:
    """
    Normalized Gini from ordered-Lorenz-curve points (`lorenz_curve_data`):
    twice the area between the diagonal (no discriminatory power) and the
    curve. Ranges roughly 0 (random ranking) to 1 (perfect separation);
    real-world frequency models for motor insurance typically land in the
    0.2-0.4 range: this is a ranking-quality metric, not the same scale as
    the deviance-based pseudo-R^2.
    """
    x = lorenz_df["cum_exposure_share"].to_numpy()
    y = lorenz_df["cum_claims_share"].to_numpy()
    area_under_curve = np.trapezoid(y, x)  # trapezoidal integration
    return 1 - 2 * area_under_curve


def plot_lorenz_curves(
    curves: dict[str, pd.DataFrame],
    ginis: dict[str, float],
    ylabel: str = "Cumulative share of observed claims",
    title: str = "Ordered Lorenz curve: frequency model discrimination",
) -> plt.Figure:
    """
    Overlay one or more models' Lorenz curves plus the diagonal
    (no-discrimination) reference line, for a direct visual comparison.

    `curves` and `ginis` are keyed by model name, e.g. {"GLM": ..., "GAM": ...}.
    `ylabel`/`title` default to the frequency-model wording this function was
    first built for (Phase 5); pass e.g. `ylabel="Cumulative share of
    observed claim cost"` when reusing it for a monetary quantity like the
    Phase 7 technical premium, so the axis actually describes what's plotted.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="No discrimination (Gini=0)")
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    for (name, curve), color in zip(curves.items(), colors):
        ax.plot(
            curve["cum_exposure_share"], curve["cum_claims_share"],
            color=color, label=f"{name} (Gini={ginis[name]:.3f})",
        )
    ax.set_xlabel("Cumulative share of exposure (policies ranked low -> high predicted risk)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
