"""
model_comparison.py
===================
The Lorenz curve and the Gini coefficient: does the model RANK the risks well?

How the Lorenz curve is built:
  1. sort the policies from the lowest to the highest predicted risk;
  2. walk through them and record, at each step, the share of exposure seen
     so far (x axis) and the share of claims seen so far (y axis).

If the model were useless, the first 50% of exposure would contain 50% of the
claims: the curve would follow the diagonal. A good model puts few claims in
the low-risk half: the curve sags below the diagonal.

    Gini = 1 - 2 * (area under the curve)

0 = random ranking, larger = better ranking. Motor frequency models usually
land between 0.2 and 0.4.
"""

import matplotlib.pyplot as plt  # charts
import numpy as np  # maths
import pandas as pd  # tables


def lorenz_curve_data(y_true, predicted_mean, exposure):
    """Points of the Lorenz curve: running share of exposure and of observed claims."""
    table = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),  # observed claims (or claim cost)
        "predicted_mean": np.asarray(predicted_mean, dtype=float),  # predicted claims (or cost)
        "exposure": np.asarray(exposure, dtype=float),  # years insured
    })
    table["predicted_rate"] = table["predicted_mean"] / table["exposure"]  # predicted risk per year
    table = table.sort_values("predicted_rate").reset_index(drop=True)  # lowest risk first
    table["cum_exposure_share"] = table["exposure"].cumsum() / table["exposure"].sum()  # x: share of exposure so far
    table["cum_claims_share"] = table["y"].cumsum() / table["y"].sum()  # y: share of claims so far
    start = pd.DataFrame({"cum_exposure_share": [0.0], "cum_claims_share": [0.0]})  # the curve starts at (0, 0)
    return pd.concat([start, table[["cum_exposure_share", "cum_claims_share"]]], ignore_index=True)


def gini_coefficient(lorenz_df):
    """Gini = 1 - 2 * area under the Lorenz curve (area computed with trapezoids)."""
    x = lorenz_df["cum_exposure_share"].to_numpy()
    y = lorenz_df["cum_claims_share"].to_numpy()
    area = np.trapezoid(y, x)  # area under the curve
    return 1 - 2 * area


def plot_lorenz_curves(curves, ginis, ylabel="Cumulative share of observed claims", title="Ordered Lorenz curve: frequency model discrimination"):
    """Draw one Lorenz curve per model, plus the diagonal (no ranking power)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="No discrimination (Gini=0)")  # the diagonal
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]  # one colour per model
    for (name, curve), color in zip(curves.items(), colors):  # each model in turn
        ax.plot(curve["cum_exposure_share"], curve["cum_claims_share"], color=color, label=f"{name} (Gini={ginis[name]:.3f})")
    ax.set_xlabel("Cumulative share of exposure (policies ranked low -> high predicted risk)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
