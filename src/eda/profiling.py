"""
profiling.py
============
Reusable exposure-weighted profiling helpers for rating-factor EDA.

THE ONE RULE THIS MODULE ENFORCES
----------------------------------
In insurance data, **never average claim frequency row-by-row**. Each row
(policy) carries a different `Exposure` (time on risk), so a policy on risk
for 1 month and a policy on risk for 12 months must not count equally toward
an average frequency. The correct estimator of claim frequency for a group of
policies is:

    frequency = sum(claim count) / sum(exposure)

not `mean(ClaimNb)` and not `mean(ClaimNb / Exposure)` (the latter blows up
for the many policies with tiny exposure). This is *the* most common mistake
in a first pass at motor pricing EDA, and getting it right: consistently,
via one shared function instead of ad hoc code in every notebook cell: is
worth calling out explicitly.

Severity has no such subtlety (it is not a rate), but it does have a
"conditional on a claim" subtlety: severity should only be averaged over
policies/claims that actually had a payment, using `sev_claim_amount` and
`sev_claim_count` from the Phase 1 processed table, not over the whole book.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def weighted_frequency_by_factor(
    df: pd.DataFrame,
    factor: str,
    claim_col: str = "ClaimNb_freq_target",
    exposure_col: str = "Exposure",
) -> pd.DataFrame:
    """
    Exposure-weighted claim frequency, grouped by `factor`.

    Returns one row per level of `factor` with:
      - exposure       : total policy-years on risk in this group
      - claims         : total claim count in this group
      - frequency      : claims / exposure  (claims per policy-year)
      - exposure_share : this group's share of total portfolio exposure
                          (needed to judge how much weight a level actually
                          carries: a dramatic frequency for a 0.1%-exposure
                          level is a very different story than for a 20% one)
    """
    grouped = df.groupby(factor, observed=True).agg(
        exposure=(exposure_col, "sum"),
        claims=(claim_col, "sum"),
    )
    grouped["frequency"] = grouped["claims"] / grouped["exposure"]
    grouped["exposure_share"] = grouped["exposure"] / df[exposure_col].sum()
    return grouped.reset_index()


def severity_by_factor(
    df: pd.DataFrame,
    factor: str,
    amount_col: str = "sev_claim_amount",
    count_col: str = "sev_claim_count",
    has_claim_col: str = "HasPaidClaim",
) -> pd.DataFrame:
    """
    Average paid severity, grouped by `factor`, computed only over policies
    with at least one paid claim (`has_claim_col`). Averaging over the whole
    book (including zero-claim policies, whose severity is undefined) would
    silently understate severity for no actuarial reason.

    Returns one row per level of `factor` with:
      - n_claims     : total number of paid claims in this group
      - total_amount : total paid amount in this group
      - avg_severity : total_amount / n_claims
    """
    claims_only = df.loc[df[has_claim_col]]
    grouped = claims_only.groupby(factor, observed=True).agg(
        n_claims=(count_col, "sum"),
        total_amount=(amount_col, "sum"),
    )
    grouped["avg_severity"] = grouped["total_amount"] / grouped["n_claims"]
    return grouped.reset_index()


def bin_continuous(
    series: pd.Series,
    bins: Sequence[float],
    labels: Optional[Sequence[str]] = None,
) -> pd.Series:
    """
    Bin a continuous rating factor (e.g. DrivAge, VehAge, Density) into
    intervals for profiling. `bins` follows `pandas.cut` convention (edges).
    Grouping continuous factors before profiling avoids a chart with one bar
    per unique value (e.g. 100+ distinct driver ages), which is unreadable
    and statistically noisy at the tails where exposure is thin.
    """
    return pd.cut(series, bins=bins, labels=labels, right=False)


def plot_frequency_severity_profile(
    freq_df: pd.DataFrame,
    sev_df: pd.DataFrame,
    factor: str,
    figsize: tuple = (11, 4),
) -> plt.Figure:
    """
    Standard two-panel profiling chart used throughout Phase 2:
      - Left panel : exposure-weighted claim frequency (bars) with exposure
                     share overlaid as a translucent bar in the background,
                     so a spike backed by 0.2% of the book reads differently
                     from one backed by 20% of it.
      - Right panel: average paid severity (bars).

    Both panels share the same category order (`freq_df[factor]`) so the two
    halves of the story, "how often" and "how much", line up visually.
    """
    fig, (ax_freq, ax_sev) = plt.subplots(1, 2, figsize=figsize)

    categories = freq_df[factor].astype(str)

    # Left panel: frequency, with exposure share as context.
    ax_exposure = ax_freq.twinx()
    ax_exposure.bar(categories, freq_df["exposure_share"], color="lightgray", alpha=0.6, zorder=1)
    ax_exposure.set_ylabel("Exposure share", color="gray")
    ax_exposure.set_ylim(0, max(freq_df["exposure_share"].max() * 3, 0.05))
    ax_exposure.tick_params(axis="y", colors="gray")

    ax_freq.plot(categories, freq_df["frequency"], marker="o", color="#1f77b4", zorder=3)
    ax_freq.set_ylabel("Claim frequency (claims / policy-year)")
    ax_freq.set_xlabel(factor)
    ax_freq.set_title(f"Frequency by {factor}")
    ax_freq.tick_params(axis="x", rotation=45)
    ax_freq.set_zorder(ax_exposure.get_zorder() + 1)
    ax_freq.patch.set_visible(False)

    # Right panel: severity.
    sev_aligned = sev_df.set_index(factor).reindex(freq_df[factor]).reset_index()
    ax_sev.bar(sev_aligned[factor].astype(str), sev_aligned["avg_severity"], color="#d62728", alpha=0.85)
    ax_sev.set_ylabel("Average severity (EUR)")
    ax_sev.set_xlabel(factor)
    ax_sev.set_title(f"Average severity by {factor}")
    ax_sev.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig
