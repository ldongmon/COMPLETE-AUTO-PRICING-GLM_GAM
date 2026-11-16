"""
profiling.py
============
Small helpers for the exploratory analysis (Phase 2): claim frequency and
average claim cost, group by group.

THE ONE RULE OF THIS MODULE: weight by exposure
-----------------------------------------------
Each policy was insured for a different length of time (its Exposure, in
years). A policy insured two weeks had almost no time to have a claim, so it
must not count as much as a policy insured all year. The claim frequency of a
group is therefore:

    frequency = total number of claims / total exposure (years insured)

and NOT the plain average of the claim counts.

Example with two policies:
    policy A: 1 claim, insured 1.0 year
    policy B: 0 claim, insured 0.1 year
    frequency     = (1 + 0) / (1.0 + 0.1) = 0.91 claim per year insured
    plain average = (1 + 0) / 2           = 0.50 claim per policy  (wrong question)

For the average claim COST (severity) we only look at policies that had a
paid claim: a policy without a claim has no claim cost.
"""

import matplotlib.pyplot as plt  # charts
import pandas as pd  # tables


def weighted_frequency_by_factor(df, factor, claim_col="ClaimNb_freq_target", exposure_col="Exposure"):
    """
    Claim frequency for each level of `factor` (e.g. each Area, each Region).

    Returns one row per level with:
      exposure        total years insured in the group
      claims          total number of claims in the group
      frequency       claims / exposure  (claims per year insured)
      exposure_share  the group's share of all years insured (its weight in the portfolio)
    """
    table = df.groupby(factor, observed=True).agg(  # one row per level of the factor
        exposure=(exposure_col, "sum"),  # add up the years insured
        claims=(claim_col, "sum"),  # add up the claims
    )
    table["frequency"] = table["claims"] / table["exposure"]  # claims per year insured
    table["exposure_share"] = table["exposure"] / df[exposure_col].sum()  # share of the whole portfolio
    return table.reset_index()  # turn the group labels back into a normal column


def severity_by_factor(df, factor, amount_col="sev_claim_amount", count_col="sev_claim_count", has_claim_col="HasPaidClaim"):
    """
    Average paid amount per claim for each level of `factor`.

    Only policies with at least one paid claim are used.

    Returns one row per level with:
      n_claims      number of paid claims in the group
      total_amount  total amount paid in the group (EUR)
      avg_severity  total_amount / n_claims  (EUR per claim)
    """
    with_claim = df[df[has_claim_col]]  # keep only the policies with a paid claim
    table = with_claim.groupby(factor, observed=True).agg(  # one row per level of the factor
        n_claims=(count_col, "sum"),  # add up the number of paid claims
        total_amount=(amount_col, "sum"),  # add up the amounts paid
    )
    table["avg_severity"] = table["total_amount"] / table["n_claims"]  # average cost of one claim
    return table.reset_index()


def bin_continuous(series, bins, labels=None):
    """
    Cut a continuous variable (age, density...) into bands, for example ages
    18-22, 23-25, ... A chart with one bar per single age would be unreadable.
    `bins` are the band edges; a value equal to an edge goes into the band that
    starts at that edge (right=False).
    """
    return pd.cut(series, bins=bins, labels=labels, right=False)


def plot_frequency_severity_profile(freq_df, sev_df, factor, figsize=(11, 4)):
    """
    The standard two-panel chart of Phase 2:
      left : claim frequency by level (blue line), with each level's exposure
             share in grey bars behind it (a high frequency on 0.1% of the
             portfolio means much less than on 20% of it);
      right: average claim cost by level (red bars).
    """
    fig, (ax_freq, ax_sev) = plt.subplots(1, 2, figsize=figsize)  # one figure, two panels side by side
    levels = freq_df[factor].astype(str)  # the level names, as text for the x axis

    # Left panel, background: grey bars for the exposure share.
    ax_share = ax_freq.twinx()  # a second y axis on the right of the left panel
    ax_share.bar(levels, freq_df["exposure_share"], color="lightgray", alpha=0.6, zorder=1)  # grey bars
    ax_share.set_ylabel("Exposure share", color="gray")  # label of the right axis
    ax_share.set_ylim(0, max(freq_df["exposure_share"].max() * 3, 0.05))  # keep the bars in the lower part of the panel
    ax_share.tick_params(axis="y", colors="gray")  # grey tick labels for the right axis

    # Left panel, foreground: the frequency line.
    ax_freq.plot(levels, freq_df["frequency"], marker="o", color="#1f77b4", zorder=3)  # blue line with dots
    ax_freq.set_ylabel("Claim frequency (claims / policy-year)")  # label of the left axis
    ax_freq.set_xlabel(factor)  # name of the variable under the x axis
    ax_freq.set_title(f"Frequency by {factor}")  # title of the panel
    ax_freq.tick_params(axis="x", rotation=45)  # tilt the level names so they do not overlap
    ax_freq.set_zorder(ax_share.get_zorder() + 1)  # draw the line in front of the grey bars
    ax_freq.patch.set_visible(False)  # transparent background so the bars stay visible

    # Right panel: average claim cost, in the same level order as the left panel.
    sev_in_order = sev_df.set_index(factor).reindex(freq_df[factor]).reset_index()  # same order as the frequency table
    ax_sev.bar(sev_in_order[factor].astype(str), sev_in_order["avg_severity"], color="#d62728", alpha=0.85)  # red bars
    ax_sev.set_ylabel("Average severity (EUR)")  # label of the y axis
    ax_sev.set_xlabel(factor)  # name of the variable
    ax_sev.set_title(f"Average severity by {factor}")  # title of the panel
    ax_sev.tick_params(axis="x", rotation=45)  # tilt the level names

    fig.tight_layout()  # avoid overlapping labels
    return fig
