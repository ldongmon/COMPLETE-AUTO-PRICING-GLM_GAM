"""
pipeline.py
===========
Phase 3: turns the clean policy-level table from Phase 1 into what a GLM/GAM
actually needs: a train/validation/test split with no leakage, and a design
matrix (X) built by a rule that is learned once on the training set and then
applied, unchanged, to every other set (validation, test, and, later, brand
new policies scored by the app).

WHY THIS IS ITS OWN MODULE, WRITTEN AS PLAIN, EXPLICIT CODE
-------------------------------------------------------------
It would be shorter to wrap everything in an `sklearn.Pipeline` +
`ColumnTransformer`. Plain code is used instead, so that it stays easy to
read and check: a `ColumnTransformer`
hides exactly the two things that matter most here: "what gets capped, at
what threshold, learned from which data" and "why exposure never appears in
X." Explicit code you can read top to bottom beats a shorter black box.

THE TWO RULES THIS MODULE ENFORCES
------------------------------------
1. **No leakage.** Every parameter used to transform a row (a capping
   threshold, the set of known categories) is computed from the *training
   set only*, stored on a fitted `PolicyFeatureEncoder`, and then applied
   identically to validation/test/new data. `.fit()` is only ever called on
   the training set: the rest of the code makes this hard to get wrong by
   accident.
2. **Exposure is never a feature.** It is returned separately, for use as the
   GLM/GAM offset (`log(Exposure)` in the linear predictor): this is the
   correct way to model a rate per unit time and is checked directly in the
   unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------- #
# Column configuration
# --------------------------------------------------------------------------- #
# Decisions below come directly from Phase 2 (notebooks/02_exploratory_data_analysis.ipynb):
#   - `Area` is dropped: Phase 2 Section 4 showed it is, empirically, a
#     discretized version of `Density` (medians span 27 -> 27,000 inhabitants/km^2
#     with zero overlap across the six area codes). Keeping both would feed a
#     GLM two copies of the same signal.
#   - `Density` is log-transformed: it spans four orders of magnitude
#     (1 to 27,000), and a GLM/GAM fits a much more stable curve on a log scale
#     than on the raw, heavily right-skewed value.
#   - `VehAge` is capped at a threshold learned from the training data
#     (see PolicyFeatureEncoder.fit): Phase 1 found values up to 100 years,
#     which is not a plausible vehicle age and is almost certainly a data
#     artifact, but *how far* to cap is a judgement call worth making
#     transparently from the training data's own tail rather than hard-coding
#     a number that might not fit a different sample.
#   - `DrivAge`, `BonusMalus`, `VehPower` pass through unchanged: Phase 1/2
#     found no data-quality issue with these ranges, and how to *shape* their
#     effect (linear term, bins, or a GAM smooth) is a Phase 4/5 modeling
#     choice, not a Phase 3 data-preparation one.

NUMERIC_PASSTHROUGH_COLS = ["DrivAge", "BonusMalus", "VehPower"]
LOG_TRANSFORM_COLS = ["Density"]
CAPPED_COLS = ["VehAge"]
CATEGORICAL_COLS = ["VehBrand", "VehGas", "Region"]

EXPOSURE_COL = "Exposure"
FREQUENCY_TARGET_COL = "ClaimNb_freq_target"
SEVERITY_TARGET_COL = "AvgClaimAmount"
SEVERITY_WEIGHT_COL = "sev_claim_count"
HAS_CLAIM_COL = "HasPaidClaim"
ID_COL = "IDpol"


# --------------------------------------------------------------------------- #
# 1. Train / validation / test split
# --------------------------------------------------------------------------- #


def _claim_bucket(claim_nb: pd.Series) -> pd.Series:
    """
    Turn the raw claim count into a small number of buckets ("0", "1", "2+")
    to stratify the split on. We stratify on this instead of on the raw
    integer count because claims are rare (95% of policies have 0): without
    stratification, a random split could easily land 5-6% more or less of
    the already-scarce claiming policies in the test set purely by chance,
    which would bias the frequency/severity metrics computed on it.
    """
    return pd.cut(claim_nb, bins=[-1, 0, 1, np.inf], labels=["0", "1", "2+"])


def train_val_test_split_policies(
    df: pd.DataFrame,
    val_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the policy-level table into train / validation / test, stratified
    on claim count (see `_claim_bucket`) so that all three sets have close
    to the same claim-occurrence rate.

    Splitting is done once per policy (`IDpol` is unique: verified in
    Phase 1), so there is no risk of the same policy appearing in two sets.

    Returns
    -------
    (train_df, val_df, test_df): three DataFrames, same columns as `df`,
    summing back to `len(df)` rows.
    """
    assert df[ID_COL].is_unique, "Expected one row per policy: split logic assumes this."

    strata = _claim_bucket(df[FREQUENCY_TARGET_COL])

    # Step 1: carve off the test set.
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=strata, random_state=random_state
    )

    # Step 2: split what's left into train / validation. The validation
    # fraction has to be re-expressed relative to the *remaining* data
    # (train_val_df), not the original total.
    val_fraction_of_remainder = val_size / (1 - test_size)
    strata_remainder = _claim_bucket(train_val_df[FREQUENCY_TARGET_COL])
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction_of_remainder,
        stratify=strata_remainder,
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compare_split_balance(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    The bias check every split should get before anyone trusts it: does
    train/validation/test actually look like three random samples of the
    *same* portfolio? Computes exposure-weighted frequency and average paid
    severity on each split side by side, so a reviewer can see directly that
    the split didn't accidentally concentrate risk in one bucket.
    """
    rows = []
    for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        exposure = part[EXPOSURE_COL].sum()
        claims = part[FREQUENCY_TARGET_COL].sum()
        paid = part.loc[part[HAS_CLAIM_COL]]
        severity = paid["sev_claim_amount"].sum() / paid[SEVERITY_WEIGHT_COL].sum()
        rows.append(
            {
                "split": name,
                "n_policies": len(part),
                "share_of_total": len(part) / (len(train_df) + len(val_df) + len(test_df)),
                "exposure_weighted_frequency": claims / exposure,
                "avg_severity": severity,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Feature encoding: fit on train only, transform everywhere
# --------------------------------------------------------------------------- #


@dataclass
class PolicyFeatureEncoder:
    """
    Learns a deterministic transformation from the raw policy table to a
    numeric design matrix `X`, using only the training data, then applies
    that exact transformation to any other set.

    Usage
    -----
        encoder = PolicyFeatureEncoder()
        encoder.fit(train_df)                 # learns capping threshold + categories from TRAIN ONLY
        X_train = encoder.transform(train_df)
        X_val   = encoder.transform(val_df)    # same threshold/categories as train: no leakage
        X_test  = encoder.transform(test_df)

    What `fit` learns (and nothing else):
      - `vehage_cap_`     : the 99.5th percentile of `VehAge` in the training
                             data: the threshold used to cap `VehAge` in
                             `transform`. Learned, not hard-coded, so it
                             reflects whatever training sample is passed in.
      - `onehot_encoder_` : an sklearn `OneHotEncoder` fit on the training
                             data's categorical columns. `handle_unknown="ignore"`
                             means a category never seen in training (e.g. a
                             brand-new `VehBrand` on a future policy) becomes
                             all-zero dummies instead of raising an error.
                             the safe behaviour for a scoring pipeline that
                             must never crash on live data.
    """

    vehage_cap_: Optional[float] = field(default=None, init=False)
    onehot_encoder_: Optional[OneHotEncoder] = field(default=None, init=False)
    feature_names_: Optional[list[str]] = field(default=None, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(self, train_df: pd.DataFrame) -> "PolicyFeatureEncoder":
        """Learn the capping threshold and categorical levels from TRAIN ONLY."""
        self.vehage_cap_ = float(train_df["VehAge"].quantile(0.995))

        self.onehot_encoder_ = OneHotEncoder(
            drop="first",  # avoid the dummy-variable trap: with an intercept in the GLM,
            # including every category's dummy would make the design matrix singular.
            handle_unknown="ignore",  # a category unseen in training -> all-zero dummies,
            # not a crash, at transform time.
            sparse_output=False,
        )
        self.onehot_encoder_.fit(train_df[CATEGORICAL_COLS])

        # Build the final column order once, so every transform() call
        # produces a DataFrame with identical columns in the same order.
        onehot_names = list(self.onehot_encoder_.get_feature_names_out(CATEGORICAL_COLS))
        self.feature_names_ = (
            NUMERIC_PASSTHROUGH_COLS
            + [f"log_{c}" for c in LOG_TRANSFORM_COLS]
            + [f"{c}_capped" for c in CAPPED_COLS]
            + onehot_names
        )
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation learned by `fit` to `df` (train, validation,
        test, or brand-new policies at scoring time: the same code path for
        all four is the point).
        """
        if not self.is_fitted_:
            raise RuntimeError("Call .fit(train_df) before .transform(). Fitting must only ever use the training set.")

        parts = [df[NUMERIC_PASSTHROUGH_COLS].reset_index(drop=True)]

        for col in LOG_TRANSFORM_COLS:
            # log1p is not needed here (Density >= 1 everywhere per Phase 1),
            # but plain log(x) would fail on an actual zero, so this is the
            # numerically safe choice if that ever changes.
            parts.append(pd.Series(np.log(df[col].to_numpy()), name=f"log_{col}"))

        for col in CAPPED_COLS:
            capped = df[col].clip(upper=self.vehage_cap_)
            parts.append(capped.rename(f"{col}_capped").reset_index(drop=True))

        onehot_array = self.onehot_encoder_.transform(df[CATEGORICAL_COLS])
        onehot_df = pd.DataFrame(
            onehot_array,
            columns=self.onehot_encoder_.get_feature_names_out(CATEGORICAL_COLS),
        )
        parts.append(onehot_df)

        X = pd.concat(parts, axis=1)
        return X[self.feature_names_]  # enforce a fixed, predictable column order


# --------------------------------------------------------------------------- #
# 3. Target / offset accessors
# --------------------------------------------------------------------------- #
# Small, explicit functions rather than bundling everything into one giant
# "get_everything" call: each one says exactly what it returns and why it's
# kept separate from X.


def get_frequency_target(df: pd.DataFrame) -> pd.Series:
    """The frequency GLM/GAM's y: claim count per policy (see Phase 1 for why `ClaimNb` was chosen)."""
    return df[FREQUENCY_TARGET_COL]


def get_exposure_offset(df: pd.DataFrame) -> pd.Series:
    """
    The frequency model's offset: `log(Exposure)`. Never part of X: Exposure
    describes *time on risk*, not a risk characteristic, and belongs in the
    linear predictor as `log(Exposure)` with a fixed coefficient of 1, which
    is exactly what statsmodels' `offset=` argument does.
    """
    return np.log(df[EXPOSURE_COL])


def get_severity_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter down to the rows a severity model can actually be trained on:
    policies with at least one paid claim (severity is undefined otherwise.
    see Phase 1). Returns the filtered DataFrame; callers then pass it
    through the same fitted `PolicyFeatureEncoder.transform` used for
    frequency, so frequency and severity features are built identically.
    """
    return df.loc[df[HAS_CLAIM_COL]].reset_index(drop=True)
