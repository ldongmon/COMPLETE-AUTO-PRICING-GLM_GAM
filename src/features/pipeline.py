"""
pipeline.py
===========
Phase 3: prepare the data for the models.

Two jobs:

1. SPLIT the policies into three sets (like exercises and exams for a pupil):
       train      (60%)  the model learns on it
       validation (20%)  a first exam the model has never seen
       test       (20%)  a second exam the model has never seen

2. ENCODE the rating factors into numbers a GLM can use. The rule for each
   factor is learned on the TRAIN set only, then applied unchanged to every
   other set. Otherwise information from the exams would leak into the
   exercises ("leakage").

How each factor is prepared (decisions taken in Phase 2):

    DrivAge, BonusMalus, VehPower  kept as they are (numbers)
    Density                        replaced by log(Density): density goes from 1 to
                                   27,000, the log turns that into 0 to 10
    VehAge                         capped at the 99.5th percentile of the train set
                                   (24 years): some ages up to 100 are not plausible
    VehBrand, VehGas, Region       one 0/1 column per level ("dummy" columns),
                                   except the first level, which is the reference
    Area                           dropped: it is a coarse copy of Density
    Exposure                       NOT a factor: it is the model's offset (see below)

Why exposure is not a factor: a policy insured for half a year is expected to
have half as many claims. That is a definition, not something to estimate.
The model receives log(Exposure) with a coefficient fixed to 1 (the
"offset"), see get_exposure_offset.
"""

import numpy as np  # numbers and maths functions (log, ...)
import pandas as pd  # tables
from sklearn.model_selection import train_test_split  # random split of a table into two parts

# The column lists below say how each rating factor is prepared.
NUMERIC_PASSTHROUGH_COLS = ["DrivAge", "BonusMalus", "VehPower"]  # used as they are
LOG_TRANSFORM_COLS = ["Density"]  # replaced by their logarithm
CAPPED_COLS = ["VehAge"]  # capped at a maximum learned on the train set
CATEGORICAL_COLS = ["VehBrand", "VehGas", "Region"]  # turned into 0/1 columns

EXPOSURE_COL = "Exposure"  # time on risk, in years
FREQUENCY_TARGET_COL = "ClaimNb_freq_target"  # what the frequency model predicts: number of claims
SEVERITY_TARGET_COL = "AvgClaimAmount"  # what the severity model predicts: average amount per claim
SEVERITY_WEIGHT_COL = "sev_claim_count"  # number of paid claims behind that average
HAS_CLAIM_COL = "HasPaidClaim"  # True when the policy has at least one paid claim
ID_COL = "IDpol"  # policy number


# --------------------------------------------------------------------------- #
# 1. Split into train / validation / test
# --------------------------------------------------------------------------- #


def claim_bucket(claim_counts):
    """Group the claim counts into 3 classes: "0", "1" and "2+" (used to balance the split)."""
    return pd.cut(claim_counts, bins=[-1, 0, 1, np.inf], labels=["0", "1", "2+"])


def train_val_test_split_policies(df, val_size=0.20, test_size=0.20, random_state=42):
    """
    Split the policies into train (60%), validation (20%) and test (20%).

    The split is "stratified" on the claim class: each set receives the same
    share of policies with 0, 1 and 2+ claims. Claims are rare (about 4% of
    policies), so a purely random split could give one set noticeably more
    claims than another by chance. `random_state` fixes the random draw, so
    the split is the same every time the code runs.
    """
    assert df[ID_COL].is_unique, "Expected one row per policy"  # each policy appears once

    # Step 1: take 20% of the policies out as the test set.
    train_val, test = train_test_split(
        df,
        test_size=test_size,  # 20% for the test set
        stratify=claim_bucket(df[FREQUENCY_TARGET_COL]),  # same claim mix in both parts
        random_state=random_state,  # same draw every run
    )

    # Step 2: split the remaining 80% into train and validation.
    # Validation must be 20% of the TOTAL, i.e. 0.20 / 0.80 = 25% of what is left.
    share_of_remainder = val_size / (1 - test_size)  # 0.25
    train, val = train_test_split(
        train_val,
        test_size=share_of_remainder,  # 25% of the remaining 80% = 20% of the total
        stratify=claim_bucket(train_val[FREQUENCY_TARGET_COL]),  # same claim mix again
        random_state=random_state,
    )

    return (  # renumber the rows of each set 0, 1, 2, ...
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def compare_split_balance(train_df, val_df, test_df):
    """
    Check that the three sets look like the same portfolio: size, claim
    frequency (claims / years insured) and average claim cost for each set.
    """
    total = len(train_df) + len(val_df) + len(test_df)  # number of policies in all three sets
    rows = []  # one result row per set
    for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        claims = part[FREQUENCY_TARGET_COL].sum()  # claims in this set
        exposure = part[EXPOSURE_COL].sum()  # years insured in this set
        paid = part[part[HAS_CLAIM_COL]]  # policies with a paid claim
        severity = paid["sev_claim_amount"].sum() / paid[SEVERITY_WEIGHT_COL].sum()  # amount paid / number of paid claims
        rows.append({
            "split": name,
            "n_policies": len(part),  # number of policies
            "share_of_total": len(part) / total,  # should be close to 60% / 20% / 20%
            "exposure_weighted_frequency": claims / exposure,  # should be about the same in all three
            "avg_severity": severity,  # can differ: a few very large claims move an average a lot
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Encode the rating factors (learned on train, applied everywhere)
# --------------------------------------------------------------------------- #


class PolicyFeatureEncoder:
    """
    Turns the policy table into the table of numbers (the "design matrix" X)
    used by the GLM.

        encoder = PolicyFeatureEncoder()
        encoder.fit(train_df)            # learn the rules on the TRAIN set only
        X_train = encoder.transform(train_df)
        X_val = encoder.transform(val_df)  # same rules, nothing learned from validation

    What `fit` learns:
      vehage_cap_    the vehicle age above which values are capped (99.5th percentile)
      categories_    the list of levels of each category column (e.g. all VehBrand codes)
      feature_names_ the final list of columns, always in the same order

    Category columns: for VehGas (Diesel, Regular) we create ONE column,
    VehGas_Regular, equal to 1 for Regular and 0 for Diesel. Diesel, the first
    level in alphabetical order, is the "reference": the GLM's intercept
    describes it, and the VehGas_Regular coefficient says how Regular differs.
    A level never seen in training gets 0 in every column, i.e. it is priced
    like the reference.
    """

    def __init__(self):
        self.vehage_cap_ = None  # learned by fit
        self.categories_ = None  # learned by fit
        self.feature_names_ = None  # learned by fit
        self.is_fitted_ = False  # becomes True after fit

    def fit(self, train_df):
        """Learn the capping threshold and the category levels from the train set."""
        self.vehage_cap_ = float(train_df["VehAge"].quantile(0.995))  # 99.5% of vehicles are younger than this

        self.categories_ = {}  # one list of levels per category column
        for col in CATEGORICAL_COLS:  # VehBrand, VehGas, Region
            levels = sorted(train_df[col].astype(str).unique())  # all levels seen in train, in alphabetical order
            self.categories_[col] = levels

        dummy_names = []  # names of the 0/1 columns
        for col in CATEGORICAL_COLS:
            for level in self.categories_[col][1:]:  # [1:] skips the first level, the reference
                dummy_names.append(f"{col}_{level}")  # e.g. "VehGas_Regular"

        self.feature_names_ = (  # the final column order
            NUMERIC_PASSTHROUGH_COLS  # DrivAge, BonusMalus, VehPower
            + ["log_Density"]  # log of density
            + ["VehAge_capped"]  # capped vehicle age
            + dummy_names  # one 0/1 column per non-reference level
        )
        self.is_fitted_ = True
        return self

    def transform(self, df):
        """Apply the rules learned by `fit` to any table (train, validation, test, new policies)."""
        if not self.is_fitted_:  # protection: fit must come first
            raise RuntimeError("Call .fit(train_df) before .transform().")

        X = pd.DataFrame(index=range(len(df)))  # an empty table with one row per policy
        for col in NUMERIC_PASSTHROUGH_COLS:  # numbers used as they are
            X[col] = df[col].to_numpy()
        X["log_Density"] = np.log(df["Density"].to_numpy())  # natural logarithm of density
        X["VehAge_capped"] = df["VehAge"].clip(upper=self.vehage_cap_).to_numpy()  # values above the cap become the cap

        for col in CATEGORICAL_COLS:  # one 0/1 column per non-reference level
            values = df[col].astype(str).to_numpy()  # the level of each policy, as text
            for level in self.categories_[col][1:]:  # skip the reference level
                X[f"{col}_{level}"] = (values == level).astype(float)  # 1.0 if the policy has this level, else 0.0

        return X[self.feature_names_]  # always the same columns, in the same order


# --------------------------------------------------------------------------- #
# 3. Target, offset and severity subset
# --------------------------------------------------------------------------- #


def get_frequency_target(df):
    """What the frequency model predicts: the number of claims of each policy."""
    return df[FREQUENCY_TARGET_COL]


def get_exposure_offset(df):
    """
    The offset of the frequency model: log(Exposure).

    The model is  log(expected claims) = log(Exposure) + b0 + b1*x1 + ...
    so            expected claims = Exposure * exp(b0 + b1*x1 + ...).
    The coefficient of log(Exposure) is fixed to 1, so a policy insured half a
    year is expected to have exactly half the claims of a full year.
    """
    return np.log(df[EXPOSURE_COL])


def get_severity_subset(df):
    """The policies a severity model can learn from: those with at least one paid claim."""
    return df[df[HAS_CLAIM_COL]].reset_index(drop=True)
