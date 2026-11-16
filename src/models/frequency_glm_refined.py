"""
frequency_glm_refined.py
========================
Phase 7: the refined frequency GLM, the bridge from the GAM back to a GLM.

THE PROBLEM
-----------
In the Phase 4 GLM, DrivAge and VehAge are straight lines on the log scale:
one coefficient each. The Phase 5 GAM shows that their real effect is a curve.

THE BRIDGE, IN FOUR STEPS
-------------------------
  Step 1  The GAM draws the curve of each factor (frequency_gam.py).
  Step 2  We read the curve and note where it changes direction
          (gam_relativity_curve in frequency_gam.py prints the values).
  Step 3  We cut the factor into BANDS at those turning points, so that the
          curve is roughly flat inside each band.
  Step 4  Each band becomes a category with its own GLM coefficient.
          The GLM can then follow the curve as a staircase, while staying a
          table of multipliers that a regulator can read.

What the GAM curves (Phase 5) show, and the bands chosen:

  DrivAge (relativity read off the GAM curve)
      18  -> 1.66 (high)     ~30 -> 0.59 (lowest)     45-50 -> 1.07
      ~65 -> 0.90            80  -> 1.10, rising after 85 (few drivers)
      bands: 18-24 | 25-34 | 35-49 | 50-64 | 65-79 | 80-100

  VehAge (capped at 24 years)
      0-10 -> about 1.17 to 1.27 (a plateau)    12 -> 1.13    15 -> 0.92
      18 -> 0.77    21-24 -> 0.74 to 0.64
      bands: 0-10 | 11-14 | 15-18 | 19-24

In the GLM formula, the straight lines are replaced by one coefficient per
band (the first band is the reference, coefficient 0):

    Phase 4:  log(mu) = log(E) + b0 + b1 * DrivAge             + ...
    Phase 7:  log(mu) = log(E) + b0 + c(25-34) * [25-34] + c(35-49) * [35-49] + ...

where [25-34] is 1 if the driver is 25 to 34 and 0 otherwise. exp(c) is the
relativity of the band against 18-24.

Every other factor is prepared as in Phase 4. BonusMalus stays a straight
line here; Phase 8 shows that this is a weakness too.
"""

import numpy as np  # maths
import pandas as pd  # tables

# Step 3: the bands, read off the GAM curves (see above).
DRIVAGE_BIN_EDGES = [18, 25, 35, 50, 65, 80, 101]  # band limits: 18 <= age < 25, 25 <= age < 35, ...
DRIVAGE_BIN_LABELS = ["18-24", "25-34", "35-49", "50-64", "65-79", "80-100"]  # band names

VEHAGE_CAP = 24.0  # vehicle ages above 24 are set to 24 (same cap as Phase 3)
VEHAGE_BIN_EDGES = [0, 11, 15, 19, 25]  # 0 <= age < 11, 11 <= age < 15, ...
VEHAGE_BIN_LABELS = ["0-10", "11-14", "15-18", "19-24"]

NUMERIC_PASSTHROUGH_COLS = ["BonusMalus", "VehPower"]  # still straight lines
LOG_TRANSFORM_COLS = ["Density"]  # still log(Density)
CATEGORICAL_COLS = ["VehBrand", "VehGas", "Region"]  # categories, as in Phase 4
BINNED_CATEGORICAL_COLS = ["DrivAge_bin", "VehAge_bin"]  # the new banded factors


def band_driver_age(drivage):
    """Put each driver age in its band, e.g. 40 -> "35-49"."""
    return pd.cut(drivage, bins=DRIVAGE_BIN_EDGES, labels=DRIVAGE_BIN_LABELS, right=False).astype(str)


def band_vehicle_age(vehage, cap=VEHAGE_CAP):
    """Cap the vehicle age at 24, then put it in its band, e.g. 12 -> "11-14"."""
    capped = vehage.clip(upper=cap)  # 30 -> 24
    return pd.cut(capped, bins=VEHAGE_BIN_EDGES, labels=VEHAGE_BIN_LABELS, right=False).astype(str)


# Older names, kept so that existing code and tests still work.
_bin_drivage = band_driver_age
_bin_vehage = band_vehicle_age


class RefinedFeatureEncoder:
    """
    Builds the columns of the refined GLM: like PolicyFeatureEncoder (Phase 3),
    but DrivAge and VehAge are bands (0/1 columns) instead of straight lines.
    Learned on train only.

    Columns produced:
      BonusMalus, VehPower, log_Density                     numbers
      VehBrand_*, VehGas_Regular, Region_*                  0/1, reference = first level
      DrivAge_bin_25-34, ..., DrivAge_bin_80-100            0/1, reference = 18-24
      VehAge_bin_11-14, VehAge_bin_15-18, VehAge_bin_19-24  0/1, reference = 0-10
    """

    def __init__(self):
        self.categories_ = None  # levels of each category column, learned by fit
        self.feature_names_ = None  # final column order
        self.is_fitted_ = False

    def category_table(self, df):
        """The five category columns of a table: the 3 original ones + the 2 bands."""
        table = df[CATEGORICAL_COLS].astype(str).copy()  # VehBrand, VehGas, Region as text
        table["DrivAge_bin"] = band_driver_age(df["DrivAge"])  # band of the driver age
        table["VehAge_bin"] = band_vehicle_age(df["VehAge"])  # band of the vehicle age
        return table

    def fit(self, train_df):
        """Learn the levels of the five category columns from the train set."""
        table = self.category_table(train_df)
        self.categories_ = {}
        dummy_names = []
        for col in CATEGORICAL_COLS + BINNED_CATEGORICAL_COLS:
            levels = sorted(table[col].unique())  # levels in alphabetical order (the bands sort correctly)
            self.categories_[col] = levels
            for level in levels[1:]:  # no column for the reference (first) level
                dummy_names.append(f"{col}_{level}")
        self.feature_names_ = NUMERIC_PASSTHROUGH_COLS + ["log_Density"] + dummy_names  # final column order
        self.is_fitted_ = True
        return self

    def transform(self, df):
        """Apply the learned rules to any table."""
        if not self.is_fitted_:
            raise RuntimeError("Call .fit(train_df) before .transform().")
        X = pd.DataFrame(index=range(len(df)))  # empty table, one row per policy
        for col in NUMERIC_PASSTHROUGH_COLS:  # BonusMalus, VehPower
            X[col] = df[col].to_numpy()
        X["log_Density"] = np.log(df["Density"].to_numpy())  # log of density
        table = self.category_table(df)  # the five category columns
        for col in CATEGORICAL_COLS + BINNED_CATEGORICAL_COLS:
            values = table[col].to_numpy()  # level of each policy
            for level in self.categories_[col][1:]:  # one 0/1 column per non-reference level
                X[f"{col}_{level}"] = (values == level).astype(float)
        return X[self.feature_names_]
