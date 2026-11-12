"""
frequency_glm_refined.py
==========================
Phase 7: closes the loop Phase 5 opened. Phase 5's GAM found that `DrivAge`
and `VehAge` both have a real shape a single straight line cannot represent
(Phase 4's linear GLM terms). Its own recommendation was: *use those curves
as evidence to re-bin the two factors for an improved GLM*: capturing most
of the GAM's accuracy while keeping a GLM's filability. This module is that
improved GLM. It is what Phase 7's technical premium actually uses for
frequency, not the original Phase 4 GLM.

WHERE THE BIN EDGES COME FROM (read off the fitted Phase 5 GAM, not guessed)
-------------------------------------------------------------------------------
`notebooks/07_technical_premium.ipynb` Section 1 loads `models/freq_gam.pkl`
and prints the exact partial-dependence values behind these choices. Summary:

`DrivAge` (centered relativity from the GAM's smooth term):
    age 18   -> 1.66  (elevated)
    age ~30  -> 0.59  (trough)
    age ~45-50 -> 1.07 (first peak)
    age ~65  -> 0.90  (second trough)
    age 80   -> 1.10, rising further after 85 with a widening confidence band (thin data)

Bin edges `[18, 25, 35, 50, 65, 80, 101]` place a boundary near each turning
point, so that within each resulting bin the GAM's fitted relativity is
roughly constant, the condition under which a piecewise-constant (binned)
GLM term is a good approximation to a smooth curve.

`VehAge` (capped at Phase 3's threshold, 24 years):
    age 0-10  -> 1.17 to 1.27  (plateau, highest around 4-5 years)
    age 12    -> 1.13
    age 15    -> 0.92
    age 18    -> 0.77
    age 21-24 -> 0.74 to 0.64

Bin edges `[0, 11, 15, 19, 25]` keep the plateau as one level and follow
the decline after 10 years in three steps.

Every other factor (`BonusMalus`, `VehPower`, `Density`, `VehBrand`,
`VehGas`, `Region`) is treated exactly as in Phase 3/4: Phase 5 found no
evidence any of them needed a different functional form, so only the two
factors with actual evidence behind them are changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# See the module docstring for the reasoning behind these specific edges.
DRIVAGE_BIN_EDGES = [18, 25, 35, 50, 65, 80, 101]
DRIVAGE_BIN_LABELS = ["18-24", "25-34", "35-49", "50-64", "65-79", "80-100"]

VEHAGE_CAP = 24.0  # reused from Phase 3's PolicyFeatureEncoder.vehage_cap_ for consistency
VEHAGE_BIN_EDGES = [0, 11, 15, 19, 25]
VEHAGE_BIN_LABELS = ["0-10", "11-14", "15-18", "19-24"]

NUMERIC_PASSTHROUGH_COLS = ["BonusMalus", "VehPower"]
LOG_TRANSFORM_COLS = ["Density"]
CATEGORICAL_COLS = ["VehBrand", "VehGas", "Region"]
BINNED_CATEGORICAL_COLS = ["DrivAge_bin", "VehAge_bin"]


def _bin_drivage(drivage: pd.Series) -> pd.Series:
    return pd.cut(drivage, bins=DRIVAGE_BIN_EDGES, labels=DRIVAGE_BIN_LABELS, right=False).astype(str)


def _bin_vehage(vehage: pd.Series, cap: float = VEHAGE_CAP) -> pd.Series:
    capped = vehage.clip(upper=cap)
    return pd.cut(capped, bins=VEHAGE_BIN_EDGES, labels=VEHAGE_BIN_LABELS, right=False).astype(str)


@dataclass
class RefinedFeatureEncoder:
    """
    Same fit-on-train-only, transform-everywhere discipline as Phase 3's
    `PolicyFeatureEncoder` (see that class's docstring): the only
    difference is that `DrivAge` and `VehAge` are binned (per the module
    docstring) instead of passed through as a single linear term / capped
    linear term.
    """

    onehot_encoder_: Optional[OneHotEncoder] = field(default=None, init=False)
    feature_names_: Optional[list[str]] = field(default=None, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def _categorical_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[CATEGORICAL_COLS].astype(str).copy()
        out["DrivAge_bin"] = _bin_drivage(df["DrivAge"])
        out["VehAge_bin"] = _bin_vehage(df["VehAge"])
        return out

    def fit(self, train_df: pd.DataFrame) -> "RefinedFeatureEncoder":
        cat_cols = CATEGORICAL_COLS + BINNED_CATEGORICAL_COLS
        cat_df = self._categorical_frame(train_df)

        self.onehot_encoder_ = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        self.onehot_encoder_.fit(cat_df[cat_cols])

        onehot_names = list(self.onehot_encoder_.get_feature_names_out(cat_cols))
        self.feature_names_ = (
            NUMERIC_PASSTHROUGH_COLS + [f"log_{c}" for c in LOG_TRANSFORM_COLS] + onehot_names
        )
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call .fit(train_df) before .transform(). Fitting must only ever use the training set.")

        parts = [df[NUMERIC_PASSTHROUGH_COLS].reset_index(drop=True)]
        for col in LOG_TRANSFORM_COLS:
            parts.append(pd.Series(np.log(df[col].to_numpy()), name=f"log_{col}"))

        cat_cols = CATEGORICAL_COLS + BINNED_CATEGORICAL_COLS
        cat_df = self._categorical_frame(df)
        onehot_array = self.onehot_encoder_.transform(cat_df[cat_cols])
        onehot_df = pd.DataFrame(onehot_array, columns=self.onehot_encoder_.get_feature_names_out(cat_cols))
        parts.append(onehot_df)

        X = pd.concat(parts, axis=1)
        return X[self.feature_names_]
