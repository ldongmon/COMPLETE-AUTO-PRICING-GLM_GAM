"""
frequency_gam.py
=================
Phase 5: a Poisson GAM (Generalized Additive Model) for claim frequency,
benchmarked against the Phase 4 GLM.

WHY A DIFFERENT FEATURE REPRESENTATION THAN THE GLM
------------------------------------------------------
`src/features/pipeline.py`'s `PolicyFeatureEncoder` was purpose-built for the
GLM: it one-hot encodes categorical factors into dummy columns, because a
GLM's linear predictor needs one coefficient per dummy. A GAM works
differently: `pygam` fits a smooth spline `s()` per continuous factor and a
per-level step function `f()` per categorical factor *natively*, without
one-hot expansion (one-hot dummies would force pygam to fit a separate,
disconnected smooth to each 0/1 indicator, which is not what we want). So
Phase 5 introduces `GAMFeatureBuilder`, a parallel encoder that reuses Phase
3's *decisions* (which factors, the `VehAge` capping threshold, the
`log(Density)` transform, dropping `Area`) but produces a different output
shape suited to `pygam`.

WHY pygam's `exposure=` INSTEAD OF A LOG-OFFSET
---------------------------------------------------
`pygam.PoissonGAM` has no `offset=` argument like `statsmodels`. Internally,
passing `exposure=` to `.fit()`/`.predict()` normalizes the response to a
rate (`y / exposure`) and weights the fit by `exposure` (see
`PoissonGAM._exposure_to_weights` in the pygam source): a different
implementation from statsmodels' `log(Exposure)` offset, but the standard,
mathematically equivalent workaround used across GAM software (R's `mgcv`
faces the same limitation and is worked around the same way) for fitting a
rate model when a package has no native offset support.

ON SMOOTHING PARAMETER SELECTION
------------------------------------
`pygam.gridsearch` (its automatic smoothing-parameter search) proved
numerically unstable on this data/term specification during development.
every candidate in the grid failed to converge, a known rough edge in this
library (pygam's own summary output carries a "KNOWN BUG" warning about its
p-values, for context on the library's maturity). Rather than spend the
project's time debugging a third-party library's internals, this module uses
pygam's sensible default smoothing (`lam=0.6` on every term) uniformly: a
documented simplification, not a silent one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pygam import PoissonGAM, f, s
from sklearn.model_selection import train_test_split

# Column order produced by GAMFeatureBuilder.transform: term indices below
# must match this order exactly.
CONTINUOUS_COLS = ["DrivAge", "BonusMalus", "VehPower", "log_Density", "VehAge_capped"]
CATEGORICAL_COLS = ["VehBrand", "VehGas", "Region"]
ALL_GAM_COLS = CONTINUOUS_COLS + CATEGORICAL_COLS


@dataclass
class GAMFeatureBuilder:
    """
    Builds the numeric array `pygam` needs: raw (transformed) continuous
    columns for smooth terms, integer category codes for factor terms.
    Fit on train only, exactly like `PolicyFeatureEncoder` (Phase 3): see
    that class's docstring for why this discipline matters.
    """

    vehage_cap_: Optional[float] = field(default=None, init=False)
    categories_: Optional[dict] = field(default=None, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(self, train_df: pd.DataFrame, vehage_cap: float) -> "GAMFeatureBuilder":
        """
        `vehage_cap` is passed in rather than recomputed, so the GLM and GAM
        use the *exact same* `VehAge` capping threshold (Phase 3's
        `PolicyFeatureEncoder.vehage_cap_`): one less place the two models
        could silently diverge on a data-preparation decision that isn't
        actually part of what's being compared.
        """
        self.vehage_cap_ = vehage_cap
        self.categories_ = {
            col: sorted(train_df[col].astype(str).unique()) for col in CATEGORICAL_COLS
        }
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Call .fit(train_df, vehage_cap) before .transform().")

        columns = [
            df["DrivAge"].to_numpy(dtype=float),
            df["BonusMalus"].to_numpy(dtype=float),
            df["VehPower"].to_numpy(dtype=float),
            np.log(df["Density"].to_numpy(dtype=float)),
            df["VehAge"].clip(upper=self.vehage_cap_).to_numpy(dtype=float),
        ]
        for col in CATEGORICAL_COLS:
            # A fixed category list learned from train. Any value unseen in training is
            # replaced with the first known category *before* building the Categorical
            # (rather than afterward, via the resulting -1 code): an arbitrary but
            # always-valid fallback, documented and safe for a scoring pipeline, matching
            # the spirit of the GLM encoder's handle_unknown="ignore".
            known_categories = self.categories_[col]
            values = df[col].astype(str)
            values = values.where(values.isin(known_categories), known_categories[0])
            codes = pd.Categorical(values, categories=known_categories).codes
            columns.append(codes.astype(float))

        return np.column_stack(columns)


def build_gam_terms(n_splines: int = 10):
    """
    5 smooth spline terms (continuous factors) + 3 factor terms (categorical
    factors), matching `ALL_GAM_COLS`'s column order exactly.

    `n_splines=10` (pygam's default is 20) is a deliberate, documented
    simplification: with ~400k rows this is still far more flexibility than
    needed to capture the shapes Phase 2 found (a young-driver spike, a
    new-vehicle spike), and halving the basis size meaningfully reduces the
    fit's memory footprint on this machine: worth doing given Phase 4's
    memory lessons, at no real cost to the comparison this phase is after.
    """
    return (
        s(0, n_splines=n_splines)  # DrivAge
        + s(1, n_splines=n_splines)  # BonusMalus
        + s(2, n_splines=n_splines)  # VehPower
        + s(3, n_splines=n_splines)  # log_Density
        + s(4, n_splines=n_splines)  # VehAge_capped
        + f(5)  # VehBrand
        + f(6)  # VehGas
        + f(7)  # Region
    )


def fit_poisson_gam(X: np.ndarray, y: np.ndarray, exposure: np.ndarray, n_splines: int = 10) -> PoissonGAM:
    """Fit a Poisson GAM with `exposure` as the actuarial offset equivalent (see module docstring)."""
    gam = PoissonGAM(build_gam_terms(n_splines=n_splines))
    gam.fit(X, y, exposure=exposure)
    return gam


# Fitting configurations to try, most statistically ambitious first: (fraction of train_df to
# sample, number of spline knots per smooth term). Each step down roughly halves the memory a fit
# needs (fewer rows *and* a narrower basis matrix), for the fallback strategy below.
DEFAULT_FIT_CONFIGS = [
    (0.25, 10),
    (0.15, 8),
    (0.10, 6),
    (0.06, 6),
    (0.03, 5),
]


def fit_poisson_gam_with_fallback(
    train_df: pd.DataFrame,
    gam_builder: "GAMFeatureBuilder",
    configs: Optional[list[tuple[float, int]]] = None,
    random_state: int = 42,
    verbose: bool = True,
):
    """
    Fits a Poisson GAM using the largest, most flexible configuration this machine can actually
    complete, by trying `configs` in order and catching `MemoryError` to fall back to a smaller one.

    WHY THIS EXISTS
    -----------------
    This project's development machine has ~8GB RAM, and how much of it is actually free varies a
    lot moment to moment with whatever else is running (confirmed during development: concurrent
    browser tabs, an IDE, and other background processes can leave well under 1GB free): the exact
    same fit call has both succeeded and failed with a `MemoryError` at different points in the same
    session, at the same sub-sample size. Rather than hand-pick one fixed, maximally conservative
    configuration "just in case" (which would silently under-use the machine's capacity whenever more
    memory happens to be free), this tries progressively smaller, cheaper configurations only as
    needed: the same discipline a production service applies under variable resource pressure,
    applied here to a notebook that has to actually finish running on real, shared hardware.

    Returns
    -------
    (gam, config_used) where `config_used` is a dict recording which (train_size, n_splines) pair
    actually succeeded and how many rows it used, since which configuration
    was used affects how much to trust the fitted curves' fine detail (Section 5 discusses this).
    """
    if configs is None:
        configs = DEFAULT_FIT_CONFIGS

    claim_bucket = pd.cut(train_df["ClaimNb_freq_target"], bins=[-1, 0, 1, np.inf], labels=["0", "1", "2+"])

    last_error: Optional[MemoryError] = None
    for train_size, n_splines in configs:
        try:
            if train_size < 1.0:
                train_sub, _ = train_test_split(
                    train_df, train_size=train_size, stratify=claim_bucket, random_state=random_state
                )
            else:
                train_sub = train_df

            X_sub = gam_builder.transform(train_sub)
            y_sub = train_sub["ClaimNb_freq_target"].to_numpy(dtype=float)
            exposure_sub = train_sub["Exposure"].to_numpy(dtype=float)

            gam = fit_poisson_gam(X_sub, y_sub, exposure_sub, n_splines=n_splines)

            config_used = {"train_size": train_size, "n_splines": n_splines, "n_rows": len(train_sub)}
            if verbose:
                print(
                    f"GAM fit succeeded: train_size={train_size:.0%} ({len(train_sub):,} rows), "
                    f"n_splines={n_splines}"
                )
            return gam, config_used

        except MemoryError as exc:
            last_error = exc
            if verbose:
                print(
                    f"MemoryError at train_size={train_size:.0%}, n_splines={n_splines} "
                    "-- falling back to a smaller configuration."
                )
            continue

    raise MemoryError(
        f"Could not fit a Poisson GAM at any configuration in {configs}; last error: {last_error}"
    )
