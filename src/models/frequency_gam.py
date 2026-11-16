"""
frequency_gam.py
================
Phase 5: the frequency GAM (Generalised Additive Model).

GLM VS. GAM, IN ONE FORMULA EACH
--------------------------------
GLM (Phase 4): each numeric factor gets ONE coefficient, i.e. a straight line
on the log scale:

    log(mu) = log(Exposure) + b0 + b1 * DrivAge + b2 * BonusMalus + ...

GAM (this phase): each numeric factor gets its own smooth CURVE f(.):

    log(mu) = log(Exposure) + b0 + f1(DrivAge) + f2(BonusMalus) + f3(VehPower)
              + f4(log Density) + f5(VehAge) + (one value per level of VehBrand, VehGas, Region)

HOW A CURVE f(x) IS BUILT: SPLINES
----------------------------------
The range of x (for instance ages 18 to 100) is cut into a few pieces. On
each piece the curve is a small polynomial, and the pieces join smoothly.
Equivalently, f(x) is a weighted sum of 10 "bump" functions (B-splines),
each covering part of the range:

    f(x) = a1 * B1(x) + a2 * B2(x) + ... + a10 * B10(x)

The GAM estimates the weights a1..a10 like GLM coefficients, with a penalty
(lam) that stops the curve from wiggling too much. With 10 bumps per factor
the curve can rise, fall and rise again: it follows the data instead of
forcing a straight line.

WHAT WE DO WITH THE GAM
-----------------------
We do NOT use the GAM as the tariff (a curve is hard to file and to explain).
We use it as a magnifying glass: we read its curves (gam_relativity_curve),
find where they change direction, and cut DrivAge and VehAge into bands at
those points. The bands then go into the GLM (frequency_glm_refined.py).

Categories (VehBrand, VehGas, Region) are "factor terms" in the GAM: one
value per level, exactly like dummy columns in a GLM.
"""

import numpy as np  # maths
import pandas as pd  # tables
from pygam import PoissonGAM, f, s  # the GAM library: s() = smooth curve term, f() = factor (category) term
from sklearn.model_selection import train_test_split  # random sub-sample

# The columns given to the GAM, in this exact order (term number = position in the list).
CONTINUOUS_COLS = ["DrivAge", "BonusMalus", "VehPower", "log_Density", "VehAge_capped"]  # terms 0 to 4: smooth curves
CATEGORICAL_COLS = ["VehBrand", "VehGas", "Region"]  # terms 5 to 7: one value per level
ALL_GAM_COLS = CONTINUOUS_COLS + CATEGORICAL_COLS


class GAMFeatureBuilder:
    """
    Prepares the table for the GAM: numbers for the curve terms, level codes
    (0, 1, 2, ...) for the category terms. Learned on train only, like the
    GLM encoder.
    """

    def __init__(self):
        self.vehage_cap_ = None  # vehicle-age cap (taken from the GLM encoder)
        self.categories_ = None  # list of levels of each category
        self.is_fitted_ = False

    def fit(self, train_df, vehage_cap):
        """Learn the category levels; reuse the GLM's vehicle-age cap so both models see the same data."""
        self.vehage_cap_ = vehage_cap  # same cap as the GLM (24 years)
        self.categories_ = {}
        for col in CATEGORICAL_COLS:
            self.categories_[col] = sorted(train_df[col].astype(str).unique())  # levels in alphabetical order
        self.is_fitted_ = True
        return self

    def transform(self, df):
        """Return a table of numbers, one column per GAM term, in the order of ALL_GAM_COLS."""
        if not self.is_fitted_:
            raise RuntimeError("Call .fit(train_df, vehage_cap) before .transform().")
        columns = [
            df["DrivAge"].to_numpy(dtype=float),  # term 0
            df["BonusMalus"].to_numpy(dtype=float),  # term 1
            df["VehPower"].to_numpy(dtype=float),  # term 2
            np.log(df["Density"].to_numpy(dtype=float)),  # term 3: log of density
            df["VehAge"].clip(upper=self.vehage_cap_).to_numpy(dtype=float),  # term 4: capped vehicle age
        ]
        for col in CATEGORICAL_COLS:  # terms 5, 6, 7
            levels = self.categories_[col]  # levels known from train
            values = df[col].astype(str)  # level of each policy
            values = values.where(values.isin(levels), levels[0])  # unknown level -> treated as the first level
            codes = pd.Categorical(values, categories=levels).codes  # level -> its position 0, 1, 2, ...
            columns.append(codes.astype(float))
        return np.column_stack(columns)  # put the columns side by side


def build_gam_terms(n_splines=10):
    """
    The GAM formula: 5 smooth curves (10 B-splines each) + 3 category terms.
    The numbers 0..7 are the column positions given by GAMFeatureBuilder.
    """
    return (
        s(0, n_splines=n_splines)  # f1(DrivAge)
        + s(1, n_splines=n_splines)  # f2(BonusMalus)
        + s(2, n_splines=n_splines)  # f3(VehPower)
        + s(3, n_splines=n_splines)  # f4(log Density)
        + s(4, n_splines=n_splines)  # f5(VehAge)
        + f(5)  # VehBrand: one value per brand
        + f(6)  # VehGas: one value per fuel
        + f(7)  # Region: one value per region
    )


def fit_poisson_gam(X, y, exposure, n_splines=10):
    """
    Fit the Poisson GAM. `exposure=` plays the role of the GLM's offset: the
    expected number of claims is exposure x exp(sum of the terms).
    """
    gam = PoissonGAM(build_gam_terms(n_splines=n_splines))  # describe the model
    gam.fit(X, y, exposure=exposure)  # estimate the curves
    return gam


# Sizes to try, from the largest to the smallest: (share of the train set, number of splines).
DEFAULT_FIT_CONFIGS = [(0.25, 10), (0.15, 8), (0.10, 6), (0.06, 6), (0.03, 5)]


def fit_poisson_gam_with_fallback(train_df, gam_builder, configs=None, random_state=42, verbose=True):
    """
    Fit the GAM on the largest sub-sample this computer can handle.

    A GAM needs much more memory than a GLM (10 columns per curve). On an 8 GB
    machine a fit on 25% of the train set sometimes fails with a MemoryError.
    We try the sizes of DEFAULT_FIT_CONFIGS in order and keep the first that
    works. The sub-sample is stratified on the claim class, like the split.
    """
    if configs is None:
        configs = DEFAULT_FIT_CONFIGS
    claim_class = pd.cut(train_df["ClaimNb_freq_target"], bins=[-1, 0, 1, np.inf], labels=["0", "1", "2+"])  # 0, 1, 2+ claims

    for train_size, n_splines in configs:  # from the largest size to the smallest
        try:
            if train_size < 1.0:  # take a stratified sub-sample
                train_sub, _ = train_test_split(train_df, train_size=train_size, stratify=claim_class, random_state=random_state)
            else:
                train_sub = train_df
            X_sub = gam_builder.transform(train_sub)  # GAM columns
            y_sub = train_sub["ClaimNb_freq_target"].to_numpy(dtype=float)  # claims
            exposure_sub = train_sub["Exposure"].to_numpy(dtype=float)  # years insured
            gam = fit_poisson_gam(X_sub, y_sub, exposure_sub, n_splines=n_splines)  # the fit itself
            if verbose:
                print(f"GAM fit succeeded: train_size={train_size:.0%} ({len(train_sub):,} rows), n_splines={n_splines}")
            return gam, {"train_size": train_size, "n_splines": n_splines, "n_rows": len(train_sub)}
        except MemoryError:  # not enough memory: try the next, smaller size
            if verbose:
                print(f"MemoryError at train_size={train_size:.0%}, n_splines={n_splines}: trying a smaller size.")
    raise MemoryError(f"Could not fit the GAM with any of the sizes {configs}")


def gam_relativity_curve(gam, term_index, x_values):
    """
    Read one curve of the GAM at chosen values of x (e.g. ages 18, 20, 25, ...).

    The GAM gives f(x) on the log scale. We turn it into a RELATIVITY, the
    multiplier on the frequency, centred so that the average over the grid is 1:

        relativity(x) = exp( f(x) - average of f over the grid )

    Returns a table: x, relativity, and the lower / upper bounds of its 95% band.
    """
    grid = gam.generate_X_grid(term=term_index, n=200)  # 200 evenly spaced points over the range of x
    f_values, band = gam.partial_dependence(term=term_index, X=grid, width=0.95)  # f(x) and its 95% band on the grid
    centre = f_values.mean()  # average of f over the grid
    x_grid = grid[:, term_index]  # the x values of the grid
    rows = []
    for x in x_values:  # read the curve at each requested x (straight-line interpolation between grid points)
        rows.append({
            "x": x,
            "relativity": float(np.exp(np.interp(x, x_grid, f_values) - centre)),
            "band_low": float(np.exp(np.interp(x, x_grid, band[:, 0]) - centre)),
            "band_high": float(np.exp(np.interp(x, x_grid, band[:, 1]) - centre)),
        })
    return pd.DataFrame(rows)
