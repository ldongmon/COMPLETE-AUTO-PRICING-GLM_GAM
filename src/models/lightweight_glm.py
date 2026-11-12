"""
lightweight_glm.py
====================
Phase 12: distills a fitted `statsmodels` GLM results object down to just
what scoring needs: the coefficient vector. Exists because of a very
concrete, measured problem: `models/freq_glm_refined.pkl` is **190MB** and
`models/sev_glm.pkl` is **15MB** on disk, because a `statsmodels` results
object retains a full copy of the training design matrix and internal
fitting arrays (Hessian, covariance, etc.): appropriate for the notebook
diagnostics that use `.summary()`, `.pvalues`, `.aic`, and so on (Phases
4-7), but completely impractical to commit to git or ship to a deployed
Streamlit app. A `LightweightGLM` built from the same fitted coefficients is
a few **kilobytes** and is mathematically identical for prediction.

DROP-IN COMPATIBLE WITH THE REAL THING
-------------------------------------------
`LightweightGLM` exposes the same two things
`scoring/score_technical_premium.py` (and everything built on it) actually
uses from a real `GLMResults` object: `.predict(X, offset=None)` and
`.model.exog_names`. Every notebook and scoring call that already works with
a full `statsmodels` results object works identically with a `LightweightGLM`
in its place: no downstream code needed to change.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd


class LightweightGLM:
    """
    A minimal, prediction-only stand-in for a fitted `statsmodels` GLM
    results object (log link only: the only link this project's frequency
    and severity GLMs use). See module docstring for why this exists.
    """

    def __init__(self, params: pd.Series):
        self.params = params.copy()
        # Mimics `results.model.exog_names`: the one other attribute
        # `score_technical_premium.py` reads off a real results object.
        self.model = SimpleNamespace(exog_names=list(self.params.index))

    def predict(self, X: pd.DataFrame, offset: np.ndarray | None = None) -> np.ndarray:
        """
        `X` must already include the columns in `self.model.exog_names`
        (typically produced by `sm.add_constant(...)`, exactly as a real
        `GLMResults.predict()` expects): reordered here to the exact
        sequence the coefficients were fit against, so column order in the
        caller's DataFrame never matters.
        """
        X_aligned = X[self.model.exog_names]
        linear_predictor = X_aligned.to_numpy() @ self.params.to_numpy()
        if offset is not None:
            linear_predictor = linear_predictor + np.asarray(offset)
        return np.exp(linear_predictor)  # log link, always, for both models this project fits

    @classmethod
    def from_results(cls, results) -> "LightweightGLM":
        """Build a `LightweightGLM` from a fitted `statsmodels` GLMResults object."""
        return cls(params=results.params)
