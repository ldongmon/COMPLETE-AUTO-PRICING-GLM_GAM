"""
lightweight_glm.py
==================
Phase 12: a GLM reduced to its coefficients, for the app.

WHY
---
A fitted statsmodels GLM keeps a copy of all its training data: the refined
frequency GLM file is about 190 MB, the severity GLM about 15 MB. Too big for
GitHub or a web app. To PRICE a policy, only the coefficients are needed:

    prediction = exp(b0 + b1 x1 + b2 x2 + ... + offset)

so this class keeps the coefficients only (a few KB) and gives exactly the
same predictions.

Example (IDpol 1006850, notebook 07): the frequency terms add up to
eta = -2.1126, so predict() returns exp(-2.1126) = 0.1209 claims per year,
the same number as the full statsmodels model.

It offers the two things the pricing function uses from a statsmodels
result: .predict(X, offset) and .model.exog_names (the column names).
"""

from types import SimpleNamespace  # a tiny object to hold model.exog_names

import numpy as np  # maths


class LightweightGLM:
    """Prediction-only GLM with a log link: just the coefficients."""

    def __init__(self, params):
        self.params = params.copy()  # the coefficients, one per column (named)
        self.model = SimpleNamespace(exog_names=list(self.params.index))  # same attribute as statsmodels: the column names

    def predict(self, X, offset=None):
        """exp(X x b + offset). X must contain the model's columns (with "const"); their order does not matter."""
        X_aligned = X[self.model.exog_names]  # put the columns in the order of the coefficients
        linear_predictor = X_aligned.to_numpy() @ self.params.to_numpy()  # b0 x 1 + b1 x x1 + ... for every row
        if offset is not None:
            linear_predictor = linear_predictor + np.asarray(offset)  # add log(exposure) when given
        return np.exp(linear_predictor)  # log link: back to the original scale

    @classmethod
    def from_results(cls, results):
        """Build it from a fitted statsmodels GLM (keeps results.params only)."""
        return cls(params=results.params)
