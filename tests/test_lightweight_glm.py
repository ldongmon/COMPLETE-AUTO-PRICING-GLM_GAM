"""
Unit tests for src/models/lightweight_glm.py: checks the distilled model
predicts identically to the real statsmodels GLM it was built from.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.models.lightweight_glm import LightweightGLM


@pytest.fixture
def fitted_glm_and_data():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({"x1": rng.uniform(-1, 1, n), "x2": rng.uniform(-1, 1, n)})
    exposure = rng.uniform(0.1, 1.0, n)
    y = rng.poisson(np.exp(-1.0 + 0.5 * X["x1"] - 0.3 * X["x2"]) * exposure)

    X_const = sm.add_constant(X, has_constant="add")
    model = sm.GLM(y, X_const, family=sm.families.Poisson(), offset=np.log(exposure))
    results = model.fit()
    return results, X_const, exposure


def test_lightweight_glm_predictions_match_the_real_model_exactly(fitted_glm_and_data):
    results, X_const, exposure = fitted_glm_and_data
    lite = LightweightGLM.from_results(results)

    real_pred = results.predict(X_const, offset=np.log(exposure))
    lite_pred = lite.predict(X_const, offset=np.log(exposure))

    np.testing.assert_allclose(real_pred, lite_pred, rtol=1e-10)


def test_lightweight_glm_predictions_match_without_offset(fitted_glm_and_data):
    results, X_const, _ = fitted_glm_and_data
    lite = LightweightGLM.from_results(results)

    real_pred = results.predict(X_const)
    lite_pred = lite.predict(X_const)

    np.testing.assert_allclose(real_pred, lite_pred, rtol=1e-10)


def test_lightweight_glm_exposes_exog_names_like_the_real_model(fitted_glm_and_data):
    results, _, _ = fitted_glm_and_data
    lite = LightweightGLM.from_results(results)
    assert lite.model.exog_names == list(results.model.exog_names)


def test_lightweight_glm_is_robust_to_column_order(fitted_glm_and_data):
    results, X_const, exposure = fitted_glm_and_data
    lite = LightweightGLM.from_results(results)

    # Shuffle the columns: predict() must reorder to match the fitted params, not assume order.
    shuffled = X_const[list(reversed(X_const.columns))]
    lite_pred = lite.predict(shuffled, offset=np.log(exposure))
    real_pred = results.predict(X_const, offset=np.log(exposure))

    np.testing.assert_allclose(real_pred, lite_pred, rtol=1e-10)


def test_lightweight_glm_params_are_a_copy_not_a_reference(fitted_glm_and_data):
    results, _, _ = fitted_glm_and_data
    lite = LightweightGLM.from_results(results)
    lite.params.iloc[0] = 999.0
    assert results.params.iloc[0] != 999.0
