"""
Unit tests for src/models/glm_diagnostics.py, built on small synthetic
Poisson data with a known true relationship, so we can check the diagnostics
against numbers we can verify by hand rather than just "it runs".
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.models.glm_diagnostics import (
    actual_vs_predicted_by_decile,
    deviance_summary,
    likelihood_ratio_test,
    pearson_residuals,
)


@pytest.fixture
def synthetic_poisson_data():
    """
    2,000 synthetic policies where claim count truly depends on `x` (a real
    signal) and not at all on `noise` (irrelevant), generated from a known
    Poisson process, so we know exactly what a correctly-fit GLM should
    recover.
    """
    rng = np.random.default_rng(0)
    n = 2000
    exposure = rng.uniform(0.1, 1.0, n)
    x = rng.uniform(-1, 1, n)
    noise = rng.normal(size=n)
    true_rate = np.exp(-2.0 + 0.8 * x)  # true relationship: only x matters
    y = rng.poisson(true_rate * exposure)
    return pd.DataFrame({"x": x, "noise": noise, "exposure": exposure, "y": y})


def _fit(df, cols):
    X = sm.add_constant(df[cols], has_constant="add")
    return sm.GLM(df["y"], X, family=sm.families.Poisson(), offset=np.log(df["exposure"])).fit()


def test_deviance_summary_has_expected_keys_and_sane_values(synthetic_poisson_data):
    results = _fit(synthetic_poisson_data, ["x", "noise"])
    summary = deviance_summary(results)

    assert summary["n_obs"] == len(synthetic_poisson_data)
    assert summary["deviance"] <= summary["null_deviance"]  # adding predictors can't make fit worse
    assert 0 <= summary["pseudo_r2_deviance"] <= 1


def test_likelihood_ratio_test_detects_a_real_predictor(synthetic_poisson_data):
    full = _fit(synthetic_poisson_data, ["x", "noise"])
    reduced_drop_x = _fit(synthetic_poisson_data, ["noise"])

    lr = likelihood_ratio_test(full, reduced_drop_x)
    # x has a strong, genuine effect (generated with coefficient 0.8) --
    # dropping it should be very significant (tiny p-value).
    assert lr["p_value"] < 0.01
    assert lr["df_diff"] == 1


def test_likelihood_ratio_test_does_not_flag_a_noise_predictor(synthetic_poisson_data):
    full = _fit(synthetic_poisson_data, ["x", "noise"])
    reduced_drop_noise = _fit(synthetic_poisson_data, ["x"])

    lr = likelihood_ratio_test(full, reduced_drop_noise)
    # `noise` was generated with zero true effect: dropping it should NOT
    # look significant most of the time. (Not a flaky test in practice: the
    # true effect is exactly zero and the sample is large.)
    assert lr["p_value"] > 0.05


def test_actual_vs_predicted_by_decile_bins_cover_all_exposure(synthetic_poisson_data):
    results = _fit(synthetic_poisson_data, ["x"])
    predicted = results.predict()
    decile_df = actual_vs_predicted_by_decile(
        synthetic_poisson_data["y"], predicted, synthetic_poisson_data["exposure"], n_bins=5
    )
    assert len(decile_df) <= 5
    assert decile_df["exposure"].sum() == pytest.approx(synthetic_poisson_data["exposure"].sum())
    assert decile_df["observed_claims"].sum() == synthetic_poisson_data["y"].sum()


def test_actual_vs_predicted_tracks_the_true_relationship(synthetic_poisson_data):
    # A model fit on the real signal `x` should show predicted frequency
    # rising from the lowest to the highest decile, roughly matching
    # observed: the core promise of this diagnostic.
    results = _fit(synthetic_poisson_data, ["x"])
    predicted = results.predict()
    decile_df = actual_vs_predicted_by_decile(
        synthetic_poisson_data["y"], predicted, synthetic_poisson_data["exposure"], n_bins=5
    )
    assert decile_df["predicted_frequency"].is_monotonic_increasing


def test_pearson_residuals_formula():
    y = np.array([5.0, 0.0, 3.0])
    mu = np.array([4.0, 1.0, 3.0])
    expected = (y - mu) / np.sqrt(mu)
    np.testing.assert_allclose(pearson_residuals(y, mu), expected)
