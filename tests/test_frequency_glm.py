"""
Unit tests for src/models/frequency_glm.py, on small synthetic data designed
so we know which factor *should* survive stepwise selection and which
shouldn't.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.frequency_glm import (
    backward_stepwise_selection,
    build_factor_groups,
    fit_poisson_glm,
    relativities_table,
)


def test_build_factor_groups_groups_one_hot_columns_together():
    feature_names = ["DrivAge", "BonusMalus", "VehPower", "log_Density", "VehAge_capped",
                      "VehBrand_B2", "VehBrand_B3", "VehGas_Regular", "Region_R21", "Region_R22"]
    groups = build_factor_groups(feature_names)

    assert groups["VehBrand"] == ["VehBrand_B2", "VehBrand_B3"]
    assert groups["Region"] == ["Region_R21", "Region_R22"]
    assert groups["Density"] == ["log_Density"]
    assert groups["VehAge"] == ["VehAge_capped"]
    # Every input column accounted for exactly once.
    all_grouped = sorted(c for cols in groups.values() for c in cols)
    assert all_grouped == sorted(feature_names)


def test_build_factor_groups_raises_on_unrecognized_column():
    with pytest.raises(ValueError):
        build_factor_groups(["DrivAge", "SomeUnmappedColumn"])


@pytest.fixture
def synthetic_policies():
    """
    3,000 synthetic policies with one genuinely predictive numeric factor
    (`real_factor`) and one categorical factor with zero true effect
    (`noise_category`): stepwise selection should keep the former and drop
    the latter.
    """
    rng = np.random.default_rng(1)
    n = 3000
    exposure = rng.uniform(0.1, 1.0, n)
    real_factor = rng.uniform(-1, 1, n)
    noise_category = rng.choice(["A", "B", "C"], n)

    true_rate = np.exp(-2.0 + 1.0 * real_factor)  # noise_category has NO effect by construction
    y = rng.poisson(true_rate * exposure)

    X = pd.DataFrame(
        {
            "real_factor": real_factor,
            "noise_category_B": (noise_category == "B").astype(float),
            "noise_category_C": (noise_category == "C").astype(float),
        }
    )
    return X, pd.Series(y, name="y"), pd.Series(np.log(exposure), name="offset")


def test_fit_poisson_glm_converges_and_includes_intercept(synthetic_policies):
    X, y, offset = synthetic_policies
    results = fit_poisson_glm(X, y, offset)
    assert results.converged
    assert "const" in results.params.index


def test_backward_stepwise_selection_drops_the_noise_factor(synthetic_policies):
    X, y, offset = synthetic_policies
    feature_groups = {
        "real_factor": ["real_factor"],
        "noise_category": ["noise_category_B", "noise_category_C"],
    }
    final_results, remaining_groups, history = backward_stepwise_selection(
        X, y, offset, feature_groups, p_threshold=0.05, verbose=False
    )

    assert "real_factor" in remaining_groups
    assert "noise_category" not in remaining_groups
    assert len(history) == 1
    assert history.iloc[0]["factor_removed"] == "noise_category"


def test_backward_stepwise_selection_keeps_a_real_factor_alone(synthetic_policies):
    X, y, offset = synthetic_policies
    # With only the real factor available, there's nothing left to drop.
    feature_groups = {"real_factor": ["real_factor"]}
    final_results, remaining_groups, history = backward_stepwise_selection(
        X[["real_factor"]], y, offset, feature_groups, verbose=False
    )
    assert remaining_groups == feature_groups
    assert len(history) == 0


def test_relativities_table_is_exp_of_coefficients(synthetic_policies):
    X, y, offset = synthetic_policies
    results = fit_poisson_glm(X[["real_factor"]], y, offset)
    table = relativities_table(results)
    np.testing.assert_allclose(table["relativity"], np.exp(table["coefficient"]))
    # A 95% CI built from +/- 1.96 SE should bracket the point estimate.
    assert (table["relativity_ci_low"] <= table["relativity"]).all()
    assert (table["relativity"] <= table["relativity_ci_high"]).all()
