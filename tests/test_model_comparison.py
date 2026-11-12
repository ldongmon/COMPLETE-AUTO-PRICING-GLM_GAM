"""
Unit tests for src/models/model_comparison.py (Gini/Lorenz) and the
out-of-sample deviance helpers added to src/models/glm_diagnostics.py.
"""

import numpy as np
import pytest

from src.models.glm_diagnostics import out_of_sample_pseudo_r2, poisson_deviance
from src.models.model_comparison import gini_coefficient, lorenz_curve_data


def test_poisson_deviance_is_zero_for_a_perfect_prediction():
    y = np.array([0.0, 1.0, 3.0, 0.0, 2.0])
    assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-9)


def test_poisson_deviance_handles_zero_observed_counts_without_warning():
    y = np.array([0.0, 0.0, 0.0])
    mu = np.array([0.5, 0.2, 0.1])
    with np.errstate(all="raise"):  # would raise if the y*log(y/mu) term mishandled y=0
        dev = poisson_deviance(y, mu)
    assert dev > 0  # predicting a positive mean when nothing occurred is still penalized


def test_poisson_deviance_increases_with_worse_predictions():
    y = np.array([0.0, 1.0, 1.0, 5.0, 0.0])
    good_pred = np.array([0.1, 0.9, 1.1, 4.8, 0.2])
    bad_pred = np.array([2.0, 0.1, 0.1, 0.5, 3.0])
    assert poisson_deviance(y, bad_pred) > poisson_deviance(y, good_pred)


def test_out_of_sample_pseudo_r2_is_zero_for_the_null_model_itself():
    rng = np.random.default_rng(0)
    exposure = rng.uniform(0.1, 1.0, 500)
    y = rng.poisson(0.1 * exposure)
    overall_rate = y.sum() / exposure.sum()
    null_pred = overall_rate * exposure  # predicting exactly the null model

    result = out_of_sample_pseudo_r2(y, null_pred, exposure)
    assert result["pseudo_r2"] == pytest.approx(0.0, abs=1e-9)


def test_out_of_sample_pseudo_r2_is_positive_for_a_model_using_real_signal():
    rng = np.random.default_rng(1)
    n = 5000
    exposure = rng.uniform(0.1, 1.0, n)
    x = rng.uniform(-1, 1, n)
    true_rate = np.exp(-2.0 + 1.0 * x)
    y = rng.poisson(true_rate * exposure)

    # A model that uses the true signal should beat the null model.
    informed_pred = true_rate * exposure
    result = out_of_sample_pseudo_r2(y, informed_pred, exposure)
    assert result["pseudo_r2"] > 0


def test_lorenz_curve_starts_at_origin_and_ends_at_one():
    rng = np.random.default_rng(2)
    n = 200
    exposure = rng.uniform(0.1, 1.0, n)
    predicted = rng.uniform(0.01, 0.5, n) * exposure
    y = rng.poisson(predicted)

    lorenz = lorenz_curve_data(y, predicted, exposure)
    assert lorenz["cum_exposure_share"].iloc[0] == 0.0
    assert lorenz["cum_claims_share"].iloc[0] == 0.0
    assert lorenz["cum_exposure_share"].iloc[-1] == pytest.approx(1.0)
    assert lorenz["cum_claims_share"].iloc[-1] == pytest.approx(1.0)


def test_gini_is_near_zero_when_predictions_carry_no_information():
    # Predicted risk uncorrelated with actual risk -> ranking is close to
    # random -> Gini should be small in magnitude.
    rng = np.random.default_rng(3)
    n = 20000
    exposure = rng.uniform(0.1, 1.0, n)
    true_rate = 0.1  # same risk for everyone
    y = rng.poisson(true_rate * exposure)
    uninformative_predicted = rng.uniform(0.01, 0.5, n) * exposure  # random, unrelated to y

    lorenz = lorenz_curve_data(y, uninformative_predicted, exposure)
    gini = gini_coefficient(lorenz)
    assert abs(gini) < 0.05


def test_gini_is_substantially_positive_when_predictions_rank_risk_correctly():
    rng = np.random.default_rng(4)
    n = 20000
    exposure = rng.uniform(0.1, 1.0, n)
    x = rng.uniform(-1, 1, n)
    true_rate = np.exp(-2.0 + 2.0 * x)  # strong, real signal
    y = rng.poisson(true_rate * exposure)
    informed_predicted = true_rate * exposure  # perfectly informed ranking

    lorenz = lorenz_curve_data(y, informed_predicted, exposure)
    gini = gini_coefficient(lorenz)
    assert gini > 0.2
