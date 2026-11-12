"""
Unit tests for the severity-specific diagnostics added to
src/models/glm_diagnostics.py: gamma_deviance, out_of_sample_pseudo_r2_gamma,
calibration_by_decile.
"""

import numpy as np
import pytest

from src.models.glm_diagnostics import (
    calibration_by_decile,
    gamma_deviance,
    out_of_sample_pseudo_r2_gamma,
)


def test_gamma_deviance_is_zero_for_a_perfect_prediction():
    y = np.array([500.0, 1200.0, 3000.0, 800.0])
    assert gamma_deviance(y, y) == pytest.approx(0.0, abs=1e-6)


def test_gamma_deviance_increases_with_worse_predictions():
    y = np.array([500.0, 1200.0, 3000.0, 800.0])
    good_pred = np.array([520.0, 1150.0, 2900.0, 810.0])
    bad_pred = np.array([2000.0, 300.0, 500.0, 4000.0])
    assert gamma_deviance(y, bad_pred) > gamma_deviance(y, good_pred)


def test_gamma_deviance_weights_scale_the_contribution():
    y = np.array([1000.0, 1000.0])
    pred = np.array([500.0, 500.0])
    unweighted = gamma_deviance(y, pred)
    weighted = gamma_deviance(y, pred, weights=np.array([1.0, 3.0]))
    assert weighted > unweighted  # the second, higher-weight observation counts for more


def test_out_of_sample_pseudo_r2_gamma_is_zero_for_the_null_model_itself():
    rng = np.random.default_rng(0)
    y = rng.gamma(shape=2.0, scale=800.0, size=500)
    overall_mean = y.mean()
    null_pred = np.full_like(y, overall_mean)

    result = out_of_sample_pseudo_r2_gamma(y, null_pred)
    assert result["pseudo_r2"] == pytest.approx(0.0, abs=1e-6)


def test_out_of_sample_pseudo_r2_gamma_is_positive_for_a_model_using_real_signal():
    rng = np.random.default_rng(1)
    n = 5000
    x = rng.uniform(-1, 1, n)
    true_mean = np.exp(7.0 + 1.0 * x)
    y = rng.gamma(shape=3.0, scale=true_mean / 3.0, size=n)

    informed_pred = true_mean
    result = out_of_sample_pseudo_r2_gamma(y, informed_pred)
    assert result["pseudo_r2"] > 0


def test_calibration_by_decile_preserves_total_weight():
    rng = np.random.default_rng(2)
    n = 300
    y = rng.gamma(2.0, 800.0, n)
    pred = rng.gamma(2.0, 800.0, n)
    weights = rng.integers(1, 4, n).astype(float)

    decile_df = calibration_by_decile(y, pred, weights, n_bins=10)
    assert decile_df["total_weight"].sum() == pytest.approx(weights.sum())
    assert decile_df["n_rows"].sum() == n


def test_calibration_by_decile_orders_predicted_increasing():
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.uniform(-1, 1, n)
    pred = np.exp(7.0 + x)
    y = pred  # perfectly matched, for a clean monotonic check

    decile_df = calibration_by_decile(y, pred, weights=None, n_bins=5)
    assert decile_df["predicted"].is_monotonic_increasing
