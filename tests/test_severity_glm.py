"""
Unit tests for src/models/severity_glm.py, on small synthetic claims data.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.severity_glm import (
    backward_stepwise_selection_gamma,
    cap_severity,
    determine_large_loss_threshold,
    fit_gamma_glm,
    large_loss_loading,
)


@pytest.fixture
def toy_severity_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "AvgClaimAmount": rng.gamma(shape=2.0, scale=800.0, size=n),
            "sev_claim_count": rng.integers(1, 3, n),
        }
    ).assign(sev_claim_amount=lambda d: d["AvgClaimAmount"] * d["sev_claim_count"])


@pytest.fixture
def toy_full_portfolio_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 2000  # includes non-claiming policies, unlike toy_severity_df
    return pd.DataFrame({"Exposure": rng.uniform(0.1, 1.0, n)})


def test_determine_large_loss_threshold_is_a_quantile_of_training_data(toy_severity_df):
    threshold = determine_large_loss_threshold(toy_severity_df, percentile=0.9)
    assert threshold == pytest.approx(toy_severity_df["AvgClaimAmount"].quantile(0.9))


def test_cap_severity_never_exceeds_the_threshold(toy_severity_df):
    threshold = determine_large_loss_threshold(toy_severity_df, percentile=0.9)
    capped = cap_severity(toy_severity_df, threshold)
    assert capped.max() <= threshold + 1e-9


def test_cap_severity_leaves_small_claims_unchanged(toy_severity_df):
    threshold = toy_severity_df["AvgClaimAmount"].max() + 1  # above every claim: nothing should be capped
    capped = cap_severity(toy_severity_df, threshold)
    pd.testing.assert_series_equal(capped, toy_severity_df["AvgClaimAmount"], check_names=False)


def test_large_loss_loading_is_zero_when_threshold_covers_every_claim(toy_severity_df, toy_full_portfolio_df):
    threshold = toy_severity_df["AvgClaimAmount"].max() + 1
    loading = large_loss_loading(toy_full_portfolio_df, toy_severity_df, threshold)
    assert loading == pytest.approx(0.0)


def test_large_loss_loading_is_positive_when_some_claims_exceed_the_threshold(toy_severity_df, toy_full_portfolio_df):
    threshold = toy_severity_df["AvgClaimAmount"].quantile(0.5)  # roughly half the claims exceed this
    loading = large_loss_loading(toy_full_portfolio_df, toy_severity_df, threshold)
    assert loading > 0


def test_large_loss_loading_scales_with_total_portfolio_exposure(toy_severity_df, toy_full_portfolio_df):
    threshold = toy_severity_df["AvgClaimAmount"].quantile(0.5)
    loading_normal = large_loss_loading(toy_full_portfolio_df, toy_severity_df, threshold)

    double_exposure_df = toy_full_portfolio_df.copy()
    double_exposure_df["Exposure"] *= 2
    loading_double_exposure = large_loss_loading(double_exposure_df, toy_severity_df, threshold)

    # same total excess euros spread over 2x the exposure -> roughly half the per-policy-year loading
    assert loading_double_exposure == pytest.approx(loading_normal / 2)


def test_fit_gamma_glm_converges_and_predicts_positive_severity(toy_severity_df):
    X = pd.DataFrame({"x": np.random.default_rng(2).uniform(-1, 1, len(toy_severity_df))})
    results = fit_gamma_glm(X, toy_severity_df["AvgClaimAmount"], toy_severity_df["sev_claim_count"])
    assert results.converged
    predicted = results.predict()
    assert (predicted > 0).all()


def test_backward_stepwise_selection_gamma_drops_a_noise_factor():
    rng = np.random.default_rng(3)
    n = 4000
    real_factor = rng.uniform(-1, 1, n)
    noise_category_B = (rng.choice(["A", "B"], n) == "B").astype(float)
    claim_count = rng.integers(1, 3, n).astype(float)

    true_mean = np.exp(7.0 + 0.5 * real_factor)  # noise_category has NO effect by construction
    y = rng.gamma(shape=3.0, scale=true_mean / 3.0, size=n)

    X = pd.DataFrame({"real_factor": real_factor, "noise_category_B": noise_category_B})
    feature_groups = {"real_factor": ["real_factor"], "noise_category": ["noise_category_B"]}

    final_results, remaining_groups, history = backward_stepwise_selection_gamma(
        X, pd.Series(y), pd.Series(claim_count), feature_groups, p_threshold=0.05, verbose=False
    )

    assert "real_factor" in remaining_groups
    assert "noise_category" not in remaining_groups
