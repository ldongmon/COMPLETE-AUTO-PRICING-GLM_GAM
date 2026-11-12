"""
Unit tests for scoring/score_technical_premium.py, built on small synthetic
fitted models so the arithmetic can be checked by hand.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import PolicyFeatureEncoder
from src.models.frequency_glm import fit_poisson_glm
from src.models.frequency_glm_refined import RefinedFeatureEncoder
from src.models.lightweight_glm import LightweightGLM
from src.models.severity_glm import fit_gamma_glm

from scoring.score_technical_premium import TechnicalPremiumModel, score_technical_premium


@pytest.fixture
def toy_policies() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "DrivAge": rng.integers(18, 90, n),
            "BonusMalus": rng.integers(50, 150, n),
            "VehPower": rng.integers(4, 15, n),
            "VehAge": rng.integers(0, 24, n),
            "Density": rng.uniform(1, 20000, n),
            "VehBrand": rng.choice(["B1", "B2", "B3"], n),
            "VehGas": rng.choice(["Diesel", "Regular"], n),
            "Region": rng.choice(["R1", "R2"], n),
            "Exposure": rng.uniform(0.1, 1.0, n),
        }
    )
    df["ClaimNb_freq_target"] = rng.poisson(0.1 * df["Exposure"])
    df["AvgClaimAmount"] = rng.gamma(2.0, 800.0, n)
    df["sev_claim_count"] = 1
    return df


@pytest.fixture
def fitted_model(toy_policies, tmp_path) -> TechnicalPremiumModel:
    freq_encoder = RefinedFeatureEncoder().fit(toy_policies)
    X_freq = freq_encoder.transform(toy_policies)
    freq_results = fit_poisson_glm(
        X_freq, toy_policies["ClaimNb_freq_target"], np.log(toy_policies["Exposure"])
    )

    sev_encoder = PolicyFeatureEncoder().fit(toy_policies)
    X_sev = sev_encoder.transform(toy_policies)
    sev_results = fit_gamma_glm(X_sev, toy_policies["AvgClaimAmount"], toy_policies["sev_claim_count"])

    return TechnicalPremiumModel(
        freq_results=freq_results,
        freq_encoder=freq_encoder,
        sev_results=sev_results,
        sev_encoder=sev_encoder,
        large_loss_loading=25.0,
    )


def test_technical_premium_equals_frequency_times_severity_plus_loading(toy_policies, fitted_model):
    scored = score_technical_premium(toy_policies, fitted_model)
    expected = scored["predicted_frequency"] * scored["predicted_severity"] + fitted_model.large_loss_loading
    np.testing.assert_allclose(scored["technical_premium"], expected)


def test_predicted_frequency_is_independent_of_actual_exposure(toy_policies, fitted_model):
    # Two otherwise-identical policies with different Exposure should get the SAME predicted
    # frequency: it's an annualized (per policy-year) rate, evaluated at offset=0 every time,
    # not scaled by whatever exposure a specific historical row happened to have.
    row_a = toy_policies.iloc[[0]].copy()
    row_b = row_a.copy()
    row_b["Exposure"] = row_a["Exposure"].iloc[0] * 3  # different exposure, same everything else

    scored_a = score_technical_premium(row_a, fitted_model)
    scored_b = score_technical_premium(row_b, fitted_model)

    np.testing.assert_allclose(scored_a["predicted_frequency"], scored_b["predicted_frequency"])
    np.testing.assert_allclose(scored_a["technical_premium"], scored_b["technical_premium"])


def test_large_loss_loading_is_constant_across_rows(toy_policies, fitted_model):
    scored = score_technical_premium(toy_policies, fitted_model)
    assert (scored["large_loss_loading"] == fitted_model.large_loss_loading).all()


def test_output_index_matches_input_index(toy_policies, fitted_model):
    subset = toy_policies.iloc[5:10]
    scored = score_technical_premium(subset, fitted_model)
    assert list(scored.index) == list(subset.index)


def test_technical_premium_is_always_positive(toy_policies, fitted_model):
    scored = score_technical_premium(toy_policies, fitted_model)
    assert (scored["technical_premium"] > 0).all()


def test_model_load_round_trips_through_disk(toy_policies, fitted_model, tmp_path):
    joblib.dump(
        {"results": fitted_model.freq_results, "encoder": fitted_model.freq_encoder},
        tmp_path / "freq_glm_refined.pkl",
    )
    joblib.dump(
        {
            "results": fitted_model.sev_results,
            "feature_columns": list(fitted_model.sev_results.model.exog_names[1:]),
            "large_loss_threshold": 16212.0,
            "large_loss_loading": fitted_model.large_loss_loading,
        },
        tmp_path / "sev_glm.pkl",
    )
    joblib.dump(fitted_model.sev_encoder, tmp_path / "feature_encoder.joblib")

    loaded = TechnicalPremiumModel.load(tmp_path)
    scored_original = score_technical_premium(toy_policies, fitted_model)
    scored_loaded = score_technical_premium(toy_policies, loaded)

    np.testing.assert_allclose(scored_original["technical_premium"], scored_loaded["technical_premium"])


def test_load_lite_matches_load_exactly(toy_policies, fitted_model, tmp_path):
    """
    The Phase 12 app loading path (`load_lite`, reading `LightweightGLM`
    artifacts from a `models/app/` subfolder) must score identically to the
    full `load()` path used by every earlier-phase notebook: the whole
    point of `src/models/lightweight_glm.py` is that distillation changes
    nothing about the numbers, only the artifact size.
    """
    app_dir = tmp_path / "app"
    app_dir.mkdir()

    joblib.dump(
        {"results": LightweightGLM.from_results(fitted_model.freq_results), "encoder": fitted_model.freq_encoder},
        app_dir / "freq_glm_refined.pkl",
    )
    joblib.dump(
        {
            "results": LightweightGLM.from_results(fitted_model.sev_results),
            "large_loss_threshold": 16212.0,
            "large_loss_loading": fitted_model.large_loss_loading,
        },
        app_dir / "sev_glm.pkl",
    )
    joblib.dump(fitted_model.sev_encoder, app_dir / "feature_encoder.joblib")

    lite = TechnicalPremiumModel.load_lite(tmp_path)
    scored_full = score_technical_premium(toy_policies, fitted_model)
    scored_lite = score_technical_premium(toy_policies, lite)

    np.testing.assert_allclose(scored_full["technical_premium"], scored_lite["technical_premium"])
    np.testing.assert_allclose(scored_full["predicted_frequency"], scored_lite["predicted_frequency"])
    np.testing.assert_allclose(scored_full["predicted_severity"], scored_lite["predicted_severity"])
