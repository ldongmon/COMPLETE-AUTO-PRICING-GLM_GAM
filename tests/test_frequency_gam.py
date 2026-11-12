"""
Unit tests for src/models/frequency_gam.py, on small synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

import src.models.frequency_gam as frequency_gam
from src.models.frequency_gam import ALL_GAM_COLS, GAMFeatureBuilder, fit_poisson_gam, fit_poisson_gam_with_fallback


@pytest.fixture
def toy_policies() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame(
        {
            "DrivAge": rng.integers(18, 80, n),
            "BonusMalus": rng.integers(50, 150, n),
            "VehPower": rng.integers(4, 15, n),
            "Density": rng.uniform(1, 20000, n),
            "VehAge": rng.integers(0, 100, n),  # includes values above any reasonable cap
            "VehBrand": rng.choice(["B1", "B2", "B3"], n),
            "VehGas": rng.choice(["Diesel", "Regular"], n),
            "Region": rng.choice(["R1", "R2", "R3"], n),
            "Exposure": rng.uniform(0.1, 1.0, n),
        }
    ).assign(ClaimNb_freq_target=lambda d: rng.poisson(0.1 * d["Exposure"]))


def test_builder_must_be_fit_before_transform(toy_policies):
    builder = GAMFeatureBuilder()
    with pytest.raises(RuntimeError):
        builder.transform(toy_policies)


def test_transform_output_shape_matches_all_gam_cols(toy_policies):
    builder = GAMFeatureBuilder().fit(toy_policies, vehage_cap=20.0)
    X = builder.transform(toy_policies)
    assert X.shape == (len(toy_policies), len(ALL_GAM_COLS))


def test_vehage_is_capped_at_the_fitted_threshold(toy_policies):
    builder = GAMFeatureBuilder().fit(toy_policies, vehage_cap=20.0)
    X = builder.transform(toy_policies)
    vehage_col_index = ALL_GAM_COLS.index("VehAge_capped")
    assert X[:, vehage_col_index].max() <= 20.0
    # the raw data does contain values above the cap: otherwise this test wouldn't prove anything
    assert toy_policies["VehAge"].max() > 20.0


def test_unseen_category_does_not_raise(toy_policies):
    builder = GAMFeatureBuilder().fit(toy_policies, vehage_cap=20.0)
    probe = toy_policies.iloc[[0]].copy()
    probe["VehBrand"] = "NEVER_SEEN"
    X = builder.transform(probe)  # must not raise
    assert np.isfinite(X).all()


def test_no_missing_or_infinite_values_in_output(toy_policies):
    builder = GAMFeatureBuilder().fit(toy_policies, vehage_cap=20.0)
    X = builder.transform(toy_policies)
    assert np.isfinite(X).all()


def test_fit_poisson_gam_converges_and_predicts_positive_rates(toy_policies):
    builder = GAMFeatureBuilder().fit(toy_policies, vehage_cap=20.0)
    X = builder.transform(toy_policies)
    y = toy_policies["ClaimNb_freq_target"].to_numpy(dtype=float)
    exposure = toy_policies["Exposure"].to_numpy(dtype=float)

    gam = fit_poisson_gam(X, y, exposure, n_splines=5)
    predicted = gam.predict(X, exposure=exposure)
    assert (predicted > 0).all()
    assert len(predicted) == len(toy_policies)


def test_fallback_uses_the_first_config_when_it_succeeds(toy_policies, monkeypatch):
    builder = GAMFeatureBuilder().fit(toy_policies, vehage_cap=20.0)
    calls = []

    def fake_fit(X, y, exposure, n_splines=10):
        calls.append(n_splines)
        return "FAKE_GAM"

    monkeypatch.setattr(frequency_gam, "fit_poisson_gam", fake_fit)
    gam, config = fit_poisson_gam_with_fallback(
        toy_policies, builder, configs=[(0.5, 6), (0.2, 4)], verbose=False
    )
    assert gam == "FAKE_GAM"
    assert config["train_size"] == 0.5
    assert config["n_splines"] == 6
    assert config["n_rows"] == pytest.approx(0.5 * len(toy_policies), abs=5)
    assert calls == [6]  # only the first, most-ambitious config was tried


def test_fallback_moves_to_the_next_config_on_memory_error(toy_policies, monkeypatch):
    builder = GAMFeatureBuilder().fit(toy_policies, vehage_cap=20.0)
    calls = []

    def flaky_fit(X, y, exposure, n_splines=10):
        calls.append(n_splines)
        if len(calls) < 3:
            raise MemoryError("simulated out-of-memory")
        return "FAKE_GAM"

    monkeypatch.setattr(frequency_gam, "fit_poisson_gam", flaky_fit)
    gam, config = fit_poisson_gam_with_fallback(
        toy_policies, builder, configs=[(0.5, 10), (0.3, 8), (0.1, 6)], verbose=False
    )
    assert gam == "FAKE_GAM"
    assert config["n_splines"] == 6  # the third (last, most conservative) config succeeded
    assert calls == [10, 8, 6]  # tried in order, one MemoryError per rejected config


def test_fallback_raises_if_every_config_fails(toy_policies, monkeypatch):
    def always_fails(X, y, exposure, n_splines=10):
        raise MemoryError("simulated out-of-memory")

    monkeypatch.setattr(frequency_gam, "fit_poisson_gam", always_fails)
    builder = GAMFeatureBuilder().fit(toy_policies, vehage_cap=20.0)

    with pytest.raises(MemoryError):
        fit_poisson_gam_with_fallback(toy_policies, builder, configs=[(0.5, 6), (0.2, 4)], verbose=False)
