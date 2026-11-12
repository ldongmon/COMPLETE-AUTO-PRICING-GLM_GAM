"""
Unit tests for src/models/severity_gam.py.
"""

import numpy as np
import pytest

from src.models.frequency_gam import GAMFeatureBuilder
from src.models.severity_gam import fit_gamma_gam


@pytest.fixture
def toy_gam_data():
    rng = np.random.default_rng(0)
    n = 500
    import pandas as pd

    df = pd.DataFrame(
        {
            "DrivAge": rng.integers(18, 80, n),
            "BonusMalus": rng.integers(50, 150, n),
            "VehPower": rng.integers(4, 15, n),
            "Density": rng.uniform(1, 20000, n),
            "VehAge": rng.integers(0, 24, n),
            "VehBrand": rng.choice(["B1", "B2"], n),
            "VehGas": rng.choice(["Diesel", "Regular"], n),
            "Region": rng.choice(["R1", "R2"], n),
        }
    )
    builder = GAMFeatureBuilder().fit(df, vehage_cap=20.0)
    X = builder.transform(df)
    y = rng.gamma(shape=2.0, scale=800.0, size=n)
    weights = rng.integers(1, 3, n).astype(float)
    return X, y, weights


def test_fit_gamma_gam_converges_and_predicts_positive_severity(toy_gam_data):
    X, y, weights = toy_gam_data
    gam = fit_gamma_gam(X, y, weights, n_splines=5)
    predicted = gam.predict(X)
    assert (predicted > 0).all()
    assert len(predicted) == len(y)


def test_fit_gamma_gam_works_without_weights(toy_gam_data):
    X, y, _ = toy_gam_data
    gam = fit_gamma_gam(X, y, weights=None, n_splines=5)
    predicted = gam.predict(X)
    assert (predicted > 0).all()
