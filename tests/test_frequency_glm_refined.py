"""
Unit tests for src/models/frequency_glm_refined.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.frequency_glm_refined import (
    DRIVAGE_BIN_LABELS,
    VEHAGE_BIN_LABELS,
    RefinedFeatureEncoder,
    _bin_drivage,
    _bin_vehage,
)


@pytest.fixture
def toy_policies() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "DrivAge": rng.integers(18, 100, n),
            "VehAge": rng.integers(0, 40, n),  # includes values above the cap, like the real data
            "BonusMalus": rng.integers(50, 150, n),
            "VehPower": rng.integers(4, 15, n),
            "Density": rng.uniform(1, 20000, n),
            "VehBrand": rng.choice(["B1", "B2", "B3"], n),
            "VehGas": rng.choice(["Diesel", "Regular"], n),
            "Region": rng.choice(["R1", "R2", "R3"], n),
        }
    )


def test_bin_drivage_covers_the_full_18_to_100_range():
    ages = pd.Series([18, 24, 25, 49, 50, 64, 65, 79, 80, 100])
    binned = _bin_drivage(ages)
    assert not binned.isin(["nan"]).any()
    assert set(binned.unique()) <= set(DRIVAGE_BIN_LABELS)


def test_bin_drivage_edge_cases_land_in_the_right_bin():
    ages = pd.Series([18, 24, 25, 34])
    binned = _bin_drivage(ages)
    assert list(binned) == ["18-24", "18-24", "25-34", "25-34"]


def test_bin_vehage_caps_before_binning():
    ages = pd.Series([0, 24, 100])  # 100 is far above the cap
    binned = _bin_vehage(ages)
    assert binned.iloc[0] == "0-10"
    assert binned.iloc[2] == binned.iloc[1] == "19-24"  # both cap to the same top bin


def test_encoder_must_be_fit_before_transform(toy_policies):
    encoder = RefinedFeatureEncoder()
    with pytest.raises(RuntimeError):
        encoder.transform(toy_policies)


def test_encoder_output_has_no_missing_values(toy_policies):
    encoder = RefinedFeatureEncoder().fit(toy_policies)
    X = encoder.transform(toy_policies)
    assert X.isnull().sum().sum() == 0


def test_encoder_includes_binned_age_dummy_columns(toy_policies):
    encoder = RefinedFeatureEncoder().fit(toy_policies)
    X = encoder.transform(toy_policies)
    assert any(c.startswith("DrivAge_bin_") for c in X.columns)
    assert any(c.startswith("VehAge_bin_") for c in X.columns)
    # the raw linear DrivAge/VehAge columns should NOT appear: they've been replaced by bins
    assert "DrivAge" not in X.columns
    assert "VehAge_capped" not in X.columns


def test_encoder_handles_unseen_category_without_error(toy_policies):
    encoder = RefinedFeatureEncoder().fit(toy_policies)
    probe = toy_policies.iloc[[0]].copy()
    probe["VehBrand"] = "NEVER_SEEN"
    X = encoder.transform(probe)  # must not raise
    brand_cols = [c for c in X.columns if c.startswith("VehBrand_")]
    assert (X[brand_cols].to_numpy() == 0).all()
