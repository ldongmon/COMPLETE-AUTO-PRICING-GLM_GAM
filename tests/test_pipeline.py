"""
Unit tests for src/features/pipeline.py.

Two things matter most here and get the most direct tests:
  1. The split is stratified correctly (claim-occurrence rate matches across
     train/validation/test).
  2. The encoder never leaks: thresholds/categories learned on one set and
     applied to another behave exactly as documented (capping uses the
     fitted threshold, not one recomputed from the new data; an unseen
     category becomes all-zero dummies instead of crashing).
"""

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import (
    PolicyFeatureEncoder,
    compare_split_balance,
    get_exposure_offset,
    get_frequency_target,
    get_severity_subset,
    train_val_test_split_policies,
)


@pytest.fixture
def toy_policies() -> pd.DataFrame:
    """
    200 synthetic policies with a deliberately rare claim rate (~5%, similar
    order of magnitude to the real portfolio) so the stratified-split test is
    meaningful: with too few positives, any split would look "balanced" by
    chance.
    """
    rng = np.random.default_rng(0)
    n = 200
    has_claim = rng.random(n) < 0.05
    return pd.DataFrame(
        {
            "IDpol": np.arange(n),
            "ClaimNb_freq_target": has_claim.astype(int),
            "Exposure": rng.uniform(0.05, 1.0, n),
            "DrivAge": rng.integers(18, 80, n),
            "BonusMalus": rng.integers(50, 150, n),
            "VehPower": rng.integers(4, 15, n),
            "VehAge": rng.integers(0, 20, n),
            "Density": rng.uniform(1, 20000, n),
            "VehBrand": rng.choice(["B1", "B2", "B3"], n),
            "VehGas": rng.choice(["Diesel", "Regular"], n),
            "Region": rng.choice(["R1", "R2"], n),
            "HasPaidClaim": has_claim,
            "sev_claim_amount": np.where(has_claim, rng.uniform(500, 5000, n), 0.0),
            "sev_claim_count": has_claim.astype(int),
            "AvgClaimAmount": np.where(has_claim, rng.uniform(500, 5000, n), np.nan),
        }
    )


def test_split_sizes_sum_to_original(toy_policies):
    train_df, val_df, test_df = train_val_test_split_policies(toy_policies, val_size=0.2, test_size=0.2)
    assert len(train_df) + len(val_df) + len(test_df) == len(toy_policies)


def test_split_has_no_duplicate_policies_across_sets(toy_policies):
    train_df, val_df, test_df = train_val_test_split_policies(toy_policies)
    train_ids, val_ids, test_ids = set(train_df["IDpol"]), set(val_df["IDpol"]), set(test_df["IDpol"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_preserves_claim_rate_reasonably_well(toy_policies):
    # With a 5% base rate and ~200 rows, exact equality across splits isn't
    # guaranteed at this small scale, but stratification should keep every
    # split within a wide, sane tolerance of the overall rate.
    overall_rate = (toy_policies["ClaimNb_freq_target"] > 0).mean()
    train_df, val_df, test_df = train_val_test_split_policies(toy_policies)
    for part in (train_df, val_df, test_df):
        part_rate = (part["ClaimNb_freq_target"] > 0).mean()
        assert abs(part_rate - overall_rate) < 0.05, f"split claim rate {part_rate} drifted too far from {overall_rate}"


def test_compare_split_balance_returns_one_row_per_split(toy_policies):
    train_df, val_df, test_df = train_val_test_split_policies(toy_policies)
    balance = compare_split_balance(train_df, val_df, test_df)
    assert list(balance["split"]) == ["train", "validation", "test"]
    assert balance["share_of_total"].sum() == pytest.approx(1.0)


def test_encoder_must_be_fit_before_transform(toy_policies):
    encoder = PolicyFeatureEncoder()
    with pytest.raises(RuntimeError):
        encoder.transform(toy_policies)


def test_encoder_transform_uses_the_fitted_threshold_not_a_recomputed_one(toy_policies):
    train_df, val_df, _ = train_val_test_split_policies(toy_policies)

    encoder = PolicyFeatureEncoder()
    encoder.fit(train_df)
    learned_cap = encoder.vehage_cap_

    # Force a row in val whose raw VehAge is far above whatever was learned
    # from train, and confirm transform() clips it to the *train* threshold.
    val_probe = val_df.copy()
    val_probe.loc[val_probe.index[0], "VehAge"] = learned_cap + 100

    X_val = encoder.transform(val_probe)
    assert X_val["VehAge_capped"].max() == pytest.approx(learned_cap)


def test_encoder_handles_unseen_category_without_error(toy_policies):
    train_df, val_df, _ = train_val_test_split_policies(toy_policies)
    encoder = PolicyFeatureEncoder()
    encoder.fit(train_df)

    probe = val_df.iloc[[0]].copy()
    probe["VehBrand"] = "NEVER_SEEN_IN_TRAINING"

    X = encoder.transform(probe)  # must not raise
    brand_cols = [c for c in X.columns if c.startswith("VehBrand_")]
    assert (X[brand_cols].to_numpy() == 0).all()


def test_encoder_output_has_no_missing_values(toy_policies):
    train_df, val_df, test_df = train_val_test_split_policies(toy_policies)
    encoder = PolicyFeatureEncoder()
    encoder.fit(train_df)
    for part in (train_df, val_df, test_df):
        X = encoder.transform(part)
        assert X.isnull().sum().sum() == 0


def test_encoder_never_includes_exposure_as_a_feature(toy_policies):
    encoder = PolicyFeatureEncoder()
    encoder.fit(toy_policies)
    X = encoder.transform(toy_policies)
    assert "Exposure" not in X.columns


def test_exposure_offset_is_log_of_exposure(toy_policies):
    offset = get_exposure_offset(toy_policies)
    np.testing.assert_allclose(offset.to_numpy(), np.log(toy_policies["Exposure"].to_numpy()))


def test_frequency_target_matches_column(toy_policies):
    y = get_frequency_target(toy_policies)
    pd.testing.assert_series_equal(y, toy_policies["ClaimNb_freq_target"])


def test_severity_subset_only_contains_paid_claims(toy_policies):
    subset = get_severity_subset(toy_policies)
    assert subset["HasPaidClaim"].all()
    assert len(subset) == toy_policies["HasPaidClaim"].sum()
