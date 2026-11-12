"""
Unit tests for src/data/load_data.py.

These tests exercise the *cached* path only (no network calls in CI/CD) by
relying on the Parquet cache populated once by an earlier run of the loader
(see docs/PROJECT_ROADMAP.md Phase 1). If the cache doesn't exist yet, the
first test call will trigger a real download: that's intentional (keeps the
test suite usable on a fresh checkout) but means the very first `pytest` run
needs internet access.
"""

import pandas as pd
import pytest

from src.data.load_data import (
    FREQ_DTYPES,
    SEV_DTYPES,
    drop_claims_only_block,
    find_claims_only_block_end,
    load_freq_raw,
    load_sev_raw,
)


@pytest.fixture(scope="module")
def freq_df() -> pd.DataFrame:
    return load_freq_raw()


@pytest.fixture(scope="module")
def sev_df() -> pd.DataFrame:
    return load_sev_raw()


def test_freq_has_expected_columns_and_dtypes(freq_df):
    assert list(freq_df.columns) == list(FREQ_DTYPES.keys())
    for col, dtype in FREQ_DTYPES.items():
        assert str(freq_df[col].dtype) == dtype, f"{col}: expected {dtype}, got {freq_df[col].dtype}"


def test_freq_ids_are_unique(freq_df):
    # Documented finding (Phase 1 / data_dictionary.md): no duplicate IDpol
    # in this OpenML pull. If a future re-download changes this, this test
    # should fail loudly rather than let a silent assumption drift.
    assert freq_df["IDpol"].is_unique


def test_freq_exposure_is_positive(freq_df):
    assert (freq_df["Exposure"] > 0).all()


def test_freq_claim_counts_are_non_negative(freq_df):
    assert (freq_df["ClaimNb"] >= 0).all()


def test_freq_has_no_missing_values(freq_df):
    assert freq_df.isnull().sum().sum() == 0


def test_sev_has_expected_columns_and_dtypes(sev_df):
    assert list(sev_df.columns) == list(SEV_DTYPES.keys())
    for col, dtype in SEV_DTYPES.items():
        assert str(sev_df[col].dtype) == dtype


def test_sev_claim_amounts_are_positive(sev_df):
    assert (sev_df["ClaimAmount"] > 0).all()


def test_claims_only_block_boundary_is_found_from_data():
    toy = pd.DataFrame({"IDpol": [1, 2, 3, 7, 8, 9], "ClaimNb": [1, 2, 1, 0, 1, 0]})
    assert find_claims_only_block_end(toy) == 3
    kept = drop_claims_only_block(toy, 3)
    assert kept["IDpol"].tolist() == [7, 8, 9]


def test_claims_only_block_on_real_data(freq_df):
    # Documented in Phase 1 / data_dictionary.md: every policy up to IDpol 24,951
    # has a reported claim; the first claim-free policy is IDpol 24,952.
    end = find_claims_only_block_end(freq_df)
    assert end == 24951
    block = freq_df[freq_df["IDpol"] <= end]
    assert (block["ClaimNb"] > 0).all()
    assert len(block) == 9387
