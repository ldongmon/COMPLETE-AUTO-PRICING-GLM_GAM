"""
Unit tests for src/eda/profiling.py: built on a small, hand-crafted
DataFrame (not the full 678k-row dataset) so the tests are fast and the
expected numbers can be checked by hand.
"""

import pandas as pd
import pytest

from src.eda.profiling import bin_continuous, severity_by_factor, weighted_frequency_by_factor


@pytest.fixture
def toy_policies() -> pd.DataFrame:
    # 4 policies, 2 in group "X" and 2 in group "Y".
    # Group X: 1 claim over 0.5 years exposure -> frequency = 1 / (0.5+0.5) = 1.0
    # Group Y: 1 claim over 2.0 years exposure  -> frequency = 1 / (1.0+1.0) = 0.5
    return pd.DataFrame(
        {
            "group": ["X", "X", "Y", "Y"],
            "ClaimNb_freq_target": [1, 0, 1, 0],
            "Exposure": [0.5, 0.5, 1.0, 1.0],
            "sev_claim_amount": [300.0, 0.0, 900.0, 0.0],
            "sev_claim_count": [1, 0, 1, 0],
            "HasPaidClaim": [True, False, True, False],
        }
    )


def test_weighted_frequency_matches_hand_calculation(toy_policies):
    result = weighted_frequency_by_factor(toy_policies, "group").set_index("group")

    # Group X: 1 claim / 1.0 total exposure = 1.0
    assert result.loc["X", "frequency"] == pytest.approx(1.0)
    # Group Y: 1 claim / 2.0 total exposure = 0.5
    assert result.loc["Y", "frequency"] == pytest.approx(0.5)


def test_weighted_frequency_is_not_a_naive_row_average(toy_policies):
    # The naive (wrong) row average of ClaimNb within group X is 0.5 --
    # the correct exposure-weighted frequency is 1.0. This test exists to
    # catch a regression that silently reverts to the naive estimator, which
    # is exactly the mistake this module exists to prevent (see its module
    # docstring and notebooks/02_exploratory_data_analysis.ipynb Section 1).
    naive_x = toy_policies.loc[toy_policies["group"] == "X", "ClaimNb_freq_target"].mean()
    weighted_x = weighted_frequency_by_factor(toy_policies, "group").set_index("group").loc["X", "frequency"]
    assert naive_x != pytest.approx(weighted_x)


def test_exposure_share_sums_to_one(toy_policies):
    result = weighted_frequency_by_factor(toy_policies, "group")
    assert result["exposure_share"].sum() == pytest.approx(1.0)


def test_severity_only_uses_paid_claims(toy_policies):
    result = severity_by_factor(toy_policies, "group").set_index("group")
    assert result.loc["X", "avg_severity"] == pytest.approx(300.0)
    assert result.loc["Y", "avg_severity"] == pytest.approx(900.0)


def test_bin_continuous_assigns_expected_labels():
    ages = pd.Series([18, 25, 40, 70])
    binned = bin_continuous(ages, bins=[18, 30, 50, 101], labels=["18-29", "30-49", "50+"])
    assert list(binned) == ["18-29", "18-29", "30-49", "50+"]
