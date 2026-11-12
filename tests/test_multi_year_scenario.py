"""
Unit tests for scoring/scenario_engine.py's project_multi_year_scenario.
"""

import numpy as np
import pandas as pd
import pytest

from scoring.scenario_engine import project_multi_year_scenario, run_repricing_scenario
from src.prospect.conversion_model import BASELINE_CONVERSION_RATE
from src.retention.renewal_model import BASELINE_RENEWAL_RATE


@pytest.fixture
def toy_segment() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    tech = rng.uniform(140, 200, n)
    comm = tech * 1.10 + rng.normal(0, 5, n)
    return pd.DataFrame(
        {
            "commercial_premium": comm,
            "technical_premium": tech,
            "tenure_years_proxy": rng.uniform(0, 13.5, n),
            "BonusMalus": rng.uniform(50, 150, n),
        }
    )


def test_returns_one_row_per_year(toy_segment):
    projection = project_multi_year_scenario(toy_segment, price_change_pct=0.10, n_years=3)
    assert list(projection["year"]) == [1, 2, 3]


def test_year_one_matches_run_repricing_scenario_exactly(toy_segment):
    projection = project_multi_year_scenario(toy_segment, price_change_pct=0.10, n_years=1)
    single_year = run_repricing_scenario(toy_segment, price_change_pct=0.10)

    year1 = projection.iloc[0]
    assert year1["policies"] == pytest.approx(single_year["total_policies"])
    assert year1["written_premium"] == pytest.approx(single_year["total_written_premium"])
    assert year1["incurred_claims"] == pytest.approx(single_year["retained_claims"] + single_year["new_claims"])


def test_year_two_policy_count_follows_the_steady_state_formula(toy_segment):
    projection = project_multi_year_scenario(toy_segment, price_change_pct=0.10, n_years=2)
    year1_policies = projection.iloc[0]["policies"]
    quote_volume = len(toy_segment)  # default quote_volume_multiplier=1.0

    expected_year2 = year1_policies * BASELINE_RENEWAL_RATE + quote_volume * BASELINE_CONVERSION_RATE
    assert projection.iloc[1]["policies"] == pytest.approx(expected_year2)


def test_atp_is_constant_across_years(toy_segment):
    projection = project_multi_year_scenario(toy_segment, price_change_pct=0.15, n_years=3)
    assert projection["atp"].nunique() == 1  # exact same avg_premium_level every year, by construction


def test_loss_ratio_is_constant_from_year_two_onward(toy_segment):
    # Year 1 mixes the repriced retained + new business at their ACTUAL realized ratio; year 2+ all
    # use the same avg_premium_level / avg_technical_cost_level, so the ratio (and hence loss ratio)
    # should be identical (up to floating-point precision: the recursive policy-count formula
    # accumulates tiny rounding differences across years, not a real change) for every year from 2
    # onward, even though the raw euro amounts keep growing.
    projection = project_multi_year_scenario(toy_segment, price_change_pct=0.15, n_years=4)
    later_years = projection.iloc[1:]
    assert later_years["loss_ratio"].max() - later_years["loss_ratio"].min() < 1e-9


def test_at_zero_price_change_year_one_uses_the_baseline_rates(toy_segment):
    projection = project_multi_year_scenario(toy_segment, price_change_pct=0.0, n_years=1)
    quote_volume = len(toy_segment)
    expected_policies = len(toy_segment) * BASELINE_RENEWAL_RATE + quote_volume * BASELINE_CONVERSION_RATE
    assert projection.iloc[0]["policies"] == pytest.approx(expected_policies)


def test_policies_grow_monotonically_when_baseline_rates_imply_growth(toy_segment):
    # BASELINE_RENEWAL_RATE (0.88) + BASELINE_CONVERSION_RATE (0.15) * (quote_volume/policies) implies
    # net growth for this toy segment's size: confirm the projection reflects that, not a typo'd
    # shrinking formula.
    projection = project_multi_year_scenario(toy_segment, price_change_pct=0.0, n_years=3)
    assert projection["policies"].is_monotonic_increasing


def test_quote_volume_multiplier_scales_new_business_inflow(toy_segment):
    low = project_multi_year_scenario(toy_segment, price_change_pct=0.10, n_years=2, quote_volume_multiplier=0.5)
    high = project_multi_year_scenario(toy_segment, price_change_pct=0.10, n_years=2, quote_volume_multiplier=1.5)
    # More new-business inflow every year -> a larger book by year 2.
    assert high.iloc[1]["policies"] > low.iloc[1]["policies"]
