"""
Unit tests for scoring/scenario_engine.py, on a small synthetic segment.
"""

import numpy as np
import pandas as pd
import pytest

from scoring.scenario_engine import run_repricing_scenario, scenario_grid
from src.retention.renewal_model import BASELINE_RENEWAL_RATE
from src.prospect.conversion_model import BASELINE_CONVERSION_RATE


@pytest.fixture
def toy_segment() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    return pd.DataFrame(
        {
            "commercial_premium": rng.uniform(150, 250, n),
            "technical_premium": rng.uniform(140, 200, n),  # deliberately underpriced on average
            "tenure_years_proxy": rng.uniform(0, 13.5, n),
            "BonusMalus": rng.uniform(50, 150, n),
        }
    )


def test_at_zero_price_change_renewals_match_the_baseline_rate(toy_segment):
    result = run_repricing_scenario(toy_segment, price_change_pct=0.0)
    # renewal_probability at 0% price change collapses to BASELINE_RENEWAL_RATE regardless of tenure
    assert result["renewal_rate"] == pytest.approx(BASELINE_RENEWAL_RATE, abs=1e-6)


def test_at_zero_price_change_conversion_matches_the_baseline_rate(toy_segment):
    result = run_repricing_scenario(toy_segment, price_change_pct=0.0)
    assert result["conversion_rate"] == pytest.approx(BASELINE_CONVERSION_RATE, abs=1e-6)


def test_expected_new_policies_scales_with_quote_volume_multiplier(toy_segment):
    result_1x = run_repricing_scenario(toy_segment, price_change_pct=0.0, quote_volume_multiplier=1.0)
    result_2x = run_repricing_scenario(toy_segment, price_change_pct=0.0, quote_volume_multiplier=2.0)
    assert result_2x["expected_new_policies"] == pytest.approx(2 * result_1x["expected_new_policies"])


def test_price_increase_reduces_renewal_and_conversion_rates(toy_segment):
    baseline = run_repricing_scenario(toy_segment, price_change_pct=0.0)
    repriced = run_repricing_scenario(toy_segment, price_change_pct=0.15)
    assert repriced["renewal_rate"] < baseline["renewal_rate"]
    assert repriced["conversion_rate"] < baseline["conversion_rate"]


def test_a_small_increase_from_underpricing_improves_contribution(toy_segment):
    # toy_segment's technical_premium is close to (sometimes above) commercial_premium: a small
    # price increase should improve contribution before retention/conversion losses dominate.
    baseline = run_repricing_scenario(toy_segment, price_change_pct=0.0)
    small_increase = run_repricing_scenario(toy_segment, price_change_pct=0.05)
    assert small_increase["total_contribution"] > baseline["total_contribution"]


def test_marginal_contribution_gain_diminishes_as_price_rises(toy_segment):
    # A real, verified property of this engine even where the *total* keeps rising within any
    # realistic range (see the module docstring / Phase 11 notebook for why: retention and conversion
    # elasticities decay slowly enough in logit space that the true mathematical peak sits at an
    # unrealistic price change, often 80-100%+, not a bug, a documented limitation of extrapolating
    # a moderately-calibrated logistic elasticity that far). What should always hold, and does, is
    # diminishing marginal returns: each further step of price increase buys a smaller improvement
    # in contribution than the previous one.
    steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    contributions = [run_repricing_scenario(toy_segment, price_change_pct=p)["total_contribution"] for p in steps]
    marginal_gains = np.diff(contributions)
    assert np.all(np.diff(marginal_gains) < 0)  # each marginal gain is smaller than the last


def test_total_written_premium_equals_retained_plus_new(toy_segment):
    result = run_repricing_scenario(toy_segment, price_change_pct=0.10)
    assert result["total_written_premium"] == pytest.approx(result["retained_premium"] + result["new_premium"])


def test_total_contribution_equals_retained_plus_new(toy_segment):
    result = run_repricing_scenario(toy_segment, price_change_pct=0.10)
    assert result["total_contribution"] == pytest.approx(result["retained_contribution"] + result["new_contribution"])


def test_scenario_grid_returns_one_row_per_price_change(toy_segment):
    price_changes = [-0.05, 0.0, 0.05, 0.10, 0.15]
    grid = scenario_grid(toy_segment, price_changes)
    assert len(grid) == len(price_changes)
    assert list(grid["price_change_pct"]) == price_changes


def test_scenario_grid_finds_the_true_interior_optimum_on_a_wide_enough_range(toy_segment):
    # The engine's retention/conversion elasticities do eventually make higher prices
    # counterproductive (see test_marginal_contribution_gain_diminishes_as_price_rises): it just
    # takes a much wider price range than any realistic single repricing action to see the turn.
    # This test confirms the grid-search machinery itself correctly locates that turn when the
    # search range is wide enough to contain it, rather than assuming a practical range must.
    price_changes = list(np.linspace(-0.10, 5.0, 60))
    grid = scenario_grid(toy_segment, price_changes)
    best_row = grid.loc[grid["total_contribution"].idxmax()]
    assert price_changes[0] < best_row["price_change_pct"] < price_changes[-1]


def test_cost_factor_scales_expected_claims(toy_segment):
    base = run_repricing_scenario(toy_segment, 0.0)
    scaled = run_repricing_scenario(toy_segment, 0.0, cost_factor=1.2)
    assert scaled["retained_claims"] == pytest.approx(1.2 * base["retained_claims"])
    assert scaled["new_claims"] == pytest.approx(1.2 * base["new_claims"])
    assert scaled["total_written_premium"] == pytest.approx(base["total_written_premium"])


def test_book_at_breakeven_loading_has_zero_contribution_at_zero_change(toy_segment):
    # Loading calibrated on a Loss Ratio of 0.9 with a 27% expense ratio; the engine,
    # given the same 0.9 as cost factor, must return zero contribution at 0% change.
    from src.pricing.commercial_premium import calibrate_breakeven_loading

    loading = calibrate_breakeven_loading(0.9, 0.27)
    seg = toy_segment.assign(commercial_premium=toy_segment["technical_premium"] * (1 + loading))
    result = run_repricing_scenario(seg, 0.0, expense_ratio=0.27, cost_factor=0.9)
    assert result["total_contribution"] == pytest.approx(0.0, abs=1e-6)
