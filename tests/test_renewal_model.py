"""
Unit tests for src/retention/renewal_model.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.retention.renewal_model import (
    BASELINE_RENEWAL_RATE,
    LOYALTY_MAX_DAMPENING,
    LOYALTY_SATURATION_YEARS,
    MALUS_DEFAULT_TENURE_YEARS,
    derive_tenure_proxy,
    loyalty_dampening,
    renewal_probability,
    simulate_renewal_outcome,
)


def test_derive_tenure_proxy_is_zero_at_the_neutral_bonus_malus():
    bm = pd.Series([100.0])
    assert derive_tenure_proxy(bm).iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_derive_tenure_proxy_matches_the_hand_computed_floor_value():
    # BonusMalus at the floor (50) -> ln(0.5) / ln(0.95) years, hand-computable.
    bm = pd.Series([50.0])
    expected = np.log(0.5) / np.log(0.95)
    assert derive_tenure_proxy(bm).iloc[0] == pytest.approx(expected)


def test_derive_tenure_proxy_increases_as_bonus_malus_decreases():
    bm = pd.Series([100.0, 90.0, 70.0, 50.0])
    tenure = derive_tenure_proxy(bm)
    assert tenure.is_monotonic_increasing  # bm is decreasing, so tenure should increase in lockstep


def test_derive_tenure_proxy_uses_the_default_for_malus_policies():
    bm = pd.Series([110.0, 150.0, 230.0])
    tenure = derive_tenure_proxy(bm)
    assert (tenure == MALUS_DEFAULT_TENURE_YEARS).all()


def test_loyalty_dampening_is_zero_for_a_brand_new_policy():
    assert loyalty_dampening(pd.Series([0.0])).iloc[0] == pytest.approx(0.0)


def test_loyalty_dampening_saturates_at_the_max():
    tenure = pd.Series([LOYALTY_SATURATION_YEARS, LOYALTY_SATURATION_YEARS * 2])
    result = loyalty_dampening(tenure)
    assert result.iloc[0] == pytest.approx(LOYALTY_MAX_DAMPENING)
    assert result.iloc[1] == pytest.approx(LOYALTY_MAX_DAMPENING)  # does not exceed the max beyond saturation


def test_loyalty_dampening_is_halfway_at_half_the_saturation_tenure():
    tenure = pd.Series([LOYALTY_SATURATION_YEARS / 2])
    assert loyalty_dampening(tenure).iloc[0] == pytest.approx(LOYALTY_MAX_DAMPENING / 2)


def test_renewal_probability_equals_baseline_at_zero_price_change():
    tenure = pd.Series([0.0, 5.0, 15.0])
    p = renewal_probability(0.0, tenure)
    np.testing.assert_allclose(p, BASELINE_RENEWAL_RATE, atol=1e-9)


def test_renewal_probability_decreases_with_a_price_increase():
    tenure = pd.Series([0.0])
    p_no_change = renewal_probability(0.0, tenure)
    p_increase = renewal_probability(0.10, tenure)
    assert p_increase[0] < p_no_change[0]


def test_renewal_probability_increases_with_a_price_decrease():
    tenure = pd.Series([0.0])
    p_no_change = renewal_probability(0.0, tenure)
    p_decrease = renewal_probability(-0.10, tenure)
    assert p_decrease[0] > p_no_change[0]


def test_long_tenure_customers_are_less_sensitive_to_a_price_increase():
    # Same price increase, different tenure: the long-tenured policy should retain a HIGHER
    # renewal probability than the brand-new one (the "loyalty penalty" / dampening effect).
    short_tenure = pd.Series([0.0])
    long_tenure = pd.Series([LOYALTY_SATURATION_YEARS])
    p_short = renewal_probability(0.15, short_tenure)
    p_long = renewal_probability(0.15, long_tenure)
    assert p_long[0] > p_short[0]


def test_renewal_probability_stays_within_valid_bounds():
    tenure = pd.Series(np.linspace(0, 30, 50))
    price_changes = np.linspace(-0.5, 0.5, 50)
    p = renewal_probability(price_changes, tenure)
    assert (p > 0).all() and (p < 1).all()


def test_simulate_renewal_outcome_is_binary():
    tenure = pd.Series([0.0] * 100)
    outcomes = simulate_renewal_outcome(np.zeros(100), tenure, random_state=0)
    assert set(np.unique(outcomes)) <= {0, 1}


def test_simulate_renewal_outcome_matches_the_probability_at_large_n():
    n = 20000
    tenure = pd.Series([0.0] * n)
    outcomes = simulate_renewal_outcome(np.zeros(n), tenure, random_state=0)
    # At price_change=0, expected renewal rate is BASELINE_RENEWAL_RATE: check within a wide,
    # non-flaky tolerance for a Bernoulli sample of this size.
    assert outcomes.mean() == pytest.approx(BASELINE_RENEWAL_RATE, abs=0.02)
