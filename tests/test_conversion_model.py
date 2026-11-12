"""
Unit tests for src/prospect/conversion_model.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.prospect.conversion_model import (
    BASELINE_CONVERSION_RATE,
    BONUS_MALUS_NEUTRAL,
    RISK_ACCESS_MAX_DAMPENING,
    RISK_ACCESS_SATURATION_BONUS_MALUS,
    conversion_probability,
    risk_access_dampening,
    simulate_conversion_outcome,
)
from src.retention.renewal_model import PRICE_ELASTICITY as RETENTION_PRICE_ELASTICITY
from src.prospect.conversion_model import CONVERSION_PRICE_ELASTICITY


def test_risk_access_dampening_is_zero_at_or_below_neutral():
    bm = pd.Series([50.0, 80.0, 100.0])
    result = risk_access_dampening(bm)
    assert (result == 0.0).all()


def test_risk_access_dampening_saturates_at_the_max():
    bm = pd.Series([RISK_ACCESS_SATURATION_BONUS_MALUS, RISK_ACCESS_SATURATION_BONUS_MALUS * 1.5])
    result = risk_access_dampening(bm)
    assert result.iloc[0] == pytest.approx(RISK_ACCESS_MAX_DAMPENING)
    assert result.iloc[1] == pytest.approx(RISK_ACCESS_MAX_DAMPENING)


def test_risk_access_dampening_is_halfway_at_the_midpoint():
    midpoint = BONUS_MALUS_NEUTRAL + (RISK_ACCESS_SATURATION_BONUS_MALUS - BONUS_MALUS_NEUTRAL) / 2
    result = risk_access_dampening(pd.Series([midpoint]))
    assert result.iloc[0] == pytest.approx(RISK_ACCESS_MAX_DAMPENING / 2)


def test_conversion_probability_equals_baseline_at_market_parity():
    bm = pd.Series([50.0, 100.0, 180.0])
    p = conversion_probability(0.0, bm)
    np.testing.assert_allclose(p, BASELINE_CONVERSION_RATE, atol=1e-9)


def test_conversion_probability_decreases_when_quoted_above_market():
    bm = pd.Series([100.0])
    p_parity = conversion_probability(0.0, bm)
    p_above = conversion_probability(0.10, bm)
    assert p_above[0] < p_parity[0]


def test_conversion_probability_increases_when_quoted_below_market():
    bm = pd.Series([100.0])
    p_parity = conversion_probability(0.0, bm)
    p_below = conversion_probability(-0.10, bm)
    assert p_below[0] > p_parity[0]


def test_high_risk_prospects_are_less_sensitive_to_a_price_increase():
    # Same quoted-above-market amount, different BonusMalus: the high-risk (high-BM) prospect
    # should retain a HIGHER conversion probability than the low-risk one (reduced market access
    # makes them less able to shop elsewhere, hence less price-sensitive at the margin).
    low_risk = pd.Series([100.0])
    high_risk = pd.Series([RISK_ACCESS_SATURATION_BONUS_MALUS])
    p_low_risk = conversion_probability(0.15, low_risk)
    p_high_risk = conversion_probability(0.15, high_risk)
    assert p_high_risk[0] > p_low_risk[0]


def test_conversion_probability_stays_within_valid_bounds():
    bm = pd.Series(np.linspace(50, 230, 50))
    price_positions = np.linspace(-0.5, 0.5, 50)
    p = conversion_probability(price_positions, bm)
    assert (p > 0).all() and (p < 1).all()


def test_conversion_elasticity_is_larger_in_magnitude_than_retention_elasticity():
    # Documented design choice: new business is more price-elastic than renewal business.
    assert abs(CONVERSION_PRICE_ELASTICITY) > abs(RETENTION_PRICE_ELASTICITY)


def test_simulate_conversion_outcome_is_binary():
    bm = pd.Series([100.0] * 100)
    outcomes = simulate_conversion_outcome(np.zeros(100), bm, random_state=0)
    assert set(np.unique(outcomes)) <= {0, 1}


def test_simulate_conversion_outcome_matches_the_probability_at_large_n():
    n = 20000
    bm = pd.Series([100.0] * n)
    outcomes = simulate_conversion_outcome(np.zeros(n), bm, random_state=0)
    assert outcomes.mean() == pytest.approx(BASELINE_CONVERSION_RATE, abs=0.01)
