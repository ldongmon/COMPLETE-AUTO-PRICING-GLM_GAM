"""
Unit tests for src/pricing/commercial_premium.py.
"""

import pandas as pd
import pytest

from src.pricing.commercial_premium import add_commercial_premium, calibrate_breakeven_loading


def test_calibrate_breakeven_loading_matches_hand_calculation():
    # LR=0.8338, expense=0.27 -> loading = 0.8338/0.73 - 1
    loading = calibrate_breakeven_loading(loss_ratio=0.8338, expense_ratio=0.27)
    assert loading == pytest.approx(0.8338 / 0.73 - 1)


def test_calibrate_breakeven_loading_is_zero_when_already_at_breakeven():
    # If LR already equals (1 - expense_ratio), no loading is needed.
    loading = calibrate_breakeven_loading(loss_ratio=0.73, expense_ratio=0.27)
    assert loading == pytest.approx(0.0, abs=1e-9)


def test_calibrate_breakeven_loading_produces_exact_breakeven_cor():
    loss_ratio, expense_ratio = 0.8338, 0.27
    loading = calibrate_breakeven_loading(loss_ratio, expense_ratio)
    new_loss_ratio = loss_ratio / (1 + loading)
    new_cor = new_loss_ratio + expense_ratio
    assert new_cor == pytest.approx(1.0)


def test_add_commercial_premium_scales_technical_premium():
    df = pd.DataFrame({"technical_premium": [100.0, 200.0]})
    out = add_commercial_premium(df, loading=0.20)
    assert out["commercial_premium"].tolist() == pytest.approx([120.0, 240.0])


def test_add_commercial_premium_does_not_mutate_input():
    df = pd.DataFrame({"technical_premium": [100.0]})
    add_commercial_premium(df, loading=0.20)
    assert "commercial_premium" not in df.columns
