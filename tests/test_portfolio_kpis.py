"""
Unit tests for src/pricing/portfolio_kpis.py, on a small hand-crafted
portfolio where every KPI can be checked by hand.
"""

import pandas as pd
import pytest

from src.pricing.portfolio_kpis import add_earned_premium, compute_kpis, kpis_by_segment


@pytest.fixture
def toy_portfolio() -> pd.DataFrame:
    # 4 policies, 2 segments ("A", "B"), hand-computable numbers.
    return pd.DataFrame(
        {
            "segment": ["A", "A", "B", "B"],
            "Exposure": [1.0, 0.5, 1.0, 1.0],
            "technical_premium": [200.0, 200.0, 100.0, 100.0],  # annualized EUR/policy-year
            "sev_claim_amount": [100.0, 0.0, 150.0, 150.0],  # actual incurred EUR for the period
        }
    )


def test_add_earned_premium_prorates_by_exposure(toy_portfolio):
    out = add_earned_premium(toy_portfolio)
    # policy 0: 200 * 1.0 = 200 ; policy 1: 200 * 0.5 = 100
    assert out["earned_premium"].tolist() == [200.0, 100.0, 100.0, 100.0]


def test_compute_kpis_loss_ratio_is_exposure_weighted_not_averaged(toy_portfolio):
    df = add_earned_premium(toy_portfolio)
    # Segment A only: earned premium = 200 + 100 = 300 ; claims = 100 + 0 = 100 -> LR = 100/300
    segment_a = df[df["segment"] == "A"]
    kpis = compute_kpis(segment_a)
    assert kpis["loss_ratio"] == pytest.approx(100.0 / 300.0)

    # This is NOT the same as averaging each policy's own ratio (100/200=0.5, 0/100=0 -> mean 0.25):
    naive_mean_ratio = ((100.0 / 200.0) + (0.0 / 100.0)) / 2
    assert kpis["loss_ratio"] != pytest.approx(naive_mean_ratio)


def test_compute_kpis_cor_equals_loss_ratio_plus_expense_ratio(toy_portfolio):
    df = add_earned_premium(toy_portfolio)
    kpis = compute_kpis(df, expense_ratio=0.30)
    assert kpis["cor"] == pytest.approx(kpis["loss_ratio"] + 0.30)


def test_compute_kpis_atp_is_mean_annualized_premium_not_earned_premium(toy_portfolio):
    df = add_earned_premium(toy_portfolio)
    kpis = compute_kpis(df)
    # ATP = mean of the annualized technical_premium column, NOT mean of earned_premium
    assert kpis["atp"] == pytest.approx((200.0 + 200.0 + 100.0 + 100.0) / 4)


def test_kpis_by_segment_returns_one_row_per_level_with_correct_exposure_share(toy_portfolio):
    df = add_earned_premium(toy_portfolio)
    result = kpis_by_segment(df, "segment")

    assert set(result["segment"]) == {"A", "B"}
    assert result["exposure_share"].sum() == pytest.approx(1.0)
    # Segment A total exposure = 1.5, Segment B = 2.0, total = 3.5
    row_a = result.loc[result["segment"] == "A"].iloc[0]
    assert row_a["exposure_share"] == pytest.approx(1.5 / 3.5)


def test_kpis_by_segment_sorted_by_loss_ratio_descending(toy_portfolio):
    df = add_earned_premium(toy_portfolio)
    result = kpis_by_segment(df, "segment")
    assert result["loss_ratio"].is_monotonic_decreasing


def test_kpis_by_segment_identifies_the_worse_segment(toy_portfolio):
    df = add_earned_premium(toy_portfolio)
    result = kpis_by_segment(df, "segment")
    # Segment B: earned premium = 100+100=200, claims=150+150=300 -> LR=1.5 (much worse than A's 0.33)
    worst = result.iloc[0]
    assert worst["segment"] == "B"
    assert worst["loss_ratio"] == pytest.approx(300.0 / 200.0)


def test_experience_cost_factor_caps_claims_and_adds_the_flat_loading():
    from src.pricing.portfolio_kpis import experience_cost_factor

    group = pd.DataFrame(
        {
            "Exposure": [1.0, 0.5],
            "technical_premium": [100.0, 200.0],
            "large_loss_loading": [20.0, 20.0],
            "sev_claim_amount": [50.0, 30000.0],  # second policy: one claim, capped at 10,000
            "sev_claim_count": [1, 1],
        }
    )
    expected = (50 + 10000 + 20 * 1.5) / (100 * 1.0 + 200 * 0.5)
    assert experience_cost_factor(group, large_loss_threshold=10000.0) == pytest.approx(expected)


def test_experience_cost_factor_is_one_when_capped_claims_match_the_attritional_premium():
    from src.pricing.portfolio_kpis import experience_cost_factor

    group = pd.DataFrame(
        {
            "Exposure": [1.0, 1.0],
            "technical_premium": [120.0, 80.0],
            "large_loss_loading": [10.0, 10.0],
            "sev_claim_amount": [110.0, 70.0],  # = technical premium minus the loading
            "sev_claim_count": [1, 1],
        }
    )
    assert experience_cost_factor(group, large_loss_threshold=10000.0) == pytest.approx(1.0)


def test_segment_x_round_trip(tmp_path):
    from src.pricing.segment_x import in_segment_x, load_segment_x, save_segment_x

    spec = {"column": "BonusMalus", "low": 51, "high": 80, "label": "BonusMalus 51-80"}
    save_segment_x(spec, tmp_path)
    loaded = load_segment_x(tmp_path)
    assert loaded == spec
    mask = in_segment_x(pd.DataFrame({"BonusMalus": [50, 51, 80, 81]}), loaded)
    assert mask.tolist() == [False, True, True, False]
