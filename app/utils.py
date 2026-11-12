"""
utils.py
=========
Phase 12: shared, cached data/model loading for every page of the
Streamlit app. Centralized here so each page imports the same cached
objects instead of reloading models or rescoring the portfolio on every
click (Streamlit re-runs the whole script on every interaction; `@st.cache_*`
is what keeps that fast).

WHY THE APP LOADS `load_lite`, NOT THE FULL RESEARCH MODELS
------------------------------------------------------------------
`models/freq_glm_refined.pkl` and `models/sev_glm.pkl` (the full,
notebook-facing `statsmodels` objects) are ~200MB combined.
`scripts/distill_app_models.py` (run once, after training) produces the
`models/app/` artifacts this file loads instead: a few KB, mathematically
identical for scoring (`src/models/lightweight_glm.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scoring.score_technical_premium import TechnicalPremiumModel, score_technical_premium  # noqa: E402
from src.pricing.commercial_premium import add_commercial_premium, calibrate_breakeven_loading  # noqa: E402
from src.pricing.portfolio_kpis import DEFAULT_EXPENSE_RATIO, add_earned_premium  # noqa: E402
from src.retention.renewal_model import derive_tenure_proxy  # noqa: E402

# Representative median population density per Area code (Phase 2, EDA notebook): lets a
# non-technical user pick a familiar "area type" instead of typing an exact density figure.
AREA_TO_DENSITY = {
    "A: Rural": 27,
    "B: Small town": 72,
    "C: Town": 222,
    "D: Suburban": 1054,
    "E: Urban": 3744,
    "F: Dense urban": 27000,
}

CATEGORICAL_ENCODER_COLS = ["VehBrand", "VehGas", "Region", "DrivAge_bin", "VehAge_bin"]


@st.cache_resource(show_spinner="Loading pricing models...")
def load_model() -> TechnicalPremiumModel:
    return TechnicalPremiumModel.load_lite(PROJECT_ROOT / "models")


@st.cache_resource
def load_category_options() -> dict[str, list[str]]:
    """VehBrand / VehGas / Region options, read directly off the fitted encoder: never hand-typed,
    so the app can never offer a category the model doesn't actually know."""
    encoder = load_model().freq_encoder
    return {
        name: list(cats)
        for name, cats in zip(CATEGORICAL_ENCODER_COLS, encoder.onehot_encoder_.categories_)
        if name in ("VehBrand", "VehGas", "Region")
    }


@st.cache_data(show_spinner="Scoring the portfolio (one-time, then cached)...")
def load_scored_portfolio() -> tuple[pd.DataFrame, float]:
    """
    The full policy-level portfolio (Phase 1), scored with the current
    technical + commercial premium and each policy's derived tenure proxy
    (Phase 9). Returns `(df, commercial_loading)`: computed once per app
    session/process, not on every widget interaction.
    """
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "policy_level_clean.parquet")
    model = load_model()
    scored = score_technical_premium(df, model)
    df = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    df["tenure_years_proxy"] = derive_tenure_proxy(df["BonusMalus"])
    df = add_earned_premium(df)

    # The technical premium is the expected claims cost (Phase 7), so the loading only covers expenses
    # (Phase 11): expected Loss Ratio of 100%.
    loading = calibrate_breakeven_loading(1.0, DEFAULT_EXPENSE_RATIO)
    df = add_commercial_premium(df, loading)
    return df, loading


def score_one_policy(
    drivage: int,
    bonus_malus: int,
    vehpower: int,
    vehage: int,
    density: float,
    vehbrand: str,
    vehgas: str,
    region: str,
    exposure: float = 1.0,
) -> dict:
    """Scores a single hypothetical policy: the Pricing Simulator page's core call."""
    model = load_model()
    policy_df = pd.DataFrame(
        [
            {
                "DrivAge": drivage,
                "BonusMalus": bonus_malus,
                "VehPower": vehpower,
                "VehAge": vehage,
                "Density": density,
                "VehBrand": vehbrand,
                "VehGas": vehgas,
                "Region": region,
                "Exposure": exposure,
            }
        ]
    )
    scored = score_technical_premium(policy_df, model)
    loading = get_commercial_loading()
    scored["commercial_premium"] = scored["technical_premium"] * (1 + loading)
    return scored.iloc[0].to_dict()


@st.cache_resource
def get_large_loss_threshold() -> float:
    """Large-loss threshold (EUR per claim) from the severity artifact (Phase 6)."""
    return float(joblib.load(PROJECT_ROOT / "models" / "app" / "sev_glm.pkl")["large_loss_threshold"])


def get_commercial_loading() -> float:
    """
    The single source of truth for the app's commercial loading: the value
    that brings the *current* portfolio to exactly 100% COR (Phase 11),
    computed once inside `load_scored_portfolio` (itself `@st.cache_data`-cached)
    and reused everywhere else in the app, including single-policy pricing.
    """
    _, loading = load_scored_portfolio()
    return loading


def format_eur(value: float) -> str:
    return f"€{value:,.0f}"


def format_pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


# --------------------------------------------------------------------------- #
# Slider + exact-value input widget
# --------------------------------------------------------------------------- #


def slider_with_input(
    label: str,
    min_value: float,
    max_value: float,
    value: float,
    step: float = 1,
    key: str = "",
    help: str | None = None,
    fmt: str | None = None,
    is_percent: bool = False,
):
    """
    A slider paired with a number box showing the exact same value, kept in
    sync: drag the slider for a quick, approximate pick, or type into the
    box next to it for an exact one (e.g. "BonusMalus = 137"), without
    hunting along the slider track.

    THE STREAMLIT GOTCHA THIS WORKS AROUND
    ---------------------------------------
    A widget's `value=` argument is only honored on the very first run; once
    its `key` has an entry in `st.session_state` (from that first run
    onward), Streamlit reads the widget's displayed value from
    `st.session_state[key]` and silently *ignores* `value=` on every rerun.
    So keeping two widgets in sync via a third, separate "shared" state
    variable does not work: each widget snaps back to its own last value.
    The fix is to never pass `value=` past the first run, and have each
    widget's `on_change` callback write straight into the *other* widget's
    own session_state key, so on the next rerun, that other widget's
    persisted state already reflects the new number.
    """
    slider_key = f"{key}_slider"
    number_key = f"{key}_number"

    # Seed both widgets' state once, on the very first run only: every later run relies
    # entirely on session_state (kept in sync by the callbacks below), never on `value=`.
    if slider_key not in st.session_state and number_key not in st.session_state:
        st.session_state[slider_key] = value
        st.session_state[number_key] = value

    def _sync_from_slider():
        st.session_state[number_key] = st.session_state[slider_key]

    def _sync_from_number():
        st.session_state[slider_key] = st.session_state[number_key]

    col_slider, col_number = st.columns([3, 1])
    with col_slider:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=slider_key,
            help=help,
            format=fmt,
            on_change=_sync_from_slider,
        )
    with col_number:
        st.number_input(
            "Exact value",
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=number_key,
            label_visibility="collapsed",
            on_change=_sync_from_number,
        )

    result = st.session_state[slider_key]
    return result / 100.0 if is_percent else result
