"""
utils.py
========
Phase 12: what every page of the app shares: loading the models and the
portfolio once (cached), pricing one policy, formatting numbers, and a
slider with an exact-value box.

Streamlit re-runs the whole page at every click; @st.cache_* keeps the
loaded models and the priced portfolio in memory so this stays fast.

The app loads the SMALL models of models/app/ (coefficients only, a few KB,
written by scripts/distill_app_models.py): same premiums as the full models.
"""

import sys  # access to Python's module search path
from pathlib import Path  # file paths

import joblib  # loads the saved models
import pandas as pd  # tables
import streamlit as st  # the app framework

APP_DIR = Path(__file__).resolve().parent  # the app/ folder
PROJECT_ROOT = APP_DIR.parent  # the project folder
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # make the project's code importable

from scoring.score_technical_premium import TechnicalPremiumModel, score_technical_premium  # noqa: E402  (Phase 7)
from src.pricing.commercial_premium import add_commercial_premium, calibrate_breakeven_loading  # noqa: E402  (Phase 11)
from src.pricing.portfolio_kpis import DEFAULT_EXPENSE_RATIO, add_earned_premium  # noqa: E402  (Phase 8)
from src.retention.renewal_model import derive_tenure_proxy  # noqa: E402  (Phase 9)

# Typical (median) population density of each Area code (Phase 2): lets a user pick an area type
# instead of typing a density.
AREA_TO_DENSITY = {
    "A: Rural": 27,
    "B: Small town": 72,
    "C: Town": 222,
    "D: Suburban": 1054,
    "E: Urban": 3744,
    "F: Dense urban": 27000,
}

CATEGORICAL_ENCODER_COLS = ["VehBrand", "VehGas", "Region", "DrivAge_bin", "VehAge_bin"]  # category columns of the refined GLM


@st.cache_resource(show_spinner="Loading pricing models...")
def load_model():
    """The small pricing models of models/app/ (loaded once)."""
    return TechnicalPremiumModel.load_lite(PROJECT_ROOT / "models")


@st.cache_resource
def load_category_options():
    """Brands, fuels and regions known by the model (read from its encoder, never typed by hand)."""
    encoder = load_model().freq_encoder  # the refined GLM's encoder
    return {name: list(encoder.categories_[name]) for name in ("VehBrand", "VehGas", "Region")}


@st.cache_data(show_spinner="Pricing the portfolio (once, then cached)...")
def load_scored_portfolio():
    """
    Every policy of the dataset, priced: technical premium (Phase 7), tenure
    (Phase 9), earned premium, commercial premium (Phase 11).
    Returns (table, commercial loading).
    """
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "policy_level_clean.parquet")  # every policy
    scored = score_technical_premium(df, load_model())  # frequency, average cost, loading, technical premium
    df = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)  # side by side
    df["tenure_years_proxy"] = derive_tenure_proxy(df["BonusMalus"])  # claim-free years from the bonus-malus
    df = add_earned_premium(df)  # technical premium x exposure
    loading = calibrate_breakeven_loading(1.0, DEFAULT_EXPENSE_RATIO)  # 1 / (1 - 27%) - 1 = 37%
    df = add_commercial_premium(df, loading)  # technical premium x 1.37
    return df, loading


def make_policy_table(drivage, bonus_malus, vehpower, vehage, density, vehbrand, vehgas, region, exposure=1.0):
    """One-row table with the same columns as the dataset, from the values typed in the app."""
    return pd.DataFrame([{
        "DrivAge": drivage, "BonusMalus": bonus_malus, "VehPower": vehpower, "VehAge": vehage,
        "Density": density, "VehBrand": vehbrand, "VehGas": vehgas, "Region": region, "Exposure": exposure,
    }])


def score_one_policy(drivage, bonus_malus, vehpower, vehage, density, vehbrand, vehgas, region, exposure=1.0):
    """Technical and commercial premium of one policy typed in the Pricing Simulator."""
    policy_df = make_policy_table(drivage, bonus_malus, vehpower, vehage, density, vehbrand, vehgas, region, exposure)
    scored = score_technical_premium(policy_df, load_model())  # same function as the notebooks
    scored["commercial_premium"] = scored["technical_premium"] * (1 + get_commercial_loading())  # + 37%
    return scored.iloc[0].to_dict()


@st.cache_resource
def get_large_loss_threshold():
    """Large-loss threshold, EUR 15,000 per claim (Phase 6), read from the saved severity model."""
    return float(joblib.load(PROJECT_ROOT / "models" / "app" / "sev_glm.pkl")["large_loss_threshold"])


def get_commercial_loading():
    """The commercial loading (37%), computed once with the portfolio and reused everywhere."""
    _, loading = load_scored_portfolio()
    return loading


def format_eur(value):
    """1234.5 -> 'EUR 1,235' style text."""
    return f"€{value:,.0f}"


def format_pct(value, decimals=1):
    """0.1234 -> '12.3%'."""
    return f"{value * 100:.{decimals}f}%"


def slider_with_input(label, min_value, max_value, value, step=1, key="", help=None, fmt=None, is_percent=False):
    """
    A slider and a number box showing the same value: drag for a quick pick,
    type for an exact one (e.g. BonusMalus = 137).

    Streamlit only uses `value=` the first time; after that it reads each
    widget's value from st.session_state. So the two widgets are kept in sync
    by writing, when one changes, directly into the other's session_state key.
    """
    slider_key = f"{key}_slider"
    number_key = f"{key}_number"

    if slider_key not in st.session_state and number_key not in st.session_state:  # first run only
        st.session_state[slider_key] = value
        st.session_state[number_key] = value

    def _sync_from_slider():  # slider moved: copy its value into the box
        st.session_state[number_key] = st.session_state[slider_key]

    def _sync_from_number():  # box changed: copy its value into the slider
        st.session_state[slider_key] = st.session_state[number_key]

    col_slider, col_number = st.columns([3, 1])
    with col_slider:
        st.slider(label, min_value=min_value, max_value=max_value, step=step, key=slider_key,
                  help=help, format=fmt, on_change=_sync_from_slider)
    with col_number:
        st.number_input("Exact value", min_value=min_value, max_value=max_value, step=step, key=number_key,
                        label_visibility="collapsed", on_change=_sync_from_number)

    result = st.session_state[slider_key]
    return result / 100.0 if is_percent else result  # percentages are returned as fractions (10 -> 0.10)
