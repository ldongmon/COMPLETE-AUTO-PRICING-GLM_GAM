"""
distill_app_models.py
=======================
Phase 12: run this once after (re-)training the frequency/severity models
(Phases 6/7) to produce the small, deployment-sized artifacts the Streamlit
app (and any other deployment context) actually needs.

Why this is a separate step, not automatic: `models/freq_glm_refined.pkl`
and `models/sev_glm.pkl` are full `statsmodels` GLMResults objects: ~190MB
and ~15MB respectively, because `statsmodels` retains a copy of the entire
training design matrix and internal fitting arrays. That's the right thing
for the Phase 4-7 notebooks, which call `.summary()`, `.pvalues`, `.aic`,
etc. on them. It is not something to commit to git or ship to a deployed
app. This script distills each one down to a `LightweightGLM`
(`src/models/lightweight_glm.py`): just the fitted coefficients, a few KB,
mathematically identical for prediction, and writes them to `models/app/`,
which the app loads via `TechnicalPremiumModel.load_lite()`.

Usage:
    python scripts/distill_app_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lightweight_glm import LightweightGLM  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
APP_MODELS_DIR = MODELS_DIR / "app"


def main() -> None:
    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Frequency: refined GLM (Phase 7) ---
    freq = joblib.load(MODELS_DIR / "freq_glm_refined.pkl")
    freq_lite = {
        "results": LightweightGLM.from_results(freq["results"]),
        "encoder": freq["encoder"],  # already tiny (a fitted OneHotEncoder + bin edges)
    }
    joblib.dump(freq_lite, APP_MODELS_DIR / "freq_glm_refined.pkl")

    # --- Severity: Gamma GLM (Phase 6) ---
    sev = joblib.load(MODELS_DIR / "sev_glm.pkl")
    sev_lite = {
        "results": LightweightGLM.from_results(sev["results"]),
        "large_loss_threshold": sev["large_loss_threshold"],
        "large_loss_loading": sev["large_loss_loading"],
    }
    joblib.dump(sev_lite, APP_MODELS_DIR / "sev_glm.pkl")

    # --- Shared feature encoder (Phase 3): already small, copied as-is for a self-contained app/ dir ---
    encoder = joblib.load(MODELS_DIR / "feature_encoder.joblib")
    joblib.dump(encoder, APP_MODELS_DIR / "feature_encoder.joblib")

    print("Distilled app-ready artifacts written to:", APP_MODELS_DIR)
    for f in sorted(APP_MODELS_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<28} {size_kb:>10,.1f} KB")

    original_total_mb = sum(
        (MODELS_DIR / name).stat().st_size for name in ["freq_glm_refined.pkl", "sev_glm.pkl", "feature_encoder.joblib"]
    ) / (1024 * 1024)
    lite_total_mb = sum(f.stat().st_size for f in APP_MODELS_DIR.glob("*")) / (1024 * 1024)
    print(f"\nOriginal artifacts: {original_total_mb:,.1f} MB  ->  distilled: {lite_total_mb:,.2f} MB")


if __name__ == "__main__":
    main()
