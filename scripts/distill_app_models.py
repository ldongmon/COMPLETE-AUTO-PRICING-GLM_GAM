"""
distill_app_models.py
=====================
Phase 12: run once after re-training the models (notebooks 06 and 07). It
writes small copies of the pricing models to models/app/, used by the app.

    models/freq_glm_refined.pkl  (~190 MB)  ->  models/app/freq_glm_refined.pkl  (a few KB)
    models/sev_glm.pkl           (~15 MB)   ->  models/app/sev_glm.pkl           (a few KB)
    models/feature_encoder.joblib            ->  copied as it is (already small)

Each GLM is replaced by a LightweightGLM (coefficients only): same premiums.

Usage:
    python scripts/distill_app_models.py
"""

import sys  # access to Python's module search path
from pathlib import Path  # file paths

import joblib  # saves and loads Python objects

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # the project folder
sys.path.insert(0, str(PROJECT_ROOT))  # make the project's code importable

from src.models.lightweight_glm import LightweightGLM  # noqa: E402  (coefficients-only GLM)

MODELS_DIR = PROJECT_ROOT / "models"  # full models (notebooks)
APP_MODELS_DIR = MODELS_DIR / "app"  # small models (app)


def main():
    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)  # create models/app/ if needed

    # Frequency: refined GLM (notebook 07) -> coefficients only, encoder kept as it is.
    freq = joblib.load(MODELS_DIR / "freq_glm_refined.pkl")
    joblib.dump({"results": LightweightGLM.from_results(freq["results"]), "encoder": freq["encoder"]},
                APP_MODELS_DIR / "freq_glm_refined.pkl")

    # Severity: Gamma GLM (notebook 06) -> coefficients only, threshold and loading kept.
    sev = joblib.load(MODELS_DIR / "sev_glm.pkl")
    joblib.dump({"results": LightweightGLM.from_results(sev["results"]),
                 "large_loss_threshold": sev["large_loss_threshold"],  # EUR 15,000
                 "large_loss_loading": sev["large_loss_loading"]},  # EUR 33.54
                APP_MODELS_DIR / "sev_glm.pkl")

    # Phase 3 encoder (used by the severity GLM): copied so that models/app/ is self-contained.
    joblib.dump(joblib.load(MODELS_DIR / "feature_encoder.joblib"), APP_MODELS_DIR / "feature_encoder.joblib")

    print("Small models written to:", APP_MODELS_DIR)
    for f in sorted(APP_MODELS_DIR.glob("*")):  # size of each file
        print(f"  {f.name:<28} {f.stat().st_size / 1024:>10,.1f} KB")
    original_mb = sum((MODELS_DIR / n).stat().st_size for n in ["freq_glm_refined.pkl", "sev_glm.pkl", "feature_encoder.joblib"]) / 1024**2
    lite_mb = sum(f.stat().st_size for f in APP_MODELS_DIR.glob("*")) / 1024**2
    print(f"\nOriginal models: {original_mb:,.1f} MB  ->  small copies: {lite_mb:,.2f} MB")


if __name__ == "__main__":
    main()
