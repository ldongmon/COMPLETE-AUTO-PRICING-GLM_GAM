"""
score_technical_premium.py
============================
Phase 7: the single, unified scoring pipeline: raw policy features in,
technical premium out. Every notebook (and, later, the Streamlit app)
calls this rather than re-implementing the frequency x severity + large-loss
combination separately: the same "one source of truth" discipline this
project has followed since `src/data/load_data.py` in Phase 1.

WHAT "TECHNICAL PREMIUM" MEANS HERE
---------------------------------------
The **annualized, per-policy-year** expected claim cost:

    technical_premium = predicted_frequency (claims / policy-year)
                       x predicted_severity (EUR / claim)
                       + large_loss_loading (EUR / policy-year, flat: Phase 6)

This is the actuarially fair rate for a *full year* of the risk: the
quantity a pricing table is actually built from, not the expected cost for
whatever partial period a specific historical policy happened to be
observed. To reconcile against a historical policy's *actual* observed
exposure period (e.g. for the validation diagnostics later in this
notebook), multiply by that policy's `Exposure`.

WHY predicted_frequency USES offset=0, NOT log(Exposure)
--------------------------------------------------------------
The frequency GLM (`src/models/frequency_glm_refined.py`) was *fit* with
`offset = log(Exposure)`, because during fitting each policy's observed
claim count does scale with how long it was on risk. But once
fitted, asking "what's this policy's expected frequency for a full year"
means evaluating the model at `Exposure = 1`, i.e. `log(Exposure) = 0`: a
zero offset at prediction time, every time, regardless of any individual
policy's actual observed exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class TechnicalPremiumModel:
    """Bundles every fitted artifact `score_technical_premium` needs, loaded once."""

    freq_results: object  # fitted statsmodels GLMResults: refined frequency GLM (Phase 7)
    freq_encoder: object  # fitted RefinedFeatureEncoder (Phase 7)
    sev_results: object  # fitted statsmodels GLMResults: severity GLM (Phase 6)
    sev_encoder: object  # fitted PolicyFeatureEncoder (Phase 3): severity was built on this encoding
    large_loss_loading: float  # EUR per policy-year, flat (Phase 6)

    @classmethod
    def load(cls, models_dir: Path) -> "TechnicalPremiumModel":
        """Loads the full research artifacts (statsmodels results objects, ~200MB): for notebooks."""
        return cls._load_from(models_dir)

    @classmethod
    def load_lite(cls, models_dir: Path) -> "TechnicalPremiumModel":
        """
        Loads the distilled, deployment-sized artifacts from `models_dir / "app"` (see
        `src/models/lightweight_glm.py` and `scripts/distill_app_models.py`): a few hundred KB
        total instead of ~200MB, for the Streamlit app (Phase 12) and any other deployment context.
        `LightweightGLM` is a drop-in replacement for a real `GLMResults` object here: every line of
        `score_technical_premium` below runs identically regardless of which `load*` was used.
        """
        return cls._load_from(models_dir / "app")

    @classmethod
    def _load_from(cls, artifact_dir: Path) -> "TechnicalPremiumModel":
        freq = joblib.load(artifact_dir / "freq_glm_refined.pkl")
        sev = joblib.load(artifact_dir / "sev_glm.pkl")
        sev_encoder = joblib.load(artifact_dir / "feature_encoder.joblib")
        return cls(
            freq_results=freq["results"],
            freq_encoder=freq["encoder"],
            sev_results=sev["results"],
            sev_encoder=sev_encoder,
            large_loss_loading=sev["large_loss_loading"],
        )


def score_technical_premium(df: pd.DataFrame, model: TechnicalPremiumModel) -> pd.DataFrame:
    """
    Score a table of raw policy features (any DataFrame with the columns
    `PolicyFeatureEncoder`/`RefinedFeatureEncoder` expect: see Phase 3/7) into
    a per-policy-year technical premium. Works identically whether `df` is a
    slice of the known dataset (for validation) or a brand-new hypothetical
    policy (for the eventual pricing app): the whole point of routing every
    caller through one function.

    Returns a DataFrame (same row index as `df`) with columns:
      - predicted_frequency  : claims per policy-year
      - predicted_severity   : EUR per claim
      - large_loss_loading   : EUR per policy-year (flat, same value every row)
      - technical_premium    : EUR per policy-year (the headline output)
    """
    # --- Frequency: refined GLM, evaluated at a full policy-year (offset = 0) ---
    X_freq = model.freq_encoder.transform(df)
    X_freq_const = sm.add_constant(X_freq, has_constant="add")
    zero_offset = np.zeros(len(df))
    predicted_frequency = model.freq_results.predict(X_freq_const, offset=zero_offset)

    # --- Severity: Gamma GLM (Phase 6), no offset: severity has none ---
    X_sev_full = model.sev_encoder.transform(df)
    sev_feature_cols = list(model.sev_results.model.exog_names[1:])  # drop 'const'
    X_sev_const = sm.add_constant(X_sev_full[sev_feature_cols], has_constant="add")
    predicted_severity = model.sev_results.predict(X_sev_const)

    technical_premium = predicted_frequency * predicted_severity + model.large_loss_loading

    return pd.DataFrame(
        {
            "predicted_frequency": np.asarray(predicted_frequency),
            "predicted_severity": np.asarray(predicted_severity),
            "large_loss_loading": model.large_loss_loading,
            "technical_premium": np.asarray(technical_premium),
        },
        index=df.index,
    )
