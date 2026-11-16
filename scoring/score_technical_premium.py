"""
score_technical_premium.py
==========================
Phase 7: the technical premium of any policy, in one function used by the
notebooks and by the app.

THE FORMULA
-----------
For one policy, per year of insurance:

    technical premium = expected frequency x expected average cost + large-loss loading

    expected frequency    = exp(b0 + sum of b_j x_j)   from the refined frequency GLM
    expected average cost = exp(c0 + sum of c_k x_k)   from the severity GLM (attritional layer)
    large-loss loading    = EUR 33.54                   the same for everybody (Phase 6)

The frequency is computed for a FULL year (offset log(1) = 0), whatever the
policy's actual exposure: a premium is quoted per year. To compare with the
claims of a policy observed for part of a year, multiply by its exposure.

Hand calculation example (IDpol 1006850 of the validation set, notebook 07:
DrivAge 24, BonusMalus 95, VehPower 9, Density 1,185, B12, Regular, R42):
    eta_freq = b0 + ... = -2.1126  ->  frequency = exp(-2.1126) = 0.1209 claims per year
    eta_sev  = c0 + ... =  7.0917  ->  average cost = exp(7.0917) = EUR 1,201.98
    technical premium    = 0.1209 x 1,201.98 + 33.54 = EUR 178.88 per year

TWO WAYS TO LOAD THE MODELS
---------------------------
  TechnicalPremiumModel.load(models_dir)       the full statsmodels results (notebooks)
  TechnicalPremiumModel.load_lite(models_dir)  coefficient-only copies in models/app/ (the app)
Both give exactly the same premiums.
"""

from pathlib import Path  # file paths

import joblib  # loads the saved models
import numpy as np  # maths
import pandas as pd  # tables
import statsmodels.api as sm  # adds the intercept column


class TechnicalPremiumModel:
    """The five pieces needed to price a policy, loaded once."""

    def __init__(self, freq_results, freq_encoder, sev_results, sev_encoder, large_loss_loading):
        self.freq_results = freq_results  # refined frequency GLM (its coefficients b)
        self.freq_encoder = freq_encoder  # builds the frequency GLM's columns (with the age bands)
        self.sev_results = sev_results  # severity GLM (its coefficients c)
        self.sev_encoder = sev_encoder  # builds the severity GLM's columns (Phase 3 encoder)
        self.large_loss_loading = large_loss_loading  # EUR per year, flat

    @classmethod
    def load(cls, models_dir):
        """Full models, from models/ (used by the notebooks)."""
        return cls.load_from_folder(Path(models_dir))

    @classmethod
    def load_lite(cls, models_dir):
        """Coefficient-only models, from models/app/ (used by the app)."""
        return cls.load_from_folder(Path(models_dir) / "app")

    @classmethod
    def load_from_folder(cls, folder):
        """Read the three saved files of a folder."""
        freq = joblib.load(folder / "freq_glm_refined.pkl")  # refined frequency GLM + its encoder
        sev = joblib.load(folder / "sev_glm.pkl")  # severity GLM + large-loss loading
        sev_encoder = joblib.load(folder / "feature_encoder.joblib")  # Phase 3 encoder, used by the severity GLM
        return cls(
            freq_results=freq["results"],
            freq_encoder=freq["encoder"],
            sev_results=sev["results"],
            sev_encoder=sev_encoder,
            large_loss_loading=sev["large_loss_loading"],
        )

    _load_from = load_from_folder  # older name, kept for existing code


def score_technical_premium(df, model):
    """
    Technical premium of every policy of `df` (one row per policy).

    Returns a table with, for each policy:
      predicted_frequency  claims per year            exp(b0 + sum b_j x_j)
      predicted_severity   EUR per claim              exp(c0 + sum c_k x_k)
      large_loss_loading   EUR per year (flat)        33.54
      technical_premium    EUR per year               frequency x severity + loading
    """
    # 1. Frequency: refined GLM, for a full year (offset = log(1) = 0).
    X_freq = sm.add_constant(model.freq_encoder.transform(df), has_constant="add")  # columns + intercept
    frequency = model.freq_results.predict(X_freq, offset=np.zeros(len(df)))  # exp(b0 + sum b_j x_j)

    # 2. Severity: Gamma GLM, on the columns it was fitted with.
    X_sev_all = model.sev_encoder.transform(df)  # every Phase 3 column
    sev_columns = list(model.sev_results.model.exog_names[1:])  # the columns kept in the severity GLM (without "const")
    X_sev = sm.add_constant(X_sev_all[sev_columns], has_constant="add")  # columns + intercept
    severity = model.sev_results.predict(X_sev)  # exp(c0 + sum c_k x_k)

    # 3. Technical premium = frequency x severity + flat large-loss loading.
    premium = frequency * severity + model.large_loss_loading

    return pd.DataFrame({
        "predicted_frequency": np.asarray(frequency),
        "predicted_severity": np.asarray(severity),
        "large_loss_loading": model.large_loss_loading,
        "technical_premium": np.asarray(premium),
    }, index=df.index)
