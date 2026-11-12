"""
severity_gam.py
=================
Phase 6: a Gamma GAM for claim severity, benchmarked against the Phase 6
Gamma GLM, mirroring Phase 5's frequency GLM-vs-GAM comparison protocol.

Reuses `GAMFeatureBuilder` from `frequency_gam.py` as-is: it builds the same
kind of feature array (smooth-term-ready continuous columns, factor-term-ready
categorical codes) regardless of which target it's eventually paired with, so
there is no severity-specific feature-building logic to duplicate here.

WHY NO FALLBACK-FITTING MACHINERY HERE, UNLIKE THE FREQUENCY GAM
---------------------------------------------------------------------
Phase 5's `fit_poisson_gam_with_fallback` exists because the frequency GAM
fits on ~100k+ rows, heavy enough to hit this machine's memory limits. The
severity GAM fits on the severity-only subset: a few thousand claiming
policies (Phase 1: ~15,000 in the training set): small enough that a full,
un-sub-sampled fit is cheap and low-risk. No fallback strategy is needed.
"""

from __future__ import annotations

import numpy as np
from pygam import GammaGAM

from src.models.frequency_gam import build_gam_terms


def fit_gamma_gam(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None, n_splines: int = 10) -> GammaGAM:
    """
    Fit a Gamma GAM. `weights` (typically `sev_claim_count`) plays the same
    role as `var_weights` in the Gamma GLM (`severity_glm.fit_gamma_glm`).
    see that module's docstring for why an averaged severity target needs
    this weighting.
    """
    gam = GammaGAM(build_gam_terms(n_splines=n_splines))
    gam.fit(X, y, weights=weights)
    return gam
