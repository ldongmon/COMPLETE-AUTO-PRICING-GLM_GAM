"""
severity_gam.py
===============
Phase 6: a Gamma GAM for the attritional claim cost, used only as a benchmark
for the Gamma GLM. Same curves and categories as the frequency GAM
(frequency_gam.build_gam_terms), with a Gamma law instead of Poisson:

    log(average capped amount) = c0 + f1(DrivAge) + f2(BonusMalus) + ... + (one value per category level)
"""

from pygam import GammaGAM  # GAM with a Gamma law and log link

from src.models.frequency_gam import build_gam_terms  # the same terms as the frequency GAM


def fit_gamma_gam(X, y, weights=None, n_splines=10):
    """Fit the Gamma GAM; `weights` = number of claims behind each average amount."""
    gam = GammaGAM(build_gam_terms(n_splines=n_splines))  # 5 curves + 3 category terms
    gam.fit(X, y, weights=weights)  # estimate them
    return gam
