"""
conversion_model.py
=====================
Phase 10: a documented, literature-calibrated **scenario simulator** for
new-business conversion (does a quoted prospect actually buy?), mirroring
Phase 9's retention simulator in spirit and in its disclosures: freMTPL2 has
no quote-to-bind outcomes and no competitor pricing data, so: exactly as
decided for Phase 9 (see `docs/PROJECT_ROADMAP.md`): this module *specifies*
a conversion-probability formula with every parameter sourced or cited,
rather than fitting one to data that cannot support it.

A note on the optional empirical anchor considered for this phase: the
roadmap flagged Kaggle's "Health Insurance Cross Sell Prediction" dataset as
a possible real-data anchor for this curve's *shape*. It was not pulled into
this project: the only copies found were unofficial third-party GitHub
mirrors of Kaggle-restricted data, and this project would rather ship a
clearly-labeled, literature-cited simulation than a real dataset of
uncertain provenance. This is a documented decision, not an oversight.

WHY CONVERSION USES A DIFFERENT BASELINE AND A DIFFERENT SEGMENT DRIVER THAN RETENTION
-------------------------------------------------------------------------------------------
- **Baseline is much lower than retention's 88%.** A prospect getting a
  single quote is, by construction, comparison-shopping across insurers.
  `BASELINE_CONVERSION_RATE` (15%) is a documented assumption in line with
  commonly-cited quote-to-bind rates for price-comparison/aggregator motor
  insurance channels (often cited in the 10-20% range), not the near-90%
  retention baseline of a customer who already has no reason to leave.
- **Elasticity is larger in magnitude than retention's.** New business is
  well documented in P&C pricing-optimization literature as *more*
  price-elastic than renewal business: a prospect has no switching cost and
  is already actively comparing offers, unlike an existing customer with
  inertia. `CONVERSION_PRICE_ELASTICITY` (-6.0, vs. retention's -4.5) is
  the *direction and relative sizing* of that well-established finding,
  translated into a moderate, stated choice, not fit from data (this
  dataset cannot support that). Because conversion's baseline (15%) is so
  much lower than retention's (88%), even this moderate logit-scale
  difference compounds into a much larger *relative* swing in probability
  terms (worked out numerically in the notebook): a real mathematical
  consequence of the logistic curve at a low base rate, not a separate
  assumption layered on top.
- **The segment driver is risk-based market access, not tenure.** A brand-new
  prospect has no tenure with us by definition, so `BonusMalus` (which a
  prospect still carries from their claims history, even as a new customer
  to us) plays a different role here: high-`BonusMalus` (high-risk) prospects
  have structurally fewer insurers willing to compete for them (the same
  reason "residual market" / assigned-risk mechanisms exist in motor
  insurance for hard-to-place risks), so they are *less* price sensitive at
  the margin, not more. `risk_access_dampening` captures this direction; its
  magnitude is this project's own illustrative assumption.

THE "MARKET BENCHMARK" ASSUMPTION
--------------------------------------
This dataset has no competitor pricing either. `price_vs_market_pct = 0` is
defined as "quoting at whatever this project's own technical premium
already implies": i.e. today's quote is assumed to sit at market parity.
and the formula measures the conversion effect of moving *away* from that
assumed parity point. Stated as an assumption, not a measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Conversion probability: literature-calibrated, not fitted (see module docstring)
# --------------------------------------------------------------------------- #

BASELINE_CONVERSION_RATE = 0.15  # documented assumption: see module docstring
CONVERSION_PRICE_ELASTICITY = -6.0  # documented assumption, larger in magnitude than renewal_model.PRICE_ELASTICITY (-4.5): see module docstring
RISK_ACCESS_SATURATION_BONUS_MALUS = 200.0  # BonusMalus at which risk-access dampening reaches its maximum
RISK_ACCESS_MAX_DAMPENING = 0.5  # a maximally hard-to-place prospect is 50% less price-sensitive
BONUS_MALUS_NEUTRAL = 100.0  # dampening starts accruing above the neutral coefficient


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def _inverse_logit(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def risk_access_dampening(bonus_malus: pd.Series) -> pd.Series:
    """
    0 for a prospect at or below the neutral `BonusMalus` (100): a
    competitive market, full price sensitivity: ramping up to
    `RISK_ACCESS_MAX_DAMPENING` for a prospect at or beyond
    `RISK_ACCESS_SATURATION_BONUS_MALUS`, reflecting structurally reduced
    insurer competition for harder-to-place risks (see module docstring).
    """
    excess = (bonus_malus.astype(float) - BONUS_MALUS_NEUTRAL).clip(lower=0)
    fraction_saturated = (excess / (RISK_ACCESS_SATURATION_BONUS_MALUS - BONUS_MALUS_NEUTRAL)).clip(upper=1.0)
    return fraction_saturated * RISK_ACCESS_MAX_DAMPENING


def conversion_probability(
    price_vs_market_pct: np.ndarray,
    bonus_malus: pd.Series,
    base_rate: float = BASELINE_CONVERSION_RATE,
    elasticity: float = CONVERSION_PRICE_ELASTICITY,
) -> np.ndarray:
    """
    Predicted conversion (quote-to-bind) probability for a prospect quoted
    `price_vs_market_pct` away from the assumed market-parity point (0.10 =
    quoted 10% above market), given their `BonusMalus` (a claims-history
    signal a prospect still carries even as a new customer to us).
    """
    dampening = risk_access_dampening(bonus_malus)
    logit_p = _logit(np.array(base_rate)) + elasticity * np.asarray(price_vs_market_pct) * (1 - dampening)
    return _inverse_logit(logit_p)


def simulate_conversion_outcome(
    price_vs_market_pct: np.ndarray,
    bonus_malus: pd.Series,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    A single Bernoulli draw per prospect from `conversion_probability`: for
    illustration, exactly as `renewal_model.simulate_renewal_outcome` is for
    retention. Phase 11's scenario engine works with the probabilities
    directly, not simulated draws.
    """
    rng = np.random.default_rng(random_state)
    p = conversion_probability(price_vs_market_pct, bonus_malus, **kwargs)
    return rng.binomial(1, p)
