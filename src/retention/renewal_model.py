"""
renewal_model.py
==================
Phase 9: a documented, literature-calibrated **scenario simulator** for
policy renewal, built because freMTPL2 has no renewal outcomes and no price-
change history to learn an elasticity from (it is a single-year
cross-section: see Phase 1). This module does not "fit" a model to
observed behavior; it *specifies* one, with every parameter either derived
from a real regulatory formula applied to this data, or cited to published
literature, and states plainly which is which. See
`docs/PROJECT_ROADMAP.md`'s Phase 9 decision note for the full rationale
behind building a simulator at all, and `notebooks/09_retention_model.ipynb`
for the worked derivation of every number below.

THE TENURE PROXY: DERIVED FROM THE REAL FRENCH BONUS-MALUS FORMULA
-----------------------------------------------------------------------
`BonusMalus` is not an arbitrary score: it is produced by a known,
published regulatory formula (the French "coefficient de réduction-
majoration"): each claim-free year multiplies the coefficient by 0.95 (a 5%
discount), down to a floor of 50; each at-fault claim multiplies it by 1.25.
Starting from the neutral value of 100, a policy with `BonusMalus <= 100`
and no claims must have had at least

    years = ln(BonusMalus / 100) / ln(0.95)

claim-free years on this scale: a **lower bound on tenure**, not a
fabricated number, derived from the same regulatory arithmetic that produced
the column in the first place. It is a *lower bound*, not an exact tenure,
for two reasons stated explicitly: (1) many policies (the large mass at the
BM=50 floor: Phase 2 found this is 62.7% of exposure) are censored at "at
least ~13.5 years," since the coefficient stops moving once it hits the
floor; (2) a policy could have switched insurers and carried its bonus with
it (common practice in France), so "years at *this* coefficient" is not
identically "years as *this insurer's* customer." For `BonusMalus > 100`
(a malus: a recent at-fault claim, only 1.15% of the book), the formula
cannot cleanly separate tenure from claims count without full claims
history, so a short default tenure is assigned instead (documented at
`MALUS_DEFAULT_TENURE_YEARS`): a deliberate, stated simplification, not an
attempt to reverse-engineer something the data can't support.

THE RENEWAL PROBABILITY FORMULA: LITERATURE-CITED, NOT FITTED
--------------------------------------------------------------------
    logit(P(renew)) = logit(BASELINE_RENEWAL_RATE)
                       + PRICE_ELASTICITY * price_change_pct * (1 - loyalty_dampening(tenure))

- `BASELINE_RENEWAL_RATE` (88%): a documented assumption in line with
  commonly-cited retention rates for mature European personal motor books
  (typically reported in the 80-90% range), not fitted from this dataset.
- `PRICE_ELASTICITY`: a documented assumption, in line with semi-elasticity
  ranges reported in P&C pricing-optimization literature (e.g. the CAS
  monograph *"Price Optimization in Property/Casualty Insurance"*): sized
  so a new, non-loyal customer facing a +10% price increase sees roughly a
  5-6 percentage point drop in renewal probability, a moderate, middle-of-
  the-cited-range elasticity, not an aggressive or conservative extreme.
- `loyalty_dampening(tenure)`: long-tenured customers are empirically *less*
  likely to shop around and switch even when price rises: the well-
  documented "loyalty penalty" / price-walking phenomenon (see the UK FCA's
  *General Insurance Pricing Practices Market Study*, 2020-2021, which found
  and ultimately acted against exactly this pattern). The specific dampening
  *magnitude* used here (`LOYALTY_MAX_DAMPENING`) is this project's own
  illustrative assumption, directionally consistent with that documented
  phenomenon, not a number taken from the FCA study itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Tenure proxy: derived from the real French bonus-malus formula
# --------------------------------------------------------------------------- #

BM_ANNUAL_DISCOUNT = 0.95  # French bonus-malus: -5% per claim-free year
BM_NEUTRAL_START = 100.0  # every policy starts here
MALUS_DEFAULT_TENURE_YEARS = 1.0  # documented simplification for BonusMalus > 100 (see module docstring)


def derive_tenure_proxy(bonus_malus: pd.Series) -> pd.Series:
    """
    Lower-bound tenure (in years), derived from `BonusMalus` via the real
    French no-claims-discount formula. See module docstring for the two
    documented reasons this is a lower bound, not an exact tenure.
    """
    bm = bonus_malus.astype(float)
    years_from_discount = np.log(bm.clip(upper=BM_NEUTRAL_START) / BM_NEUTRAL_START) / np.log(BM_ANNUAL_DISCOUNT)
    return pd.Series(
        np.where(bm <= BM_NEUTRAL_START, years_from_discount, MALUS_DEFAULT_TENURE_YEARS),
        index=bonus_malus.index,
        name="tenure_years_proxy",
    )


# --------------------------------------------------------------------------- #
# Renewal probability: literature-calibrated, not fitted (see module docstring)
# --------------------------------------------------------------------------- #

BASELINE_RENEWAL_RATE = 0.88  # documented assumption: see module docstring
PRICE_ELASTICITY = -4.5  # documented assumption: see module docstring
LOYALTY_SATURATION_YEARS = 10.0  # tenure at which loyalty dampening reaches its maximum
LOYALTY_MAX_DAMPENING = 0.6  # a fully "saturated" long-tenured customer is 60% less price-sensitive


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def _inverse_logit(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def loyalty_dampening(tenure_years: pd.Series) -> pd.Series:
    """
    0 for a brand-new customer (no dampening: full price sensitivity) up to
    `LOYALTY_MAX_DAMPENING` for a customer at or beyond `LOYALTY_SATURATION_YEARS`
    of tenure, ramping linearly in between.
    """
    fraction_saturated = (tenure_years / LOYALTY_SATURATION_YEARS).clip(upper=1.0)
    return fraction_saturated * LOYALTY_MAX_DAMPENING


def renewal_probability(
    price_change_pct: np.ndarray,
    tenure_years: pd.Series,
    base_rate: float = BASELINE_RENEWAL_RATE,
    elasticity: float = PRICE_ELASTICITY,
) -> np.ndarray:
    """
    Predicted renewal probability under a given price change, for policies
    with the given tenure. `price_change_pct` is a fraction (0.10 = +10%),
    broadcastable against `tenure_years` (a scalar or an array of the same
    length).
    """
    dampening = loyalty_dampening(tenure_years)
    logit_p = _logit(np.array(base_rate)) + elasticity * np.asarray(price_change_pct) * (1 - dampening)
    return _inverse_logit(logit_p)


def simulate_renewal_outcome(
    price_change_pct: np.ndarray,
    tenure_years: pd.Series,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    A single Bernoulli draw per policy from `renewal_probability`: for
    demonstrating what a *realized* (not just expected) renewal outcome
    looks like under a scenario. Phase 11's scenario engine works with the
    probabilities directly (`renewal_probability`), not simulated draws;
    this function exists for illustration and testing.
    """
    rng = np.random.default_rng(random_state)
    p = renewal_probability(price_change_pct, tenure_years, **kwargs)
    return rng.binomial(1, p)
