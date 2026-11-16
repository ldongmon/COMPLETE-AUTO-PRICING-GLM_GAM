"""
renewal_model.py
================
Phase 9: if we raise the price of a segment, how many existing customers
leave?

WHY A SIMULATOR AND NOT A FITTED MODEL
--------------------------------------
freMTPL2 is a one-year photograph: it has no renewal outcome and no price
change history, so there is nothing to learn an elasticity from. This module
therefore SPECIFIES the renewal behaviour with stated assumptions, instead of
estimating it. Every parameter is either computed from a real rule (the
French bonus-malus) or an assumption in the range quoted by the literature.

STEP 1: HOW LONG HAS THE CUSTOMER BEEN CLAIM-FREE? (tenure proxy)
-----------------------------------------------------------------
The French bonus-malus starts at 100 and is multiplied by 0.95 after every
claim-free year (floor 50). Read backwards:

    BonusMalus = 100 x 0.95^years   =>   years = ln(BonusMalus / 100) / ln(0.95)

Example: BonusMalus 57 -> ln(0.57) / ln(0.95) = -0.5621 / -0.0513 = 10.96 years.
At the floor (50): 13.5 years, the most the formula can tell ("at least
13.5 years"), so it is a lower bound. Above 100 (a recent claim, 1.15% of the
book) tenure cannot be separated from the claims: 1 year is used.

STEP 2: THE RENEWAL PROBABILITY
-------------------------------
A logistic curve, the usual shape for a probability:

    logit(p) = ln(p / (1 - p))                  (the "log-odds")
    logit(P(renew)) = logit(88%) + (-4.5) x price change x (1 - loyalty)
    P(renew) = 1 / (1 + exp(-logit))

    loyalty = 0.6 x min(tenure / 10, 1)         (0 for a new customer, 0.6 after 10 years)

Hand example, +10% price:
    logit(88%) = ln(0.88 / 0.12) = 1.9924
    new customer (loyalty 0):   1.9924 - 4.5 x 0.10 x 1.0 = 1.5424 -> P = 82.4%
    10+ years   (loyalty 0.6):  1.9924 - 4.5 x 0.10 x 0.4 = 1.8124 -> P = 86.0%
    no price change: logit stays 1.9924 -> P = 88% for everybody.

THE ASSUMPTIONS
---------------
  88%   baseline renewal rate: usual range for a mature European motor book (80-90%)
  -4.5  price elasticity: a new customer facing +10% loses about 5-6 points of
        renewal probability, a middle value of the pricing literature
        (e.g. CAS monograph "Price Optimization in Property/Casualty Insurance")
  60% / 10 years  loyalty: long-standing customers shop around less (the
        "loyalty penalty" documented by the UK FCA market study, 2020-2021);
        the size used here is this project's own illustrative choice.
"""

import numpy as np  # maths
import pandas as pd  # tables

# --- Step 1: tenure proxy from the French bonus-malus rule ---
BM_ANNUAL_DISCOUNT = 0.95  # each claim-free year multiplies the bonus-malus by 0.95
BM_NEUTRAL_START = 100.0  # every driver starts at 100
MALUS_DEFAULT_TENURE_YEARS = 1.0  # tenure used above 100 (recent claim)


def derive_tenure_proxy(bonus_malus):
    """Claim-free years implied by the bonus-malus: ln(BM / 100) / ln(0.95), or 1 year above 100."""
    bm = bonus_malus.astype(float)
    years_from_discount = np.log(bm.clip(upper=BM_NEUTRAL_START) / BM_NEUTRAL_START) / np.log(BM_ANNUAL_DISCOUNT)  # the formula read backwards
    return pd.Series(
        np.where(bm <= BM_NEUTRAL_START, years_from_discount, MALUS_DEFAULT_TENURE_YEARS),  # 1 year for a malus
        index=bonus_malus.index,
        name="tenure_years_proxy",
    )


# --- Step 2: renewal probability ---
BASELINE_RENEWAL_RATE = 0.88  # renewal rate when the price does not change
PRICE_ELASTICITY = -4.5  # effect of the price change on the log-odds
LOYALTY_SATURATION_YEARS = 10.0  # loyalty is at its maximum after 10 years
LOYALTY_MAX_DAMPENING = 0.6  # a 10+ year customer is 60% less sensitive to price


def _logit(p):
    """Log-odds: ln(p / (1 - p))."""
    return np.log(p / (1 - p))


def _inverse_logit(x):
    """Back from log-odds to a probability: 1 / (1 + exp(-x))."""
    return 1 / (1 + np.exp(-x))


def loyalty_dampening(tenure_years):
    """0.6 x min(tenure / 10, 1): from 0 (new customer) to 0.6 (10 years or more)."""
    fraction_saturated = (tenure_years / LOYALTY_SATURATION_YEARS).clip(upper=1.0)  # share of the 10 years reached
    return fraction_saturated * LOYALTY_MAX_DAMPENING


def renewal_probability(price_change_pct, tenure_years, base_rate=BASELINE_RENEWAL_RATE, elasticity=PRICE_ELASTICITY):
    """
    P(renew) for a price change (0.10 = +10%) and a tenure in years.
        logit(P) = logit(base rate) + elasticity x price change x (1 - loyalty)
    """
    dampening = loyalty_dampening(tenure_years)  # loyalty of each customer
    logit_p = _logit(np.array(base_rate)) + elasticity * np.asarray(price_change_pct) * (1 - dampening)  # log-odds after the change
    return _inverse_logit(logit_p)  # back to a probability


def simulate_renewal_outcome(price_change_pct, tenure_years, random_state=42, **kwargs):
    """One random draw per policy (1 = renews, 0 = leaves), for illustration; Phase 11 uses the probabilities."""
    rng = np.random.default_rng(random_state)  # reproducible random numbers
    p = renewal_probability(price_change_pct, tenure_years, **kwargs)
    return rng.binomial(1, p)  # 1 with probability p
