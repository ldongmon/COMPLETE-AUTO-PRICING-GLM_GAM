"""
conversion_model.py
===================
Phase 10: if we quote a prospect above the market, how likely is he or she
to buy? Same approach as the renewal model (Phase 9): freMTPL2 has no quotes
and no competitor prices, so the behaviour is SPECIFIED with stated
assumptions, not estimated.

THE FORMULA (same shape as renewal)
-----------------------------------
    logit(P(convert)) = logit(15%) + (-6.0) x price vs market x (1 - risk access)
    P(convert) = 1 / (1 + exp(-logit))

    risk access = 0.5 x min(max(BonusMalus - 100, 0) / 100, 1)
                  0 up to BonusMalus 100, 0.5 from BonusMalus 200

Hand example, quoted 10% above the market:
    logit(15%) = ln(0.15 / 0.85) = -1.7346
    low risk (BM 100, access 0):   -1.7346 - 6.0 x 0.10 x 1.0 = -2.3346 -> P = 8.8%
    high risk (BM 200, access 0.5): -1.7346 - 6.0 x 0.10 x 0.5 = -2.0346 -> P = 11.6%

THREE DIFFERENCES WITH RENEWAL, AND WHY
---------------------------------------
  baseline 15% (renewal 88%): a prospect asking for one quote is comparing
      insurers; quote-to-sale rates on comparison channels are about 10-20%.
  elasticity -6.0 (renewal -4.5): new business reacts more to price than
      renewals (no switching cost, already comparing offers).
  driver = BonusMalus, not tenure: a prospect has no tenure with us. A
      high-risk driver finds fewer insurers willing to cover him or her, so is
      LESS able to walk away from a high quote (less price-sensitive).

"Market price" = the price this project's technical premium implies today
(no competitor data exists): an assumption, stated as such.
"""

import numpy as np  # maths

BASELINE_CONVERSION_RATE = 0.15  # conversion rate at the market price
CONVERSION_PRICE_ELASTICITY = -6.0  # stronger than renewal's -4.5
RISK_ACCESS_SATURATION_BONUS_MALUS = 200.0  # risk access is at its maximum from BonusMalus 200
RISK_ACCESS_MAX_DAMPENING = 0.5  # a very hard-to-place prospect is 50% less sensitive to price
BONUS_MALUS_NEUTRAL = 100.0  # no effect up to BonusMalus 100


def _logit(p):
    """Log-odds: ln(p / (1 - p))."""
    return np.log(p / (1 - p))


def _inverse_logit(x):
    """Back from log-odds to a probability: 1 / (1 + exp(-x))."""
    return 1 / (1 + np.exp(-x))


def risk_access_dampening(bonus_malus):
    """0.5 x min(max(BM - 100, 0) / 100, 1): 0 up to BM 100, 0.5 from BM 200."""
    excess = (bonus_malus.astype(float) - BONUS_MALUS_NEUTRAL).clip(lower=0)  # points above 100
    fraction_saturated = (excess / (RISK_ACCESS_SATURATION_BONUS_MALUS - BONUS_MALUS_NEUTRAL)).clip(upper=1.0)  # share of the way to 200
    return fraction_saturated * RISK_ACCESS_MAX_DAMPENING


def conversion_probability(price_vs_market_pct, bonus_malus, base_rate=BASELINE_CONVERSION_RATE, elasticity=CONVERSION_PRICE_ELASTICITY):
    """
    P(convert) for a quote `price_vs_market_pct` away from the market (0.10 = 10% above).
        logit(P) = logit(base rate) + elasticity x price vs market x (1 - risk access)
    """
    dampening = risk_access_dampening(bonus_malus)  # risk access of each prospect
    logit_p = _logit(np.array(base_rate)) + elasticity * np.asarray(price_vs_market_pct) * (1 - dampening)  # log-odds after the price move
    return _inverse_logit(logit_p)  # back to a probability


def simulate_conversion_outcome(price_vs_market_pct, bonus_malus, random_state=42, **kwargs):
    """One random draw per prospect (1 = buys, 0 = does not), for illustration; Phase 11 uses the probabilities."""
    rng = np.random.default_rng(random_state)  # reproducible random numbers
    p = conversion_probability(price_vs_market_pct, bonus_malus, **kwargs)
    return rng.binomial(1, p)  # 1 with probability p
