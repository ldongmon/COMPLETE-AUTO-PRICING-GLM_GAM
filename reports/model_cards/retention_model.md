# Model Card: Retention / Renewal Model (Phase 9)

**Module:** [`src/retention/renewal_model.py`](../../src/retention/renewal_model.py)
**Built in:** [`notebooks/09_retention_model.ipynb`](../../notebooks/09_retention_model.ipynb)
**Output:** [`data/processed/tenure_proxy.parquet`](../../data/processed/tenure_proxy.parquet)
**Status:** documented scenario simulator: **not** a model trained on observed behavior (see below)

## Purpose

Answers the second half of the project's original business question: *"if we raise price on a
segment, how many existing customers do we lose?"*: for Segment X (`BonusMalus` 51-80, identified in
Phase 8) and, more generally, for any future repricing scenario Phase 11's engine needs to evaluate.

## Why this is a simulator, not a fitted model

freMTPL2 is a single-year cross-section (Phase 1): it has no renewal outcome and no price-change
history, so there is no real elasticity signal in this dataset to learn from. No public, policy-level
dataset linking premium change to renewal for motor insurance is known to exist either: elasticity
curves are commercially sensitive and insurers do not publish them. This decision is recorded in
`docs/PROJECT_ROADMAP.md`'s Phase 9 entry: rather than fabricate a "fit", this module
**specifies** a renewal-probability formula with every parameter sourced or cited, and states which is
which.

## The tenure proxy: derived from a real regulatory formula, not simulated noise

France's motor no-claims-discount system is a published formula: `BonusMalus` starts at 100 (neutral),
multiplies by **0.95** per claim-free year (floor: 50), and multiplies by 1.25 per at-fault claim.
Solving backward for `BonusMalus <= 100`:

```
tenure_years = ln(BonusMalus / 100) / ln(0.95)
```

This is a **lower bound** on tenure (not an exact figure), for two documented reasons: (1) the
62.7%-of-exposure mass at the `BonusMalus = 50` floor (Phase 2) is censored at "at least ~13.5 years";
(2) a driver can carry their bonus-malus coefficient to a new insurer. For the 1.2% of policies with
`BonusMalus > 100` (a recent at-fault claim), the formula can't cleanly separate tenure from claims
count, so a short default (1 year) is used: a stated simplification.

**Result on the real portfolio:** median tenure proxy 13.5 years (floor-censored), mean 10.6 years.

## The renewal probability formula

```
logit(P(renew)) = logit(baseline) + elasticity * price_change_pct * (1 - loyalty_dampening(tenure))
```

| Parameter | Value | Source |
|---|---|---|
| `BASELINE_RENEWAL_RATE` | 88% | Documented assumption, in line with commonly-cited retention rates for mature European personal motor books (80-90% typical range) |
| `PRICE_ELASTICITY` | -4.5 | Documented assumption, calibrated so a new customer facing +10% sees a ~5-6pt renewal-probability drop: in line with semi-elasticity ranges reported in P&C pricing-optimization literature (e.g. CAS monograph *"Price Optimization in Property/Casualty Insurance"*) |
| `LOYALTY_SATURATION_YEARS` | 10 | Illustrative assumption for this simulator |
| `LOYALTY_MAX_DAMPENING` | 60% | Illustrative magnitude, directionally consistent with the documented "loyalty penalty"/price-walking phenomenon (UK FCA *General Insurance Pricing Practices Market Study*, 2020-2021): the *direction* (long-tenured customers are less price-sensitive) is the cited, real finding; the specific 60% figure is this project's own choice, not taken from the FCA study |

None of these four parameters were estimated from freMTPL2.

## Validation performed (of the formula's behavior, not of predictive accuracy: there is no ground truth to validate against)

- At 0% price change, every tenure level converges to exactly the 88% baseline (checked directly on
  the full portfolio: mean = std = 0, all 668,626 policies at 88.0%).
- +10% price change: new customer 82.4% (5.6pt drop, matches the calibration target), 10+-year tenure
  86.0% (2.0pt drop): the loyalty dampening effect visibly at work.
- Stays within valid probability bounds across a wide range (-20% to +30% price change tested).

## Applied to Segment X

`BonusMalus` 51-80 (28.3% of portfolio exposure), under the **+17%** correction indicated by Phase 8
(the segment's measured cost relativity):

| | Current pricing | +17% correction |
|---|---|---|
| Mean P(renew) | 88.0% | 83.2% |
| Expected policies lost | n/a | about 10,100 (4.8% of the segment) |

The segment's tenure proxy is shorter than the book's (8.9 vs. 10.6 years), so it reacts more than the
average policy. **Phase 11** combines this effect with the conversion effect (Phase 10) and the premium
change.

## Known limitations

1. **This is a simulator, calibrated to external benchmarks, not a model fit to observed renewal
   behavior.**
2. The tenure proxy is a lower bound, heavily censored at the BonusMalus floor for the majority of the
   book: real tenure variation among "13.5+ years" policies is invisible to this proxy.
3. `PRICE_ELASTICITY` and the loyalty-dampening magnitude are illustrative, not empirically estimated.
   a real deployment should replace them with parameters fit on real renewal experience as soon as
   it exists (see the roadmap's "test-and-learn" recommendation below).
4. The formula assumes a single, uniform elasticity shape across all risk segments except for the
   tenure-driven loyalty effect: no interaction with `BonusMalus` level, region, or other rating
   factors beyond what tenure itself captures.

## The realistic path to a real model (recorded, not implemented here)

The usual fix for missing elasticity data is not a better public dataset (none
exists) but a **test-and-learn pricing program**: randomized rate variation on a slice of quotes/
renewals with a control cell, recalibrating this formula's parameters on real experience as it
accumulates. This project's `scoring/` pipeline design (Phase 7) is built so a real, experience-based
retention model could later replace `renewal_probability()` without changing any downstream code
(Phase 11's scenario engine, the app).

## Reproducing

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/09_retention_model.ipynb
```

Requires `data/processed/policy_level_clean.parquet` (Phase 1) only: no fitted-model dependencies,
since this phase builds no fitted artifact.
