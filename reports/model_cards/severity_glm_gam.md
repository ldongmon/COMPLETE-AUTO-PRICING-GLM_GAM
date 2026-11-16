# Model Card: Severity Models (Gamma GLM and Gamma GAM)

**Model files:** [`models/sev_glm.pkl`](../../models/sev_glm.pkl) (dict: `results`, `feature_columns`,
`large_loss_threshold`, `large_loss_loading`), [`models/sev_gam.pkl`](../../models/sev_gam.pkl) (dict:
`gam`, `feature_builder`, `large_loss_threshold`)
**Built in:** [`notebooks/06_severity_model_glm_gam.ipynb`](../../notebooks/06_severity_model_glm_gam.ipynb)
**Status:** GLM used in Phase 7; GAM built for comparison only

## Purpose

Predicts average claim severity (`AvgClaimAmount`) for policies with at least one paid claim: the
severity half of the technical premium (Phase 7).

## Data

- Severity subset of `train.parquet`: **14,792 policies** with at least one paid claim.
- Validated on the severity subset of `val.parquet` (4,928 policies).
- Weighted by `sev_claim_count`: an average over several claims is more precise than over one.

## Large-loss treatment

- Threshold: **EUR 15,000**, the 99th percentile of training severity (a round settlement amount).
- 146 of 14,792 training claims (0.99%) exceed it.
- The models below are fitted on severity **capped** at the threshold.
- **Large-loss loading: EUR 33.54 per policy-year**, added flat to every policy: training excess above
  the threshold divided by training exposure, like a burning-cost estimate for an excess-of-loss layer.

## Specification

- **GLM:** Gamma, log link, `var_weights = sev_claim_count`.
- **GAM:** Gamma, same feature builder as the Phase 5 frequency GAM, `n_splines=8`,
  `weights = sev_claim_count`.

## Factor selection (GLM)

Backward, factor-level stepwise elimination (as in Phase 4) drops **`DrivAge`**, **`Density`** and
**`VehGas`**. The univariate gap in *mean* severity between fuel types (Phase 2) does not survive
capping and controlling for `VehBrand` and `Region`. Kept: `BonusMalus`, `VehPower`, `VehAge`,
`VehBrand`, `Region`.

## Performance

| Metric | Naive (train average) | Gamma GLM | Gamma GAM |
|---|---|---|---|
| In-sample pseudo-R² | n/a | 1.4% | 1.9% |
| **Validation pseudo-R²** | **-0.11%** | **-0.09%** | **+0.04%** |

Neither model improves meaningfully on a flat average out of sample. The rating factors differentiate
claim occurrence well, not claim size; severity also has far less data (14,792 vs. 401,175 rows).

## Conclusion

- **Gamma GLM used in Phase 7:** not worse than the naive average, plausible relativities, and the
  standard form. The GAM brings no measured advantage.
- **Flat large-loss loading** on top.
- **Risk differentiation in the technical premium comes almost entirely from frequency.**

## Known limitations

1. The GLM's relativities are directional at best, especially for small `Region` and `VehBrand` levels.
2. The 99th-percentile threshold is a convention, not a formal tail-index estimate.
3. GAM smoothing fixed at `lam=0.6`, as for the frequency GAM.

## Reproducing

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/06_severity_model_glm_gam.ipynb
```

Requires `train.parquet`, `val.parquet` and `models/feature_encoder.joblib` (Phase 3).

## Worked example 1: cutting a real claim in two (IDpol 2261431)

**The policy.** A train policy with one paid claim of EUR 40,000 (shown highlighted in notebook 06).

| Step | Calculation | Where it comes from |
|---|---|---|
| Claim paid | EUR 40,000.00 | column `sev_claim_amount` |
| Attritional part | min(40,000, 15,000) = **EUR 15,000** | goes into the Gamma GLM |
| Large-loss part | max(40,000 - 15,000, 0) = **EUR 25,000** | goes into the shared pot |

## Worked example 2: the large-loss loading = EUR 33.54 per year

| Step | Calculation | Where it comes from |
|---|---|---|
| The pot | sum of the large-loss parts of the train claims = EUR 7,112,014 | columns `sev_claim_amount`, `sev_claim_count` |
| Years insured | sum of Exposure of all 401,175 train policies = 212,076.8 | column `Exposure` |
| **Loading** | 7,112,014 / 212,076.8 = **EUR 33.54 per year** | pot / years insured |

**Reading.** Every insured year pays its share of the large claims, whether or not the policy had a claim
(mutualisation, the burning cost of an excess-of-loss layer).

## Worked example 3: the expected average cost of IDpol 137202 = EUR 1,480.00

**The policy.** The first validation policy with a paid claim (shown highlighted in notebook 06):
BonusMalus 50, power 6, car 2 years old, brand B1 (reference), region R52, one claim of EUR 2,256.25.

| Term | Calculation | Where it comes from |
|---|---|---|
| Intercept | 7.1063 | Gamma GLM |
| BonusMalus | 0.00207 x 50 = +0.1035 | dataset, Gamma GLM |
| VehPower | 0.01377 x 6 = +0.0826 | same |
| VehAge | -0.0090 x 2 = -0.0181 | same |
| Region R52 | +0.0253 | same |
| **Total** | **7.2998** | |

Expected average cost = exp(7.2998) = **EUR 1,480.00** per claim (statsmodels: EUR 1,480.00).

**Reading.** The policy actually paid EUR 2,256.25. One claim is far from its expectation: the model gives
an average over many similar policies, not a forecast of each claim.
