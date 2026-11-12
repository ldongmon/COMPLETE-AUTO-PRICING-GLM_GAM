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
