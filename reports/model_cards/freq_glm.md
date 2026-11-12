# Model Card: Frequency Model (Poisson GLM)

**Model file:** [`models/freq_glm.pkl`](../../models/freq_glm.pkl) (a dict: `results`, the fitted
`statsmodels` `GLMResults`, plus `feature_columns` and `factor_groups`)
**Built in:** [`notebooks/04_frequency_model_glm.ipynb`](../../notebooks/04_frequency_model_glm.ipynb)
**Status:** baseline frequency model, refined in Phase 7

## Purpose

Predicts the expected claim count (`ClaimNb`) per policy from the rating factors: the frequency half
of the technical premium (Phase 7).

## Data

- Trained on `data/processed/train.parquet` (401,175 policies, 60% of the Phase 1 table).
- Validated on `data/processed/val.parquet` (133,725 policies), not used for fitting.
- Features from `models/feature_encoder.joblib` (Phase 3): `DrivAge`, `BonusMalus`, `VehPower`
  (numeric); `log(Density)`; `VehAge` capped at the 99.5th training percentile (24 years); `VehBrand`,
  `VehGas`, `Region` (one-hot, `drop="first"`). `Area` excluded (redundant with `Density`, Phase 2).
  `Exposure` is only the offset.

## Specification

Poisson GLM, log link, `offset = log(Exposure)`. All 8 rating factors kept (37 design columns plus the
intercept).

## Performance

| Metric | Value |
|---|---|
| Training rows | 401,175 |
| Deviance / null deviance | 97,826 / 101,963 |
| Pseudo-R² (deviance-based, in sample) | 4.1% |
| Pseudo-R² (validation) | 4.53% |
| AIC | 128,030 |
| Validation calibration | Predicted frequency rises from decile 1 (3.8%) to decile 10 (17.9%); observed follows the same order (3.7% to 18.2%) |
| Pearson residuals (validation) | mean 0.023, std 1.44 |

## Factor selection

Backward, factor-level stepwise elimination (likelihood-ratio test, 5%) on a stratified 25% training
sub-sample (memory constraint), checked against the full-data model. The search keeps all eight
factors, and the full-data p-values confirm the two weakest ones (`VehGas`, `VehPower`).

## Coherence check against Phase 2

`DrivAge`'s multivariate coefficient is **positive** (+0.0058 per year), the opposite of its univariate
relationship. Fitted alone it is negative (-0.0100); it turns positive once `BonusMalus` is in the
model. This is confounding: `BonusMalus` already carries much of the young-driver effect. The other
directions (`BonusMalus`, `VehBrand` B12, `VehGas`, `Density`, `VehAge`) match Phase 2.

## Known limitations

1. **Mild overdispersion** (Pearson residual std 1.44). A Negative Binomial or quasi-Poisson variance
   would be the standard next step.
2. **Linear terms** cannot follow the shapes of `DrivAge` (young-driver drop, then flat) and `VehAge`
   (plateau, then decline), nor the concave shape of `BonusMalus` (Phases 5 and 8).
3. **Small regions** (e.g. `Region_R21`, `Region_R43`, under 0.5% of exposure) have wide confidence
   intervals; a credibility weighting would be appropriate if the model were taken further.
4. Significant is not the same as material at this sample size: `VehPower` adds about 3.4% frequency
   per fiscal-power point.

## Reproducing

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/04_frequency_model_glm.ipynb
```

Requires `data/processed/train.parquet`, `val.parquet` and `models/feature_encoder.joblib` (Phase 3).
