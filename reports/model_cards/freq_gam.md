# Model Card: Frequency Model (Poisson GAM)

**Model file:** [`models/freq_gam.pkl`](../../models/freq_gam.pkl) (a dict: `gam`, the fitted
`pygam.PoissonGAM`, plus `feature_builder`, the fitted `GAMFeatureBuilder`)
**Built in:** [`notebooks/05_frequency_model_gam.ipynb`](../../notebooks/05_frequency_model_gam.ipynb)
**Status:** benchmark; its curves are used to design the Phase 7 bins

## Purpose

Tests whether smooth, non-linear effects improve on the Phase 4 GLM, and shows the shape of each
factor's effect.

## Data and features

- Fitted on a stratified 25% sub-sample of `train.parquet` (100,293 policies, memory constraint).
- Validated on the full `val.parquet` (133,725 policies).
- `GAMFeatureBuilder` (`src/models/frequency_gam.py`): `DrivAge`, `BonusMalus`, `VehPower`,
  `log(Density)`, `VehAge` (capped at 24 years) as spline terms; `VehBrand`, `VehGas`, `Region` as
  factor terms. `Exposure` enters through `pygam`'s `exposure=` argument (equivalent to the offset).

## Specification

Poisson GAM: 5 spline terms (`n_splines=10`) and 3 factor terms, smoothing `lam=0.6` for every term.

## Fitting strategy

`fit_poisson_gam_with_fallback` tries configurations from 25% of train with 10 knots down to 3% with
5 knots, moving on after a `MemoryError`. This run used the largest configuration.

## Out-of-sample performance (validation set)

| Metric | GLM (Phase 4) | GAM |
|---|---|---|
| Validation deviance | 32,563 | 32,236 |
| Validation pseudo-R² | 4.53% | 5.49% |
| Validation Gini | 0.285 | 0.311 |

The in-sample comparison (GAM 10.0% vs. GLM 4.1%) is not valid across different training sizes and is
shown in the notebook only to explain why it is not used.

## Partial dependence

- **`DrivAge`:** high at 18 (relativity about 1.66), minimum near 30 (0.59), about 1.07 at 45-50, about
  0.90 near 65, then rising after 75 (1.10 at 80, 1.22 at 85). The 95% band widens sharply beyond 85.
- **`VehAge`:** plateau from 0 to about 10 years (1.17 to 1.27), then decline to about 0.64 at 24 years.
- **`BonusMalus`:** concave, steep from 50 to about 70 then nearly flat to 90, unlike the GLM's
  straight line (examined in Phase 8).

## Conclusion

The GAM fits better out of sample, but its curves must be binned before they can serve as a rate
table, and `pygam`'s p-values carry a library caveat. Phase 7 uses the `DrivAge` and `VehAge` curves to
re-bin those factors in a refined GLM.

## Known limitations

1. Smoothing not optimized: `pygam.gridsearch` did not converge on this specification; `lam=0.6` used
   for every term.
2. Fitted on a sub-sample; the GLM it is compared with uses the full training set.
3. `pygam`'s p-values are not relied on.
4. The `DrivAge` curve beyond 85 rests on thin data.

## Reproducing

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/05_frequency_model_gam.ipynb
```

Requires the Phase 3 files and `models/freq_glm.pkl` (Phase 4). On a machine with different free
memory, the fallback may select another configuration; the notebook prints the one used.
