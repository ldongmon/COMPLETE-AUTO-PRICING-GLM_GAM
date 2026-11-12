# Model Card: Technical Premium (Refined GLM x Severity GLM)

**Model files:**
[`models/freq_glm_refined.pkl`](../../models/freq_glm_refined.pkl) (dict: `results`, `encoder`),
[`models/sev_glm.pkl`](../../models/sev_glm.pkl) (Phase 6),
[`scoring/score_technical_premium.py`](../../scoring/score_technical_premium.py) (scoring pipeline)
**Built in:** [`notebooks/07_technical_premium.ipynb`](../../notebooks/07_technical_premium.ipynb)
**Status:** technical premium used by every later phase and the app

## Purpose

Combines frequency and severity into a per-policy-year **technical premium** (expected claims cost for
a full year of the risk), through one scoring function used everywhere.

```
technical_premium = predicted_frequency (claims / policy-year)
                   x predicted_severity (EUR / claim)
                   + large_loss_loading (EUR / policy-year, flat, Phase 6)
```

## Refined frequency GLM

Phase 5's GAM showed shapes that a straight line cannot follow for `DrivAge` and `VehAge`. The refined
GLM replaces those two linear terms by bins read off the GAM's curves:

| Factor | Curve shape (centered relativity) | Bin edges | Bins |
|---|---|---|---|
| `DrivAge` | 1.66 at 18, 0.59 near 30, 1.07 at 45-50, 0.90 near 65, rising after 75 | `[18, 25, 35, 50, 65, 80, 101]` | 18-24, 25-34, 35-49, 50-64, 65-79, 80-100 |
| `VehAge` (capped at 24) | plateau 1.17-1.27 from 0 to 10, then down to 0.64 at 24 | `[0, 11, 15, 19, 25]` | 0-10, 11-14, 15-18, 19-24 |

Other factors are unchanged from Phase 4; `BonusMalus` stays linear (its shape is examined in Phase 8).

## Data and fitting

- **Frequency:** fitted on a stratified 40% sub-sample of `train.parquet` (160,470 policies; full-size
  fits were unreliable on the 8GB development machine), validated on the full `val.parquet`.
- **Severity:** Phase 6 Gamma GLM, capped at EUR 15,000.
- **Large-loss loading:** EUR 33.54 per policy-year, flat.

## Validation (`val.parquet`, 133,725 policies)

| Frequency model | Validation pseudo-R² | Gini |
|---|---|---|
| GLM, original (Phase 4) | 4.53% | 0.285 |
| **GLM, refined** | **4.72%** | **0.295** |
| GAM (Phase 5, benchmark) | 5.49% | 0.311 |

The refined GLM closes part of the gap to the GAM (about a fifth on pseudo-R², a third on Gini). Most
of the remainder is `BonusMalus`: with `BonusMalus` in bands, the validation pseudo-R² reaches 5.28%
(Phase 8).

| Technical premium built from | Gini (vs. actual claim cost) |
|---|---|
| Original GLM + Phase 6 severity | 0.556 |
| **Refined GLM + Phase 6 severity** | **0.580** |

Median change in premium from the refinement: +0.8% (10th to 90th percentile: -14% to +13%).

**Balance:** paid claims / technical premium = 99.7% on the training set, 98.3% on the test set.

A Gini on monetary claim cost is not on the same scale as a frequency Gini: it also reflects how
concentrated claim euros are. It does not mean severity adds predictive signal.

## Known limitations

1. Refined GLM fitted on a 40% training sub-sample (machine constraint).
2. Bin edges read from the GAM curves by inspection: reasonable, not unique.
3. `BonusMalus` linear: under-prices the 51-80 range and extrapolates steeply at high malus values
   (maximum predicted frequency 12.9 claims per year).
4. Inherits the Phase 4-6 limitations (overdispersion, weak severity differentiation, threshold by
   convention).

## Reproducing

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/07_technical_premium.ipynb
```

Requires the Phase 3 files and the Phase 4, 5 and 6 model artifacts.
