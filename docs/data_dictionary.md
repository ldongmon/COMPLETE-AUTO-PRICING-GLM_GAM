# Data Dictionary

Reference for every table this project reads or produces: the two source files, the processed tables
built from them, and the columns the scoring pipeline and app add.

## Dataset inventory

| File | Rows | Level | Produced by | Committed to git? |
|---|---|---|---|---|
| `data/raw/freMTPL2freq.parquet` | 678,013 | 1 row / policy | `src/data/load_data.py` (OpenML fetch) | No, re-fetched automatically |
| `data/raw/freMTPL2sev.parquet` | 26,639 | 1 row / claim payment | `src/data/load_data.py` (OpenML fetch) | No, re-fetched automatically |
| `data/processed/policy_level_clean.parquet` | 668,626 | 1 row / policy | Notebook 01 (joins the two raw tables, excludes the claims-only block, adds derived columns, §2) | **Yes** (`.gitignore` exception: the deployed app reads it directly) |
| `data/processed/train.parquet` / `val.parquet` / `test.parquet` | 401,175 / 133,725 / 133,726 | 1 row / policy | Notebook 03 (60/20/20 policy-level split of the table above) | No, re-derived |
| `data/processed/tenure_proxy.parquet` | 668,626 | 1 row / policy | Notebook 09 (`derive_tenure_proxy`, §4) | No, re-derived |
| `models/app/*.pkl`, `*.joblib` | n/a | fitted model coefficients | `scripts/distill_app_models.py` | **Yes** (the app needs them, see the app model card) |

**Source of the two raw tables:** [OpenML](https://www.openml.org/), `freMTPL2freq` (data id `41214`)
and `freMTPL2sev` (data id `41215`), originally distributed in the R package `CASdatasets`. It is a
standard benchmark dataset for motor pricing, used in a large published literature (for example
Wüthrich & Merz, *Statistical Foundations of Actuarial Learning and its Applications*).

**Scope:** one accounting year of a French private motor third-party-liability book.
`freMTPL2freq` has one row per policy; `freMTPL2sev` has one row per claim payment and is aggregated
to policy level before joining.

`src/data/load_data.py` fetches both tables via `sklearn.datasets.fetch_openml` and caches them as
parquet under `data/raw/`, so later runs do not hit the network.

---

## 1. Raw source data

### `freMTPL2freq`: one row per policy

| Column | Type | Description | Notes |
|---|---|---|---|
| `IDpol` | int | Policy identifier | Join key to `freMTPL2sev`. Unique in this OpenML copy. |
| `ClaimNb` | int | Number of claims during the exposure period | The frequency target. |
| `Exposure` | float | Time on risk, in years (e.g. `0.5` = 6 months) | Should be in `(0, 1]`; a few values exceed 1 (quirk 2). Used as the GLM **offset**, never as a feature. |
| `Area` | categorical (`A` to `F`) | Area code, ordered proxy for population density | Ordinal. |
| `VehPower` | int | Vehicle power rating | Ordinal, French fiscal-power scale. |
| `VehAge` | int | Vehicle age, in years | Continuous/ordinal. |
| `DrivAge` | int | Age of the main driver, in years | Continuous/ordinal. |
| `BonusMalus` | int | French bonus-malus coefficient | `100` = neutral, `<100` = bonus, `>100` = malus; floor at 50. Encodes the driver's own claim history. |
| `VehBrand` | categorical | Vehicle brand, anonymized (`B1`, `B2`, ...) | Nominal. |
| `VehGas` | categorical (`Diesel` / `Regular`) | Fuel type | Nominal binary. |
| `Density` | float | Population density (inhabitants/km²) of the policyholder's town | Heavily right-skewed, log-transformed for modeling. |
| `Region` | categorical | French administrative region (pre-2016 naming) | Nominal. |

### `freMTPL2sev`: one row per claim payment

| Column | Type | Description | Notes |
|---|---|---|---|
| `IDpol` | int | Policy the payment belongs to | Many-to-one back to `freMTPL2freq`. |
| `ClaimAmount` | float | Payment amount, in euros | The severity target once aggregated to policy level. |

---

## 2. Processed dataset: `policy_level_clean.parquet`

Built by [`notebooks/01_data_acquisition_and_quality_checks.ipynb`](../notebooks/01_data_acquisition_and_quality_checks.ipynb):
excludes the leading claims-only block (quirk 3), joins `freMTPL2sev` onto `freMTPL2freq` (aggregated
to one row per policy first) and adds the columns below. **668,626 rows**, one per policy. Every later
phase and the app are built from this table.

| Column | Type | Description | Derivation |
|---|---|---|---|
| *(the 12 `freMTPL2freq` columns)* | | See §1 | Carried through as-is |
| `Exposure_raw` | float | Original, uncapped `Exposure` | Kept so the capping decision is auditable |
| `Exposure` | float | `Exposure_raw` capped at `1.0` | A policy cannot be on risk longer than the one-year window (quirk 2) |
| `sev_claim_count` | int | Number of matching payment rows in `freMTPL2sev` | `0` if none |
| `sev_claim_amount` | float | Sum of `ClaimAmount` over those rows | `0.0` if none |
| `ClaimNb_freq_target` | int | Frequency model target | `= ClaimNb` |
| `AvgClaimAmount` | float | Average paid amount per claim, for this policy | `sev_claim_amount / sev_claim_count` where `sev_claim_count > 0`, otherwise `NaN` (not applicable, not zero) |
| `HasPaidClaim` | bool | At least one paid claim | `sev_claim_count > 0`, the filter for the severity training set (Phase 6) |

## 3. Train / validation / test splits

`train.parquet` (60%), `val.parquet` (20%), `test.parquet` (20%), built by
[`notebooks/03_data_preparation_and_split.ipynb`](../notebooks/03_data_preparation_and_split.ipynb).
Same columns as §2; stratified on claim count at the **policy level** to prevent leakage. Exposure-weighted
frequency is 7.40-7.41% on all three sets and the median paid claim is EUR 1,172 on all three.

## 4. Columns added by the scoring pipeline

Computed by [`scoring/score_technical_premium.py`](../scoring/score_technical_premium.py) and
[`src/pricing/`](../src/pricing/); not stored, but present in every scored DataFrame from Phase 7 on.

| Column | Type | Description | Produced by |
|---|---|---|---|
| `predicted_frequency` | float | Predicted claims per policy-year, at a full year of exposure (offset = 0) | `score_technical_premium()`, Phase 7 refined frequency GLM |
| `predicted_severity` | float | Predicted EUR per claim (attritional layer) | `score_technical_premium()`, Phase 6 severity GLM |
| `large_loss_loading` | float | Flat EUR per policy-year for the large-loss layer (EUR 33.54, Phase 6) | Same value for every policy |
| `technical_premium` | float | `predicted_frequency × predicted_severity + large_loss_loading` | Expected claims cost per policy-year |
| `earned_premium` | float | `technical_premium × Exposure` | [`src/pricing/portfolio_kpis.py`](../src/pricing/portfolio_kpis.py) |
| `commercial_premium` | float | `technical_premium × (1 + loading)` | [`src/pricing/commercial_premium.py`](../src/pricing/commercial_premium.py), expense loading of 37% (Phase 11) |
| `tenure_years_proxy` | float | Lower-bound tenure in years, derived from `BonusMalus` with the French bonus-malus formula | [`src/retention/renewal_model.py`](../src/retention/renewal_model.py), `derive_tenure_proxy` |

---

## Known data quirks (measured on this copy)

Measured on the files downloaded from OpenML (data ids 41214 / 41215) in
[`notebooks/01_data_acquisition_and_quality_checks.ipynb`](../notebooks/01_data_acquisition_and_quality_checks.ipynb).

1. **Duplicate `IDpol`: none.** Some redistributions of this dataset are reported to contain duplicate
   policy IDs; in this copy all 678,013 `IDpol` values are unique.
2. **`Exposure` above 1 year: 1,224 policies (0.18%)**, up to 2.01, capped at 1.0. No policy has
   `Exposure <= 0`.
3. **A claims-only block at the start of the file (excluded).** Every policy with `IDpol` up to 24,951
   (9,387 policies, 1.34% of exposure) has at least one reported claim: there is no claim-free policy in
   that range, the implied claim frequency is above 200%, and only 3.1% of its 9,942 reported claims have
   a payment in `freMTPL2sev`. Beyond that point, 96.3% of policies are claim-free and 99.9% of reported
   claims have a payment. The block is a claims extract appended to the portfolio, not a sample of it.
   Kept, it would inflate modeled frequency without the matching cost and distort rating relativities
   (it is two thirds Regular fuel, against about half elsewhere). It is detected from the data by
   `find_claims_only_block_end` and excluded in Phase 1.
4. **`ClaimNb` vs. `freMTPL2sev` after the exclusion.** 22 policies still have a reported claim with no
   payment (negligible); no policy has payments without a reported claim; 6 `IDpol` values in
   `freMTPL2sev` have no matching policy and are dropped. `ClaimNb` is the frequency target;
   severity is built from the payments, for the 24,651 policies with at least one.
5. **Claim amounts.** No zero or negative `ClaimAmount` (`min = EUR 1`). Heavy-tailed: median
   EUR 1,172, mean EUR 2,279, max EUR 4,075,401. A few amounts recur thousands of times (EUR 1,204,
   EUR 1,128.12, EUR 1,172), which looks like standard settlement values.
6. **`ClaimNb` tail.** 96.3% of the retained policies have no claim; the maximum is 16 claims on one
   policy, and 8 policies have more than 4.
7. **Missing values: none**, in any column of either table.
