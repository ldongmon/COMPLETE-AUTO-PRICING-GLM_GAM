# Methodology Notes: Auto Insurance Pricing & Retention Engine

**Prepared by:** Dongmo Nanda, Actuary / Pricing Data Scientist
**Date:** 2026-11-12

A single reference for the scope, data, methods, assumptions, limitations and validation of the
project. Worked numbers are in the notebooks (`notebooks/01` to `11`) and the model cards
(`reports/model_cards/`).

---

## 1. Purpose and scope

**Business question:** *a segment of the motor book has a Loss Ratio above the rest of the book. Should
it be repriced, by how much, and what would that do to retention, new-business conversion and
profitability?*

**In scope:** frequency and severity modeling (GLM and GAM), technical premium, portfolio KPIs and
segment diagnosis, retention and conversion simulators, commercial premium, a repricing scenario
engine, and an interactive app.

**Out of scope:** reserving, capital, reinsurance, rate filing, and any connection to a policy or
quoting system. This is a pricing analysis on one public research dataset.

## 2. Data

**Source:** French Motor Third-Party Liability dataset on [OpenML](https://www.openml.org/),
`freMTPL2freq` (id `41214`) and `freMTPL2sev` (id `41215`). One accounting year; 678,013 policies and
26,639 claim payments in the raw files. Details: [`docs/data_dictionary.md`](data_dictionary.md).

**Exclusion of a claims-only block.** The first 9,387 policies of the file (IDpol up to 24,951) all
have a reported claim, with an implied frequency above 200% and only 3.1% of their claims paid in
freMTPL2sev; elsewhere, 96% of policies are claim-free and 99.9% of claims are paid. This block is a
claims extract, not part of the insured population. Kept, it inflated modeled frequency without the
matching cost and distorted rating relativities, notably by fuel type. It is excluded in Phase 1;
**668,626 policies** remain.

**Limitations that bound everything downstream:**
- **One year, cross-sectional:** no renewal history, no price changes, no quote outcomes, hence
  simulators rather than fitted models in Phases 9-10.
- **No premium data:** the "technical premium" is model output, so Loss Ratio measures the fit of the
  expected cost to actual claims, not realized profitability.
- **No expense data:** the 27% expense ratio is an assumption.

**Split:** 60/20/20 train/validation/test at policy level, stratified on claim count. Exposure-weighted
frequency is 7.40-7.41% on all three sets and the median paid claim is EUR 1,172 on all three; the
difference in mean severity comes from a few very large claims, mainly in the validation set.

## 3. Model inventory

| # | Model | Type | Module | Role |
|---|---|---|---|---|
| 1 | Frequency GLM | Poisson GLM, log link, `log(Exposure)` offset | [`frequency_glm.py`](../src/models/frequency_glm.py) | Baseline |
| 2 | Frequency GAM | Poisson GAM (`pygam`) | [`frequency_gam.py`](../src/models/frequency_gam.py) | Benchmark, source of the bins |
| 3 | Refined frequency GLM | Poisson GLM, `DrivAge` and `VehAge` binned | [`frequency_glm_refined.py`](../src/models/frequency_glm_refined.py) | **Frequency model used** |
| 4 | Severity GLM | Gamma GLM, log link, capped severity | [`severity_glm.py`](../src/models/severity_glm.py) | **Severity model used** |
| 5 | Severity GAM | Gamma GAM | [`severity_gam.py`](../src/models/severity_gam.py) | Benchmark |
| 6 | Technical premium | frequency x severity + flat large-loss loading | [`score_technical_premium.py`](../scoring/score_technical_premium.py) | Expected cost per policy |
| 7 | Portfolio KPIs, Segment X | Weighted aggregates, A/E by segment | [`portfolio_kpis.py`](../src/pricing/portfolio_kpis.py), [`segment_x.py`](../src/pricing/segment_x.py) | Segment diagnosis |
| 8 | Retention | Documented logistic simulator | [`renewal_model.py`](../src/retention/renewal_model.py) | Renewal under a price change |
| 9 | Conversion | Documented logistic simulator | [`conversion_model.py`](../src/prospect/conversion_model.py) | Quote-to-bind under a price change |
| 10 | Commercial premium | Expense loading | [`commercial_premium.py`](../src/pricing/commercial_premium.py) | Chargeable premium |
| 11 | Scenario engine | Deterministic combination of 6-10 | [`scenario_engine.py`](../scoring/scenario_engine.py) | Repricing scenarios, 1 to 3 years |

One model card per row in [`reports/model_cards/`](../reports/model_cards/).

## 4. Methods by stage

### 4.1 Frequency

`ClaimNb ~ Poisson`, `log(E[ClaimNb]) = log(Exposure) + Xβ`. Eight rating factors kept after
factor-level likelihood-ratio elimination (search on a 25% sub-sample for memory reasons, confirmed on
the full training set). Validation pseudo-R² 4.53%, Gini 0.285. `DrivAge`'s multivariate sign is
positive, opposite to its univariate trend: confounding with `BonusMalus` (r = -0.48).

A **Poisson GAM** (smoothing `lam=0.6`; automatic selection did not converge) does better out of sample
(pseudo-R² 5.49%, Gini 0.311). Its curves give the bins of the **refined GLM** for `DrivAge` (6 bins)
and `VehAge` (4 bins): pseudo-R² 4.72%, Gini 0.295. Most of the remaining gap to the GAM is the shape of
`BonusMalus`, kept linear in the tariff and examined in §4.4.

*Limitation:* mild overdispersion (Pearson residual std 1.44); a Negative Binomial or quasi-Poisson
variance would be the next step.

### 4.2 Severity

`AvgClaimAmount ~ Gamma`, log link, weighted by claim count, on policies with a paid claim, capped at
EUR 15,000 (99th percentile of training severity). Out of sample, neither the GLM (-0.09%) nor the GAM
(+0.04%) improves meaningfully on a naive average (-0.11%). The GLM is used (not worse, plausible
relativities); the large-loss layer is charged as a flat loading of EUR 33.54 per policy-year.

### 4.3 Technical premium

`technical_premium = predicted_frequency x predicted_severity + large_loss_loading`, from the refined
frequency GLM and the severity GLM. Paid claims / technical premium: 99.7% on the training set, 98.3% on
the test set. Gini against actual claim cost: 0.580 (0.556 with the original GLM).

### 4.4 Portfolio KPIs and Segment X

Ratios are always `sum(claims) / sum(premium)`, never averaged per policy. A raw Loss Ratio scan flags
`Region` R24 (146.7%, 28.5% of exposure); on the attritional layer it is in line with the book (100.8%
vs. 100.5%), its raw figure coming from large losses including the largest claim in the dataset.

Attritional actual-to-expected by segment, computed separately on train, validation and test, shows one
stable gap: `BonusMalus` 51-60 and 61-80 at about 1.15 to 1.25 in all three sets, while the bonus floor
(50) is at about 0.88. Fuel type shows no stable gap: Diesel and Regular have the same rate of large
claims, and the cost of those claims changes direction from one split to the next. Mechanism: the GLM's
straight-line `BonusMalus` term against the concave curve of the GAM; with `BonusMalus` in bands, the
refined GLM's validation pseudo-R² rises from 4.72% to 5.28%.

**Segment X = `BonusMalus` 51-80** (28.3% of exposure). Out-of-sample cost relativity, with attritional
claims from experience and large losses at the flat loading: **1.17**.

### 4.5 Retention simulator

Not a fitted model. Tenure proxy from the French bonus-malus formula
(`years = ln(BonusMalus/100) / ln(0.95)`, a lower bound, censored at the floor for 62.7% of exposure);
baseline renewal 88%; price elasticity -4.5 (consistent with pricing-optimization literature); loyalty
dampening up to 60% at 10+ years (direction consistent with the UK FCA pricing-practices study,
magnitude assumed).

### 4.6 Conversion simulator

Same approach: baseline 15% quote-to-bind; elasticity -6.0 (set after an initial -9.0 gave implausible
drops); `BonusMalus`-based market-access dampening instead of tenure. Conversion's relative sensitivity
is 7 to 11 times retention's at matched price moves.

### 4.7 Commercial premium

`commercial_premium = technical_premium x (1 + loading)`, with
`loading = expected LR / (1 - expense ratio) - 1 = 1 / 0.73 - 1 = 37.0%`. The technical premium is the
expected cost (expected LR = 100%); the realized book Loss Ratio (112.7%, driven by a few very large
claims in the validation set) is not used, so as not to price one year's large losses into every
policy.

### 4.8 Scenario engine

For a segment and a price change: retained policies (§4.5) and new policies (§4.6) at the repriced
commercial premium; contribution = `written_premium - expected_claims - expenses`. Expected claims =
`technical_premium x cost_factor`, with a factor of 1 for the book (the loading's basis) and the cost
relativity for a segment, so that the whole book at 0% change has a contribution of zero. Stated
simplifications: quote volume = current segment policy count; no adverse selection in who lapses or
converts. Multi-year: Year 1 row by row; from Year 2, with no further change, both simulators return
their baseline rate, so later years follow from Year 1 aggregates (a KPI outlook under stable
conditions, not a forecast).

## 5. Assumptions

| # | Assumption | Value | Basis | Where |
|---|---|---|---|---|
| A1 | Expense ratio | 27% | Illustrative, French personal motor range | [`portfolio_kpis.py`](../src/pricing/portfolio_kpis.py) |
| A2 | Large-loss threshold | EUR 15,000 (99th percentile) | Convention, not a tail-index estimate | [`severity_glm_gam.md`](../reports/model_cards/severity_glm_gam.md) |
| A3 | Baseline renewal rate | 88% | Commonly reported European motor retention | [`retention_model.md`](../reports/model_cards/retention_model.md) |
| A4 | Renewal price elasticity | -4.5 | Pricing-optimization literature, moderate | idem |
| A5 | Loyalty dampening | up to 60% at 10+ years | Direction from the UK FCA study; magnitude assumed | idem |
| A6 | Baseline conversion rate | 15% | Price-comparison quote-to-bind range | [`conversion_model.md`](../reports/model_cards/conversion_model.md) |
| A7 | Conversion price elasticity | -6.0 | Relative sizing vs. renewal; set once during development | idem |
| A8 | New-business quote volume | = segment policy count | No quote data | [`scenario_engine.md`](../reports/model_cards/scenario_engine.md) |
| A9 | Risk mix of retained and new business | = segment average | No adverse selection modeled | idem |
| A10 | Market parity | today's technical premium | No competitor prices | [`conversion_model.md`](../reports/model_cards/conversion_model.md) |
| A11 | Commercial loading | 37.0%, one for the book | Expected LR 100%, expenses 27% | [`scenario_engine.md`](../reports/model_cards/scenario_engine.md) |

## 6. Limitations

- **No observed retention, conversion, premium or expense data.** Phases 9-11 rest on simulators and
  assumptions; their outputs are directional. A test-and-learn cell on the indicated correction would
  provide the data to recalibrate them.
- **Statistical simplifications:** frequency overdispersion not modeled; no credibility weighting for
  small `Region` and `VehBrand` levels; large-loss threshold by convention; GAM smoothing fixed;
  `BonusMalus` linear in the tariff (extrapolates steeply at high malus values).
- **Scope:** one portfolio-wide loading, no lifetime value, no competitor reaction; the scenario engine
  is meaningful only within about -20% to +30%.
- **App:** read-only analysis over a cached snapshot.

## 7. Validation summary

| Claim | Evidence | Where |
|---|---|---|
| Frequency model generalizes | Refined GLM: pseudo-R² 4.72%, Gini 0.295 on 133,725 validation policies | [`technical_premium.md`](../reports/model_cards/technical_premium.md) |
| GAM gain is real, not in-sample | Out of sample: 5.49% (GAM) vs. 4.53% (GLM) | [`freq_gam.md`](../reports/model_cards/freq_gam.md) |
| Severity has weak explanatory power | GLM -0.09%, GAM +0.04%, naive -0.11% out of sample | [`severity_glm_gam.md`](../reports/model_cards/severity_glm_gam.md) |
| Technical premium is balanced | Paid claims / technical premium: 99.7% train, 98.3% test | [`07_technical_premium.ipynb`](../notebooks/07_technical_premium.ipynb), [`08_portfolio_kpis.ipynb`](../notebooks/08_portfolio_kpis.ipynb) |
| Segment X gap is not noise | `BonusMalus` 51-80 A/E 1.15-1.25 on train, validation and test | [`portfolio_kpis.md`](../reports/model_cards/portfolio_kpis.md) |
| Split is unbiased | Frequency 7.40-7.41% and median severity EUR 1,172 on all three sets | [`03_data_preparation_and_split.ipynb`](../notebooks/03_data_preparation_and_split.ipynb) |
| Loading and engine share one cost basis | Whole book at 0% change: contribution 0 (notebook check and unit test) | [`scenario_engine.md`](../reports/model_cards/scenario_engine.md) |
| App scoring matches the notebooks | Exact-match unit test on the coefficient-only models | [`streamlit_app.md`](../reports/model_cards/streamlit_app.md) |
| Code | 151 unit tests passing (`pytest tests/`) | [`tests/`](../tests/) |

## 8. Result

**Reprice `BonusMalus` 51-80 (28.3% of exposure) by +17%**, the segment's measured out-of-sample cost
relativity. Under the simulators' assumptions, about 4.8% of the segment's existing policies lapse,
new-business conversion falls from 15.0% to 6.0%, and the segment's expected annual contribution moves
from about EUR -5.7M to about breakeven. In practice this is a correction of the `BonusMalus` relativity
for that band; the next tariff revision should model `BonusMalus` in bands. Derivation:
[`notebooks/11_commercial_premium_and_scenario_engine.ipynb`](../notebooks/11_commercial_premium_and_scenario_engine.ipynb).

## 9. Reproducibility

All figures come from the notebooks in this repository, executed end to end; library versions are
pinned in [`requirements.txt`](../requirements.txt). Instructions: [`README.md`](../README.md#how-to-reproduce).

## 10. Document map

| Document | Contents |
|---|---|
| [`README.md`](../README.md) | Overview, business question, results, how to run |
| [`docs/PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md) | Phase plan and design decisions |
| [`docs/data_dictionary.md`](data_dictionary.md) | Fields and measured data quirks |
| `docs/methodology.md` (this document) | Methods, assumptions, limitations, validation |
| [`reports/model_cards/`](../reports/model_cards/) | One card per model |
| [`notebooks/`](../notebooks/) | Full analysis, one notebook per stage |
