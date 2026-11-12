# Project Plan: Auto Insurance Pricing & Retention Engine

Scope and sequencing of the project. Each phase states the question it answers, the method, and its
deliverables. The order follows the dependencies of a pricing exercise: a technical premium needs
frequency and severity models; a repricing scenario needs retention and conversion assumptions; an
interactive tool needs all of the above as tested code. Results are in the notebooks and model cards,
not repeated here.

All phases (0 to 13) are complete.

---

## Phase 0: Project scaffolding

- Folder structure: `src/` (reusable code), `notebooks/` (analysis), `models/` (fitted artifacts),
  `scoring/` (shared scoring pipeline), `app/` (Streamlit), `docs/`, `reports/`.
- `requirements.txt` pinned to the tested versions.

## Phase 1: Data acquisition and quality checks

**Question:** what data do we have, and can we trust it?

freMTPL2freq (policies, exposure, claim counts) and freMTPL2sev (claim payments), one accounting year
of a French motor third-party-liability book, from OpenML (data ids 41214 and 41215).

- `src/data/load_data.py`: fetch-and-cache from OpenML to local parquet.
- Reconcile `ClaimNb` with the payments in freMTPL2sev; structural checks (duplicates, exposure,
  missing values).
- Exclusion of the leading claims-only block (IDpol up to 24,951), detected from the data.
- `docs/data_dictionary.md`.

Notebook: [`01_data_acquisition_and_quality_checks.ipynb`](../notebooks/01_data_acquisition_and_quality_checks.ipynb)

## Phase 2: Exploratory data analysis

**Question:** which rating factors move frequency and severity, and how?

- Exposure-weighted frequency and average severity for every candidate factor.
- Exposure distribution; redundancy check (`Area` vs. `Density`); correlations (`DrivAge` vs.
  `BonusMalus`).

Notebook: [`02_exploratory_data_analysis.ipynb`](../notebooks/02_exploratory_data_analysis.ipynb)

## Phase 3: Data preparation and split

**Question:** how do we split and encode the data so that evaluation is reliable?

- 60/20/20 train/validation/test split at policy level, stratified on claim count; balance checks.
- Encoder fitted on train only (`src/features/pipeline.py`); exposure as the GLM offset, not a feature.

Notebook: [`03_data_preparation_and_split.ipynb`](../notebooks/03_data_preparation_and_split.ipynb)

## Phase 4: Frequency model, Poisson GLM

**Question:** what is the expected claim count per policy?

- Poisson GLM, log link, `log(Exposure)` offset; factor-level stepwise selection.
- Coefficient signs checked against Phase 2; actual vs. predicted by decile on the validation set.

Notebook: [`04_frequency_model_glm.ipynb`](../notebooks/04_frequency_model_glm.ipynb) ·
Card: [`freq_glm.md`](../reports/model_cards/freq_glm.md)

## Phase 5: Frequency model, GAM and benchmark

**Question:** do smooth, non-linear effects improve on the GLM, and at what cost in interpretability?

- `pygam` Poisson GAM; out-of-sample comparison (deviance, pseudo-R², Gini, deciles).
- Partial dependence curves, used to define bins for an improved GLM.

Notebook: [`05_frequency_model_gam.ipynb`](../notebooks/05_frequency_model_gam.ipynb) ·
Card: [`freq_gam.md`](../reports/model_cards/freq_gam.md)

## Phase 6: Severity model, Gamma GLM and GAM

**Question:** given a claim, how large is it expected to be?

- Large-loss threshold (99th percentile) and flat large-loss loading; Gamma GLM and GAM on capped
  severity; comparison with a naive baseline.

Notebook: [`06_severity_model_glm_gam.ipynb`](../notebooks/06_severity_model_glm_gam.ipynb) ·
Card: [`severity_glm_gam.md`](../reports/model_cards/severity_glm_gam.md)

## Phase 7: Technical premium

**Question:** what is the expected claims cost per policy?

- Refined frequency GLM (`DrivAge` and `VehAge` binned from the GAM curves).
- `scoring/score_technical_premium.py`: frequency x severity + large-loss loading; Gini and calibration
  against actual claim cost; balance check.

Notebook: [`07_technical_premium.ipynb`](../notebooks/07_technical_premium.ipynb) ·
Card: [`technical_premium.md`](../reports/model_cards/technical_premium.md)

## Phase 8: Portfolio KPIs and Segment X

**Question:** which segment has a Loss Ratio above the rest of the book, and is it a real pricing gap?

- Loss Ratio, COR, ATP by rating factor; attritional vs. large-loss decomposition; actual-to-expected
  by segment on train, validation and test; large losses by fuel type.
- Segment X and its out-of-sample cost relativity, saved to `reports/figures/segment_x.json`.

Notebook: [`08_portfolio_kpis.ipynb`](../notebooks/08_portfolio_kpis.ipynb) ·
Card: [`portfolio_kpis.md`](../reports/model_cards/portfolio_kpis.md)

## Phase 9: Retention simulator

**Question:** if we raise the price of a segment, how many existing customers do we lose?

freMTPL2 has no renewal outcome and no price history, and no public policy-level dataset of motor
renewal elasticity is available. **Decision:** a documented simulator rather than a fitted model:
tenure proxy from the French bonus-malus formula, baseline renewal rate, price elasticity and loyalty
dampening, each sourced or stated as an assumption. The usual way to replace it is a test-and-learn
program (randomized rate variation with a control cell); `renewal_probability()` can then be swapped
without changing downstream code.

Notebook: [`09_retention_model.ipynb`](../notebooks/09_retention_model.ipynb) ·
Card: [`retention_model.md`](../reports/model_cards/retention_model.md)

## Phase 10: Conversion simulator

**Question:** if we quote above the market, how many prospects still convert?

Same approach as Phase 9, with a lower baseline, a larger elasticity and a risk-based market-access
term instead of tenure. A public cross-sell dataset was considered as an anchor and not used (only
unofficial copies found).

Notebook: [`10_prospect_conversion_model.ipynb`](../notebooks/10_prospect_conversion_model.ipynb) ·
Card: [`conversion_model.md`](../reports/model_cards/conversion_model.md)

## Phase 11: Commercial premium and scenario engine

**Question:** what price change on Segment X, and with what effect on retention, conversion and
contribution?

- Commercial premium = technical premium + expense loading.
- `scoring/scenario_engine.py`: retained and new business, contribution, on one cost basis with the
  loading; scenario grid and indicated correction.

Notebook: [`11_commercial_premium_and_scenario_engine.ipynb`](../notebooks/11_commercial_premium_and_scenario_engine.ipynb) ·
Card: [`scenario_engine.md`](../reports/model_cards/scenario_engine.md)

## Phase 12: Streamlit app

**Question:** can a non-technical user run this analysis?

- Pricing simulator, retention and conversion explorer, portfolio scenario with a 1 to 3-year outlook,
  all built on the Phase 7-11 code.
- `models/app/`: coefficient-only copies of the GLMs (`src/models/lightweight_glm.py`,
  `scripts/distill_app_models.py`), committed with `policy_level_clean.parquet` so the hosted app runs
  from a clone.

Code: [`app/`](../app/) · Card: [`streamlit_app.md`](../reports/model_cards/streamlit_app.md)

## Phase 13: Documentation

- [`docs/methodology.md`](methodology.md): methodology, assumptions, limitations, validation summary.
- Model cards in `reports/model_cards/`; README.

---

## Design decisions

- **Exposure as the GLM offset, never as a feature.**
- **Frequency and severity modeled separately** rather than a single Tweedie model on pure premium:
  the two are driven by different factors here, and separate models give clearer diagnostics. A
  Tweedie comparison would be a natural extension.
- **GLM and GAM run side by side:** the GAM informs the GLM's specification; the GLM remains the rate
  table.
- **Retention and conversion are explicit, documented simulators:** freMTPL2 cannot support an
  elasticity estimate.
- **Segment decisions rest on out-of-sample evidence:** a segment gap counts only if it holds on the
  validation and test sets.
