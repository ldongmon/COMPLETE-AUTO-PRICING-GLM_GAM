# Auto Insurance Pricing & Retention: GLM/GAM Pricing Engine with Scenario Analysis

End-to-end motor pricing project on the French Motor TPL dataset: data quality, frequency and severity
models (GLM and GAM), technical premium, portfolio diagnosis, retention and conversion simulators,
commercial premium, a repricing scenario engine and a Streamlit app. All phases (0 to 13) are complete.
Methodology, assumptions and limitations: [`docs/methodology.md`](docs/methodology.md).

## The business question

> **A segment of the book has a Loss Ratio above the rest. Should we reprice it, by how much, and
> what would that do to retention, new-business conversion and profitability?**

**Answer:** reprice **`BonusMalus` 51-80** (28.3% of exposure) by **+17%**. On the validation and test
sets, this segment costs 15% to 25% more than its technical premium predicts on the attritional layer,
because the frequency GLM treats `BonusMalus` as a straight line while its effect is concave (Phase 8).
Under the documented retention and conversion simulators, the correction loses about 4.8% of the
segment's policies and cuts new-business conversion from 15.0% to 6.0%, and brings the segment's
expected annual contribution from about EUR -5.7M to about breakeven (Phase 11).

## Data

[French Motor Third-Party Liability dataset](https://www.openml.org/): `freMTPL2freq` (OpenML id
`41214`, one row per policy) and `freMTPL2sev` (OpenML id `41215`, one row per claim payment), one
accounting year, 678,013 policies and 26,639 claim payments in the raw files. Downloaded automatically by
[`src/data/load_data.py`](src/data/load_data.py).

The first 9,387 policies of the file (IDpol up to 24,951) are excluded: they all carry a reported claim
(implied frequency above 200%) and only 3.1% of their claims are paid in `freMTPL2sev`. They are a
claims extract appended to the portfolio, not insured policies like the others. **668,626 policies**
are modeled. Details: [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Phases

| Phase | Question | Notebook / code |
|---|---|---|
| 0. Setup | Reproducible environment, repository structure | `requirements.txt` |
| 1. Data quality | What data do we have, and can we trust it? | [`01`](notebooks/01_data_acquisition_and_quality_checks.ipynb) |
| 2. EDA | Which rating factors move frequency and severity? | [`02`](notebooks/02_exploratory_data_analysis.ipynb) |
| 3. Preparation and split | Leak-free split and encoding | [`03`](notebooks/03_data_preparation_and_split.ipynb) |
| 4. Frequency GLM | Expected claim count per policy | [`04`](notebooks/04_frequency_model_glm.ipynb) |
| 5. Frequency GAM | Do non-linear effects improve on the GLM? | [`05`](notebooks/05_frequency_model_gam.ipynb) |
| 6. Severity GLM and GAM | How large is a claim, given it happened? | [`06`](notebooks/06_severity_model_glm_gam.ipynb) |
| 7. Technical premium | Expected claims cost per policy | [`07`](notebooks/07_technical_premium.ipynb) |
| 8. Portfolio KPIs | Which segment is under-priced, and is it real? | [`08`](notebooks/08_portfolio_kpis.ipynb) |
| 9. Retention | How many customers do we lose if we reprice? | [`09`](notebooks/09_retention_model.ipynb) |
| 10. Conversion | How many prospects still convert? | [`10`](notebooks/10_prospect_conversion_model.ipynb) |
| 11. Commercial premium and scenarios | What correction, with what effect? | [`11`](notebooks/11_commercial_premium_and_scenario_engine.ipynb) |
| 12. Streamlit app | Can a non-technical user run this? | [`app/`](app/) |
| 13. Documentation | Methodology, model cards | [`docs/`](docs/), [`reports/model_cards/`](reports/model_cards/) |

## Main findings

- **Data:** exposure-weighted claim frequency is **7.40%** per policy-year; the plain average of
  `ClaimNb` (3.9%) is claims per policy, not a frequency. `BonusMalus` is the strongest frequency factor
  (5.2% at the bonus floor to 61% above 160); `Area` duplicates `Density`.
- **Split:** 60/20/20, stratified on claim count; frequency 7.40-7.41% and median paid claim EUR 1,172
  on all three sets.
- **Frequency GLM (Phase 4):** ranks risk correctly on the validation set (deciles from 3.8% to 17.9%).
  `DrivAge`'s multivariate sign is positive, opposite to its univariate trend: confounding with
  `BonusMalus` (r = -0.48).
- **GAM (Phase 5):** better out of sample (pseudo-R² 5.49% vs. 4.53%, Gini 0.311 vs. 0.285). Its curves
  define the `DrivAge` and `VehAge` bins of the refined GLM (Phase 7: 4.72%, 0.295).
- **Severity (Phase 6):** neither the Gamma GLM nor the GAM improves meaningfully on a naive average out
  of sample; large losses above EUR 15,000 are charged as a flat EUR 33.54 per policy-year. Risk
  differentiation comes almost entirely from frequency.
- **Technical premium (Phase 7):** reconciles with paid claims on the training set (99.7%) and the test
  set (98.3%).
- **Segment diagnosis (Phase 8):** raw Loss Ratio flags `Region` R24, but on the attritional layer it is
  in line with the book; its figure comes from large losses. Fuel type shows no stable gap (same rate of
  large claims for Diesel and Regular). The one stable gap, in and out of sample, is `BonusMalus` 51-80;
  modeling `BonusMalus` in bands raises the validation pseudo-R² from 4.72% to 5.28%.
- **Retention and conversion (Phases 9-10):** documented simulators, since the dataset has no renewal or
  quote history. Tenure proxy from the French bonus-malus formula; conversion 7 to 11 times more
  price-sensitive than retention.
- **Commercial premium and scenarios (Phase 11):** loading of 37% for the 27% expense assumption; one
  cost basis for the loading and the scenario engine (the whole book at 0% change has a contribution of
  zero); +17% on Segment X brings it to about breakeven.

## Conclusion

`BonusMalus` 51-80 should be repriced by about +17%, which amounts to correcting the `BonusMalus`
relativity for that band; the next tariff revision should model `BonusMalus` in bands. The retention and
conversion effects come from simulators calibrated on literature values, not from observed behavior: the
correction would be applied with a test-and-learn cell, and the simulators recalibrated on the observed
experience. The Streamlit app lets a user run the same analysis on another segment or another price
change.

## Architecture

```
notebooks/   analysis, one notebook per stage (11, all executed)
src/         code shared by the notebooks and the app
  data/      OpenML loader, claims-only block detection
  eda/       exposure-weighted profiling
  features/  encoding and train/validation/test split
  models/    GLM/GAM training and diagnostics
  pricing/   portfolio KPIs, Segment X, commercial premium
  retention/ renewal simulator
  prospect/  conversion simulator
scoring/     technical premium scoring and scenario engine
app/         Streamlit app (3 pages)
tests/       pytest unit tests
docs/        methodology, data dictionary, project plan
reports/     model cards and exported tables
```

## How to reproduce

```bash
git clone https://github.com/ldongmon/COMPLETE-AUTO-PRICING-GLM_GAM.git
cd COMPLETE-AUTO-PRICING-GLM_GAM
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The data is downloaded automatically by the first notebook run (or `python -m src.data.load_data`).
To run the notebooks with the project's environment:

```bash
python -m ipykernel install --user --name=auto-pricing-venv --display-name="Python (auto-pricing venv)"
jupyter lab
```

Tests (151):

```bash
pytest tests/ -v
```

## Run the app

```bash
streamlit run app/streamlit_app.py
```

Three pages (Pricing Simulator, Retention & Conversion, Portfolio Scenario) at `http://localhost:8501`.
The coefficient-only pricing models (`models/app/`) and the policy table
(`data/processed/policy_level_clean.parquet`) are committed, so the app runs right after installation.

## Links

- **Presentation page:** [ldongmon.github.io/COMPLETE-AUTO-PRICING-GLM_GAM](https://ldongmon.github.io/COMPLETE-AUTO-PRICING-GLM_GAM/)
- **Full PDF report:** sent on request, via the button on the presentation page.
- **Streamlit app:** [launch the app](https://ldongmon-complete-auto-pricing-glm-gam-appstreamlit-app-jajm0q.streamlit.app/)
  (on the free hosting tier the app sleeps when unused; waking it takes a few seconds)

## Author

Dongmo Nanda, Actuary / Pricing Data Scientist
