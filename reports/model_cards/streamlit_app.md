# Model Card: Streamlit Dashboard (Phase 12)

**Modules:** [`app/streamlit_app.py`](../../app/streamlit_app.py), [`app/pages/`](../../app/pages/),
[`app/utils.py`](../../app/utils.py), [`src/models/lightweight_glm.py`](../../src/models/lightweight_glm.py),
[`scripts/distill_app_models.py`](../../scripts/distill_app_models.py)
**Extends:** [`scoring/scenario_engine.py`](../../scoring/scenario_engine.py) (`project_multi_year_scenario`)

## Purpose

An interface to the models of Phases 7-11 for a user who does not read code:

1. **Pricing Simulator:** technical and commercial premium for a client profile.
2. **Retention & Conversion:** how a price change moves renewal and conversion probabilities.
3. **Portfolio Scenario:** pick a segment (whole book, `BonusMalus` band, fuel type, region or brand)
   and a price change, and project Loss Ratio, COR, ATP and contribution over 1 to 3 years.

The app contains no modeling logic of its own: every page calls the tested `src/` and `scoring/`
modules.

## Architecture

- **`app/utils.py`:** cached loading of the pricing model, of the categories the fitted encoder knows,
  and of the scored portfolio; the commercial loading (37%, expenses only, Phase 11).
- **`app/streamlit_app.py`:** home page.
- **`app/pages/1_Pricing_Simulator.py`**, **`2_Retention_Conversion.py`**, **`3_Portfolio_Scenario.py`**.

## Cost basis of the scenario page

Expected claims are `technical_premium x cost_factor`, where the cost factor is the selected segment's
cost relative to the book: attritional claims from the segment's own experience, large losses at the
flat loading (`experience_cost_factor`). The whole book is at 1.0, the basis of the commercial loading,
so the whole book at a 0% price change shows a contribution of zero. The default price change is the
Phase 8 indication for Segment X (+17%).

## Inputs: slider plus exact value

Each numeric input pairs a slider with a number box kept in sync (`utils.slider_with_input`), so an exact
value such as `BonusMalus = 137` can be typed. Streamlit ignores a widget's `value=` once its key is in
`st.session_state`, so each widget's `on_change` writes into the other widget's key.

## Model artifacts for the app

The fitted `statsmodels` results keep their training design matrix and are too large for the
repository. `LightweightGLM` keeps only the coefficients behind the same `.predict()` interface;
`scripts/distill_app_models.py` writes them to `models/app/`. Scoring is identical (exact-match unit
test `test_load_lite_matches_load_exactly`). `models/app/` and
`data/processed/policy_level_clean.parquet` are committed so that a hosted deployment, which only
clones the repository, can run without re-training.

## Multi-year projection

- **Year 1:** the row-level Phase 11 engine.
- **Year 2 onward:** with no further price change, the retention and conversion formulas return their
  baseline rate for every policy, so later years follow from Year 1 aggregates. Tested in
  [`tests/test_multi_year_scenario.py`](../../tests/test_multi_year_scenario.py).
- A book renewing and converting at flat baseline rates grows without limit: read the projection as a
  KPI outlook under stable conditions, not a market-share forecast.

## Known limitations

1. Retention and conversion are simulated (Phases 9-10); the Retention & Conversion page says so.
2. The multi-year projection is a single-year dataset extended by assumption, not observed experience.
3. One portfolio-wide loading; no segment-specific expenses.
4. Read-only tool over a cached snapshot; no authentication, no link to a policy system.
5. On the free hosting tier, the app sleeps after a period without visits and needs a few seconds to
   wake up.

## Reproducing the distillation step

```bash
python scripts/distill_app_models.py
```

Requires the Phase 7 artifacts (`models/freq_glm_refined.pkl`, `models/sev_glm.pkl`,
`models/feature_encoder.joblib`); re-run after any re-training.
