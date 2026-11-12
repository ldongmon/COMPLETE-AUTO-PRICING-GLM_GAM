# Model Card: Commercial Premium and Repricing Scenario Engine (Phase 11)

**Modules:** [`src/pricing/commercial_premium.py`](../../src/pricing/commercial_premium.py),
[`scoring/scenario_engine.py`](../../scoring/scenario_engine.py)
**Built in:** [`notebooks/11_commercial_premium_and_scenario_engine.ipynb`](../../notebooks/11_commercial_premium_and_scenario_engine.ipynb)
**Output:** [`reports/figures/segment_x_repricing_scenario_grid.csv`](../figures/segment_x_repricing_scenario_grid.csv)

## Purpose

Combines the technical premium (Phase 7), the segment measured in Phase 8, the retention simulator
(Phase 9) and the conversion simulator (Phase 10) to answer: *"Segment X has a Loss Ratio above the
rest of the book. Should we reprice it, by how much, and what would that do to retention, conversion
and profitability?"*

## Commercial premium

`commercial_premium = technical_premium x (1 + loading)`. The technical premium is the expected claims
cost (it reconciles with paid claims on the training and test sets), so the loading only covers the
27% expense assumption: `loading = 1 / (1 - 0.27) - 1 = 37.0%`. The realized Loss Ratio of the whole
book (112.7%) is not used, because it is driven by a few very large claims in the validation set
(Phase 8).

## The scenario engine

For a segment and a price change, the engine projects **retained** policies (Phase 9's
`renewal_probability`) and **new** policies (Phase 10's `conversion_probability`) and sums
`written_premium - expected_claims - expenses` on each side.

Expected claims are `technical_premium x cost_factor`. The cost factor is 1 for the book (the basis of
the loading) and equals the segment's cost relativity for a segment: attritional claims from the
segment's own experience, large losses at the flat loading (`experience_cost_factor` in
`src/pricing/portfolio_kpis.py`). With this single cost basis, the whole book at a 0% price change has
a contribution of zero (checked in the notebook and by a unit test).

Stated simplifications: new-business quote volume equals the segment's current policy count
(`quote_volume_multiplier = 1.0`, no quote data exists); retained and new policies carry the segment's
average risk (no adverse selection modeled).

## Segment X at current pricing

`BonusMalus` 51-80 (211,105 policies, 28.3% of exposure), cost relativity 1.17 measured out of sample
in Phase 8. At the 37% loading, its average margin after expenses is about EUR -26 per policy, an
expected annual contribution of about EUR -5.7M.

## Behavior over the price range

From -10% to +30%, expected contribution improves steadily with a larger correction; there is no peak
inside that range. The retention and conversion curves are logistic in the price change while the
premium grows linearly, so a peak exists further out: on a -10% to +300% grid it sits at +95%. That
value is outside the range the Phase 9-10 assumptions cover and is not a recommendation.

## Indicated correction

**+17%**, equal to the segment's measured cost relativity (loaded from `reports/figures/segment_x.json`):

| | Current pricing | +17% correction |
|---|---|---|
| Renewal rate | 88.0% | 83.2% |
| Conversion rate | 15.0% | 6.0% |
| Expected renewed policies | 185,772 | 175,697 |
| Expected new policies | 31,666 | 12,630 |
| Segment written premium | EUR 47.0M | EUR 47.5M |
| **Segment expected annual contribution** | **EUR -5.7M** | **EUR 0.1M** |

The correction brings the segment to about breakeven, under the simulators' assumptions.

## Known limitations

1. Retention and conversion come from documented simulators (Phases 9-10), not from observed price
   response; results are directional.
2. Quote volume and average risk mix are stated simplifications.
3. Outside roughly -20% to +30%, the engine still computes numbers but they are not meaningful.
4. One portfolio-wide loading; no segment-specific expenses, no multi-year customer value, no
   competitor reaction.

## Next step

Apply the correction as a `BonusMalus` relativity change for the 51-80 band, monitored with a
test-and-learn cell, and recalibrate `renewal_model.py` and `conversion_model.py` on the observed
retention and conversion.

## Reproducing

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/11_commercial_premium_and_scenario_engine.ipynb
```

Requires the Phase 7 artifacts (`models/freq_glm_refined.pkl`, `models/sev_glm.pkl`,
`models/feature_encoder.joblib`), `data/processed/policy_level_clean.parquet` (Phase 1) and
`reports/figures/segment_x.json` (Phase 8).
