# Model Card: Prospect / New-Business Conversion Model (Phase 10)

**Module:** [`src/prospect/conversion_model.py`](../../src/prospect/conversion_model.py)
**Built in:** [`notebooks/10_prospect_conversion_model.ipynb`](../../notebooks/10_prospect_conversion_model.ipynb)
**Status:** documented scenario simulator: **not** a model trained on observed behavior (same standard as Phase 9)

## Purpose

Answers *"if we quote a segment at a given price relative to the market, how many prospects convert?"*
the new-business counterpart to Phase 9's retention model, for Segment X (`BonusMalus` 51-80, Phase 8)
and any future scenario Phase 11's engine needs to evaluate.

## Why this is a simulator, not a fitted model

Same reasoning as Phase 9, not repeated in full here: freMTPL2 has no quote-to-bind outcomes and no
competitor pricing data, so there is nothing to fit a conversion elasticity to. This module specifies a
formula with every parameter sourced or cited, exactly matching Phase 9's disclosure standard.

## The empirical anchor that was considered, and not used

The roadmap flagged Kaggle's *"Health Insurance Cross Sell Prediction"* dataset as a possible real-data
anchor for this curve's shape (different market/product, but structurally a premium-vs-purchase-decision
dataset). It was **not pulled into this project**: a web search for a direct download turned up only
unofficial third-party GitHub mirrors of Kaggle-restricted data, and this project would rather ship a
clearly-labeled simulation than a real dataset of uncertain licensing provenance. Recorded here as a
deliberate decision, not a gap that was overlooked.

## The formula, and three deliberate differences from Phase 9

```
logit(P(convert)) = logit(baseline) + elasticity * price_vs_market_pct * (1 - risk_access_dampening(BonusMalus))
```

| | Retention (Phase 9) | Conversion (this phase) | Why different |
|---|---|---|---|
| Baseline | 88% | **15%** | A prospect getting one quote is, by construction, comparison-shopping: commonly-cited quote-to-bind rates for price-comparison motor channels are typically 10-20% |
| Elasticity | -4.5 | **-6.0** (larger magnitude) | New business is well documented in P&C pricing-optimization literature as more price-elastic than renewal: no switching cost, actively comparing offers |
| Segment driver | Tenure (loyalty dampens sensitivity) | **`BonusMalus`** (risk-based market-access dampening) | A new prospect has no tenure by definition; but a high-risk prospect has structurally fewer insurers willing to compete for them (the same logic behind residual/assigned-risk markets for hard-to-place risks), so they're *less* price-sensitive, not more |

`price_vs_market_pct = 0` means "quoting at whatever this project's technical premium already implies"
an assumption, since this dataset has no real competitor pricing either.

**A recalibration worth recording:** the first elasticity tried (-9.0, roughly double the retention
value in logit units) produced an implausibly severe result: a 71% relative conversion drop from a
mere +15% price move. Recalibrated to -6.0 (still larger in magnitude than retention's -4.5, but not
double it) after checking the actual resulting probabilities, not just the raw coefficient: a reminder
that a logit-scale parameter's real-world effect has to be checked numerically, especially at a low
baseline probability where the logistic curve compounds sensitivity.

## Validation performed (of the formula's behavior: there is no ground truth to validate predictive accuracy against)

- At market parity (0% price position), every risk level converges to exactly the 15% baseline.
- +10% above market: low-risk prospect falls to 8.8% (41% relative drop); high-risk prospect only
  falls to 11.6% (23% relative drop): the risk-access dampening effect, working as designed.
- **New business vs. renewal, quantified directly:** at matched price moves (+5% to +20%), conversion's
  relative probability drop is consistently **7x to 11x larger** than retention's: both the direction
  and the general scale are consistent with the well-documented "new business is far more price-elastic
  than renewal" pattern this notebook cites, even though the specific multiple is a consequence of this
  project's own calibration choices, not an estimated value.

## Applied to Segment X

`BonusMalus` 51-80 (28.3% of portfolio exposure), same **+17%** correction as Phase 9:

| | Market parity | +17% correction |
|---|---|---|
| Mean P(convert) | 15.0% | 6.0% |
| Relative drop | n/a | 60.1% |

For the same move, Phase 9 projects about 4.8% of existing policies lapsing: the conversion channel is
much more exposed to a price correction than the renewal book. This assumes competitors price the
segment like today's tariff; if the market already charges these risks more, the real drop would be
smaller.

## Known limitations

1. **This is a simulator, calibrated to external benchmarks and internal consistency checks, not a
   model fit to observed conversion behavior.**
2. `price_vs_market_pct = 0` assumes today's technical premium sits at market parity: this project has
   no real competitor pricing to verify that assumption against.
3. `CONVERSION_PRICE_ELASTICITY` and `RISK_ACCESS_MAX_DAMPENING` are illustrative, recalibrated once
   already during this phase's own development after an initial choice proved implausible (see above).
   a real deployment should replace both with parameters fit on real quote and conversion experience.
4. No empirical anchor dataset was used (see above): the formula's shape is entirely literature- and
   consistency-check-driven, not benchmarked against any real conversion data, even from a different
   market.

## The realistic path to a real model (same as Phase 9)

A test-and-learn pricing program: randomized quote pricing across a slice of new business, with a
control cell: is the usual way to replace this simulator's assumptions with real
elasticity estimates. `src/prospect/conversion_model.py`'s `conversion_probability()` interface is
designed so a future, experience-based conversion model could replace it without changing Phase 11's
downstream scenario-engine code.

## Reproducing

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/10_prospect_conversion_model.ipynb
```

Requires `data/processed/policy_level_clean.parquet` (Phase 1) only.
