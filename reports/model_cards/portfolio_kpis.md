# Model Card: Portfolio KPIs (Loss Ratio, COR, ATP) and Segment X

**Modules:** [`src/pricing/portfolio_kpis.py`](../../src/pricing/portfolio_kpis.py),
[`src/pricing/segment_x.py`](../../src/pricing/segment_x.py)
**Built in:** [`notebooks/08_portfolio_kpis.ipynb`](../../notebooks/08_portfolio_kpis.ipynb)
**Outputs:** [`kpis_by_region.csv`](../figures/kpis_by_region.csv),
[`attritional_ae_by_segment.csv`](../figures/attritional_ae_by_segment.csv),
[`large_losses_by_vehgas.csv`](../figures/large_losses_by_vehgas.csv),
[`segment_x.json`](../figures/segment_x.json)

## Purpose

Portfolio KPIs (Loss Ratio, Combined Operating Ratio, Average Technical Premium) at book level and by
rating factor, and the search for the segment to reprice. Scored on the full book (668,626 policies),
with every segment result also split by train / validation / test.

## Two assumptions (freMTPL2 has no premium and no expense data)

1. **Earned premium = Phase 7 technical premium x exposure.** Loss Ratio therefore measures how well the
   expected cost matches actual claims, not realized profit.
2. **Expense ratio = 27%**, a flat assumption in the range commonly cited for French personal motor.

## Book level

Loss Ratio 112.7%, COR 139.7%, ATP EUR 158 per policy-year (all data). The technical premium reconciles
with paid claims on the training set (99.7%) and the test set (98.3%); the excess on the whole book
comes from the large-loss layer in the validation set, which holds the largest claims of the dataset.

## Segment search

**First pass (raw Loss Ratio):** `Region` R24 looks like a major problem (LR 146.7%, 28.5% of exposure).
**Check:** the large-loss loading is flat, so raw Loss Ratio by segment mostly reflects where a few
large claims fell. On the attritional layer (claims capped at EUR 15,000), R24 is at 100.8% against
100.5% for the rest of the book, with no pricing gap; its raw figure comes from large losses (49% of
its claim cost), including the largest claim in the dataset (EUR 4.08M).

**Attritional A/E relative to the book, by factor, in and out of sample:** most levels stay near 1.00
or change direction between validation and test. One factor stands out:

| `BonusMalus` band | Exposure | A/E all data | A/E validation | A/E test |
|---|---|---|---|---|
| 50 | 62.7% | 0.89 | 0.86 | 0.88 |
| 51-60 | 13.4% | 1.21 | 1.15 | 1.22 |
| 61-80 | 14.9% | 1.22 | 1.21 | 1.25 |
| 81-100 | 8.0% | 0.93 | 1.05 | 0.96 |
| >100 | 1.0% | 0.98 | 1.05 | 0.87 |

**Fuel type:** Diesel and Regular show the same rate of large claims (0.7-1.0 per 1,000 policy-years in
each split); the cost of those claims changes direction from one split to the next. Fuel type is not a
repricing candidate.

**Mechanism:** the frequency GLM treats `BonusMalus` as a straight line on the log scale, while the
Phase 5 GAM shows a concave curve (steep from 50 to about 70, flat after). Replacing the linear term by
bands raises the refined GLM's validation pseudo-R² from 4.72% to 5.28% (GAM: 5.49%).

## Segment X for Phases 9-11

**`BonusMalus` 51-80**, 28.3% of exposure. Out-of-sample cost relativity (validation + test,
attritional claims from experience, large losses at the flat loading): **1.17**, so an indicated price
change of **+17%**, saved to `reports/figures/segment_x.json` and loaded by the later phases.

## Known limitations

1. Premium and expenses are assumptions; the KPIs compare segments, they do not measure historical
   profitability.
2. Book-level figures include rows the models were fitted on; the segment decision rests on the
   validation and test columns.
3. The large-loss threshold (EUR 15,000, 99th percentile of training severity) is a convention, not a
   fitted tail estimate.
4. `Region` R11 (8.4% of exposure, A/E 1.09 / 1.16 / 1.09) is a smaller, secondary signal, not
   investigated further.

## Reproducing

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/08_portfolio_kpis.ipynb
```

Requires `data/processed/policy_level_clean.parquet` (Phase 1), the train/val/test files (Phase 3) and
the Phase 7 model artifacts.

## Worked example: the cost relativity of Segment X = 1.1842 / 1.0155 = 1.166

**The group.** Not one policy but a group: every policy with BonusMalus 51 to 80, on the data the model
did not see (validation + test). The question: does this group cost more than its premium, compared with
the rest of the portfolio?

| Step | Segment X | Book | Where it comes from |
|---|---|---|---|
| Claims capped at EUR 15,000 | EUR 5,988,212 | EUR 16,643,671 | column `sev_claim_amount`, each claim up to EUR 15,000 |
| + EUR 33.54 x years insured | EUR 1,343,494 | EUR 4,744,650 | flat large-loss loading (notebook 06), column `Exposure` |
| Earned premium | EUR 6,191,086 | EUR 21,062,209 | technical premium (notebook 07) x `Exposure` |
| Cost / premium | (5,988,212 + 1,343,494) / 6,191,086 = **1.1842** | (16,643,671 + 4,744,650) / 21,062,209 = **1.0155** | the three lines above |

**Result.** Cost relativity = 1.1842 / 1.0155 = **1.166**, indicated price change **+17%**.

**Reading.** The segment costs 1.1842 times its premium, the book 1.0155 times. Relative to the book, the
segment costs 1.166 times more: its premium is about 17% too low, hence +17%. Large claims are replaced
by the flat EUR 33.54 per year (mutualisation), so a EUR 4M claim that fell in the group by chance does
not distort the measure. After +17%, the segment costs 1.012 times its premium, in line with the book.
Notebook 08 shows three policies of the segment (IDpol 24956, 24976, 24980) highlighted in the dataset,
with their corrected premiums.
