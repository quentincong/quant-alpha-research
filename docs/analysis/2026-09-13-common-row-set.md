# Deep dive: the common-row-set trap (0.32% of rows, 5.9 bps)

*Written 2026-09-13. The body is frozen; corrections go to [Errata](#errata) at the end.*
*Research log entry: [R42](../research_log.md#r42).*
*Numbers: every value is followed by its row ID in [`results_provenance.md`](../results_provenance.md).
Terms follow [`glossary.md`](../glossary.md).*

**Summary.** To compare eleven models on the same sample, I intersected their row sets. The
intersection dropped 5,744 of the headline model's 1,814,693 rows, 0.32% (E6). Measured on the
intersection, the headline long-short net@10 fell from +23.19 to +17.30 bps, a loss of 5.9 bps (E7).
The dropped rows were not random: they are the limit-down exit failures that one label gate removes,
and their mean absolute return is 5.1x the overall mean (E6). An innocent alignment step had
reintroduced a label defect I had already fixed. The rule I took from it: before trusting any number
computed on an intersection, look at what the intersection removed.

---

## Problem

[R42](../research_log.md#r42) built a pool of eleven CN prediction sets: three GBDT seeds on the
default `vwap × entry` label, four GBDT runs on other label variants (`close × entry`, `close × full`,
`hlc3 × full`, `vwap × full`), three MLP seeds, and a ridge model. The goal was a map of diversity:
the daily rank correlation ρ between every pair of members, and next to it each member's strength
(RankIC, net@10) under one common ruler (the headline label and the frozen `ema5/exit10` variant).

A correlation matrix is only self-consistent if every pair is computed on the same rows. The label
variants do not keep exactly the same rows, so `model/pool_corr.py` takes the intersection: the
`(date, instrument)` keys that every member has. It has 1,808,949 rows (E1). My first version then
computed the strength columns on that same intersection, because "same sample for everything" sounded
like the careful choice.

The headline member came out at net@10 +17.30. Its established value, reproduced by more than one
script, is +23.19 (P1, T6).

## Method

A gap of that size between two runs of the same predictions is a measurement difference, not a model
difference, so I looked at the rows instead of the models.

1. **Which rows?** The keys of `model/preds_walkforward_cn_vwapentry.parquet` that are missing from at
   least one of the eleven member files (E6).
2. **What do they look like?** Their `label_raw` distribution against the whole file: mean absolute
   value and median (E6).
3. **Why these rows?** Match them against the label definitions (glossary, *label variant*).
4. **How much do they matter?** Compute the headline member's portfolio both ways, on its own rows and on
   the common rows, with the same label, variant, and code (E7).

## Result

**The dropped rows** (E6):

- 5,744 of 1,814,693 rows, 0.32%. The arithmetic closes: 1,814,693 own rows (D6) minus 5,744 dropped
  rows is the 1,808,949 common rows (E1).
- Mean `|label_raw|` 0.0816 against 0.0160 overall, 5.1x.
- Median `label_raw` −0.085: typically a large loss over the two-day holding period.

**Why these rows.** The pool contains members built on `full`-gate labels. The `full` gate keeps a row
only if the stock can be bought on T+1 **and** sold on T+2, so those label files lack every row where
the stock was at limit-down on T+2. The default `entry` gate keeps those rows, because the position was
already open and its loss is real (the exit-gate defect in the
[label deep dive](2026-09-13-label-vwap-entry.md)). Intersecting an `entry` member with any `full`
member therefore removes exactly the exit-failure rows, the left tail that the `entry` gate had put
back. In the result file, every `entry`-gate member loses net@10 on the common rows, while the three
`full`-gate members, whose rows define the intersection, are unchanged (`strength.<member>.own` against
`.common` in `model/pool_corr_results.json`).

**How much they matter** (headline GBDT seed 0, long-short `ema5/exit10`, CN 2020–2025) (E7):

| row set | gross | net@10 | IR@10 |
|---|---|---|---|
| own rows | +27.50 | +23.19 | +3.50 (P1) |
| common rows | +21.74 | **+17.30** (−5.9 bps) | +2.64 |

Removing large losses *lowered* a long-short return. A limit-down loss is a loss only in the long leg; in
the short leg it is a gain. The direction of the change says that, on net, the model held those stocks
on the short side. That fits the shape of the portfolio: the short side of the universe mean carries 79%
of the `vwap × full` spread (L16), and a filter that removes limit-down losses removes short-leg profits.

**The fix.** `model/pool_corr.py` now computes the two kinds of number on different row sets and prints
both:

- **ρ matrix: common rows.** Every pair must use the same sample.
- **strength: each member's own rows**, joined only to the headline label so that the bps scale is
  the same for all members. The common-row value is printed next to it as a diagnostic column.

With strength on own rows the pool reproduces the established values, e.g. GBDT seeds +23.19 / +20.82 /
+23.78 (P2) and ridge +17.91 (M8).

## What it overturned

- **The headline measured on the common row set, net@10 +17.30** (X17). Corrected to own rows, +23.19
  (E7, P1).
- **The assumption that an intersection is a neutral, conservative choice.** Here the intersection was
  selective in the worst direction: it removed the tail rows that carry the money.
- **A fix I thought was complete.** Making `vwap × entry` the default label removed the exit-gate defect
  from the headline, but any later step that combines an `entry` member with a `full` member can silently
  bring the defect back.

## Limits

- **The ρ matrix still excludes those rows.** Correlations on the common row set describe the models on
  all rows except the exit failures. Two models could agree or disagree more on precisely those names.
- **Own-row strength is not gate-neutral either.** A `full`-gate member's own rows still lack the exit
  failures, so its long-short strength carries the same downward bias that E7 shows for the headline.
  For example `vwap × full` long-short net@10 is +18.98 (L14) against +23.19 for `vwap × entry` (P1);
  that gap mixes a different model with a different row set, so it is not a clean model comparison.
- **One member, one seed quantified.** The sheet records the −5.9 bps for GBDT seed 0 only (E7). The
  other members show the same direction in the result file, but I quote no sizes for them.
- **Intersections elsewhere.** Other tools also align inputs: the paired block bootstrap intersects
  days, not rows, and prints any days it drops (`model/block_bootstrap.py`). Earlier ensemble and
  reversal comparisons used aligned prediction copies that no shipped script produces; I no longer
  quote their numbers, and the reversal audit was re-run on the walk-forward predictions
  (see the [reversal deep dive](2026-09-13-reversal-exposure.md)). Any new alignment step needs the same
  dropped-row check.
- **The label caps the tail at two days.** An exit-failure row is charged the two-day label return, but
  a stock that stays at limit-down for several days can keep falling. A long position stuck in it loses more
  than the label records, so even own-row numbers are optimistic for the long leg.

## Errata

- **2026-09-13 · Holding period.** "The two-day holding period" (Result) and "the label caps the tail at two
  days … the two-day label return" (Limits) should say one day: `label_raw` runs from T+1 to T+2
  (glossary, *label timing*). The limit is unchanged in substance: an exit-failure row is charged only the
  T+1 → T+2 return, while a stock stuck at limit-down can keep falling afterwards.
