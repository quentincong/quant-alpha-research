# Deep dive: block bootstrap, a retracted model comparison, and two kinds of error bar

*Written 2026-09-13. The body is frozen; corrections go to [Errata](#errata) at the end.*
*Research log entry: [R37](../research_log.md#r37) (context: [R34](../research_log.md#r34),
[R35](../research_log.md#r35), [R41](../research_log.md#r41)).*
*Numbers: every value is followed by its row ID in [`results_provenance.md`](../results_provenance.md).
Terms follow [`glossary.md`](../glossary.md).*

**Summary.** Until this point my only error bar was the spread across training seeds. I added a
circular block bootstrap over test days, which answers a different question: would the result hold on
a different stretch of history? The CN headline intervals exclude zero; the US portfolio intervals do
not (P9, N9). The new tool also forced a retraction. I had written that GBDT beats the MLP on
long-short net@10 by "more than ten seed sds" (X14). With GBDT's own seed spread included the gap is
2.15σ (T11), and the paired bootstrap interval for the difference contains zero (T10). The lesson is
that seed sd and a bootstrap interval measure different uncertainties, and a model comparison has to
report both.

---

## Problem

**Naive t-statistics overstate precision.** `model/evaluate.py` reports a naive t for the daily RankIC
series, and the IR@c values are naive annualized ratios (glossary). Both treat the test days as
independent. They are not:

- the label spans two days but the portfolio rebalances daily, so consecutive holdings overlap;
- EMA smoothing carries yesterday's score into today's;
- market regimes persist.

With positive autocorrelation the true standard error is larger than the naive one.

**Comparisons had only one kind of error bar.** In [R35](../research_log.md#r35) I compared the MLP
and GBDT at portfolio level. The MLP had three seeds, long-short net@10 +19.45 / +20.96 / +20.78 (M4).
GBDT had predictions on disk for seed 0 only, +23.19 (P1). I divided the gap by the MLP's seed sd and
wrote that GBDT wins by more than ten seed sds (X14). There was no GBDT seed spread in that
denominator, and no allowance at all for the possibility that the six test years were a lucky sample.

## Method

`model/block_bootstrap.py` implements a circular moving-block bootstrap over test days.

- **Resampling.** For n days and block length L, draw ⌈n/L⌉ uniformly random start days, take L
  consecutive days from each (wrapping from the end back to the start), concatenate, and truncate to n.
  Resampling blocks rather than single days keeps the dependence inside each block (Künsch, 1989);
  wrapping around a circle gives every day the same chance of selection, where a non-circular scheme
  under-samples the first and last days (Politis & Romano, 1992).
- **Settings.** L = 20 trading days, B = 2,000 resamples, bootstrap seed 0, percentile 95% intervals.
  The script repeats the intervals at L = 5, 10, 40 to show whether the conclusion depends on L.
- **Statistics.** RankIC, ICIR, and net@10 and IR@10 for the two frozen variants (long-short
  `ema5/exit10`, long-only `ema5/exit30`). The daily series come from the existing functions
  (`evaluate.py`, `turnover_study.py`, `longonly_study.py`), and the script asserts that its daily RankIC
  equals `evaluate.py`'s, so the bootstrap does not introduce a second definition of any metric. The point
  estimate runs through the same code as the resamples (the identity resample).
- **An approximation I have to state.** Turnover is path dependent: day t's turnover depends on day
  t−1's holdings. At each block boundary the resampled "yesterday" is not the real one, which affects one
  day in L. The script prints the bootstrap mean minus the point estimate for every statistic, so a
  distortion of the centre would be visible.
- **Paired mode** (`--paired`). Both models are evaluated on the **same** resampled days (the same index
  matrix), and the interval is taken on the difference. Holdings, turnover, and costs of both models move
  with the same days, so costs do not leave a systematic offset in the difference. This is why a paired
  bootstrap can test a gap while the within-day permutation null cannot
  (see the [selection deep dive](2026-09-13-selection-procedure-test.md)).
- **Seed-level comparison.** Separately, I re-computed the seed-axis comparison with both models at three
  seeds: difference of the 3-seed means divided by √(sd²_GBDT/3 + sd²_MLP/3) (T11).

## Result

**Levels, CN headline** (GBDT seed 0, 2020–2025; 95% intervals, L = 20) (P9):

| statistic | interval |
|---|---|
| RankIC | [+0.0334, +0.0486] |
| long-short net@10 | [+16.97, +29.49] |
| long-short IR@10 | [+2.56, +4.50] |
| long-only net@10 | [+0.78, +5.50] |
| long-only IR@10 | [+0.29, +2.12] |

The RankIC, long-short net@10 and long-only net@10 intervals also exclude zero at L = 5, 10 and 40 (P9).
The long-only interval is wide: its lower end is close to zero.

**Levels, US negative control** (N9). RankIC [+0.0007, +0.0134]; long-short net@10 [−8.05, +4.08];
long-short IR@10 [−1.18, +0.50]; long-only net@10 [−1.40, +5.13]; long-only IR@10 [−0.37, +1.24]. All four
portfolio intervals contain zero; the RankIC interval does not. The honest description of the control is
therefore "a weak but non-zero ranking signal whose portfolios are indistinguishable from zero", not
"no signal".

**Paired difference, MLP − GBDT** (seed 0 each, CN 2020–2025) (T10):

| statistic | difference | 95% interval |
|---|---|---|
| RankIC | +0.0043 | [−0.0008, +0.0094] |
| long-short net@10 | −3.74 | [−7.87, +0.51] |
| long-only net@10 | +0.27 | [−1.88, +2.45] |

On this pair of seeds none of the three differences is resolved: the sample period could reverse any of
them.

**Seed axis, with GBDT at three seeds** (T11). GBDT long-short net@10 is +23.19 / +20.82 / +23.78, mean
+22.59, seed sd 1.57 (P2); the MLP is +19.45 / +20.96 / +20.78, mean +20.40, seed sd 0.83 (M4). GBDT's
second seed lands inside the MLP's range. The difference of means is +2.20 bps with standard error 1.02:
**2.15σ**. For IR@10 the difference is +0.65 with standard error 0.12, 5.2σ (T11).

**The two error bars can disagree, and why.** RankIC shows the pattern clearly. Across seeds the MLP's
lead is consistent: +0.0466 ± 0.0014 against +0.0400 ± 0.0015, and the MLP's 3-seed mean beats GBDT's in
6 of 6 folds, sign test p = 0.031 (M1, M2, M3). Yet the paired bootstrap interval for one seed pair
contains zero (T10). The reverse also happens: when I paired two GBDT seeds, the bootstrap resolved their
small long-short difference more easily than the larger MLP − GBDT difference
([R37](../research_log.md#r37)). The mechanism is the variance of the daily difference. Two seeds of the
same model produce almost the same daily return series, so their difference is a quiet series and even
a small mean difference has a tight interval. Two model families disagree more from day to day, so the
difference series is noisy and a larger mean gap is not resolved. The paired bootstrap's resolving power
depends on how similar the two models are, not on how meaningful the comparison is.

So the two error bars answer different questions:

- **seed sd**: if I retrain, do I get the same model? (training randomness);
- **block bootstrap interval**: holding the trained models fixed, would the result, or the difference,
  hold on another stretch of history? (sample-period uncertainty).

A comparison between model families has to pass both. Reporting only the bootstrap can certify seed
noise as "significant"; reporting only seed sd ignores whether the result is specific to these six years.

## What it overturned

- **"GBDT beats the MLP long-short by more than ten seed sds"** (X14). The denominator lacked GBDT's
  own seed spread, which is larger than the MLP's (P2, M4), and GBDT's one available seed sat above its
  later 3-seed mean. Corrected: 2.15σ on net@10, 5.2σ on IR@10
  (T11), and the paired bootstrap interval for net@10 contains zero (T10).
- **An implicit claim that the headline had no sample-period uncertainty.** It now carries an interval,
  [+16.97, +29.49] bps for long-short net@10 (P9).
- **"No signal" for the US control**, which becomes "weak non-zero ranking, portfolios indistinguishable
  from zero" (N9).
- **A related, later correction** ([R41](../research_log.md#r41)): the 3-seed sd itself is unreliable.
  The same MLP quantity measured 0.00143 in one set of runs and 0.00031 in another (M9, X16), so I switched
  to the pooled within-fold sd, which has 12 degrees of freedom (M10, M11).
- **Two smaller claims from the same session** ([R37](../research_log.md#r37)): "the best epoch is 1" was
  the argmax of a flat validation curve, not a sign of overfitting; and "fixed seed, new learning rate" is
  not a controlled experiment on GPU, because non-deterministic kernels make runs diverge.

## Limits

- **T11 still rests on 2-df sds.** Its standard error uses 3-seed sds, the quantity M9 shows to be
  unreliable. The 2.15σ is better than the retracted claim but is not itself a precise number.
- **Models are held fixed.** The bootstrap resamples test days of stored predictions. It does not refit
  the models on resampled history, so it says nothing about training-set variation.
- **One sample of history.** Six test years of one market. A block bootstrap cannot manufacture regimes
  that never occurred, and blocks of 20 days do not capture dependence that lasts longer than that.
- **Path-dependent turnover** is only approximated at block boundaries (Method).
- **Percentile intervals** without bias correction; I chose them because I can explain them, not because
  they have the best coverage.
- **Paired results are per seed pair.** T10 is MLP seed 0 against GBDT seed 0. The GBDT seed-pair
  comparison that shows the resolution effect is described here without a number because it is not yet in
  the provenance sheet; it is reproducible with
  `python model/block_bootstrap.py --preds model/preds_walkforward_cn_vwapentry.parquet:GBDT_s0 --preds model/preds_walkforward_cn_vwapentry_s1.parquet:GBDT_s1 --paired`.
- **US has one seed**, so the negative control has no seed axis at all.

## References

- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of
  Statistics*, 17(3), 1217–1241.
- Politis, D. N., & Romano, J. P. (1992). A circular block-resampling procedure for stationary data. In
  R. LePage & L. Billard (Eds.), *Exploring the Limits of Bootstrap* (pp. 263–270). Wiley.

## Errata

- **2026-09-13 · Overlapping holdings.** The first reason listed under Problem, "the label spans two days
  but the portfolio rebalances daily, so consecutive holdings overlap", is wrong. `label_raw` runs from
  T+1 to T+2, one trading day, so consecutive daily returns cover back-to-back intervals (glossary,
  *label timing*). The other two reasons, EMA smoothing and persistent regimes, still make the daily
  series autocorrelated, so the case for a block bootstrap and every interval in this note are unchanged.
