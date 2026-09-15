# Deep dive: testing the selection procedure, not just the selected variant

*Written 2026-09-13. The body is frozen; corrections go to [Errata](#errata) at the end.*
*Research log entries: [R23](../research_log.md#r23), [R35](../research_log.md#r35).*
*Numbers: every value is followed by its row ID in [`results_provenance.md`](../results_provenance.md).
Terms follow [`glossary.md`](../glossary.md).*

**Summary.** My headline portfolio variants were picked from a grid, so the reported number is the
maximum of many correlated draws. I tested the procedure that picked them in two ways. Nested
reselection measures what the choice added: +0.64 bps, about 3% of the long-short result (T2). A
permutation null that re-runs the whole grid search on noise shows the headline level cannot come from
picking the best of noise (T6). Along the way, the statistic I had designed to measure selection
("best − fixed baseline") turned out to measure trading-cost arithmetic instead (T7). The permutation
null tests levels only, and neither test reaches choices made before the grid existed.

---

## Problem

Daily turnover of 110.0% made costs the main obstacle on the original label (L12). In
[R23](../research_log.md#r23) I post-processed stored predictions with two turnover controls, an EMA
of each stock's score and an exit band (`emaS/exitQ`, glossary), over a 24-variant long-short grid and
a 12-variant long-only grid. I then reported the best variants: long-short `ema5/exit10`, net@10
+23.19, IR@10 +3.50 (P1), against +17.97 for the unsmoothed `ema1/exit10` (P7); and long-only
`ema5/exit30`, net@10 +3.11, IR@10 +1.19 (P3).

Those variants were not specified in advance. The best of 24 correlated noisy estimates is biased
upward even when nothing underneath is real. This is the data-snooping problem: once the same data is
used both to search a specification space and to judge the winner, a good result may come from the
search (White, 2000). The same bias appears in machine learning when the error of a tuned model is
estimated on the data used to tune it (Varma & Simon, 2006).

Two earlier checks did not answer this:

- `model/longonly_null.py` shuffles returns for **one fixed variant**. It asks whether that variant's
  stock selection beats random, not whether the variant was picked from a grid. It also passes on the US
  negative control (T8, T9), so it is not evidence of economic significance (X9).
- My first plan was to read "best variant − fixed baseline `ema1/exit10`" as the value of choosing.
  That turned out to be wrong; see the result on gaps below.

## Method

Both modes are in `model/selection_null.py`. They use the same grids as the original scans (long-short:
spans 1, 2, 3, 5, 10, 20 × exits 10–40%; long-only: spans 1, 3, 5, 10 × exits 10–30%), and the script's
portfolio numbers match `turnover_study.py` and `longonly_study.py` for the headline variants.

**1. Nested reselection** (`--mode nested`). Move the variant choice inside the walk-forward. For each
test year k from the second year on, pick the variant with the best net@10 over all years before k,
and apply it to year k only. Joining those years gives a result that is out of sample for the choice
too, the same idea as nested cross-validation (Varma & Simon, 2006). Over the same years I compare three
numbers:

- full-sample best: the variant that is best over all years, evaluated on 2021–2025 (the current headline);
- nested: the year-by-year out-of-sample choice;
- fixed baseline: `ema1/exit10`, no choice at all.

*Selection bias* = full-sample best − nested. *Honest turnover gain* = nested − fixed baseline. 2020 has
no prior year to choose from, so every number in this mode covers 2021–2025.

**2. Selection permutation null** (`--mode permute`). Under the null that scores carry no information
about returns, shuffle `label_raw` across stocks within each day, rescan the **whole** grid, keep the
best variant's net@10, and repeat (n = 200). The null side carries the same best-of-24 maximization as
the observed side, so the two are comparable. Holdings depend only on the scores, so they are computed
once; each permutation only re-aggregates returns, and all variants share the same permutation.

**3. The negative control.** Both modes also run on the US `hlc3` predictions. A test that the US also
passes cannot separate signal from no signal.

## Result

**Nested reselection, long-short** (CN, GBDT seed 0, 2021–2025) (T1, T2):

| | net@10 | IR@10 |
|---|---|---|
| full-sample best `ema5/exit10` | +22.39 | +3.43 |
| nested reselection | +21.76 | +3.34 |
| fixed `ema1/exit10` | +17.00 | +2.78 |

- **Selection bias is +0.64 bps**, about 3% of +22.39 (T2). It is small because the choice is stable:
  the nested procedure picks `ema5/exit10` in 4 of 5 years, and `ema2/exit10`, a neighbour, in 2021 (T2).
- The honest gain from turnover reduction is +4.76 bps, IR +0.56 (T3).

**Nested reselection, long-only** (T4). Full-sample best `ema10/exit30` +2.20, nested +1.78, fixed
−0.82: selection bias +0.42, turnover gain +2.59 (IR +0.90). Note which variant wins: by net@10 the
full-sample best on these years is `ema10/exit30`, not my headline `ema5/exit30`. On the full
2020–2025 long-only grid `ema10/exit30` also has slightly higher net@10 (+3.15 against +3.11), while
`ema5/exit30` has the higher IR@10 (+1.19 against +1.18) (P5). I keep `ema5/exit30` as the long-only
headline because I chose it by IR@10; the difference in net@10 is small next to the bootstrap interval
of the long-only result (P9).

**Permutation null, level** (CN long-short, 2020–2025) (T6). The observed best-of-24 is +23.19. On
shuffled returns the best-of-24 has mean −0.67, sd 0.47, and 95th percentile +0.06; no permutation
reached the observed level (p = 0.005, 0 of 200). Picking the best variant on noise does not come
close to the headline level.

**Permutation null, gap: the test I designed and had to withdraw** (T7). For "best − fixed baseline"
the observed gap is +5.22, but the null gap averages +9.40 (95th percentile +10.42). Read naively, the
null "does better" than the data. The reason is mechanical. Shuffling returns leaves holdings and
turnover unchanged. Slow variants trade less, so on noise they lose less to costs than `ema1/exit10`,
and best-of-24 beats the baseline by several bps with no signal at all. The gap is set by cost
arithmetic, not by skill at choosing, so the two sides are not comparable. The rule I took from this:
**a permutation null tests levels, never gaps, unless the two things being subtracted have identical
turnover.** To measure selection, nest it.

**The same cost arithmetic on the negative control** (T5). On US 2021–2025, nested reselection gains
+8.78 bps (IR +1.63) long-short and +4.45 (IR +1.11) long-only over the fixed baseline, larger than on CN
(T3, T4). US turnover is higher (unsmoothed 124.4% (N3) against 101.4% on CN (P6)), so there is more
cost to save. The nested US results themselves are still negative: long-short −2.67, long-only −0.35
(T5). The turnover gain is therefore not a CN finding and I do not report it as alpha; the negative
control holds on levels.

## What it overturned

- **"Best − fixed baseline measures selection bias"** (X15). It measures cost arithmetic (T7); the
  nested value is +0.64 (T2).
- **The worry that the headline is a best-of-grid artefact.** Selection bias is +0.64 (T2) and the
  null's 95th percentile is +0.06 against +23.19 (T6).
- **Reading the turnover gain as a result.** It is larger on the negative control (T5), so it is cost
  saving that any high-turnover score would enjoy.
- **An earlier over-reading of a single-variant null** was already withdrawn: the within-day null passes
  on US too, z = +9.2 against +11.9 on CN (T8, T9, X9).

## Limits: what permutation, and this whole test, cannot reach

- **Gaps.** As above, a within-day permutation cannot test a difference between variants with different
  turnover. Model-to-model differences need a different tool: the paired block bootstrap, which
  resamples days so that holdings and costs move together
  (see the [block bootstrap deep dive](2026-09-13-block-bootstrap.md)).
- **What the null hypothesis is.** Shuffling returns within a day breaks every link between scores and
  returns at once. A score that only tilts towards a known effect (such as short-term reversal) is also
  "not noise" and passes. Passing the permutation null says the level is not luck; it does not say the
  level is new information. That question is the
  [reversal-exposure audit](2026-09-13-reversal-exposure.md).
- **Choices upstream of the grid.** The nested test covers only the choice among the 24 (or 12)
  variants. It does not cover choices made earlier with knowledge of full-sample results: the label
  (four variants, see the [label deep dive](2026-09-13-label-vwap-entry.md)), the grid's own ranges, the
  10 bps cost level, the model family, or the decision to smooth at all. Those are outside any test here.
- **Short history for the choice.** The nested procedure makes five yearly choices, and the earliest is
  made from a single prior year. That is enough to show the choice is stable, not enough to estimate
  selection bias precisely.
- **One seed, one market.** Every number here is GBDT seed 0 on CN, plus a single US run. The seed
  spread of the long-short headline itself is 1.57 bps (P2), larger than the selection bias.
- **Permutation count.** With n = 200 the smallest attainable p is 0.005 (T6); the null's 95th percentile
  is the more informative statistic.
- **Frozen variants are a policy, not a test.** Every later model uses the same `ema5/exit10` and
  `ema5/exit30` without re-picking (glossary, *headline variants*). That avoids a new choice per model,
  but it means those models are evaluated at a variant chosen for GBDT seed 0.

## References

- Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model
  selection. *BMC Bioinformatics*, 7, 91. https://doi.org/10.1186/1471-2105-7-91
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126.

## Errata

*None yet.*
