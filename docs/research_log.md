# Research log

This log records how the project developed, one entry per working session, in date order. It covers
every session at the same level of detail, including dead ends and claims I later withdrew. I did not
pick only the good stories, because a log that keeps only the best results has the same selection
bias that the project's own tests are built to catch.

**How to read an entry.** Each entry gives the problem, the method, the result, what was overturned
(by a later entry or by the same one), and the files involved. Entry IDs (`R01`–`R49`) are stable
anchors, e.g. `docs/research_log.md#r27`.

**Numbers.** Every number is quoted from [`results_provenance.md`](results_provenance.md), and the row
ID follows it in brackets, e.g. "+0.0400 ± 0.0015 (M1)". Rows in section X of that sheet are values I
once reported and later corrected; they appear here only next to the value that replaced them.
Sessions whose results are not yet tied to a result file or a script run are described without
numbers. Terms follow [`glossary.md`](glossary.md).

**Status** of each entry's main claim, as of the last entry:
**holds** · **partly corrected** · **overturned** · **superseded plan** (historical value only).
An arrow points to the entry that made the change.

**★ Key entries** (read these first): [R15](#r15), [R18](#r18), [R19](#r19), [R21](#r21),
[R23](#r23), [R27](#r27), [R31](#r31), [R33](#r33), [R34](#r34), [R35](#r35), [R37](#r37),
[R38](#r38), [R40](#r40), [R42](#r42).

---

## July 2026: first pipeline, US only

<a id="r01"></a>
### R01 · 2026-07-05 · Project scope and first plan
*Planning · superseded plan (→ R16, R44)*
- **Problem.** I wanted a daily cross-sectional equity alpha project built end to end from public data,
  where I can explain every step.
- **Method.** Planned steps: freeze the data and my own Alpha158-style features; train an MLP on one
  trading day per batch; switch to a correlation (IC-type) loss; build a decile backtest; add models
  with different mechanisms; combine them with greedy ensemble selection (Caruana et al., 2004).
- **Result.** Decided to keep my own feature code instead of an external factor library, and to
  disclose survivorship bias rather than fix it at this stage.
- **Overturned.** The model-first order was replaced by "evaluation first" (R16).
- **Files.** `dataset/alphaFactor/features.py`, `dataset/alphaFactor/build_dataset.py`.

<a id="r02"></a>
### R02 · 2026-07-05 · Data-loading design for cross-sectional losses
*Design · superseded plan (→ R05)*
- **Problem.** How to feed the panel to a neural network when the loss is defined per trading day.
- **Method.** Compared a sample-level loader (shuffle rows) with a day-level loader (one batch = one
  full cross-section).
- **Result.** A per-day correlation loss needs the whole day's cross-section in one batch; a row-level
  loss such as MSE does not.
- **Overturned.** The planned separate loader module was never built; one day per batch was
  implemented directly in the MLP training script.
- **Files.** `model/train_mlp.py`.

<a id="r03"></a>
### R03 · 2026-07-05 · Label standardization: robust z-score, not ranks
*Decision · holds (price definition later changed → R27, R30)*
- **Problem.** The first label was a daily rank, which throws away how large a return was.
- **Method.** Compared three options: plain z-score (its denominator blows up on fat tails),
  winsorize then z-score, and robust z-score `(x − median) / (1.4826·MAD)` clipped at ±3, the same
  transform the features use.
- **Result.** Adopted the robust z-score. It keeps magnitude, resists fat tails, and matches the features.
- **Overturned.** Nothing about the standardization. The underlying price and gate changed later.
- **Files.** `dataset/alphaFactor/build_dataset.py`.

<a id="r04"></a>
### R04 · 2026-07-05 · Full US dataset built
*Experiment · partly corrected (same entry)*
- **Problem.** Build and check the full US panel before modeling.
- **Method.** Ran the builder on the US universe (1,298 tickers downloaded, D1) and checked missing
  values, label distribution, and the train/validation/test split.
- **Result.** No missing feature values; the label is missing only on each ticker's last two days, as
  the forward-looking label requires.
- **Overturned.** I had guessed that 1–2% of rows would hit the ±3 clip (X18). The measured share
  was 4.55% (L1): fat tails, not a bug.
- **Files.** `dataset/rawdata/download.py`, `dataset/alphaFactor/build_dataset.py`.

<a id="r05"></a>
### R05 · 2026-07-07 · MLP, IC loss, GBDT, and a horizon sweep
*Experiment · overturned (→ R18)*
- **Problem.** First models: is the bottleneck the model, the loss, the label horizon, or the features?
- **Method.** MLP with MSE; MLP with a Soft IC loss (1 − daily Pearson correlation); LightGBM with
  early stopping on daily RankIC; GBDT retrained on longer label horizons.
- **Result.** Everything seemed stuck near a validation RankIC of 0.01 (GBDT best +0.0105; MSE MLP best
  +0.0125, ending at +0.0035, X1). A longer horizon did not raise it. I concluded the features were the bottleneck.
- **Overturned.** These were tier A selection scores, the best point of each training curve (R18). The
  tier C US value is +0.0033 (N1), and the same features give +0.0236 on CN (L4), so "features too
  weak" holds only for US (X3).
- **Files.** `model/train_mlp.py`, `model/train_lgbm.py`, `model/horizon_sweep.py`,
  `model/experiments.jsonl`, `model/horizon_sweep.jsonl`.

<a id="r06"></a>
### R06 · 2026-07-08 · A conservative MLP training recipe
*Experiment · overturned (→ R07)*
- **Problem.** The Soft IC MLP diverged: validation RankIC fell as training went on.
- **Method.** Applied a standard conservative recipe: tapered hidden layers 256-128-64-32 with
  dropout, cosine learning-rate decay, gradient clipping, stronger weight decay, early stopping on a
  smoothed validation curve, and a very low learning rate.
- **Result.** On the small smoke dataset the Soft IC run no longer drifted negative.
- **Overturned.** On full data the very low learning rate underfit (R07). Scale-dependent
  hyperparameters do not transfer between datasets, and smoke-set numbers mean nothing.
- **Files.** `model/train_mlp.py`.

<a id="r07"></a>
### R07 · 2026-07-08 · Full runs and a learning-rate scan
*Experiment · partly corrected (same entry → R18)*
- **Problem.** Does the conservative recipe work at full scale?
- **Method.** Full US runs at the low learning rate, then a scan of larger learning rates with the rest
  of the recipe fixed.
- **Result.** The low rate underfit, with validation RankIC near zero. A rate of 1e-3 was stable and
  became the default (I6). The recipe stopped the collapse but did not raise the level, which stayed
  below GBDT. All values were tier A/B.
- **Overturned.** The positive smoke-set result from R06 was an artifact of a tiny cross-section.
  Rule adopted: smoke runs test code, never numbers.
- **Files.** `model/train_mlp.py`, `model/experiments.jsonl`.

<a id="r08"></a>
### R08 · 2026-07-08 · Loss × recipe grid completed; walk-forward built
*Tool · partly corrected (→ R17)*
- **Problem.** The single 2023 validation year was the largest untested assumption behind every conclusion.
- **Method.** Finished the MSE vs Soft IC comparison under the new recipe, then wrote an expanding
  walk-forward: train on all years before Y and test on Y. The validation segment is cut from the end
  of the training window, with an embargo at the year boundary.
- **Result.** Soft IC beat MSE on validation, but both MLPs stayed below GBDT (tier A/B). The
  walk-forward tool became the basis of every later tier C number.
- **Overturned.** There was no embargo between the fit and validation segments inside the training
  window. Fixed in R17.
- **Files.** `model/walkforward.py`.

<a id="r09"></a>
### R09 · 2026-07-08 · US mini walk-forward
*Experiment · overturned (→ R17, R18)*
- **Problem.** Is "about 0.01" a stable property across years, or bad luck in 2023?
- **Method.** Six expanding folds, test years 2020–2025, GBDT, US.
- **Result.** First tier C numbers: mean +0.0046 (X2). 2023 was not unusual. I read this as a stable
  ceiling and "weak features confirmed", and noted that validation and test rankings disagreed across years.
- **Overturned.** The "0.01 ceiling" came from tier A scores, and this run lacked the fit/validation
  embargo. The corrected US value is +0.0033 (N1), and CN with the same features gives +0.0236 (L4).
- **Files.** `model/walkforward.py`, `model/walkforward.jsonl`.

<a id="r10"></a>
### R10 · 2026-07-08 · Direction: no formulaic-alpha expansion; build backtest, pool, ensemble
*Decision · partly corrected (→ R18, R19)*
- **Problem.** Given "weak features", should I add more factors or build the portfolio and ensemble layers?
- **Method.** Weighed adding formulaic alphas from the published 101 Formulaic Alphas (Kakushadze, 2016)
  against building a backtest engine, a model pool, and ensemble selection.
- **Result.** Dropped factor expansion and chose backtest → model pool → ensemble.
- **Overturned.** The premise ("features are weak") was false outside the US (R18). The closure was
  later kept for a different reason, priority: the measurement was the bottleneck (R19).
- **Files.** None.

<a id="r11"></a>
### R11 · 2026-07-08 · Rolling to-do list
*Planning · superseded plan (→ R16, R44)*
- **Problem.** Keep track of open work after the direction decision.
- **Method.** Listed: close the MLP–GBDT gap (seed averaging, ranking losses), backtest engine,
  model pool, ensemble selection, and a China A-share (CN) dataset to test whether the US level is market-specific.
- **Result.** The CN dataset became the active item.
- **Overturned.** Replaced by the evaluation-first order (R16) and later the unified tracker (R44).
- **Files.** None.

<a id="r12"></a>
### R12 · 2026-07-08 · CN dataset design
*Planning · partly corrected (→ R24, R27)*
- **Problem.** Design a CN daily panel that the frozen feature code can reuse, with realistic tradability.
- **Method.** Universe: union of historical CSI 300 and CSI 500 constituents, sampled quarterly to
  limit survivorship bias. Adjusted prices. Limit-up/limit-down detection by board, ST status, and
  date; suspensions. A VWAP-based label was planned. Open question: drop rows whose exit cannot be sold,
  or roll the exit to the next tradable day?
- **Result.** A design ready for implementation, apart from the exit question.
- **Overturned.** The VWAP label was not implemented at the time (R24); it arrived only in R27.
- **Files.** `dataset/rawdata_cn/download_cn.py`, `dataset/alphaFactor/build_dataset.py`.

<a id="r13"></a>
### R13 · 2026-07-09 · CN dataset implemented
*Tool · partly corrected (→ R27, R30)*
- **Problem.** Implement the CN downloader and label with the tradability gate.
- **Method.** Downloader with resumable manifest and the same column layout as US; builder option
  `--market cn`; label kept only when the stock can be bought on T+1 and sold on T+2, plus the
  `|label_raw| ≤ 0.8` filter; same close-to-close formula as US. For unsellable exits I chose to drop
  the row rather than roll it. Synthetic tests covered board-specific and date-specific thresholds.
- **Result.** A working CN panel; the synthetic gate tests passed.
- **Overturned.** Dropping rows that cannot be sold on T+2 deletes losses on positions already held,
  which cuts only the left tail. The entry-only gate became the default (R27, R30).
- **Files.** `dataset/rawdata_cn/download_cn.py`, `dataset/alphaFactor/build_dataset.py`.

<a id="r14"></a>
### R14 · 2026-07-09 · Data-source switch and download robustness
*Tool · holds (failures → survivorship bias, R21)*
- **Problem.** The first CN data source was slow, and a second one became unreachable from my network.
- **Method.** Debugged layer by layer (proxy settings, name resolution, TLS fingerprinting), then added
  a `--source` switch with three interchangeable backends that share one output schema.
- **Result.** 1,313 of 1,377 tickers downloaded; 64 failed (4.6%), and the failures behave like
  delisted names (D2).
- **Overturned.** Nothing here, but the failed tickers are the CN half of the survivorship bias found in R21.
- **Files.** `dataset/rawdata_cn/download_cn.py`, `dataset/rawdata_cn/manifest_cn.csv`.

<a id="r15"></a>
### R15 ★ · 2026-07-09 · CN looks far stronger than US, until it is broken down
*Experiment / correction · overturned (→ R18, R20)*
- **Problem.** The first CN GBDT test RankIC (train before 2023, test 2024–2025) looked several times
  larger than the US figure.
- **Method.** I doubted the number and broke it down: median vs mean, share of positive days,
  trimming, removing the best days, and quarter-by-quarter values.
- **Result.** The effect was not one outlier day, but it was concentrated in 2025. I concluded that CN
  was about as weak as US in ordinary years, with a strong 2025 regime.
- **Overturned.** The comparison set a tier C CN value against a tier A US value (R18). Walk-forward
  then gave CN positive RankIC in all six test years, worst fold +0.0061 (L4), and the cleanest decile
  structure was in 2020–2022, not 2025 (R20).
- **Files.** `model/compare_lgbm_cn_us.py`.

## Late August 2026: rebuilding the measurement

<a id="r16"></a>
### R16 · 2026-08-29 · Move to Linux; publication boundary; evaluation first
*Decision · partly corrected (→ R17)*
- **Problem.** I saw three gaps (few models, no hyperparameter search, weak evaluation) and needed an order.
- **Method.** Reasoned about what each step does. Adding models and tuning are both selection
  procedures: generate candidates, keep the best score. If the score is unreliable, both maximize noise.
- **Result.** New order: evaluation → nested hyperparameter search → models → competitions. Planned
  RankIC/ICIR, deciles, turnover and cost curves, block bootstrap, and permutation nulls. A
  "60-second rule": every method in the repository must be explainable unprompted, and methods I cannot
  defend go into Limitations. Moved the working tree to a Linux filesystem for faster I/O.
- **Overturned.** "Migration complete" meant only that the files were copied; the environment could not
  run the code (R17).
- **Files.** None.

<a id="r17"></a>
### R17 · 2026-08-30 · Environment rebuild, thread count, fit/validation embargo fix
*Tool · holds*
- **Problem.** No script ran in the new environment, and reruns of the US walk-forward did not match July.
- **Method.** Installed and pinned dependencies; timed LightGBM at several thread counts on a hybrid
  performance/efficiency-core CPU; reran the walk-forward several times; checked every split boundary.
- **Result.** Using all cores was far slower than six threads (now the documented setting). The rerun
  was deterministic, but after the library/platform change two of six folds changed sign: per-fold
  noise was as large as the effect. The fit and validation segments had no embargo. After the one-line
  fix the US mean went from +0.0030 to +0.0033 and the fold range narrowed from [−0.0038, +0.0113] to
  [−0.0021, +0.0071] (X2, N1). I also decided not to reframe the project as a null result.
- **Overturned.** Nothing; the fix is part of every later number.
- **Files.** `model/walkforward.py`, `requirements.txt`.

<a id="r18"></a>
### R18 ★ · 2026-08-30 · Number audit: selection score vs measurement
*Audit · holds (one caveat withdrawn → R24)*
- **Problem.** I suspected the earlier numbers in my notes.
- **Method.** Read how each script reported its metric. The training scripts reported the maximum of
  the validation curve, and the experiment log had no test fields at all. I defined three tiers:
  A selection score, B last-epoch validation, C measurement (see glossary).
- **Result.** The "0.01 ceiling" was a tier A value, +0.0105 (X1); on the same data the tier C US value
  is +0.0033 (N1). "Features are weak" held only for US: CN walk-forward with the same features and
  parameters gave +0.0236, 6/6 folds positive, worst fold +0.0061 (L4). From here on every number carries a tier.
- **Overturned.** Two earlier conclusions (R05, R09). This entry's caveat that "the two markets use
  different label prices" was itself withdrawn in R24.
- **Files.** `model/train_lgbm.py`, `model/train_mlp.py`, `model/experiments.jsonl`, `model/walkforward.jsonl`.

<a id="r19"></a>
### R19 ★ · 2026-08-30 · US becomes the negative control; formulaic alphas closed; "7x" retired
*Decision · holds*
- **Problem.** How should the two markets be used now that they differ so much?
- **Method.** Asked what each market can prove. With the same script, parameters, and embargo, the
  dataset is the only variable, so US is the only evidence that the CN signal is not a pipeline bug.
- **Result.** CN is the development market; US is kept as a negative control (see glossary). The
  cross-market ratio "CN is 7x more predictable" (X4) is retired: labels, gates, and universes differ,
  so it is not comparable. Formulaic alphas stay closed for a stated reason, priority rather than
  feasibility: more features only give a selection procedure more noise to pick from.
- **Overturned.** Nothing. One supporting reason ("different label prices") was withdrawn in R24;
  the decision does not depend on it.
- **Files.** None.

<a id="r20"></a>
### R20 · 2026-08-30 · One evaluation script: deciles, turnover, costs, baselines
*Tool / experiment · overturned (→ R27, R32, R33)*
- **Problem.** RankIC alone cannot show whether a signal can be traded.
- **Method.** The walk-forward now saves its predictions, and `evaluate.py` computes RankIC and ICIR,
  decile returns and monotonicity, turnover, a cost curve, per-year breakdowns, and single-factor
  baselines. Its self-check reproduces the training log exactly.
- **Result.** On `close × full` the ICIR was +0.202 (L11), but net@10 was only +0.15 at 110.0% turnover
  (L12). I concluded that a 10 bps cost wipes out the signal (X5). The ROC5 reversal factor seemed to
  reach 76% of the model's RankIC (X11). US showed a negative spread of −12.40 with monotonicity +0.067 (X6).
- **Overturned.** All three. With the VWAP label, net@10 is +12.72 (L13) (R27). The US pattern was one
  price splice; capped, it is +3.13 with monotonicity +0.491 (N3) (R31, R32). Re-measured, the model's
  decile spread is only 1.42x the ROC5 spread, and ROC5 alone earns net@10 +11.57 (A1, A2) (R33).
- **Files.** `model/evaluate.py`, `model/walkforward.py`.

<a id="r21"></a>
### R21 ★ · 2026-08-30 · Isolation audit: clean splits, survivor-biased universes
*Audit · holds (bias not fixed)*
- **Problem.** Every year from 2020 to 2025 is used somewhere. Is the isolation really clean?
- **Method.** I checked in the code that each year is a test year exactly once and that the code constructs
  ordered fit < validation < test masks with a 2-day embargo at both boundaries (D4). Checked that
  standardization uses only the same day's cross-section, that features use only past shifts, and that the
  label looks forward. Then checked how each universe was built.
- **Result.** The time splits are clean. Both universes are survivor-biased: the US list is an
  end-of-sample index snapshot (a spot check of names that left the index during the sample found none
  of them in the list), and on CN 64 of 1,377 tickers failed to download (D2). The bias makes results
  look better; it makes the US negative control more conservative, not less.
- **Overturned.** Nothing. The bias is a documented limitation.
- **Files.** `model/walkforward.py`, `dataset/alphaFactor/features.py`, `dataset/alphaFactor/build_dataset.py`,
  `dataset/rawdata/universe_full.txt`, `dataset/rawdata_cn/universe_full_cn.txt`.

<a id="r22"></a>
### R22 · 2026-08-30 · Session summary; working notes stay private
*Decision · holds (→ R47)*
- **Problem.** After a day of corrections, what are the real assets, and what can be published?
- **Method.** Reviewed the session's work: environment and embargo fix, number audit, market roles,
  evaluation script, isolation audit.
- **Result.** The strongest assets were the self-corrections, the negative control, and cost-aware
  evaluation. My working notes would stay private, and the public story would be written separately in
  English. A repository boundary controls which files can be committed, not what committed prose says.
  The largest gap was that none of this was readable from outside.
- **Overturned.** Nothing; R47 turned it into a publication plan.
- **Files.** None.

<a id="r23"></a>
### R23 ★ · 2026-08-30 · Turnover reduction: EMA smoothing and an exit band
*Experiment · partly corrected (→ R27, R35)*
- **Problem.** Daily turnover of 110.0% (L12) made costs the main obstacle.
- **Method.** Post-processed the stored predictions with a 24-variant grid of `emaS/exitQ` (glossary)
  in `turnover_study.py`, and ran the same grid on US.
- **Result.** Smoothing improved both RankIC and turnover on the close label, and `ema5/exit10` gave the
  best net@10; on US it created no signal. On the current label the same variant gives gross +27.50,
  turnover 43.2%, net@10 +23.19, IR@10 +3.50 (P1), against +17.97 for the unsmoothed `ema1/exit10` (P7).
- **Overturned.** The close-label numbers were replaced by the VWAP label (R27). The concern that the
  variant was picked from the grid was settled by the nested test: selection bias +0.64 (T2). The gain
  from smoothing is cost arithmetic, not alpha: it is larger on US (+8.78, T5) than on CN (+4.76, T3) (R35).
- **Files.** `model/turnover_study.py`.

<a id="r24"></a>
### R24 · 2026-08-30 · Correction: the CN label was close-to-close, not VWAP
*Correction · holds (one claim corrected → R29)*
- **Problem.** A public draft described the CN label as true VWAP.
- **Method.** Read the builder code instead of the design notes.
- **Result.** The CN label used adjusted close, the same formula as US; the traded amount was
  downloaded but used only for tradability. The one real design difference between markets was the
  CN tradability gate. I withdrew the "different label prices" caveat (R18, R19).
- **Overturned.** This entry's claim that raw-label standard deviations were nearly equal across markets
  was wrong: on the full panels US is 0.1685 and CN 0.0259, 6.5x, because only CN filters extreme
  returns (L2) (R29). Rule: check every implementation statement against code, not against plans.
- **Files.** `dataset/alphaFactor/build_dataset.py`.

<a id="r25"></a>
### R25 · 2026-08-30 · Multi-seed GBDT ensemble
*Experiment (null) · holds*
- **Problem.** Does averaging GBDT seeds improve the tradable result?
- **Method.** Made the seed and parameters overridable from the command line (`--seed`, `--set`, `--tag`)
  and averaged several seeds' raw scores, with and without smoothing.
- **Result.** The seed ensemble raised RankIC on its own, but after EMA smoothing it added nothing to
  net@10. Both average away the same prediction variance. Not adopted.
- **Overturned.** Nothing. A side lesson: explicitly setting LightGBM's derived seeds changes the random
  stream and breaks reproducibility, so only the base seed is set.
- **Files.** `model/walkforward.py`.

<a id="r26"></a>
### R26 · 2026-08-30 · GBDT hyperparameter scan on CN
*Experiment (null) · holds*
- **Problem.** The GBDT parameters had been set for US and never tuned on CN. Many boosting rounds
  looked like a capacity limit.
- **Method.** A small grid of leaves, minimum child samples, and learning rate. Selection by in-fold
  validation RankIC; test values were read only after selection.
- **Result.** The validation-selected configuration did not beat the existing parameters on test.
  More leaves and weaker regularization did not help, so the capacity hypothesis was rejected. Test
  results varied within a narrow range, and validation was too noisy to rank configurations.
- **Overturned.** Nothing. A robust response surface is itself evidence against a knife-edge result.
- **Files.** `model/tune_gbdt.py`.

<a id="r27"></a>
### R27 ★ · 2026-08-30 · VWAP label: RankIC goes up; the short leg never had a limit gate
*Experiment · partly corrected (→ R29, R34)*
- **Problem.** I expected a VWAP label to lower RankIC, because close prices carry bid-ask bounce that
  looks like reversal.
- **Method.** Two controls first: rebuilding the old label with the new script is bit-identical
  (`--verify`), and the new `--label-file` path reproduces the baseline. Then only the label varies.
- **Result.** RankIC: `close × full` +0.0236 (L4), `close × entry` +0.0278 (L5), `vwap × full` +0.0369
  (L6), `vwap × entry` +0.0409 (seed 0, L7); close → VWAP is +57% (L8). net@10 went from +0.15 to
  +12.72 (L12, L13), mostly via the short leg (L15). Because the label starts at T+1, bounce cannot
  align with day-T features; it only adds label noise, which attenuates the correlation. The short leg's
  entry was never gated; gating it cuts the `vwap × full` spread from +23.63 to +20.19 (A3).
- **Overturned.** The seed-0 +0.0409 is the top of three seeds; the honest value is +0.0400 ± 0.0015 (M1,
  X12) (R34). "About 70% of the spread comes from the short leg" was measured as 79% (L16, X7) (R29).
- **Files.** `dataset/alphaFactor/make_vwap_label.py`, `model/walkforward.py`, `model/limit_audit.py`, `model/evaluate.py`.

<a id="r28"></a>
### R28 · 2026-08-30 · MLP single-fold timing and a compute budget
*Experiment · overturned (→ R34)*
- **Problem.** I could not judge an MLP search budget without knowing the cost of one run.
- **Method.** Timed one walk-forward fold of the MLP on CN and extrapolated linearly with fit days.
- **Result.** A budget per configuration and seed. The MLP beat GBDT on that one fold. One day per
  batch makes GPU kernel launches, not compute, the bottleneck. I set a rule: report the whole response
  surface of a scan, and fix the headline selection rule before looking at test values.
- **Overturned.** One fold supports no conclusion, and the MLP value was the low end of its own
  run-to-run noise band (R34).
- **Files.** `model/walkforward_mlp.py`.

<a id="r29"></a>
### R29 · 2026-08-30 · VWAP look-ahead audit, hlc3 proxy calibration, long-only view
*Audit · partly corrected (→ R33)*
- **Problem.** Does the VWAP label use future data? Can US test the same effect without traded amounts?
  And how much of the spread needs shorting?
- **Method.** Four look-ahead checks: features never touch traded amount; hand-computed timestamps; a
  whole-pipeline control with the label shuffled across stocks within each day; adjustment-factor
  anchoring. Calibrated the typical price `hlc3` on CN, the only market with both prices. Wrote a
  long-only study against the universe mean.
- **Result.** No look-ahead: the shuffled-label control gives −0.0012, 2/6 folds positive (L10). On CN,
  `hlc3` gives +0.0407 (L9) vs VWAP +0.0369 (L6), so it is a valid proxy; US `hlc3` gives +0.0071, ICIR
  +0.054 (N2). The short leg carries 79% of the `vwap × full` spread (L16). The close label also inflates
  the universe mean itself (L17). Full-panel label sds are 0.1685 vs 0.0259 (L2).
- **Overturned.** The inflation was first put at about 500 bps/year from a cross-gate comparison (X8);
  with the gate held fixed it was put at about 450 (R33), which annualized a one-day figure as if it
  covered two days (X19). Correctly annualized it is about 890 (L17).
- **Files.** `dataset/alphaFactor/make_vwap_label.py`, `model/longonly_study.py`.

<a id="r30"></a>
### R30 · 2026-08-30 · Entry-only gate becomes the default; headline moved to long-only
*Decision · partly corrected (→ R31)*
- **Problem.** Keep the `full` gate or switch to `entry`? And should RankIC stay the headline when
  shorting individual A-shares is hard?
- **Method.** Argued from correctness: an exit blocked on T+2 is a position already held, so its loss is
  real, and the exit gate removes only the left tail. Checked that the change does not flatter the result.
- **Result.** `vwap × entry` became the CN default. It raised RankIC from +0.0369 to +0.0409 (L6, L7)
  while slightly lowering the long-only result, so I did not choose it for its numbers. I moved the headline to long-only
  `ema5/exit30`: active +4.13, net@10 +3.11, IR@10 +1.19 (P3), backed by a within-day permutation null
  that looked like significance (X9).
- **Overturned.** Dropping long-short from the headline was an overreaction (R31). The null is not
  evidence of significance: the US control passes it too (T8, T9) (R31).
- **Files.** `dataset/alphaFactor/make_vwap_label.py`, `model/longonly_null.py`.

<a id="r31"></a>
### R31 ★ · 2026-08-30 · Null withdrawn; two headlines; a price splice in the US data
*Correction · partly corrected (→ R33)*
- **Problem.** Is the US top-decile spike a signal? And does the long-only null mean anything?
- **Method.** Broke the US decile return down by year, row, and ticker; traced the largest row to raw
  prices. Ran the long-only null on the negative control.
- **Result.** One ticker's bankruptcy relisting was spliced into its price history: close $0.12 → $31.00
  with an unchanged adjustment factor, `label_raw` +257.3 (N10). RankIC, being rank-based, was
  unaffected; the bps numbers were contaminated. The null passes on US as well: z = +9.2 on US vs +11.9
  on CN (T8, T9). It shows only that stock selection is not random. Headline set to two views with their
  assumptions: long-short `ema5/exit10` net@10 +23.19, IR@10 +3.50 (P1) and long-only `ema5/exit30`
  net@10 +3.11, IR@10 +1.19 (P3). The US long-only +1.67 (N8) is not claimed as alpha (survivorship bias).
- **Overturned.** Two numbers in this entry's table: ICIR +0.315 was copied from the `vwap × full` row
  and is +0.335 (X13, L11); US long-short −2.39 (N5) was CN's variant applied to US, not US's own best
  (X10) (R33).
- **Files.** `model/longonly_null.py`, `model/evaluate.py`.

<a id="r32"></a>
### R32 · 2026-08-30 · A shared return cap for US
*Tool · holds*
- **Problem.** The US label had no filter for extreme returns, so every US bps number was exposed to splices.
- **Method.** One cap function shared by all evaluation tools, default 0.8 for US and none for CN, where
  the label already filters. I state the two thresholds' reasons separately: on CN, daily price limits
  make larger T+1 → T+2 returns implausible, while on US the cap is a judgment call. Added `--cap-scan`.
- **Result.** The cap removes 20 rows (D7). D10−D1 is +15.89 uncapped and +3.13 / +3.42 / +4.03 at
  caps 0.8 / 0.5 / 0.2, with RankIC unchanged (N4). The corrected US picture: monotonicity +0.491,
  spread +3.13, long-short net@10 −9.30 (N3), a weak monotone signal that does not cover costs.
- **Overturned.** The old US finding of a negative spread and flat deciles (X6).
- **Files.** `model/evaluate.py`, `model/turnover_study.py`, `model/longonly_study.py`, `model/longonly_null.py`.

<a id="r33"></a>
### R33 ★ · 2026-08-30 · README rewrite; three corrections found by re-measuring
*Correction · partly corrected (→ R34)*
- **Problem.** Every headline number in the README was out of date.
- **Method.** Re-ran the evaluations instead of copying numbers from my notes.
- **Result.** Three notes were wrong. ICIR is +0.335, not +0.315 (L11, X13). US's own best long-short
  variant is `ema10/exit20` at +0.59, IR +0.09, with 6 of 24 variants positive (N6), about 40x below CN
  (N7). The close-label inflation is not 500 bps/year (X8); this entry put it at 450,
  itself later corrected to about 890 (X19, L17). The re-measured baseline cut
  the model's lead: ROC5 has RankIC +0.0268, D10−D1 +19.70, net@10 +11.57 (A1), and the model spread is
  only 1.42x ROC5's (A2). On the long-only grid `ema10/exit30` has slightly higher net@10 than
  `ema5/exit30` (P5). Rule: notes record the process; they are not a data source.
- **Overturned.** The README still carried the seed-0 RankIC (X12) until later corrections (R34).
- **Files.** `README.md`, `model/evaluate.py`, `model/turnover_study.py`, `model/longonly_study.py`.

<a id="r34"></a>
### R34 ★ · 2026-08-30 · The MLP through the same instrument; the US control does not clear it
*Experiment · partly corrected (→ R41)*
- **Problem.** Measure the MLP exactly as GBDT is measured, with error bars.
- **Method.** Unit-checked the segmented Soft IC loss; measured run-to-run noise on the GPU (repeats
  inside one process understate it); tested whether multi-day batches are free (they cost RankIC, so I
  kept one day per batch); ran 3 seeds on CN with the same folds as GBDT, and the US control.
- **Result.** CN MLP +0.0466 ± 0.0014 vs GBDT +0.0400 ± 0.0015 (M1, M2); the MLP wins 6 of 6 folds,
  sign test p = 0.031 (M3). GBDT's +0.0409 was the top of its own seed band (X12). On US the MLP also
  wins, +0.0108 vs +0.0071, 5 of 6 folds (M6). "MLP beats GBDT on RankIC" holds, but "the MLP extracts
  more CN alpha" does not: the likely reason is that the MLP optimizes a correlation directly.
- **Overturned.** With two seeds the MLP looked less seed-sensitive than GBDT; the third seed removed
  that. The 3-seed sd itself turned out unreliable (M9, X16) (R41).
- **Files.** `model/walkforward_mlp.py`, `model/walkforward_mlp.jsonl`, `model/train_mlp.py`.

## September 2026: portfolio-level tests, audits, and the pool

<a id="r35"></a>
### R35 ★ · 2026-09-06 · MLP at portfolio level; equal-weight blend; selection test; meta layer rejected
*Experiment / audit · partly corrected (→ R37)*
- **Problem.** Does the MLP's RankIC lead reach the portfolio? Was the headline variant selected on noise?
- **Method.** MLP at the frozen variants (no re-picking); an equal-weight MLP+GBDT blend; a small nested
  meta-MLP over both models' daily ranks; `selection_null.py` (nested reselection and permutation null).
- **Result.** The lead splits by leg. MLP long-short net@10 is +20.40 (M4) vs GBDT +23.19 (P1); long-only
  +3.96 (M4) vs +3.11 (P3); MLP is better in D10 (+6.17 vs +5.31) and worse in D1 (−20.45 vs −22.61)
  (M5); on US long-only it loses, +0.62 vs +1.67 (M7). The blend only lowered variance; the meta-MLP moved
  weight between legs at zero net gain and was never best, so neither was adopted. Selection bias is +0.64
  (full-sample +22.39, nested +21.76; T1, T2); the null's 95th percentile for the best of 24 is +0.06 (T6).
- **Overturned.** Same entry: "best − fixed baseline" cannot measure selection; the observed +5.22 is below
  the null's +9.40, because slower variants save costs (T7, X15). Later, R37 retracted a model comparison (X14).
- **Files.** `model/selection_null.py`, `model/meta_mlp.py`, `model/longonly_study.py`.

<a id="r36"></a>
### R36 · 2026-09-06 · Reading published methods: five candidate mechanisms
*Reading · overturned (→ R38, R40)*
- **Problem.** Several model-side attempts had changed the model while the objective and the
  evaluation stayed fixed. What had I not tried?
- **Method.** Reviewed published work for mechanisms missing from the pipeline: exposure to short-term
  reversal (Jegadeesh, 1990; Lehmann, 1990); losses built on portfolio returns (Zhang et al., 2020);
  low-learning-rate training; comparing models at equal turnover; greedy ensemble selection (Caruana et al., 2004).
- **Result.** Ranked five actions, reversal exposure first (the only one that could overturn the headline);
  a larger pool with ensemble selection came last.
- **Overturned.** Three of five were closed, two because my own reasoning was wrong (R40). "Reversal was
  never checked" was half true: the factor had been checked, not the model's exposure to it (R38).
- **Files.** None.

<a id="r37"></a>
### R37 ★ · 2026-09-06 · Block bootstrap; a model comparison retracted; two kinds of error bar
*Tool / correction · partly corrected (same entry)*
- **Problem.** The naive t-statistic ignores autocorrelation, and comparisons had no sample-period uncertainty.
- **Method.** `block_bootstrap.py` (circular moving blocks; paired mode resamples the same days for both
  models, see glossary). Also a nested learning-rate check and a direct look at MLP validation curves.
- **Result.** CN RankIC [+0.0334, +0.0486] and long-short net@10 [+16.97, +29.49] exclude 0 (P9). US:
  RankIC excludes 0, every portfolio interval contains it (N9). Paired MLP − GBDT long-short net@10 is
  −3.74 [−7.87, +0.51] (T10). Learning rate made no detectable difference; "best epoch = 1" was the
  argmax of flat, non-reproducible validation curves.
- **Overturned.** "GBDT beats MLP by more than ten seed sds" (X14): GBDT's own seed sd is 1.57 (P2), so the
  gap is 2.15σ for net@10, 5.2σ for IR (T11). Rule: report both error bars. Seed sd measures training
  randomness; the paired bootstrap measures sample period and resolves gaps between near-identical models
  (two seeds) more easily than between families. Also withdrawn: "fixed seed, new learning rate" is not a
  controlled experiment, because GPU non-determinism makes runs diverge.
- **Files.** `model/block_bootstrap.py`, `model/walkforward_mlp.py`.

<a id="r38"></a>
### R38 ★ · 2026-09-06 · Reversal-exposure audit: the headline is not a reversal proxy
*Audit · partly corrected (→ R40, R42)*
- **Problem.** Short-term reversal is a well-documented effect (Jegadeesh, 1990; Lehmann, 1990). Is the
  model an expensive copy of it?
- **Method.** Daily regression of the standardized score on standardized ROC5; orthogonalize with an
  expanding-window β (year k uses only earlier years); run the reversal score alone through the same portfolio.
- **Result.** Exposure is systematic: mean β +0.373, R² 0.159, β > 0 on 98.6% of days (A4). Removing it
  costs −3.18 bps (14.3%): long-short net@10 +22.20 → +19.03, still well above pure reversal's +13.11 (A5).
  Correlated with reversal is not the same as reducible to it.
- **Overturned.** The session's own figures came from an aligned prediction copy no shipped script
  produces; the reproducible run above replaces them and shows a larger cost. "The MLP's higher RankIC
  comes from more reversal" was weakened (R40), though its exposure is higher (R² 0.194–0.215 vs 0.158, E8).
- **Files.** `model/reversal_exposure.py`, `model/evaluate.py`, `model/turnover_study.py`.

<a id="r39"></a>
### R39 · 2026-09-06 · Rule: public documents must show the research process
*Decision · holds (→ R47)*
- **Problem.** If the working notes stay private, a public README with only results would read as a
  scorecard, with no process behind it.
- **Method.** Wrote down what every public document must answer: what problem came up, what was tried,
  what the result was, what was falsified, what succeeded, and what comes next (and why).
- **Result.** A README checklist: state the problem before the method; include failed attempts; give each
  number its market, years, seeds, and cost assumption; list withdrawn claims with before/after values;
  state limits; keep private material out.
- **Overturned.** Nothing. R47 extended it from one README to a README, this log, and deep dives.
- **Files.** None.

<a id="r40"></a>
### R40 ★ · 2026-09-06 · The five mechanisms settled: one delivered, one pending, three closed
*Correction · holds*
- **Problem.** Judge each action from R36 against the evidence so far.
- **Method.** Checked each mechanism's arithmetic or code path before building it.
- **Result.** Reversal audit delivered (R38). Learning-rate regime closed: no detectable effect (R37).
  Equal-turnover comparison closed: the frozen variants already fix turnover, and the turnover
  confound is a small share of the model gap. Portfolio-return loss closed: in a synthetic check it
  puts less gradient on the tails than Soft IC, and it weights the whole cross-section linearly by rank,
  not the decile book I trade. Pool + ensemble pending; diversity is strongest across model families
  (GBDT–MLP ρ 0.569–0.621, E2) and weakest across seeds (GBDT seeds 0.794–0.851, E3).
- **Overturned.** Two of the closures were my own reasoning errors from R36. I also weakened R38's
  "the MLP relies more on reversal", which had compared against the luckiest GBDT seed.
- **Files.** None.

<a id="r41"></a>
### R41 · 2026-09-06 · Robust epoch selection; a 3-seed sd is not trustworthy
*Experiment · holds (no rule adopted)*
- **Problem.** If the best epoch is picked on a flat, noisy curve, can a smoother rule reduce seed spread?
- **Method.** Added `--epoch-dump` (off by default; verified bit-identical on CPU) to save per-epoch
  validation curves and weights, retrained 3 seeds, and compared 13 rules (M12) offline in `epoch_rule.py`.
  All rules read only the validation curve.
- **Result.** Baseline argmax: mean +0.0468, pooled within-fold sd 0.00306 (M10). Weight averaging lowers
  the spread, e.g. `wavg_run_k10` 0.00165, while `plateau_med_z1` raises it to 0.00803 (M11). Only
  `wavg_top_k*` keeps the mean RankIC (M12). No rule adopted: choosing among 13 rules is itself a selection procedure.
- **Overturned.** The same 3-seed sd measured 0.00143 and 0.00031 in two runs (M9), so a 2-df sd is
  unreliable (X16). From here on I report the pooled within-fold sd (12 df).
- **Files.** `model/walkforward_mlp.py`, `model/epoch_rule.py`, `model/epoch_rule_results.json`.

<a id="r42"></a>
### R42 ★ · 2026-09-06 · Ridge floor, an 11-member correlation pool, and the common-row-set trap
*Experiment / audit · holds*
- **Problem.** Add a linear floor model and measure how different the pool members really are.
- **Method.** Closed-form ridge with the same folds, alpha chosen on validation RankIC. The grid's upper
  bound must scale with sample size, because alpha is not unitless. `pool_corr.py` computes daily rank
  correlations, raw and after reversal orthogonalization.
- **Result.** Ridge RankIC +0.0434, 6/6 folds positive, worst +0.0310; long-short net@10 +17.91, IR@10
  +2.30 (M8). 11 members, 1,808,949 common rows (E1). Ridge has the lowest mean ρ, 0.537, and also the
  lowest IR (E4): low correlation is not the same as usefulness. Orthogonalization barely moves the matrix
  (0.6300 → 0.6253); only ridge ↔ GBDT drops (0.553 → 0.494) (E5). Ridge is the most reversal-exposed
  member, R² 0.391 (E8).
- **Overturned.** In the same entry: strength first computed on the common row set dropped 5,744 rows
  (0.32%) whose mean `|label_raw|` is 5.1x the overall mean (E6). That moved the headline from +23.19 to
  +17.30, −5.9 bps (E7, X17). Rule: audit the rows any intersection drops.
- **Files.** `model/walkforward_ridge.py`, `model/walkforward_ridge.jsonl`, `model/pool_corr.py`, `model/pool_corr_results.json`.

<a id="r43"></a>
### R43 · 2026-09-06 · Backlog: loss-diverse MLP members and stage-two ensembles
*Planning · holds (not started → R46)*
- **Problem.** The three MLPs in the pool differ only by seed, the weakest diversity axis (E3).
- **Method.** Planned MLPs that differ only in their loss (e.g. rank IC, tail-weighted IC, listwise ranking
  such as ListMLE, Xia et al., 2008), with architecture, folds, and label unchanged. Stage two: greedy
  ensemble selection with net@10 as the objective, a meta-MLP on the larger pool, and a linear stacker.
- **Result.** Each loss enters as a pool member, not as a headline candidate. Choosing a loss or a stacker
  must go through nested selection, and acceptance uses the paired bootstrap, not a 2-df seed sd.
- **Overturned.** Nothing yet.
- **Files.** None.

<a id="r44"></a>
### R44 · 2026-09-06 · One progress tracker
*Planning · holds (standing tracker)*
- **Problem.** Several plans written on different dates no longer showed where the project stood.
- **Method.** Merged them into one status table and decoded the numbering used in earlier plans.
- **Result.** Evaluation layer complete; GBDT hyperparameter search done (null); pool stage one done;
  ensemble not started; writing far behind. The shape of the deviation: I traded model breadth for
  measurement rigor. Every model-side attempt so far was null, while the one measurement change, the
  label, gave the largest single gain (+57%, L8).
- **Overturned.** Nothing; this is the tracker later plans update.
- **Files.** None.

<a id="r45"></a>
### R45 · 2026-09-10 · Publication-readiness review
*Planning · partly superseded (→ R47)*
- **Problem.** Can the repository be made public as it is?
- **Method.** A read-only review of the code, the git history and tracked files, and the README.
- **Result.** The research was ready but the repository was not: the history needed a clean start, the
  README contradicted the current state, redistributing downloaded vendor data was unclear, there was no
  license, and there were no tests. Proposed order: publication safety → README accuracy → clean-clone reproducibility.
- **Overturned.** R47 turned the proposal into a decision; tests and synthetic fixtures were deferred.
- **Files.** None.

<a id="r46"></a>
### R46 · 2026-09-13 · Research plan after publication
*Planning · holds*
- **Problem.** What should the next research step be: more architectures, or something else?
- **Method.** Asked which additions would strengthen the existing measurement rather than widen the model zoo.
- **Result.** Order: loss diversity first (only the objective changes); then nested ensembles (equal
  weight → ridge / non-negative least squares → greedy selection → meta-MLP only if the linear methods add
  value); then minimal engineering (one experiment config, a machine-readable result registry, regression
  tests); architectures last, and only for diversity. An ensemble is promoted only if it beats the best
  single member on net@10, its paired-bootstrap interval for the difference excludes 0, it passes the
  common-row audit, it does not rest on one seed or one year, and the US control shows no gain of the same shape.
- **Overturned.** Nothing.
- **Files.** None.

<a id="r47"></a>
### R47 · 2026-09-13 · Publication plan: allowlist export to a fresh repository
*Decision · partly superseded (→ R48)*
- **Problem.** The development history contained private working files and commit metadata that break my
  authorship rules; rewriting that history would be hard to prove clean.
- **Method.** Compared rewriting history with exporting an allowlist of files into a new repository.
- **Result.** Fresh repository from an allowlisted export; working notes stay private and become this
  English log, covering every session evenly so the narrative is not a best-of-K selection; frozen deep
  dives with appended errata; MIT license for code only; the reversal deep dive is grounded in published
  reversal literature. The machine-readable result archive was deferred, so portfolio numbers are
  transcribed from run output with a provenance sheet.
- **Overturned.** The writing workflow was replaced by R48.
- **Files.** `LICENSE`, `docs/`.

<a id="r48"></a>
### R48 · 2026-09-13 · Writing workflow: edit the public copy directly
*Decision · holds*
- **Problem.** The drafting workflow from R47 assumed the writer could not see the repository.
- **Method.** Replaced it with direct editing of a public working copy under a written task list: setup,
  deletions, code hygiene, writing, verification, publish.
- **Result.** Three issues found while writing the list: the public copy needs its own working rules;
  code comments pointed to private note entries and must be repointed to this log; a checkpoint path
  pointed to a private scratch directory.
- **Overturned.** Nothing.
- **Files.** None.

<a id="r49"></a>
### R49 · 2026-09-13 · Task-list review: agreement is not evidence
*Decision · holds*
- **Problem.** Who should do which publication task, and is the task list itself correct?
- **Method.** Collected independent reviews of the task list.
- **Result.** The reviewers agreed on most assignments, but each said its view came from the same prior.
  Agreement between estimators with a shared bias adds little information, the same structure as two
  highly correlated seeds in R37. The reviews found ordering bugs: a code-comment task scheduled before
  its target existed; an acceptance check that referenced a list defined later; authors auditing their own
  work (now a rule: author ≠ auditor); a task that needed write access nobody had; and commit metadata,
  now blocked by a commit hook. I decided against a pilot, and to publish from a single fresh commit.
- **Overturned.** Nothing.
- **Files.** None.

---

## References

- Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). Ensemble selection from libraries of models. *ICML*.
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3).
- Kakushadze, Z. (2016). 101 Formulaic Alphas. *Wilmott*, 2016(84). arXiv:1601.00991.
- Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics*, 105(1).
- Xia, F., Liu, T.-Y., Wang, J., Zhang, W., & Li, H. (2008). Listwise approach to learning to rank: theory and algorithm. *ICML*.
- Zhang, Z., Zohren, S., & Roberts, S. (2020). Deep learning for portfolio optimization. arXiv:2005.13665.
