# Results provenance

This sheet lists every number the public documents use. Each row gives the value, what it measures,
the market and period, the seed count and spread, and the source: a result file shipped in this
repository, or the exact script and arguments that print it. Terms follow [`glossary.md`](glossary.md).

**Rules for the other documents.**
- Quote numbers only from this sheet, and cite the row ID when it helps (e.g. `P1`).
- Section X lists values I once reported and later corrected. They may appear only as the thing that
  was corrected, next to the corrected value.
- Section U lists values I believe but cannot yet tie to a shipped result file or script run. Do not
  quote them as numbers until they are re-run and moved into a normal section.

**How the numbers were checked.** Walk-forward RankICs are read straight from the result files
(`model/walkforward*.jsonl`, `model/pool_corr_results.json`, `model/epoch_rule_results.json`). All
decile, portfolio, null and bootstrap numbers come from post-processing scripts that read stored
predictions. I re-ran each of those commands on 2026-09-13 against the stored prediction files and
copied the printed values here. Those scripts print to the terminal only, so for now these values are
transcribed from run output; a machine-readable archive is planned. Prediction and label files are
data and are not in the repository. Section 0 shows how to regenerate them.

**Conventions.**
- Returns are in basis points (bps) per day, using the forward return `label_raw` (enter T+1, exit T+2:
  one trading day of holding).
- IR values are the naive annualized ratio (see glossary).
- Unless a row says otherwise, "GBDT" is LightGBM with the fixed parameters in `model/walkforward.py`, and
  the CN label is `vwap × entry`.
- "fold mean" and "pooled daily mean" are the two RankIC averages defined in the glossary.
- Every value is tier C (measurement) unless marked tier A or B.

---

## 0. Inputs: how the stored files are produced

Run commands from the repository root. File names are the ones the evaluation commands below expect.
Environment, runtimes and hardware notes are in [`REPRODUCE.md`](REPRODUCE.md).

| ID | File | Command |
|---|---|---|
| I1 | `dataset/alphaFactor/dataset_alpha158.parquet` (US) | `python dataset/rawdata/download.py --tickers-file dataset/rawdata/universe_full.txt --out dataset/rawdata`, then `python dataset/alphaFactor/build_dataset.py --market us` |
| I2 | `dataset/alphaFactor/dataset_alpha158_cn.parquet` (CN) | `python dataset/rawdata_cn/download_cn.py`, then `python dataset/alphaFactor/build_dataset.py --market cn` |
| I3 | `dataset/alphaFactor/label_<market>_<price>_<gate>.parquet` | `python dataset/alphaFactor/make_vwap_label.py --market <cn\|us> --price <close\|vwap\|hlc3> --gate <full\|entry\|none>` (check first with `--verify`) |
| I4 | `model/preds_walkforward_cn_<tag>.parquet` (GBDT, CN) | `python model/walkforward.py --data dataset/alphaFactor/dataset_alpha158_cn.parquet --label-file dataset/alphaFactor/label_cn_<price>_<gate>.parquet --seed <s> --tag _<tag>`. Tags used: `_closefull`, `_closeentry`, `_vwapfull`, `_hlc3` (label `cn_hlc3_full`), `_vwapentry` (seed 0), `_vwapentry_s1`, `_vwapentry_s2`. |
| I5 | `model/preds_walkforward_us_hlc3.parquet` (GBDT, US) | `python model/walkforward.py --data dataset/alphaFactor/dataset_alpha158.parquet --label-file dataset/alphaFactor/label_us_hlc3_none.parquet --tag _hlc3`. With no `--label-file` and no tag the output is `model/preds_walkforward_us.parquet` (built-in close label). |
| I6 | `model/preds_wf_mlp_cn_vwapentry_s<s>.parquet` (MLP, CN) | `python model/walkforward_mlp.py --data dataset/alphaFactor/dataset_alpha158_cn.parquet --label-file dataset/alphaFactor/label_cn_vwap_entry.parquet --seed <s> --tag _vwapentry_s<s>` (defaults: soft-IC loss, hidden 256-128-64-32, lr 1e-3, weight decay 1e-3, dropout 0.1, 1 day per batch, up to 100 epochs, patience 15) |
| I7 | `model/preds_wf_mlp_us_hlc3.parquet` (MLP, US) | `python model/walkforward_mlp.py --data dataset/alphaFactor/dataset_alpha158.parquet --label-file dataset/alphaFactor/label_us_hlc3_none.parquet --tag _hlc3` |
| I8 | `model/preds_wf_mlp_cn_epsel_s<s>.parquet` + per-epoch checkpoints | `python model/walkforward_mlp.py --data dataset/alphaFactor/dataset_alpha158_cn.parquet --label-file dataset/alphaFactor/label_cn_vwap_entry.parquet --seed <s> --tag _epsel_s<s> --epoch-dump <ckpt-dir>` |
| I9 | `model/preds_wf_ridge_cn_vwapentry.parquet` (ridge, CN) | `python model/walkforward_ridge.py --data dataset/alphaFactor/dataset_alpha158_cn.parquet --label-file dataset/alphaFactor/label_cn_vwap_entry.parquet --tag _vwapentry` |
| I10 | `model/pool_corr_results.json` | `python model/pool_corr.py --data dataset/alphaFactor/dataset_alpha158_cn.parquet --headline-label dataset/alphaFactor/label_cn_vwap_entry.parquet --add ridge_vwapentry=model/preds_wf_ridge_cn_vwapentry.parquet` |
| I11 | `model/epoch_rule_results.json` | `python model/epoch_rule.py --ckpt-dir <ckpt-dir> --tag-glob '_epsel_s*' --data dataset/alphaFactor/dataset_alpha158_cn.parquet --label-file dataset/alphaFactor/label_cn_vwap_entry.parquet` |

Each run of the three walk-forward scripts appends one record to its result file, identified by
`time`, `market`, `label_file`, and `seed`. The rows below name the record they use.

---

## D. Data and setup

| ID | Value | What it measures | Market · period | Source |
|---|---|---|---|---|
| D1 | 1,298 of 1,335 tickers downloaded (37 failed) | US universe after download | US · 2018–2025 | `dataset/rawdata/manifest.csv`, column `status` |
| D2 | 1,313 of 1,377 tickers downloaded; 64 failed (4.6%) | CN universe after download; the failures behave like delisted names (survivorship bias) | CN · 2018–2025 | `dataset/rawdata_cn/manifest_cn.csv`, column `status`; `dataset/rawdata_cn/universe_full_cn.txt` (1,377 lines) |
| D3 | 2018-01-02 to 2025-12-29 | First and last label dates in both markets | CN, US | `date` column of the label files (I3) |
| D4 | 6 folds, test years 2020–2025; expanding window; 120 validation days; embargo 2 days | Walk-forward setup | CN, US | `window`, `val_days`, `embargo`, `results[].test_year` in every record of `model/walkforward*.jsonl` |
| D5 | 1,453 test days (CN); 1,506 test days (US) | Days evaluated | 2020–2025 | Printed by `python model/evaluate.py --preds model/preds_walkforward_cn_vwapentry.parquet` / `--preds model/preds_walkforward_us_hlc3.parquet` |
| D6 | 1,814,693 rows (CN, `vwap × entry`); 1,841,554 rows (US) | Stock-days with a test prediction | 2020–2025 | Same commands as D5 |
| D7 | 20 rows (0.0011%) | US rows removed by the evaluation cap `\|label_raw\| ≤ 0.8` | US · 2020–2025 | Same US command as D5 |

## L. Labels

| ID | Value | What it measures | Market · period | Seeds | Source |
|---|---|---|---|---|---|
| L1 | 4.55% (US); 5.42% (CN) | Share of rows with `\|label\| > 2.99`, i.e. the ±3 clip binds (fat tails, not a bug) | Full panels 2018–2025, built-in close labels | — | `label` column of I1 / I2: `(df.label.abs() > 2.99).mean()` over non-missing rows |
| L2 | 0.1685 (US close) vs 0.0259 (CN close × full), 6.5x | Standard deviation of `label_raw`. The US label has no return filter, so extreme corporate-action rows stay in | Full panels 2018–2025 | — | `label_raw` column of I1 / I2: `df.label_raw.std()` |
| L3 | 0.0229 (CN vwap × entry); 0.1461 (US hlc3 × none) | Same statistic for the current labels | Full panels 2018–2025 | — | `label_raw` of `label_cn_vwap_entry.parquet` / `label_us_hlc3_none.parquet` |
| L4 | +0.0236, 6/6 folds positive, worst fold +0.0061 | GBDT RankIC with the original CN label `close × full` (fold mean) | CN · 2020–2025 | 1 | `model/walkforward.jsonl`, record `2026-08-30 21:24:31` (`label_cn_close_full`); same values in the built-in-label record `2026-08-30 17:37:06` |
| L5 | +0.0278 | GBDT RankIC, `close × entry` (fold mean) | CN · 2020–2025 | 1 | `model/walkforward.jsonl`, record `2026-08-30 21:30:25` |
| L6 | +0.0369 | GBDT RankIC, `vwap × full` (fold mean) | CN · 2020–2025 | 1 | `model/walkforward.jsonl`, record `2026-08-30 21:21:22` |
| L7 | +0.0409 (seed 0) | GBDT RankIC, `vwap × entry` (fold mean; pooled daily mean also +0.0409) | CN · 2020–2025 | 1 | `model/walkforward.jsonl`, record `2026-08-30 21:32:20`; see M1 for 3 seeds |
| L8 | +57% (0.0369 / 0.0236 = 1.57x) | RankIC gain from switching the price close → VWAP with the gate fixed at `full` (L6 vs L4, seed 0). Unrounded, fold mean and pooled daily mean both give +56.6% | CN · 2020–2025 | 1 | Derived from L4, L6 |
| L9 | +0.0407 | GBDT RankIC, `hlc3 × full` (VWAP proxy check on CN) | CN · 2020–2025 | 1 | `model/walkforward.jsonl`, record `2026-08-30 21:52:54` |
| L10 | −0.0012, 2/6 folds positive | GBDT RankIC trained and tested on a permuted `vwap × full` label (control: a label with no link to the features gives no signal) | CN · 2020–2025 | 1 | `model/walkforward.jsonl`, record `2026-08-30 21:45:38` (`label_cn_vwap_full_PERM`). The permuted label file was made outside the shipped scripts |
| L11 | ICIR +0.202 (close × full); +0.315 (vwap × full); +0.335 (vwap × entry) | Daily RankIC mean / sd | CN · 2020–2025 | 1 | `python model/evaluate.py --preds model/preds_walkforward_cn_<closefull\|vwapfull\|vwapentry>.parquet` |
| L12 | net@10 +0.15 (gross +11.14, turnover 110.0%) | Long-short, no smoothing, `close × full` | CN · 2020–2025 | 1 | `python model/evaluate.py --preds model/preds_walkforward_cn_closefull.parquet` |
| L13 | net@10 +12.72 (gross +23.63, turnover 109.2%) | Long-short, no smoothing, `vwap × full` | CN · 2020–2025 | 1 | `python model/evaluate.py --preds model/preds_walkforward_cn_vwapfull.parquet` |
| L14 | net@10 +18.98, IR@10 +3.02 | Long-short `ema5/exit10`, `vwap × full` predictions | CN · 2020–2025 | 1 | `model/pool_corr_results.json`, `strength.gbdt_vwapfull.own` (on these rows `label_raw` is identical under both gates) |
| L15 | D10 +7.34, D1 −16.29, D10−D1 +23.63; universe mean +2.30 | Decile returns, `vwap × full` | CN · 2020–2025 | 1 | Deciles: L13 command. Universe mean: `python model/longonly_study.py --preds model/preds_walkforward_cn_vwapfull.parquet --spans 1 --exits 0.10` (column "benchmark") |
| L16 | 79% short leg / 21% long leg | Split of D10−D1 against the universe mean: (mean − D1) = +18.59, (D10 − mean) = +5.04 | CN · 2020–2025, `vwap × full` | 1 | Derived from L15 |
| L17 | +3.98 (close) vs +0.44 (VWAP) → 3.54 bps per day ≈ 890 bps/year (× 252) | Upward bias in the universe-mean return from a close-price label (noise in the denominator), with the gate held at `entry` | CN · 2020–2025 | — | "benchmark" column of `python model/longonly_study.py --preds model/preds_walkforward_cn_closeentry.parquet --spans 1 --exits 0.10` of the P3 command |

## P. Headline portfolios (CN, GBDT, `vwap × entry`, 2020–2025)

| ID | Value | What it measures | Seeds / spread | Source |
|---|---|---|---|---|
| P1 | gross +27.50; turnover 43.2%; net@5 +25.34; **net@10 +23.19**; net@20 +18.87; **IR@10 +3.50** | Long-short `ema5/exit10` (assumes shorting) | seed 0 | `python model/turnover_study.py --preds model/preds_walkforward_cn_vwapentry.parquet` (row `ema5/exit10`); also `model/pool_corr_results.json` `strength.gbdt_vwapentry_s0.own` |
| P2 | net@10 +23.19 / +20.82 / +23.78 → mean +22.59, seed sd 1.57; IR@10 mean +3.48 | P1 over 3 GBDT seeds | 3 seeds (2 df) | `model/pool_corr_results.json` `strength.gbdt_vwapentry_s{0,1,2}.own` |
| P3 | active +4.13; turnover 10.2%; **net@10 +3.11**; **IR@10 +1.19** | Long-only `ema5/exit30` (no shorting) | seed 0 | `python model/longonly_study.py --preds model/preds_walkforward_cn_vwapentry.parquet` (row `ema5/exit30`) |
| P4 | net@10 +3.11 / +2.55 / +3.61 → mean +3.09, seed sd 0.53 | P3 over 3 GBDT seeds | 3 seeds (2 df) | `python model/longonly_study.py --preds model/preds_walkforward_cn_vwapentry{,_s1,_s2}.parquet --spans 5 --exits 0.30` |
| P5 | `ema10/exit30` net@10 +3.15 (IR +1.18) vs `ema5/exit30` +3.11 (IR +1.19) | On the full-sample long-only grid, `ema5/exit30` is best by IR@10, not by net@10 | seed 0 | P3 command, all 12 rows |
| P6 | gross +27.93; turnover 101.4%; net@10 +17.79; IR@10 +2.87 | Long-short with no smoothing and plain deciles (`evaluate.py` construction) | seed 0 | `python model/evaluate.py --preds model/preds_walkforward_cn_vwapentry.parquet` |
| P7 | gross +28.10; turnover 101.3%; net@10 +17.97; IR@10 +2.89 | Grid baseline `ema1/exit10` (`turnover_study.py` construction; decile edges differ slightly from P6) | seed 0 | P1 command, row `ema1/exit10` |
| P8 | D1 −22.61 … D10 +5.31; D10−D1 +27.93; monotonicity +0.988 | Decile returns | seed 0 | P6 command |
| P9 | RankIC [+0.0334, +0.0486]; LS net@10 [+16.97, +29.49]; LS IR@10 [+2.56, +4.50]; LO net@10 [+0.78, +5.50]; LO IR@10 [+0.29, +2.12]. The RankIC, LS net@10 and LO net@10 intervals also exclude 0 at block lengths 5, 10, 40 | 95% block-bootstrap intervals (L = 20, B = 2,000, bootstrap seed 0) | seed 0; sample-period uncertainty | `python model/block_bootstrap.py --preds model/preds_walkforward_cn_vwapentry.parquet` |

## N. Negative control (US)

| ID | Value | What it measures | Period | Seeds | Source |
|---|---|---|---|---|---|
| N1 | +0.0033, 4/6 folds positive, worst −0.0021 | GBDT RankIC, close label, after the fit/validation embargo fix (fold mean) | 2020–2025 | 1 | `model/walkforward.jsonl`, record `2026-08-30 21:48:49` (`label_us_close_none`); same values in built-in-label record `2026-08-30 16:05:06` |
| N2 | +0.0071, 5/6 folds positive; ICIR +0.054 | GBDT RankIC, `hlc3 × none` label | 2020–2025 | 1 | `model/walkforward.jsonl`, record `2026-08-30 21:50:12`; ICIR from `python model/evaluate.py --preds model/preds_walkforward_us_hlc3.parquet` |
| N3 | D10−D1 +3.13; monotonicity +0.491; turnover 124.4%; long-short net@10 (no smoothing) −9.30 | Deciles and plain long-short, `hlc3` label, cap 0.8 | 2020–2025 | 1 | `python model/evaluate.py --preds model/preds_walkforward_us_hlc3.parquet --cap-scan` |
| N4 | D10−D1 +15.89 (no cap), +3.13 (0.8), +3.42 (0.5), +4.03 (0.2); RankIC +0.0071–0.0072 at every cap | Sensitivity to the cap threshold | 2020–2025 | 1 | N3 command, `cap-scan` block |
| N5 | net@10 −2.39, IR@10 −0.32 | Long-short `ema5/exit10` (the variant chosen on CN, applied to US) | 2020–2025 | 1 | `python model/turnover_study.py --preds model/preds_walkforward_us_hlc3.parquet` |
| N6 | best `ema10/exit20`: net@10 +0.59, IR@10 +0.09; 6 of 24 variants have net@10 > 0 | US's own best long-short variant | 2020–2025 | 1 | N5 command, all 24 rows |
| N7 | ≈ 40x (+23.19 / +0.59) | CN headline vs the best US variant | 2020–2025 | 1 | Derived from P1, N6 |
| N8 | active +3.18; net@10 +1.67; IR@10 +0.43 | Long-only `ema5/exit30` | 2020–2025 | 1 | `python model/longonly_study.py --preds model/preds_walkforward_us_hlc3.parquet` |
| N9 | RankIC [+0.0007, +0.0134]; LS net@10 [−8.05, +4.08]; LS IR@10 [−1.18, +0.50]; LO net@10 [−1.40, +5.13]; LO IR@10 [−0.37, +1.24] | 95% block-bootstrap intervals (L = 20, B = 2,000). All four portfolio intervals contain 0; the RankIC interval does not | 2020–2025 | 1 | `python model/block_bootstrap.py --preds model/preds_walkforward_us_hlc3.parquet` |
| N10 | close $0.12 (2020-11-19) → $31.00 (2020-11-20) with the adjustment factor unchanged (0.6133); `label_raw` +257.3 (+25,733%) on 2020-11-18, the largest absolute value in the US panel | One corporate-action price splice (ticker CHRD, bankruptcy and relisting) that inflated every uncapped US bps number | US · 2020 | — | `close_adj`, `raw_close`, `adjustFactor`, `label_raw` of I1 for `instrument == "CHRD"` |

## T. Tests of the selection procedure and of significance

| ID | Value | What it measures | Market · period | Seeds | Source |
|---|---|---|---|---|---|
| T1 | full-sample best `ema5/exit10` +22.39 (IR +3.43); nested reselection +21.76 (IR +3.34); fixed `ema1/exit10` +17.00 (IR +2.78) | Long-short net@10 over the same years, three ways | CN · 2021–2025 (the first test year has no prior year) | seed 0 | `python model/selection_null.py --preds model/preds_walkforward_cn_vwapentry.parquet --mode nested --legs longshort` |
| T2 | **selection bias +0.64 bps** (≈3% of +22.39); nested picks `ema5/exit10` in 4 of 5 years (2021 picks `ema2/exit10`) | What choosing the variant added | CN · 2021–2025 | seed 0 | T1 command |
| T3 | +4.76 bps (IR +0.56) | Honest turnover-reduction gain, long-short (nested − fixed `ema1/exit10`) | CN · 2021–2025 | seed 0 | T1 command |
| T4 | full-sample best `ema10/exit30` +2.20; nested +1.78; fixed −0.82; selection bias +0.42; turnover gain +2.59 (IR +0.90) | Same three-way test, long-only grid | CN · 2021–2025 | seed 0 | `python model/selection_null.py --preds model/preds_walkforward_cn_vwapentry.parquet --mode nested --legs long` |
| T5 | nested +8.78 bps (IR +1.63) long-short; +4.45 (IR +1.11) long-only; US nested long-short −2.67, long-only −0.35 | Turnover-reduction gain on the negative control (larger than CN, because US turnover is higher: N3 124.4% vs P6 101.4%). This is cost arithmetic, not alpha | US · 2021–2025 | 1 | `python model/selection_null.py --preds model/preds_walkforward_us_hlc3.parquet --mode nested --legs longshort` and `--legs long` |
| T6 | observed best-of-24 +23.19; null mean −0.67, sd 0.47, 95th percentile **+0.06**; p = 0.005 (0 of 200) | Selection permutation null: could best-of-grid on pure noise reach the headline level? | CN · 2020–2025 | seed 0; 200 permutations | `python model/selection_null.py --preds model/preds_walkforward_cn_vwapentry.parquet --mode permute --legs longshort --n 200` |
| T7 | observed gap +5.22 vs null gap mean +9.40 (95th percentile +10.42) | "best − fixed baseline" under the same null. Cost arithmetic makes it larger with no signal, so a gap cannot measure selection | CN · 2020–2025 | seed 0 | T6 command, reference row |
| T8 | net@10 null mean −1.01, sd 0.35, observed +3.11, z = +11.9 (0 of 2,000) | Within-day permutation null for long-only `ema5/exit30` | CN · 2020–2025 | seed 0 | `python model/longonly_null.py --preds model/preds_walkforward_cn_vwapentry.parquet --span 5 --exit 0.30 --n 2000` |
| T9 | net@10 null mean −1.50, sd 0.35, observed +1.67, z = +9.2 (0 of 2,000) | The same null passes on the negative control, so it is not evidence of significance | US · 2020–2025 | 1 | `python model/longonly_null.py --preds model/preds_walkforward_us_hlc3.parquet --span 5 --exit 0.30 --n 2000` |
| T10 | MLP − GBDT (seed 0 each): RankIC +0.0043 [−0.0008, +0.0094]; LS net@10 −3.74 [−7.87, +0.51]; LO net@10 +0.27 [−1.88, +2.45] | Paired block bootstrap of the model difference (same resampled days) | CN · 2020–2025 | seed 0 each | `python model/block_bootstrap.py --preds model/preds_walkforward_cn_vwapentry.parquet:GBDT_s0 --preds model/preds_wf_mlp_cn_vwapentry_s0.parquet:MLP_s0 --paired` |
| T11 | LS net@10 GBDT − MLP = +2.20, SE 1.02 → **2.15σ**; IR@10 +0.65, SE 0.12 → 5.2σ | Seed-level model difference: difference of 3-seed means / √(sd²_GBDT/3 + sd²_MLP/3) | CN · 2020–2025 | 3 + 3 seeds | Derived from P2 and M4 (`model/pool_corr_results.json`) |

## M. Models

| ID | Value | What it measures | Market · period | Seeds / spread | Source |
|---|---|---|---|---|---|
| M1 | +0.0409 / +0.0383 / +0.0408 → **+0.0400 ± 0.0015** (seed sd); pooled within-fold sd 0.0023 | GBDT RankIC, `vwap × entry` (fold means) | CN · 2020–2025 | 3 seeds; 12 df for pooled | `model/walkforward.jsonl`, records `2026-08-30 21:32:20` (seed 0), `23:05:12` (seed 1), `23:07:37` (seed 2) |
| M2 | +0.0452 / +0.0481 / +0.0466 → **+0.0466 ± 0.0014** (seed sd); pooled within-fold sd 0.0032 | MLP RankIC, `vwap × entry` (fold means) | CN · 2020–2025 | 3 seeds | `model/walkforward_mlp.jsonl`, records `2026-08-30 23:25:27` (seed 0), `23:11:36` (seed 1), `23:09:56` (seed 2) |
| M3 | 6 of 6 folds; two-sided sign test p = 0.031 | Folds where the 3-seed MLP mean beats the 3-seed GBDT mean | CN · 2020–2025 | 3 + 3 seeds | Derived from M1, M2 (per-fold `test_rankic`) |
| M4 | LS net@10 +19.45 / +20.96 / +20.78 → mean +20.40, seed sd 0.83 (IR@10 mean +2.83); LO net@10 +3.38 / +4.18 / +4.32 → mean +3.96, seed sd 0.51 | MLP portfolios at the frozen variants | CN · 2020–2025 | 3 seeds | LS: `model/pool_corr_results.json` `strength.mlp_vwapentry_s{0,1,2}.own`. LO: `python model/longonly_study.py --preds model/preds_wf_mlp_cn_vwapentry_s<s>.parquet --spans 5 --exits 0.30` |
| M5 | MLP D10 +6.17, D1 −20.45 (3-seed means) vs GBDT D10 +5.31, D1 −22.61 (seed 0) | Where each model earns: MLP is better at the top, GBDT at the bottom | CN · 2020–2025 | 3 / 1 | `python model/evaluate.py --preds model/preds_wf_mlp_cn_vwapentry_s<s>.parquet`; GBDT from P8 |
| M6 | US MLP +0.0108 vs GBDT +0.0071; MLP wins 5 of 6 folds | The same model comparison on the negative control (so an MLP advantage is not specific to CN) | US · 2020–2025 | 1 seed each, no error bar | `model/walkforward_mlp.jsonl` record `2026-08-30 23:26:33`; N2 |
| M7 | US long-only net@10: MLP +0.62 vs GBDT +1.67 | Long-only on the negative control | US · 2020–2025 | 1 | `python model/longonly_study.py --preds model/preds_wf_mlp_us_hlc3.parquet --spans 5 --exits 0.30`; N8 |
| M8 | +0.0434, 6/6 folds positive, worst +0.0310 (pooled daily +0.0433); LS net@10 +17.91, IR@10 +2.30 | Ridge RankIC and long-short portfolio | CN · 2020–2025 | deterministic | `model/walkforward_ridge.jsonl` record `2026-09-06 20:43:23`; `model/pool_corr_results.json` `strength.ridge_vwapentry` |
| M9 | fold-mean seed sd 0.00143 (runs of 2026-08-30) vs 0.00031 (same config, runs of 2026-09-06) | Two estimates of the same 3-seed sd differ by 4.6x, so a 2-df sd is unreliable | CN · 2020–2025 | 3 + 3 seeds | M2 records vs `model/walkforward_mlp.jsonl` records `2026-09-06 20:15:28`, `20:25:49`, `20:35:38` |
| M10 | argmax: mean +0.0468, pooled within-fold sd 0.00306 | MLP epoch selection by best validation RankIC (baseline rule) | CN · 2020–2025 | 3 seeds | `model/epoch_rule_results.json`, `rules.argmax` (pooled sd computed from `per_fold` as defined in the glossary) |
| M11 | pooled within-fold sd: wavg_run_k10 0.00165, wavg_top_k3 0.00198, wavg_top_k5 0.00198, wavg_run_k5 0.00206, wavg_top_k10 0.00209, wavg_run_k3 0.00276, wavg_best_k10 0.00302, wavg_best_k5 0.00332, wavg_best_k3 0.00402, plateau_med_z2 0.00429, plateau_med_z0.5 0.00540, plateau_med_z1 0.00803 | Seed spread under alternative epoch rules | CN · 2020–2025 | 3 seeds | `model/epoch_rule_results.json`, `rules.<rule>.per_fold` |
| M12 | mean RankIC wavg_top_k3 +0.0469, k5 +0.0468, k10 +0.0465 vs argmax +0.0468; the other nine rules +0.0403 to +0.0451 | Only `wavg_top_k*` keeps the mean RankIC. No rule is adopted (choosing among 13 is itself a selection procedure) | CN · 2020–2025 | 3 seeds | `model/epoch_rule_results.json`, `rules.<rule>.mean_rankic` |

## E. Pool, correlation, and the common-row-set check

| ID | Value | What it measures | Market · period | Source |
|---|---|---|---|---|
| E1 | 11 members; 1,808,949 common rows | Size of the correlation pool and of its common row set | CN · 2020–2025 | `model/pool_corr_results.json`, `members`, `n_common_rows` |
| E2 | GBDT seed 0 vs MLP seeds: ρ 0.569, 0.611, 0.621 | Daily rank correlation between the two model families | CN · 2020–2025 | `rho_raw_all` |
| E3 | GBDT seed–seed ρ 0.794–0.851; MLP seed–seed ρ 0.748–0.783 | Seed-only diversity | CN · 2020–2025 | `rho_raw_all` |
| E4 | ridge mean ρ to the other members 0.537 (lowest), with the lowest IR@10 (+2.30) | Least correlated member vs its strength | CN · 2020–2025 | `strength.ridge_vwapentry.mean_rho_raw`; M8 |
| E5 | mean off-diagonal ρ 0.6300 → 0.6253 after reversal orthogonalization (−0.005); ridge ↔ GBDT seed 0: 0.553 → 0.494 | Whether shared reversal exposure explains the low correlations (it does not, except for ridge) | CN · 2021–2025 subset | `rho_raw_sub` vs `rho_ortho` |
| E6 | 5,744 rows dropped of 1,814,693 (0.32%); mean `\|label_raw\|` 0.0816 vs 0.0160 overall (5.1x); median `label_raw` −0.085 | The rows the intersection removes from the headline member. The `full`-gate members lack exactly the limit-down exit-failure rows, so the dropped rows are large losses | CN · 2020–2025 | `(date, instrument)` keys of `model/preds_walkforward_cn_vwapentry.parquet` that are missing from any of the 11 member files listed in `members` (I4, I6, I9); `label_raw` of the same file |
| E7 | own rows: gross +27.50, net@10 +23.19; common rows: gross +21.74, net@10 **+17.30** (−5.9 bps), IR@10 +2.64 | Effect of measuring the headline on the intersection | CN · 2020–2025 | `strength.gbdt_vwapentry_s0.own` vs `.common` |
| E8 | mean R² 0.158 (GBDT seed 0), 0.194–0.215 (MLP seeds), 0.391 (ridge) | Share of each member's daily cross-sectional score variance explained by ROC5 reversal | CN · 2020–2025 common rows | `betas.<member>.mean_r2_full` |

## A. Audits and baselines

| ID | Value | What it measures | Market · period | Seeds | Source |
|---|---|---|---|---|---|
| A1 | RankIC +0.0268, ICIR +0.158; D10−D1 +19.70; net@10 +11.57 (IR +1.50) | Single-factor ROC5 reversal baseline through the same evaluation, `vwap × entry` rows | CN · 2020–2025 | — | `python model/evaluate.py --preds model/preds_walkforward_cn_vwapentry.parquet --baselines --data dataset/alphaFactor/dataset_alpha158_cn.parquet` |
| A2 | 1.42x (+27.93 / +19.70) | Model decile spread over the ROC5 spread | CN · 2020–2025 | seed 0 | Derived from P8, A1 |
| A3 | D10−D1 +23.63 → **+20.19** (−3.44) with a short-entry gate (`vwap × full`); +11.14 → +9.78 (−1.36) for `close × full`. Short-leg untradable share is highest in D1 (3.08% for `vwap × full`) | Cost of gating the short leg's entry (cannot sell short on a limit-down T+1) | CN · 2020–2025 | 1 | `python model/limit_audit.py --preds model/preds_walkforward_cn_vwapfull.parquet` and `--preds model/preds_walkforward_cn_closefull.parquet` |
| A4 | mean β +0.373; mean R² **0.159**; mean Spearman +0.266; β > 0 on 98.6% of days | Exposure of the headline GBDT score to ROC5 reversal (daily cross-sectional regression) | CN · 2020–2025 | seed 0 | `python model/reversal_exposure.py --preds model/preds_walkforward_cn_vwapentry.parquet --data dataset/alphaFactor/dataset_alpha158_cn.parquet` |
| A5 | long-short `ema5/exit10` net@10 +22.20 (raw) → +19.03 after orthogonalization with nested β: **−3.18 bps (14.3%)**; leaky full-sample β +18.88; pure reversal score alone +13.11 (IR +1.75) | Cost of removing the reversal component (the first year has no prior year to estimate β, so it is dropped) | CN · 2021–2025 | seed 0 | A4 command |

---

## X. Superseded values (quote only as the thing that was corrected)

| ID | Earlier value | Why it was wrong | Corrected by |
|---|---|---|---|
| X1 | "~0.01 IC ceiling": GBDT best validation RankIC +0.0105; MLP (MSE) +0.0125, which ends at +0.0035 | Tier A selection scores (`model/experiments.jsonl` `best_val_rankic` / `final_val_rankic`; `model/horizon_sweep.jsonl`) | N1 (tier C US +0.0033); L4 (CN +0.0236) |
| X2 | US walk-forward +0.0046 (first run, fold range [−0.0037, +0.0113]) and +0.0030 (re-run, range [−0.0038, +0.0113]) | No embargo between fit and validation segments (`model/walkforward.jsonl` records `2026-07-08 15:14:39`, `2026-08-30 14:51:16`) | N1: +0.0033, range [−0.0021, +0.0071] |
| X3 | "Features are too weak" | True only for US; the same features give CN +0.0236 | L4 |
| X4 | "CN is 7x more predictable than US" (+0.0236 / +0.0033) | Not comparable: labels, gates, and universes differ. Do not use | — |
| X5 | "A 10 bps cost wipes out the CN signal": net@10 +0.15 | Close-price label noise | L13 (+12.72), P1 |
| X6 | US D10−D1 −12.40, monotonicity +0.067, highest return in the lowest decile | One uncapped price splice (N10) in the close-label deciles (`python model/evaluate.py --preds model/preds_walkforward_us.parquet --cap 0`) | N3 (+3.13, +0.491) |
| X7 | "About 70% of the spread comes from the short leg" | Rough read | L16 (79%) |
| X8 | Close-label benchmark inflation ≈ 500 bps/year: +6.15 vs +2.30 | Compared across gates (`close × full` vs `vwap × full`, `longonly_study.py --spans 1 --exits 0.10`), and annualized × 126 as in X19 | L17 (≈ 890, same gate) |
| X9 | Long-only within-day null "p < 0.001" read as significance | The negative control passes the same test | T8, T9 |
| X10 | US long-short net@10 −2.39 "never positive" | That is CN's chosen variant applied to US | N6 |
| X11 | ROC5 reaches 76% of the model RankIC (+0.0179 / +0.0236); spread 1/2.8 (+3.97 vs +11.14) | Close-label numbers (`evaluate.py --baselines` on `preds_walkforward_cn_closefull.parquet`) | A1, A2 (1.42x; ROC5 net@10 +11.57) |
| X12 | Headline RankIC +0.0409 | Seed 0 is the top of three seeds | M1 (+0.0400 ± 0.0015) |
| X13 | ICIR +0.315 for `vwap × entry` | Copied from the `vwap × full` row | L11 (+0.335) |
| X14 | GBDT beats MLP long-short by "more than ten seed sds" | Used MLP's sd only, with GBDT at one seed | T11 (2.15σ), T10 (interval contains 0) |
| X15 | "best − fixed baseline" measures selection bias | Cost arithmetic (T7) | T2 (nested: +0.64) |
| X16 | 3-seed sd as an acceptance threshold | 2 df; the same quantity measured 0.00143 and 0.00031 | M9, M10, M11 (pooled within-fold sd) |
| X17 | Headline measured on the common row set: net@10 +17.30 | Intersection drops loss-making tail rows | E6, E7 (own rows +23.19) |
| X18 | Expected share of `\|label\| > 2.99` about 1–2% | Guess before measuring | L1 (4.55%) |
| X19 | Close-label benchmark inflation ≈ 450 bps/year (3.54 bps × 126) | Treated each daily observation as a two-day period. `label_raw` runs from T+1 to T+2, one trading day, so a daily mean annualizes × 252 | L17 (≈ 890) |

## U. Not yet sourced (do not quote as numbers)

These values were measured once, but no shipped result file or script run reproduces them. The run was
unlogged, done with ad-hoc code, or used an aligned prediction copy that the shipped scripts do not produce.
To use one, re-run it, record the output, and move it to a normal section.

| ID | Claim | Missing piece |
|---|---|---|
| U1 | Of the 82 formulaic alphas in the published 101-alphas paper, 67 are computable on the CN data | Feasibility check not in the repository |
| U2 | `--batch-days k > 1` costs 0.002–0.006 RankIC; the same seed has cross-process sd 0.0005 (range 0.0011); repeats inside one process understate it about 4x | MLP runs with `--no-log` / `--only-year` were not logged |
| U3 | Nested learning-rate choice: paired lr 1e-4 − 1e-3 = −0.0002 ± 0.0019 (0.23σ) | lr 1e-4 walk-forward runs are not in `model/walkforward_mlp.jsonl` |
| U4 | MLP single fold +0.0378 vs GBDT +0.0300 (center of the noise band ≈ +0.0382) | Single-fold MLP runs not logged (the GBDT value is fold 2020 of L4) |
| U5 | GBDT hyperparameter scan: range 0.0015; seed ensembles add nothing once EMA smoothing is applied | `model/tune_gbdt.py` and ensemble outputs not logged |
| U6 | Change of label measurement vs model improvement: 37% / 63% | Ad-hoc decomposition |
| U7 | A portfolio-PnL loss puts 0.71x of Soft IC's gradient share on the tails | Ad-hoc gradient check |
| U8 | Turnover confound in the model comparison: 0.24 bps (6% of the gap) | Ad-hoc check |
| U9 | Equal-weight GBDT+MLP ensemble and meta-MLP numbers (e.g. meta beats equal-weight by +1.40 ± 0.50 long-short; long-short +20.50 vs GBDT +22.33; long-only +2.74 vs equal-weight +3.14) | Computed on aligned 2021–2025 prediction copies that the shipped scripts do not produce |
| U10 | Reversal audit on the aligned 2021–2025 GBDT copy: R² 0.148; removing reversal costs −2.06 bps (9.3%) | Same aligned copy. A4–A5 are the reproducible replacement and give a larger cost (−3.18 bps, 14.3%) |
| U11 | The splice ticker (N10) carries 58.5% of the uncapped US D10 return | Share not re-run |
| U12 | Smoke-set soft-IC +0.0216 | Smoke logs are not published |
