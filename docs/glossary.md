# Glossary

These are the fixed names I use in every public document in this repository. When a term has a
precise implementation, the definition follows the code (file named in brackets), not my notes.
Every number quoted in the docs is listed, with its source, in
[`results_provenance.md`](results_provenance.md).

## Markets, data, and folds

**Main market / negative control.**
China A-shares (CN) are the main market, where I develop and report. US large caps (US) are a
*negative control*: the same pipeline runs there, and its only job is to show that a market with
almost no signal gives almost no signal. **Rule:** I run every new test on the negative control too.
If the control also passes, the test cannot tell signal from no signal, and I do not use it as evidence.

**Universe.** The ticker lists in `dataset/rawdata/universe_full.txt` (US) and
`dataset/rawdata_cn/universe_full_cn.txt` (CN). Download outcomes per ticker are recorded in `manifest.csv` / `manifest_cn.csv`.
The US list is a snapshot of index members at the end of the sample. The CN list is the union of historical
CSI 300 / CSI 500 constituents, but some CN tickers failed to download. Both final panels therefore mostly
contain stocks that survived to the end of the sample (*survivorship bias*).

**Features.** Alpha158-style price/volume factors that I implemented myself
(`dataset/alphaFactor/features.py`). Each feature is a cross-sectional robust z-score clipped at ±3,
with missing values filled with 0 (`build_dataset.py`).

**Walk-forward fold.** One test calendar year Y from 2020 to 2025, six folds per run
(`model/walkforward.py`, and the `_mlp` / `_ridge` variants). The window is *expanding*: I train on all
years before Y (at least two). The last `--val_days 120` trading days of that window are the
validation segment, used only for early stopping (or for picking the ridge alpha). The test year is
predicted once and plays no part in any choice made inside the fold.

**Embargo.** The label looks two trading days ahead (see *label*), so I drop the last
`--embargo 2` days before each boundary. There are two such boundaries: training window → test year,
and fit segment → validation segment.

## Labels

**Label timing.** Features are known at the close of day T. The forward return is
`label_raw = P(T+2) / P(T+1) − 1`: enter on T+1, exit on T+2. It is a **one-day** return that starts one
day after the features are known, so consecutive days' returns cover back-to-back intervals and never
overlap. The label reaches two trading days past T, which is what the embargo protects. `label` is the same return turned into
a cross-sectional robust z-score, `(x − median) / (1.4826·MAD)`, clipped at ±3. Models train on
`label` and RankIC uses `label`. Anything reported in bps (deciles, portfolios, costs) uses `label_raw`.

**Label variant `price × gate`** (`dataset/alphaFactor/make_vwap_label.py`, files
`label_<market>_<price>_<gate>.parquet`):

| Part | Value | Meaning |
|---|---|---|
| price | `close` | Adjusted close price. |
| price | `vwap` | True VWAP = `amount / volume`, scaled by the same-day adjustment factor. CN only: the US source has no traded-amount field. |
| price | `hlc3` | Typical price `(high + low + close) / 3`, a VWAP proxy. On CN I use it only to check how good the proxy is; on US it is the best price I have. |
| gate | `full` | CN limit-up/limit-down gate: keep a row only if the stock can be bought on T+1 **and** sold on T+2. This was the original CN label. |
| gate | `entry` | Keep only the entry check (can be bought on T+1). |
| gate | `none` | No gate (US has no daily price limits). |

- **CN default label: `vwap × entry`.** The `full` gate also removed rows where the stock could not be sold on
  T+2 (limit-down). By then the position already exists and its loss is real, so dropping those rows
  cut off only the left tail of returns. `entry` puts those losses back.
- **CN return filter.** CN labels drop `|label_raw| > 0.8` (trading halts and resumption gaps).
- **US cap.** US labels have no such filter in the label itself. For every bps metric on US, the evaluation
  drops `|label_raw| > 0.8` (`--cap`, default 0.8 for US in `model/evaluate.py`). This removes corporate-action
  price splices. RankIC is rank-based and barely changes.

## Signal metrics

**RankIC.** For each day, the Spearman correlation between the model score and `label` across stocks.
The reported value is an average over days. Two averages appear, and they differ slightly (in the
fourth decimal):
- *fold mean*: the mean of the six per-year test RankICs recorded in the walk-forward logs
  (`model/walkforward*.jsonl`);
- *pooled daily mean*: the mean over all test days, printed by `model/evaluate.py` and stored in
  `model/pool_corr_results.json`.
The provenance sheet says which one each number is.

**ICIR.** Mean of the daily RankIC series divided by its standard deviation (`model/evaluate.py`,
not annualized). It measures how steady the signal is, not how large.

**Decile spread (D10−D1) and monotonicity.** Each day I sort stocks by score into ten equal groups
and average `label_raw` in each group, then average over days. D10−D1 is the top-minus-bottom
difference in bps. Monotonicity is the Spearman correlation between decile index (1–10) and decile mean return.

**Single-factor reversal baseline (ROC5).** `ROC5 = close(T−5) / close(T)`: a high value means the stock
fell recently. With its sign unchanged it is a short-term reversal score. I run it through the same
evaluation as the model (`model/evaluate.py --baselines`).

## Portfolio metrics

**Long-short.** Each day, equal-weight long the top decile and short the bottom decile, rebalanced
daily (`model/evaluate.py`, `model/turnover_study.py`). Gross return = mean `label_raw` of the long leg
minus that of the short leg. **Assumes shorting is possible**, which is hard for individual A-shares.

**Long-only (active).** Hold only the top decile, equal-weight. Active return = mean return of the held
stocks minus the equal-weight mean of the whole universe that day (`model/longonly_study.py`). Only the long
leg's turnover is charged. No shorting or leverage is needed.

**Turnover.** Per leg, `Σ|w_t − w_{t−1}| / 2`, the fraction of the leg's capital replaced that day.
Long-short turnover adds both legs. The first day (portfolio build) is not counted.

**net@c (e.g. net@10).** Mean daily net return in bps after charging `c` bps per unit of turnover:
`net_t = gross_t − turnover_t × c / 10⁴`. **net@10** is the headline cost level.

**IR@c.** `mean(net) / sd(net) × √252` for the same daily series. Each day's return covers one trading
day, so the √252 scaling fits the horizon, but it treats days as independent. EMA smoothing and the exit
band carry holdings across days, and market regimes persist, so daily returns can be autocorrelated;
positive autocorrelation makes this ratio overstate. That makes it *naive*: a rough guide only.

**Variant `emaS/exitQ`** (`model/turnover_study.py`). Two turnover controls applied to stored predictions:
- `emaS`: exponential moving average of each stock's score over time with span S (causal; `ema1` = no smoothing);
- `exitQ`: a no-trade band. Enter the top 10%, but leave only when a held stock drops out of the top Q%:
  holdings = (yesterday's holdings ∩ top Q) ∪ top 10%. `exit10` = no band.

The long-short grid has 24 variants (spans 1, 2, 3, 5, 10, 20 × exits 10–40%). The long-only grid has 12
(spans 1, 3, 5, 10 × exits 10–30%).

**Headline variants.** Long-short `ema5/exit10`; long-only `ema5/exit30`. I report both, each with its own
assumptions. I keep these same variants for every later model and never re-pick them per model, because
re-picking would give each model another free choice from the grid.

## Metric tiers (what kind of number is this?)

| Tier | Name | Definition | Counts as performance? |
|---|---|---|---|
| A | **selection score** | The best value seen during training, on the same validation segment used for early stopping (e.g. `best_val_rankic`, `val_rankic_SELECTION_SCORE`). | No. Taking the maximum of a noisy curve is biased upward, even with no signal. |
| B | **last-epoch validation** | The final value on the early-stopping validation segment (e.g. `final_val_rankic`). | No. It is less biased, but the same data chose the model. |
| C | **measurement** | Test-year value from walk-forward, predicted once and used in no choice (`test_rankic`). | Yes. |

All headline and finding numbers in the public docs are tier C unless a row says otherwise. Early
logs (`model/experiments.jsonl`, `model/horizon_sweep.jsonl`) contain only tiers A and B.

## Testing the selection procedure

**Selection procedure.** Any step that picks one option from several using the data: a variant from the grid,
a hyperparameter, an epoch rule, a stacker. The best of several correlated draws is biased upward.

**Nested reselection** (`model/selection_null.py --mode nested`). For each test year k from the
second year on, pick the variant with the best net@10 using only years before k, then apply it to year k.
Joining those years gives a number that is out of sample for the variant choice too.
**Selection bias** = (best variant chosen on the full sample) − (nested reselection). This is how I measure
what the choice itself added.

**Selection permutation null** (`model/selection_null.py --mode permute`). Shuffle `label_raw` across stocks
within each day, rescan the whole grid, keep the best variant, and repeat N times. This shows how good
"best of the grid" looks with no signal. **It tests levels only, never gaps.** Shuffling returns leaves holdings and turnover
unchanged, so without any signal slower variants still save costs. A gap such as "best − fixed baseline" is
driven by cost arithmetic, not skill.

**Within-day permutation null for one variant** (`model/longonly_null.py`). The same shuffle, applied to a single
fixed variant. Passing shows that stock selection is not random. It is **not** evidence of economic
significance, because the negative control passes too.

## Uncertainty

**Seed sd.** The standard deviation across training seeds of a fold-mean metric. It measures training
randomness. With 3 seeds it has only 2 degrees of freedom and is unreliable.

**Pooled within-fold sd.** For each fold, the across-seed variance of that fold's test RankIC. Average these
variances over the six folds and take the square root. With 3 seeds × 6 folds it has 12 degrees of freedom.
This is the seed spread I report.

**Block bootstrap CI** (`model/block_bootstrap.py`). Circular moving-block bootstrap over test days (main block
length 20 days, 2,000 resamples, percentile 95% interval). It measures sample-period uncertainty: would
the result hold on a different stretch of history? That is a different question from seed sd, and I never
merge the two. With `--paired`, both models use the same resampled days, so costs move together and
a *difference* between models can be tested.

## Pooling and correlation

**Own row set vs common row set** (`model/pool_corr.py`). The *common* row set is the intersection of rows
that every pool member has. I use it only for the correlation matrix, where each pair must use the same
sample. Strength (RankIC, net@10) is always measured on each member's *own* rows. The rows the
intersection drops are not random. Before trusting any number computed on an intersection, I check what
was dropped.

**Pairwise ρ.** The daily cross-sectional Spearman correlation between two members' scores, averaged over days.

**Reversal exposure** (`model/reversal_exposure.py`). A daily cross-sectional regression of the standardized
score on standardized ROC5: `pred_z = α + β·rev_z`. The *orthogonalized* score is `pred_z − β·rev_z`, with β
estimated on an expanding window (year k uses only earlier years).
