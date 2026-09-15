# Deep dive: is the model an expensive copy of short-term reversal?

*Written 2026-09-13. The body is frozen; corrections go to [Errata](#errata) at the end.*
*Research log entries: [R36](../research_log.md#r36), [R38](../research_log.md#r38),
[R40](../research_log.md#r40), [R42](../research_log.md#r42).*
*Numbers: every value is followed by its row ID in [`results_provenance.md`](../results_provenance.md).
Terms follow [`glossary.md`](../glossary.md).*

**Summary.** Short-term reversal is one of the oldest documented return anomalies, and my model's
scores are clearly exposed to it: regressed day by day on a 5-day reversal score, the GBDT score has a
mean R² of 0.159, and the exposure is positive on 98.6% of days (A4). The question that matters is whether the portfolio's return is reducible to that exposure. It is not.
Removing the reversal component with an out-of-sample β costs 3.18 bps, 14.3% of long-short net@10,
and what remains is well above what reversal alone earns through the same portfolio (A5). Correlated
with reversal is not the same as reducible to it.

---

## Problem

**The literature.** Individual stock returns reverse at short horizons. Jegadeesh (1990) documented
highly significant negative first-order serial correlation in the monthly returns of individual US
stocks, and decile portfolios formed on the resulting return forecasts differed substantially in later
abnormal returns. Lehmann (1990) studied an even shorter horizon: stocks that were winners or losers in one week tend to reverse
sharply in the next week. He read this as evidence against market efficiency, and reported that the
contrarian profits survive corrections for thin trading, bid-ask spreads, and plausible transaction
costs. Two alternative sources of such profits are also well documented. Bid-ask bounce alone makes
successive observed price changes negatively autocorrelated (Roll, 1984). And a contrarian portfolio
can earn money from lead-lag effects across stocks, where large stocks' returns lead small stocks',
even if no individual stock's returns are negatively autocorrelated (Lo & MacKinlay, 1990).

**Why it matters here.** My features are Alpha158-style price and volume factors, which include
several past-return windows, and the label is a two-day forward return. A gradient-boosted model
trained on that setup could easily learn "buy recent losers, sell recent winners". If that were most of
what it does, the headline would be a well-known effect wearing a model's clothes, and its cost
of implementation would be the real question, not the model.

**What I had and had not checked.** When I listed candidate mechanisms from published work
([R36](../research_log.md#r36)), I wrote that reversal had never been checked. That was half true.
The single-factor reversal baseline had been checked: the ROC5 score through the same evaluation
has RankIC +0.0268, ICIR +0.158, D10−D1 +19.70 bps, net@10 +11.57 (IR +1.50), and the model's decile
spread is 1.42x ROC5's (A1, A2). That asks what reversal is worth. It does not ask how much of the
model is reversal.

## Method

`model/reversal_exposure.py` post-processes stored predictions; it retrains nothing.

**Reversal score.** `ROC5 = close(T−5) / close(T)`, the project's existing baseline factor (glossary).
A high value means the stock fell over the last five trading days, so with its sign unchanged it is a
reversal score. Five trading days is Lehmann's one-week formation window.

**1. Exposure.** Each day, standardize the model score and ROC5 across stocks (`pred_z`, `rev_z`) and
fit `pred_z = α + β·rev_z`. With both sides standardized, β equals the daily Pearson correlation and
R² = β². I also report the daily Spearman correlation, since the project's signal metric is rank based.
This step is descriptive and makes no modelling choice.

**2. Orthogonalization.** Form `pred_z − β·rev_z` and run it through the headline long-short portfolio
(`ema5/exit10`, the same `smooth` and `portfolio` functions as the headline). β must not be estimated on
the year it cleans, or the cleaned score would use future information. So β for year k is the mean daily
β over all years before k (an expanding window, the same rule as nested reselection in
`model/selection_null.py`). 2020 has no earlier year, so this comparison covers 2021–2025. For contrast
the script also prints the result with the full-sample mean β, which uses future data.

**3. Reference.** Run `rev_z` alone through the same portfolio: what does pure reversal earn under
exactly this construction?

**Pool extension** ([R42](../research_log.md#r42)). `model/pool_corr.py` applies the same exposure
regression and nested orthogonalization to all eleven pool members, and recomputes their daily rank
correlations after orthogonalization.

## Result

**Exposure of the headline GBDT** (CN, `vwap × entry`, seed 0, 2020–2025) (A4):

| statistic | value |
|---|---|
| mean β | +0.373 |
| mean R² | 0.159 |
| mean Spearman | +0.266 |
| days with β > 0 | 98.6% |

The exposure is systematic, not noise.

**What removing it costs** (long-short `ema5/exit10`, net@10, 2021–2025) (A5):

| score | net@10 |
|---|---|
| model score, standardized (raw) | +22.20 |
| orthogonalized, nested β | +19.03 (**−3.18 bps, 14.3%**) |
| orthogonalized, full-sample β (uses future data) | +18.88 |
| reversal score alone | +13.11 (IR +1.75) |

Three readings:

- **Most of the return is not on the reversal direction.** The score's mean daily R² against ROC5 is
  0.159 (A4), and removing that part costs 14.3% of net@10 (A5). The cost is of the same order as the
  exposure, and the remainder is most of the result.
- **The remainder beats reversal.** The orthogonalized score earns +19.03 against +13.11 for reversal
  alone under the same construction (A5).
- **The nested β does not flatter the result.** The leaky full-sample β gives a slightly lower number,
  +18.88 (A5), so the out-of-sample estimate is not what keeps the remainder high.

**Across the pool** (CN, common row set). Mean R² against ROC5 over 2020–2025 is 0.158 for GBDT seed 0,
0.194–0.215 for the three MLP seeds, and 0.391 for ridge (E8). The linear model is the most
reversal-like member, which is consistent with a linear fit to features that include past-return
ratios such as ROC5 itself. Orthogonalizing every member over 2021–2025 barely changes how similar they
are: the mean off-diagonal correlation moves from 0.6300 to 0.6253; only ridge ↔ GBDT seed 0 drops noticeably, 0.553 → 0.494 (E5).

## What it overturned

- **"Reversal was never checked"** ([R36](../research_log.md#r36)). The factor had been checked (A1);
  the model's exposure to it had not. Both are now measured.
- **The worry that the headline is a reversal proxy.** Exposure is real (A4), but the orthogonalized
  portfolio keeps most of its return and still beats pure reversal (A5).
- **The close-label baseline comparison.** There, ROC5 reached 76% of the model RankIC and 1/2.8 of its
  decile spread (X11). On the current label ROC5 has RankIC +0.0268 and the model's spread is only 1.42x
  ROC5's (A1, A2). In bps, reversal alone is closer to the model than the old spread ratio suggested,
  which is exactly why the exposure question needed an answer.
- **The first figures of this audit.** The session that introduced the audit ran it on an aligned
  prediction copy that no shipped script produces. The reproducible run on the walk-forward predictions
  replaces them, and it shows a larger cost of removing reversal, not a smaller one (A4, A5).
- **"The MLP's higher RankIC comes from leaning more on reversal."** Its exposure is indeed higher
  (E8), but that claim had compared against the luckiest GBDT seed and was weakened
  ([R40](../research_log.md#r40)).
- **"The low correlations between model families are shared reversal exposure."** Not supported: the
  pool's correlations barely move after orthogonalization, except for ridge (E5).

## Limits

- **One reversal definition.** ROC5 is a 5-day, close-price, raw-return score. I did not test other
  horizons (1-day, monthly as in Jegadeesh (1990)), industry- or market-adjusted reversal, or
  volume-conditioned versions. A model could carry exposure to those that ROC5 does not capture.
- **ROC5 itself contains bounce.** It uses `close(T)`, so by Roll (1984) part of its signal is
  microstructure. Removing it therefore removes some bounce-driven information along with reversal proper.
- **Linear, yearly β.** Orthogonalization subtracts one β per year times `rev_z`. It removes the linear
  projection on average, not day by day, and not any non-linear use of past returns (for example,
  reversal that only applies to high-volatility names).
- **Standardize, then smooth.** The audit standardizes scores before EMA smoothing, while the headline
  smooths raw scores. Daily standardization keeps each day's ranking, but the EMA runs across days whose
  scales differ, so the order of operations changes the portfolio. That is why the raw row in A5
  (+22.20) differs from the 2021–2025 headline in T1 (+22.39). Comparisons inside A5 are like for like;
  comparisons between A5 and other rows are not.
- **Seed and market.** A4–A5 are GBDT seed 0 on CN only. I have not recorded the same audit for other
  seeds or for the US control in the provenance sheet.
- **Exposure over time.** The script prints exposure by year, but the yearly values are not in the
  provenance sheet, so I make no claim here about a trend. Recording that breakdown is the next check,
  because a result that leans on one known effect can weaken when that effect does.
- **Costs are a flat 10 bps per unit of turnover.** I did not model market impact or capacity, which
  matter most for the high-turnover parts of a score.

## References

- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*,
  45(3), 881–898.
- Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics*,
  105(1), 1–28.
- Lo, A. W., & MacKinlay, A. C. (1990). When are contrarian profits due to stock market overreaction?
  *Review of Financial Studies*, 3(2), 175–205.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market.
  *Journal of Finance*, 39(4), 1127–1139.

## Errata

- **2026-09-13 · Label horizon.** "The label is a two-day forward return" (Problem) should read: the
  label is a one-day return entered on T+1 and exited on T+2 (glossary, *label timing*). Nothing else in
  this note depends on it.
