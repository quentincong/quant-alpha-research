# Deep dive: the CN label, from `close × full` to `vwap × entry`

*Written 2026-09-13. The body is frozen; corrections go to [Errata](#errata) at the end.*
*Research log entries: [R27](../research_log.md#r27), [R29](../research_log.md#r29), [R30](../research_log.md#r30).*
*Numbers: every value is followed by its row ID in [`results_provenance.md`](../results_provenance.md).
Terms follow [`glossary.md`](../glossary.md).*

**Summary.** I changed how the China A-share label measures a two-day return, and nothing else.
Switching the price from the close to the day's VWAP raised GBDT RankIC by +57% (L8). Dropping the
exit half of the price-limit gate raised it again (L7). I had predicted the opposite for the price
change. The mechanism is attenuation: close-price noise in the label lowers a correlation, and the
design of the label timing leaves noise no other way to act. The gate change fixed a real defect:
the original gate threw away losses that had already happened.

---

## Problem

The label is the return from entering on T+1 to exiting on T+2, `label_raw = P(T+2) / P(T+1) − 1`,
with features known at the close of day T (glossary, *label timing*). The original CN label used the
adjusted close for `P` and a price-limit gate that kept a row only if the stock could be bought on
T+1 **and** sold on T+2 (`close × full`). With that label the GBDT had RankIC +0.0236, 6/6 folds
positive (L4), but after a 10 bps cost the unsmoothed long-short portfolio earned net@10 +0.15 bps
(L12). The obvious reading was "the signal is real but too weak to trade" (X5).

Two things about the label bothered me.

1. **Price.** A close is one trade. A portfolio that rebalances a few hundred names a day cannot fill
   them all at the close; VWAP is what a working order spread over the day can get. Closing prices also
   carry bid-ask bounce: a close that happens to print at the bid or the ask makes successive price
   changes negatively autocorrelated even in an efficient market (Roll, 1984). That looks like
   short-term reversal, and the CN model has a large reversal component (A4). So I expected part of
   the +0.0236 to be bounce, and **I expected a VWAP label to lower RankIC**.
2. **Gate.** The two halves of the gate mean different things. A stock at limit-up on T+1 cannot be
   bought, so the position never exists and dropping the row is right. A stock at limit-down on T+2
   cannot be sold, but the position was already bought on T+1 and the loss has happened. Dropping that
   row pretends the position never existed. The gate tests only limit-down on the exit, so it removes
   large losses and never the matching large gains: a one-sided cut of the left tail.

## Method

**Controls before any comparison.** A new label path can create differences through bugs, so I first
showed it reproduces the old label when nothing should change.

- `dataset/alphaFactor/make_vwap_label.py --verify` rebuilds `close × full` through the new code (same
  gate derivation, the same shift alignment on the dataset's own row grid, the same `|label_raw| ≤ 0.8`
  filter, the same cross-sectional robust z-score) and requires bit-identical `label_raw` and `label`
  against the built-in label.
- Feeding that rebuilt label back through `model/walkforward.py --label-file` reproduces the
  built-in-label run fold by fold (L4 lists both records).

**One variable at a time.** Then I built the label variants `price × gate` (glossary) and ran the
same walk-forward GBDT on each, with the same folds, features, and parameters:

| | gate `full` | gate `entry` |
|---|---|---|
| price `close` | L4 | L5 |
| price `vwap` | L6 | L7 |

`vwap` is `amount / volume` multiplied by the same-day adjustment factor; without that factor the
unadjusted average price jumps on ex-dividend days.

**Look-ahead checks** ([R29](../research_log.md#r29)). A higher RankIC after a label change is exactly
what a leak would produce, so I checked: the features never use traded amount; label timestamps by hand;
a whole-pipeline control that trains and tests on a label permuted across stocks within each day; and
the anchoring of the adjustment factor.

**Proxy calibration.** The US data has no traded amount, so the US can only use the typical price
`hlc3 = (high + low + close) / 3`. CN is the only market with both prices, so I ran `hlc3 × full` on CN
to see whether the proxy reproduces the VWAP effect before reading anything into the US result.

**Portfolio view.** `model/evaluate.py` for deciles and the unsmoothed long-short portfolio;
`model/longonly_study.py` for the universe mean; `model/limit_audit.py` for the short leg's
tradability (below).

## Result

**RankIC went up, in every cell of the design** (GBDT, seed 0, fold mean, CN 2020–2025):

| label | RankIC | ICIR |
|---|---|---|
| `close × full` (original) | +0.0236, 6/6 folds positive, worst +0.0061 (L4) | +0.202 (L11) |
| `close × entry` | +0.0278 (L5) | — |
| `vwap × full` | +0.0369 (L6) | +0.315 (L11) |
| `vwap × entry` (new default) | +0.0409 (L7) | +0.335 (L11) |

- Price alone, with the gate held at `full`: +57% (0.0369 / 0.0236 = 1.57x) (L8).
- The +0.0409 is seed 0; over three seeds the value is +0.0400 ± 0.0015 (M1).
- The typical price reproduces the effect on CN: `hlc3 × full` +0.0407 (L9) against `vwap × full`
  +0.0369 (L6), so on the US `hlc3` is a meaningful stand-in. The US `hlc3 × none` label gives +0.0071,
  ICIR +0.054 (N2).
- The permuted-label control gives −0.0012, 2/6 folds positive (L10): the pipeline does not produce
  signal from a label with no link to the features.

**The money moved even more than the correlation.** Unsmoothed long-short net@10 went from +0.15
(gross +11.14, turnover 110.0%) to +12.72 (gross +23.63, turnover 109.2%) (L12, L13). Turnover barely
changed, so the gain was not bought with trading. The `vwap × full` deciles are D10 +7.34 and
D1 −16.29 against a universe mean of +2.30 (L15): 79% of the spread comes from the short side of the
mean and 21% from the long side (L16).

**Why the prediction was wrong: attenuation.** Bounce can create *false predictability* only if the
same noisy print enters both the features and the label. Here the features end at the close of T and
the label starts at T+1. In Roll's model the trade direction behind each close is independent from day
to day, so the bounce in `close(T+1)` and `close(T+2)` is unrelated to anything known at T. Its only
effect is extra variance in the label, and noise in one variable that is independent of the other
lowers their measured correlation (Spearman, 1904). A VWAP averages many trades and removes most of
that noise, so the measured correlation rises. The T+1 start was chosen to avoid look-ahead; it also
closed the channel through which bounce could have inflated the old result. Two consequences are
visible in the sheet:

- the raw label is less dispersed: `label_raw` sd 0.0259 for `close × full` (L2) against 0.0229 for
  `vwap × entry` (L3). The two rows also differ in gate, so this is an illustration, not a controlled
  comparison. Because bps scales change with the label, I compare portfolios only within one label;
- a close-price label also inflates the universe-mean return itself. With the gate held at `entry`, the
  2-day mean is +3.98 bps with the close against +0.44 with VWAP, about 450 bps a year (L17).

**The exit gate, and a short leg that was never gated.** Putting the exit-failure rows back
(`full` → `entry`) raised RankIC from +0.0236 to +0.0278 with the close and from +0.0369 to +0.0409
with VWAP (L4–L7). I made `vwap × entry` the default on correctness, not on those numbers: the same change slightly lowered the long-only result ([R30](../research_log.md#r30)).

While auditing the gate I found the mirror-image gap. The gate describes a **long** round trip. The
decile long-short portfolio also shorts D1, whose round trip needs "not limit-down on T+1" to open and
"not limit-up on T+2" to close, and nothing in the project checked either. `model/limit_audit.py`
applies the same entry/exit logic to the short leg: it drops short entries that were impossible
(limit-down on T+1) and keeps short exits that were blocked (the loss is real). Gating the short entry
cuts the `vwap × full` D10−D1 spread from +23.63 to +20.19 (−3.44) and the `close × full` spread from
+11.14 to +9.78 (−1.36); the untradable short share is highest in D1, 3.08% for `vwap × full` (A3).

## What it overturned

- **"A 10 bps cost wipes out the CN signal"** (net@10 +0.15, X5). That was label noise; the same model
  class on the VWAP label earns +12.72 unsmoothed (L13) and the smoothed long-short headline is
  +23.19 (P1).
- **My own prediction** that a VWAP label would lower RankIC. The direction was wrong, for the timing
  reason above.
- **The original CN gate.** `full` hides realized limit-down losses; `entry` is the default from here on.
- **"About 70% of the spread comes from the short leg"** was a rough read; measured, it is 79% (X7, L16).
- **Close-label benchmark inflation of about 500 bps a year** (X8) compared across gates; with the gate
  fixed it is about 450 (L17).
- **The seed-0 RankIC** +0.0409 as a headline (X12); it is the top of three seeds, +0.0400 ± 0.0015 (M1).

## Limits

- **Idealized fills.** The VWAP label assumes both legs fill exactly at the day's VWAP. VWAP is a
  standard execution benchmark, but real orders deviate from it and pay impact. That changes the level of
  every bps number; it does not by itself create cross-sectional predictability.
- **Shorting.** 79% of the `vwap × full` spread sits on the short side (L16), and shorting individual
  A-shares is hard. The long-short headline (P1) assumes it is possible and, like every portfolio script
  except `limit_audit.py`, it does not gate short entries. A3 measures that gate on `vwap × full` deciles
  only, not on the smoothed `vwap × entry` headline.
- **Exit failures are still optimistic.** A position that cannot be sold on T+2 is charged the T+2
  return, but in practice it stays stuck and can lose more on later days.
- **One seed per label.** L4–L7 are single GBDT runs; only `vwap × entry` has three seeds (M1). The seed
  sd there, 0.0015 (M1), is small next to the close → VWAP gap (L4, L6), but the `full` → `entry`
  step (L6, L7) is only a few seed sds wide, so its size is less certain than its direction.
- **Measurement vs. training.** The gain can come from a better ruler (the same predictions scored
  against a cleaner label) or from a better model (training on a cleaner target). Scoring close-trained
  predictions against the VWAP label separates the two, but I have not re-run that split with a shipped
  script, so I do not quote it.
- **Selection.** The label was chosen after seeing the full-sample results of four variants. I argued
  the gate change from correctness, but no nested test covers the label choice
  (see the [selection deep dive](2026-09-13-selection-procedure-test.md)).
- **Permuted-label control.** The permuted label file (L10) was made outside the shipped scripts.

## References

- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market.
  *Journal of Finance*, 39(4), 1127–1139.
- Spearman, C. (1904). The proof and measurement of association between two things. *American Journal
  of Psychology*, 15(1), 72–101.

## Errata

- **2026-09-13 · Label horizon and benchmark inflation.** The Summary says the label measures "a two-day
  return". `label_raw = P(T+2) / P(T+1) − 1` is a one-day return, entered on T+1 and exited on T+2
  (glossary, *label timing*). For the same reason the Result section's "2-day mean" is a daily mean, and
  the benchmark inflation of 3.54 bps per day is about **890 bps a year** (× 252, L17), not "about 450"
  (X19). The "What it overturned" bullet should read: about 500 across gates (X8) → about 890 with the
  gate fixed (L17). The direction and the mechanism are unchanged.
