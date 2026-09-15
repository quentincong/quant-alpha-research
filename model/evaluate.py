"""
evaluate.py — unified evaluation metric (the one ruler)

**Why this file exists** (following the 2026-08-30 number audit):
  Previously the project had only one metric: daily RankIC averaged across days. A correlation
  coefficient can be driven by a handful of outlier stocks, and IC does not reveal whether this
  is happening. In addition, `best_val_*` reported by the scripts is the **maximum over training
  history** (a Tier A selection score), not measured performance.

  This file consumes only **predictions** saved by `walkforward.py` (Tier C: predict the test year
  once, with no role in selection) and centralizes every evaluation definition. **Train once;
  evaluate any number of times.**

**Self-check**: annual RankIC computed here must be **elementwise identical** to the values
  recorded during training in `walkforward.jsonl`. A mismatch indicates a problem in the dump
  or evaluation definition. This is a free correctness check and must pass before interpreting
  other results. IC directly reuses `train_lgbm.daily_ic` rather than implementing another version,
  preventing two metrics from disagreeing.

**Do not mix the two label versions**:
  - `label`     = cross-sectional robust z-score clipped to ±3. **Use it for IC**; a same-day
    monotone transform leaves RankIC unchanged.
  - `label_raw` = actual forward return. **Use it for decile returns, turnover, and costs** because
    only actual returns answer how many basis points a group of stocks earned.
    **The definition changes with the label**: the built-in label buys at the T+1 close and sells
    at the T+2 close; external labels from `make_vwap_label.py` may use vwap pricing or a different
    gate. **Compare the standard deviation of `label_raw` before comparing bps across labels**.
    Price limits mechanically compress vwap tails, so bps on different scales are not directly comparable.

Usage (from the repository root, after conda activate torch-env):
  python model/evaluate.py --preds model/preds_walkforward_cn.parquet
  python model/evaluate.py --preds model/preds_walkforward_cn.parquet --baselines \
         --data dataset/alphaFactor/dataset_alpha158_cn.parquet
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lgbm import build_day_groups, daily_ic     # Reuse the same daily IC to preserve the metric definition.

COST_BPS = [0, 5, 10, 20]
N_DECILES = 10

# ---------------------------------------------------------------------------
# Outlier-return safeguards have **different rationales** in the two markets
# ---------------------------------------------------------------------------
# · **A-shares**: `build_dataset.add_label_cn` already applies `|rtn|<=0.8` in the label.
#   Its basis is **institutional**: with price limits, a normally traded T+1→T+2 return **cannot
#   reach** 0.8. This safeguard catches only suspension/reopen gaps and data errors. **It follows
#   from market structure rather than parameter tuning.**
#
# · **US equities**: `add_label` has **no safeguard**. Without price limits, legitimate T+1→T+2
#   moves above 80% can occur, for example after binary biotechnology events or acquisitions.
#   **Corporate-action splices** therefore flow directly into portfolio metrics.
#   Verified example on 2026-08-30: `CHRD` closed at $0.12 on 2020-11-19 and opened at $27.15
#   on 11-20 with an **unchanged adjustment factor**. Oasis Petroleum relisted as Chord Energy
#   after bankruptcy reorganization, old equity was cancelled, new shares were issued, and
#   yfinance directly spliced the two price series, producing a **+22,310%** label. This one stock
#   accounts for **58.5%** of total US D10 return, and the top 18 rows account for 56.7%.
#   Thus **all US bps/turnover/cost values were once contaminated**; rank-based RankIC is unaffected.
#
# **The US cap is therefore an outlier-screening judgment, not an institutional deduction.**
# The rationales must remain separate; otherwise the A-share rationale improperly justifies a
# US judgment. This module **applies the US cap by default and prints a notice**, with sensitivity
# available under --cap-scan. Measured D10−D1 at cap=0.8/0.5/0.2 is +3.06/+3.34/+3.95, so the
# **conclusion does not depend on the threshold choice**.
DEFAULT_CAP = {"us": 0.8, "cn": None}      # CN already safeguards at the label layer; do not apply twice.


def apply_cap(df: pd.DataFrame, market: str, cap: float | None = None,
              quiet: bool = False) -> pd.DataFrame:
    """Apply a market-specific `label_raw` safeguard. cap=None uses market default; cap<=0 explicitly disables it."""
    if cap is None:
        cap = DEFAULT_CAP.get(market)
    if cap is None or cap <= 0:
        return df
    n0 = len(df)
    out = df[df["label_raw"].abs() <= cap]
    if not quiet:
        print(f"  [cap] market={market} applying |label_raw|<={cap} → "
              f"removed {n0 - len(out):,} rows ({1 - len(out)/n0:.4%})"
              f"{'  ← required for US: original label lacks a safeguard and contains bankruptcy-reorganization price splices' if market == 'us' else ''}")
    return out

# Single factor for the trivial baseline: ROC5 = close.shift(5)/close.
# If a stock rose over five days, close > close.shift(5), so ROC5 < 1. Therefore,
# **high ROC5 = a larger recent decline = a loser**. Short-horizon reversal predicts losers rebound,
# so the **unchanged positive sign** of ROC5 is the reversal signal. Sign reversal is the classic
# mistake in this baseline, so the derivation is fixed explicitly here.
BASELINE_FACTOR = "ROC5"


# ----------------------------------------------------------------------------
# Daily IC
# ----------------------------------------------------------------------------
def ic_panel(df: pd.DataFrame) -> dict:
    """Daily RankIC series → mean / standard deviation / ICIR / t / share of positive-IC days.

    **ICIR = mean/std** can be understood as a Sharpe ratio for the IC series. It measures
    **stability**, not level. A steady +0.005 every day is more useful than mean +0.008 with ±0.05
    fluctuations, while the mean alone cannot distinguish them.
    """
    day_id = pd.factorize(df["date"], sort=True)[0]
    groups = build_day_groups(day_id)
    pred, lab = df["pred"].to_numpy(), df["label"].to_numpy()
    series = np.array([daily_ic(pred[g], lab[g], [np.arange(len(g))])[1] for g in groups])
    series = series[~np.isnan(series)]
    n = len(series)
    mean, sd = float(series.mean()), float(series.std(ddof=1))
    return {"n_days": n, "rankic_mean": mean, "rankic_std": sd,
            "icir": mean / sd if sd > 0 else np.nan,
            # Naive t: **IC series are autocorrelated, so this t is inflated**.
            # The corresponding error bar comes from block_bootstrap.py.
            "t_naive": mean / (sd / np.sqrt(n)) if sd > 0 else np.nan,
            "pct_positive_days": float((series > 0).mean()),
            "series": series}


# ----------------------------------------------------------------------------
# Decile portfolio + monotonicity
# ----------------------------------------------------------------------------
def decile_panel(df: pd.DataFrame, n_dec: int = N_DECILES) -> dict:
    """Sort predictions into n_dec groups daily and test whether each group's **actual return** rises left to right.

    This is more intuitive than IC: if IC is reasonable but only the tails move, the "signal"
    is driven by a handful of outlier stocks.
    """
    d = df.copy()
    r = d.groupby("date")["pred"].rank(method="first")
    cnt = d.groupby("date")["pred"].transform("size")
    d["decile"] = np.floor((r - 1) * n_dec / cnt).astype(int).clip(0, n_dec - 1)
    per_day = (d.groupby(["date", "decile"])["label_raw"].mean()
                 .unstack().reindex(columns=range(n_dec)))  # Fill missing groups with NaN to avoid silent misalignment.
    means = per_day.mean()                                  # Actual return by group, averaged across days.
    idx = np.arange(n_dec)
    mono = float(pd.Series(idx).corr(pd.Series(means.values), method="spearman"))
    return {"decile_mean_bps": (means * 1e4).to_dict(),
            "monotonicity_spearman": mono,
            "top_minus_bottom_bps": float((means.iloc[-1] - means.iloc[0]) * 1e4),
            "per_day": per_day}


# ----------------------------------------------------------------------------
# Long-short portfolio: turnover + cost curve
# ----------------------------------------------------------------------------
def portfolio_panel(df: pd.DataFrame, n_dec: int = N_DECILES) -> dict:
    """Equal-weight long the top decile and short the bottom decile, rebalancing daily.

    **Turnover**: for each leg, compute Σ|w_t − w_{t−1}|/2, the share of capital replaced,
    then sum both legs.
    **Cost**: net_t = gross_t − turnover_t × bps/1e4.
    **Definition warning**: label_raw is a **one-day** return (T+1→T+2), so daily returns do not overlap,
    but the √252 annualization treats days as **independent** while holdings and regimes persist.
    Annualized IR is therefore **naive** and only a rough reference.
    """
    d = df.copy()
    r = d.groupby("date")["pred"].rank(method="first")
    cnt = d.groupby("date")["pred"].transform("size")
    dec = np.floor((r - 1) * n_dec / cnt).astype(int).clip(0, n_dec - 1)
    d["leg"] = np.where(dec == n_dec - 1, 1, np.where(dec == 0, -1, 0))
    d = d[d["leg"] != 0]

    gross, turn = [], []
    prev = {1: {}, -1: {}}
    # Group once; do not apply full-table Boolean filtering for each day inside the loop, which is O(n×days).
    for _, day in d.groupby("date", sort=True):
        # Both legs must contain stocks; otherwise the day is not a long-short portfolio.
        # With only five smoke tickers, the top decile may be empty. Without this guard, gross
        # silently becomes a short-only, incorrect value.
        if not (day["leg"] == 1).any() or not (day["leg"] == -1).any():
            continue
        g, t_day = 0.0, 0.0
        for side in (1, -1):
            names = day.loc[day["leg"] == side, "instrument"].to_numpy()
            rets = day.loc[day["leg"] == side, "label_raw"].to_numpy()
            if len(names) == 0:
                continue
            w = {nm: 1.0 / len(names) for nm in names}
            g += side * float(rets.mean())
            allnm = set(w) | set(prev[side])
            t_day += sum(abs(w.get(k, 0.0) - prev[side].get(k, 0.0)) for k in allnm) / 2
            prev[side] = w
        gross.append(g); turn.append(t_day)
    gross, turn = np.asarray(gross, dtype=float), np.asarray(turn, dtype=float)
    if gross.size == 0:
        # No valid day: the cross-section is too small for disjoint top/bottom deciles
        # (the smoke set has only five tickers). Return NaN with a reason, **not** zero,
        # which would be interpreted as a genuine no-profit result.
        return {"n_days": 0, "gross_mean_bps": np.nan, "gross_std_bps": np.nan,
                "turnover_mean": np.nan, "cost_curve": {b: {"net_mean_bps": np.nan,
                "ir_annualized_naive": np.nan} for b in COST_BPS},
                "note": "cross-section too small to construct a ten-decile long-short portfolio"}
    turn[0] = np.nan                                        # First day establishes positions and does not count as turnover.

    out = {"n_days": len(gross), "gross_mean_bps": float(gross.mean() * 1e4),
           "gross_std_bps": float(gross.std(ddof=1) * 1e4),
           "turnover_mean": float(np.nanmean(turn)), "cost_curve": {}}
    for bps in COST_BPS:
        net = gross - np.nan_to_num(turn, nan=0.0) * bps / 1e4
        mu, sd = net.mean(), net.std(ddof=1)
        out["cost_curve"][bps] = {"net_mean_bps": float(mu * 1e4),
                                  "ir_annualized_naive": float(mu / sd * np.sqrt(252)) if sd > 0 else np.nan}
    return out


# ----------------------------------------------------------------------------
# Period breakdown
# ----------------------------------------------------------------------------
def period_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, grp in [("year", df.groupby(df["date"].dt.year)),
                     ("quarter", df.groupby(df["date"].dt.to_period("Q")))]:
        for label, g in grp:
            if g["date"].nunique() < 20:
                continue
            ic = ic_panel(g); dc = decile_panel(g)
            rows.append({"scope": key, "period": str(label), "n_days": ic["n_days"],
                         "rankic": ic["rankic_mean"], "icir": ic["icir"],
                         "mono": dc["monotonicity_spearman"],
                         "d10_minus_d1_bps": dc["top_minus_bottom_bps"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Trivial baselines (the most valuable part of this file)
# ----------------------------------------------------------------------------
def make_baselines(df: pd.DataFrame, data_path: str | None, seed: int = 0) -> dict:
    """Feed **trivial signals** through the same metric.

    Why it matters: every July comparison that "confirmed a weak-feature bottleneck" was
    **model versus model**, never against a trivial baseline. If a raw reversal factor achieves
    a similar score, features are not the bottleneck; the model is destroying information.
    This comparison **favors the model** because it tried dozens of configurations while the baseline gets one attempt.
    """
    out = {}
    rng = np.random.default_rng(seed)
    rnd = df.copy(); rnd["pred"] = rng.standard_normal(len(rnd))
    out["random"] = rnd
    if data_path:
        cols = ["date", "instrument", BASELINE_FACTOR]
        raw = pd.read_parquet(data_path, columns=[BASELINE_FACTOR]).reset_index()
        raw["date"] = pd.to_datetime(raw["date"]); raw["instrument"] = raw["instrument"].astype(str)
        m = df.merge(raw[cols], on=["date", "instrument"], how="left")
        m = m[m[BASELINE_FACTOR].notna()].copy()
        m["pred"] = m[BASELINE_FACTOR]                      # Positive sign = reversal; see the constant derivation above.
        out[f"single_factor_{BASELINE_FACTOR}"] = m
    return out


# ----------------------------------------------------------------------------
def self_check(df: pd.DataFrame, market: str, smoke: bool, label_file: str = "builtin") -> None:
    """Require this script's annual RankIC to match values recorded during training in walkforward.jsonl.

    Smoke predictions must match the smoke log. Comparing smoke predictions with the full log
    necessarily fails because the **comparison target is wrong**, not because the pipeline is broken.
    This mistake occurred during the first run on 2026-08-30.
    """
    path = "model/walkforward_smoke.jsonl" if smoke else "model/walkforward.jsonl"
    if not os.path.exists(path):
        print(f"  [self-check] cannot find {path}; skipping"); return
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    recs = [r for r in recs if r.get("market") in (market, None)
            and bool(r.get("smoke")) == smoke
            # A different label is a different experiment. Comparing a vwap log with close predictions
            # necessarily mismatches because the target is wrong, not because the pipeline failed,
            # analogous to the smoke/full mistake.
            and r.get("label_file", "builtin") == label_file]
    if not recs:
        print(f"  [self-check] walkforward.jsonl has no market={market} "
              f"label={label_file} record; skipping"); return
    logged = {r["test_year"]: r["test_rankic"] for r in recs[-1]["results"]}
    ok = True
    for yr, g in df.groupby("test_year"):
        mine = ic_panel(g)["rankic_mean"]
        ref = logged.get(int(yr))
        if ref is None:
            continue
        d = abs(mine - ref)
        flag = "OK" if d < 1e-9 else "MISMATCH"
        if d >= 1e-9:
            ok = False
        print(f"  [self-check] {yr}: evaluate={mine:+.6f}  walkforward={ref:+.6f}  {flag}")
    print(f"  [self-check] {'all match — dump and evaluation definition are credible' if ok else 'mismatch — investigate here first'}")


def report(name: str, df: pd.DataFrame) -> dict:
    ic = ic_panel(df); dc = decile_panel(df); pf = portfolio_panel(df)
    print(f"\n{'=' * 78}\n[{name}]  n={len(df):,} rows, {ic['n_days']} days\n{'=' * 78}")
    print(f"  RankIC mean {ic['rankic_mean']:+.4f} | standard deviation {ic['rankic_std']:.4f} | "
          f"ICIR {ic['icir']:+.3f} | naive t {ic['t_naive']:+.2f} | positive-IC days {ic['pct_positive_days']:.1%}")
    print(f"\n  Actual decile returns (bps, daily then averaged across days) — monotonic means increasing left to right:")
    vals = [dc['decile_mean_bps'][k] for k in sorted(dc['decile_mean_bps'])]
    print("    " + "  ".join("D%d:%s" % (i + 1, "   n/a " if pd.isna(v) else f"{v:+7.2f}")
                            for i, v in enumerate(vals)))
    print(f"    monotonicity Spearman = {dc['monotonicity_spearman']:+.3f}   "
          f"D10−D1 = {dc['top_minus_bottom_bps']:+.2f} bps")
    print(f"\n  Long-short portfolio (daily rebalance, equal-weight long D10 / short D1):")
    if pf.get("note"):
        print(f"    [skip] {pf['note']}")
        return {"ic": {k: v for k, v in ic.items() if k != "series"},
                "decile": dc["decile_mean_bps"], "mono": dc["monotonicity_spearman"]}
    print(f"    gross return {pf['gross_mean_bps']:+.2f} bps/day | average daily turnover {pf['turnover_mean']:.1%}")
    print(f"    {'cost(bps)':>10} {'net return(bps/day)':>16} {'annualized IR(naive)':>14}")
    for bps, v in pf["cost_curve"].items():
        print(f"    {bps:>10} {v['net_mean_bps']:>16.2f} {v['ir_annualized_naive']:>14.2f}")
    return {"ic": {k: v for k, v in ic.items() if k != "series"}, "decile": dc["decile_mean_bps"],
            "mono": dc["monotonicity_spearman"], "portfolio": {k: v for k, v in pf.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="prediction parquet saved by walkforward.py")
    ap.add_argument("--data", default=None, help="source dataset from which the single-factor baseline reads ROC5")
    ap.add_argument("--baselines", action="store_true", help="also evaluate random / single-factor baselines")
    ap.add_argument("--by-period", action="store_true", help="print annual/quarterly breakdown")
    ap.add_argument("--label-file", dest="label_file", default=None,
                    help="external label used to train these predictions; used only to select the correct self-check log record")
    ap.add_argument("--cap", type=float, default=None,
                    help="|label_raw| outlier safeguard; market default if omitted (us=0.8, cn=None); 0 disables it")
    ap.add_argument("--cap-scan", action="store_true",
                    help="print cap sensitivity to show the conclusion does not depend on the threshold")
    args = ap.parse_args()

    df = pd.read_parquet(args.preds)
    df["date"] = pd.to_datetime(df["date"])
    base = os.path.basename(args.preds)
    market = "cn" if "_cn" in base else "us"
    smoke = "_smoke" in base
    print(f"[data] {args.preds}  market={market}  {len(df):,} rows  "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    print("\n[self-check] matching annual RankIC recorded during training:")
    # Run the self-check **before applying the cap** because it matches annual RankIC recorded
    # during training on all untrimmed rows. Trimming first necessarily mismatches and is a false alarm.
    self_check(df, market, smoke, args.label_file or "builtin")

    if args.cap_scan:
        print(f"\n[cap-scan] outlier-safeguard sensitivity (rank-based RankIC barely moves; bps is contaminated)")
        print(f"    {'cap':>8} {'dropped':>10} {'RankIC':>9} {'D10-D1(bps)':>13} {'monotonicity':>9}")
        for c in [0, 0.8, 0.5, 0.2]:
            d = apply_cap(df, market, c if c > 0 else -1, quiet=True)
            dc = decile_panel(d)
            print(f"    {('none' if c == 0 else c):>8} {1-len(d)/len(df):>10.4%} "
                  f"{ic_panel(d)['rankic_mean']:>+9.4f} "
                  f"{dc['top_minus_bottom_bps']:>+13.2f} {dc['monotonicity_spearman']:>+9.3f}")

    df = apply_cap(df, market, args.cap)
    report("MODEL (LightGBM walk-forward)", df)
    if args.baselines:
        for nm, bdf in make_baselines(df, args.data).items():
            report(f"BASELINE: {nm}", bdf)
    if args.by_period:
        print(f"\n{'=' * 78}\n[period breakdown]\n{'=' * 78}")
        print(period_breakdown(df).to_string(index=False,
              float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
