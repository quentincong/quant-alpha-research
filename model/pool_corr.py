"""
pool_corr.py — pairwise daily cross-sectional rank-correlation matrix for the candidate-member pool

**Why run this analysis** (planned in docs/research_log.md#r40, run in docs/research_log.md#r42):
  An ensemble adds value only when its members **make different mistakes**. The ρ matrix maps
  where to look for diversity. But low ρ does not imply useful diversity; the least-correlated
  member may simply have **higher error variance**. Therefore, next to ρ the script **prints each
  member's strength under the same metric** (RankIC and net@10), making it immediately visible
  whether a member sees something different or is merely worse. This script **does not decide
  which members to include**.

**Compute two versions** (the finding in docs/research_log.md#r38: MLP relies more on reversal, so some "diversity"
  is shared exposure to the same reversal direction):
  1. ρ for **raw pred**;
  2. ρ **after reversal orthogonalization** (`pred_z − β·rev_z`), estimating β annually with an
     **expanding window** by reusing `reversal_exposure.daily_beta / expanding_beta`.
  If ρ rises substantially after orthogonalization, the original low correlation reflected
  **different reversal exposures**, not independent information.

**Read ROC5 only once**: the source dataset is 2.4GB, so reading it per member is wasteful.
  Read `(date, instrument, ROC5)` once and share it across all members.

**Sample definition**: the ρ matrix uses the **common row set with predictions from every member**
  because row sets differ slightly across label files. Every pair then uses the same sample,
  keeping the matrix internally consistent; the retained share is printed. Pairwise intersections
  would make each matrix cell come from a different sample and complicate comparison.

**Same metric**: strength columns use `turnover_study.portfolio` at **ema5/exit10** and replace
  every label with the **headline label** (`label_cn_vwap_entry.parquet`). Otherwise each member's
  bps comes from its own label definition and is not comparable.

Usage:
  python model/pool_corr.py --headline-label dataset/alphaFactor/label_cn_vwap_entry.parquet \\
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
from evaluate import BASELINE_FACTOR, apply_cap
from reversal_exposure import daily_beta, expanding_beta, _csnorm
from turnover_study import portfolio, smooth
from train_mlp import _pearson, _rank

HEADLINE_SPAN, HEADLINE_EXIT = 5, 0.10

# Default member pool — name: prediction parquet.
# Deliberately **exclude** ens5 / ens5rank (already ensemble outputs), permentry (permutation-null control),
# preds_cmp_* / preds_ens_* / preds_meta_* (aligned derived copies from earlier comparisons), and
# seed1–4 (seed variants with the built-in close label, highly redundant with closefull).
DEFAULT_POOL = [
    ("gbdt_vwapentry_s0", "model/preds_walkforward_cn_vwapentry.parquet"),
    ("gbdt_vwapentry_s1", "model/preds_walkforward_cn_vwapentry_s1.parquet"),
    ("gbdt_vwapentry_s2", "model/preds_walkforward_cn_vwapentry_s2.parquet"),
    ("gbdt_closeentry",   "model/preds_walkforward_cn_closeentry.parquet"),
    ("gbdt_closefull",    "model/preds_walkforward_cn_closefull.parquet"),
    ("gbdt_hlc3",         "model/preds_walkforward_cn_hlc3.parquet"),
    ("gbdt_vwapfull",     "model/preds_walkforward_cn_vwapfull.parquet"),
    ("mlp_vwapentry_s0",  "model/preds_wf_mlp_cn_vwapentry_s0.parquet"),
    ("mlp_vwapentry_s1",  "model/preds_wf_mlp_cn_vwapentry_s1.parquet"),
    ("mlp_vwapentry_s2",  "model/preds_wf_mlp_cn_vwapentry_s2.parquet"),
]


def load_member(name: str, path: str) -> pd.DataFrame | None:
    """Read one member's (date, instrument, pred); return None and log if missing/unreadable."""
    if not os.path.exists(path):
        print(f"  [skip] {name}: file does not exist {path}")
        return None
    try:
        d = pd.read_parquet(path, columns=["date", "instrument", "pred"])
    except Exception as e:                       # noqa: BLE001 — Handle missing columns/schema mismatches here.
        print(f"  [skip] {name}: read failed {type(e).__name__}: {e}")
        return None
    d["date"] = pd.to_datetime(d["date"])
    d["instrument"] = d["instrument"].astype(str)
    d["pred"] = d["pred"].astype(np.float32)     # Halve memory for 12 members × 2.3M rows.
    d = d.rename(columns={"pred": name})
    print(f"  [ok]   {name:<20} {len(d):>9,} rows  "
          f"{d['date'].min().date()} → {d['date'].max().date()}")
    return d


def daily_rank_corr(wide: pd.DataFrame, cols: list[str], min_n: int = 20) -> pd.DataFrame:
    """Rank each column cross-sectionally by day, compute K×K correlation matrices, then average over days.

    The common row set gives every column the same daily sample, so one `rank` and one `corrcoef`
    compute all pairs for the day without pairwise loops, about 50x faster.
    """
    K = len(cols)
    acc = np.zeros((K, K))
    n_day = 0
    for _, g in wide.groupby("date", sort=True):
        if len(g) < min_n:
            continue
        R = np.array(g[cols].rank(method="average"), dtype=np.float64)  # Explicit copy:
        # pandas sometimes returns a read-only to_numpy view for a single-dtype block; mutation follows.
        R -= R.mean(axis=0)
        sd = R.std(axis=0)
        if (sd < 1e-12).any():
            continue
        R /= sd
        acc += (R.T @ R) / len(g)
        n_day += 1
    return pd.DataFrame(acc / max(n_day, 1), index=cols, columns=cols), n_day


def print_matrix(M: pd.DataFrame, title: str):
    print(f"\n{title}")
    w = max(len(c) for c in M.columns) + 1
    print(" " * 20 + "".join(f"{c[:8]:>9}" for c in M.columns))
    for r in M.index:
        print(f"{r:<20}" + "".join(f"{M.loc[r, c]:>9.3f}" for c in M.columns))
    off = M.to_numpy()[~np.eye(len(M), dtype=bool)]
    print(f"  off-diagonal: mean {off.mean():.3f} | min {off.min():.3f} | max {off.max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-json", dest="pool_json", default=None,
                    help="JSON: [[name, path], ...], overriding the default member pool")
    ap.add_argument("--add", action="append", metavar="NAME=PATH", default=[],
                    help="append a member to the default pool; repeatable")
    ap.add_argument("--data", required=True, help="source dataset for ROC5, read only once")
    ap.add_argument("--headline-label", dest="headline_label", required=True,
                    help="label parquet used to unify strength metrics (date,instrument,label,label_raw)")
    ap.add_argument("--out-json", dest="out_json", default="model/pool_corr_results.json")
    ap.add_argument("--no-strength", dest="no_strength", action="store_true",
                    help="skip portfolio-strength columns for faster ρ-only output")
    args = ap.parse_args()

    pool = DEFAULT_POOL if args.pool_json is None else \
        [tuple(x) for x in json.load(open(args.pool_json, encoding="utf-8"))]
    pool = list(pool) + [tuple(a.split("=", 1)) for a in args.add]

    print(f"[pool] {len(pool)} candidate members")
    frames, names = [], []
    for name, path in pool:
        d = load_member(name, path)
        if d is not None:
            frames.append(d); names.append(name)
    if len(names) < 2:
        sys.exit("fewer than two members are available")

    # ---- Wide table + common row set ----
    wide = frames[0]
    for d in frames[1:]:
        wide = wide.merge(d, on=["date", "instrument"], how="outer")
    n_union = len(wide)
    wide = wide.dropna(subset=names)
    print(f"\n[align] union {n_union:,} rows → common row set {len(wide):,} rows "
          f"({len(wide)/n_union:.1%}), {wide['date'].nunique()} days")
    for name, d in zip(names, frames):
        print(f"  {name:<20} own {len(d):>9,} rows → common set retains {len(wide)/len(d):.1%}")
    del frames

    # ---- Unified label used for strength columns and orthogonalization ----
    lab = pd.read_parquet(args.headline_label,
                          columns=["date", "instrument", "label", "label_raw"])
    lab["date"] = pd.to_datetime(lab["date"])
    lab["instrument"] = lab["instrument"].astype(str)
    wide = wide.merge(lab, on=["date", "instrument"], how="left")
    n0 = len(wide)
    wide = wide[wide["label"].notna()]
    print(f"[label] merged headline label → {len(wide):,} rows ({n0-len(wide):,} rows without labels dropped)")
    del lab

    # ---- Read ROC5 once ----
    print(f"[rev] reading {BASELINE_FACTOR} (once, shared by all members)")
    raw = pd.read_parquet(args.data, columns=[BASELINE_FACTOR]).reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["instrument"] = raw["instrument"].astype(str)
    wide = wide.merge(raw, on=["date", "instrument"], how="left")
    del raw
    n0 = len(wide)
    wide = wide[wide[BASELINE_FACTOR].notna()]
    print(f"[rev] {n0:,} → {len(wide):,} rows have ROC5")
    wide["rev_z"] = wide.groupby("date", sort=False)[BASELINE_FACTOR].transform(_csnorm)
    wide["year"] = wide["date"].dt.year

    # ---- Each member's pred_z and nested orthogonalization ----
    ortho_cols, betas = [], {}
    for name in names:
        wide[name + "_z"] = wide.groupby("date", sort=False)[name].transform(_csnorm)
        # Build only a three-column table for daily_beta; renaming wide would copy it in full.
        bt = daily_beta(pd.DataFrame({"date": wide["date"].to_numpy(),
                                      "pred_z": wide[name + "_z"].to_numpy(),
                                      "rev_z": wide["rev_z"].to_numpy()}))
        eb = expanding_beta(bt)                      # Estimate year k's β using only years before k.
        betas[name] = {"mean_beta_full": float(bt["beta"].mean()),
                       "mean_r2_full": float(bt["r2"].mean()),
                       "expanding": {int(k): v for k, v in eb.items()}}
        wide[name + "_o"] = wide[name + "_z"] - wide["year"].map(eb) * wide["rev_z"]
        ortho_cols.append(name + "_o")
        print(f"  [beta] {name:<20} full-sample mean β {bt['beta'].mean():+.4f} "
              f"(R² {bt['r2'].mean():.4f}) | expanding window " +
              " ".join(f"{y}:{b:+.3f}" for y, b in eb.items()))

    ortho_years = sorted(betas[names[0]]["expanding"].keys())
    sub = wide[wide["year"].isin(ortho_years)].copy()
    print(f"\n[ortho] orthogonalized matrix covers only {ortho_years} (first year lacks preceding years for β and is dropped)"
          f" → {len(sub):,} rows")

    # ---- Three matrices ----
    M_raw_all, nd1 = daily_rank_corr(wide, names)
    M_raw_sub, nd2 = daily_rank_corr(sub, names)
    # Build the orthogonalized matrix separately to avoid collisions with raw columns in sub.
    ortho_frame = sub[["date"]].copy()
    for name in names:
        ortho_frame[name] = sub[name + "_o"].to_numpy()
    M_ortho, nd3 = daily_rank_corr(ortho_frame, names)
    del ortho_frame
    print_matrix(M_raw_all, f"[1] raw pred, all years ({nd1} days)")
    print_matrix(M_raw_sub, f"[2] raw pred, {ortho_years[0]}–{ortho_years[-1]} ({nd2} days; same period as [3])")
    print_matrix(M_ortho, f"[3] after reversal orthogonalization (nested β), same period ({nd3} days)")

    d = M_ortho.to_numpy() - M_raw_sub.to_numpy()
    off = ~np.eye(len(names), dtype=bool)
    print(f"\n[3]−[2] off-diagonal change: mean {d[off].mean():+.3f} | "
          f"min {d[off].min():+.3f} | max {d[off].max():+.3f}★")
    print("  Positive means members become more similar after removing shared reversal exposure, so the original"
          " low correlation reflected different reversal exposures rather than independent information.")

    # ---- Strength columns: same metric ----
    # Metric warning discovered on 2026-09-06: **never compute strength on the common row set**.
    # The `full` gates under different labels drop precisely the **failed-exit rows pinned at limit-down**.
    # The 5,744 dropped rows have mean |label_raw| **0.0816**, **5.1 times** the full-sample 0.0160,
    # and median −0.085, almost entirely affecting the short leg. Removing them from the headline
    # member changes gross 27.50 → 21.74 and net@10 **23.19 → 17.30**.
    # Therefore, compute strength on **each member's own row set**, merging only the headline label
    # to unify label definitions. Use the common row set only for the ρ matrix, where identical samples
    # are required for consistency. Print both columns to expose the difference.
    strength = {}
    if not args.no_strength:
        print(f"\n{'='*112}\n[4] strength under the same metric (headline label + ema{HEADLINE_SPAN}/"
              f"exit{int(HEADLINE_EXIT*100)}, variant not reselected)\n{'='*112}")
        print("  own = member's own row set (official definition; matches existing headline values);"
              " common = common row set (tail-contaminated by the full gate; comparison only)")
        print(f"{'member':<20} {'RankIC':>9} {'gross':>9} {'turn':>8} {'net@10':>9} "
              f"{'IR@10':>8} | {'net@10_com':>11} {'delta':>8} {'mean_rho':>9}")
        lab2 = pd.read_parquet(args.headline_label,
                               columns=["date", "instrument", "label", "label_raw"])
        lab2["date"] = pd.to_datetime(lab2["date"])
        lab2["instrument"] = lab2["instrument"].astype(str)
        base_com = wide[["date", "instrument", "label", "label_raw"]]
        for name, path in [(n, p_) for n, p_ in pool if n in names]:
            own = pd.read_parquet(path, columns=["date", "instrument", "pred"])
            own["date"] = pd.to_datetime(own["date"])
            own["instrument"] = own["instrument"].astype(str)
            own = own.merge(lab2, on=["date", "instrument"], how="inner")
            own = apply_cap(own, "cn", None, quiet=True)
            ric = float(np.nanmean([_pearson(_rank(g["pred"].to_numpy()),
                                             _rank(g["label"].to_numpy()))
                                    for _, g in own.groupby("date", sort=True) if len(g) >= 5]))
            p_own = portfolio(smooth(own, HEADLINE_SPAN), q_enter=0.10, q_exit=HEADLINE_EXIT)
            del own

            d1 = base_com.copy()
            d1["pred"] = wide[name].to_numpy()
            d1 = apply_cap(d1, "cn", None, quiet=True)
            p_com = portfolio(smooth(d1, HEADLINE_SPAN), q_enter=0.10, q_exit=HEADLINE_EXIT)
            del d1

            mr = float(M_raw_all.loc[name].drop(name).mean())
            strength[name] = {"rankic_headline_own": ric, "mean_rho_raw": mr,
                              "own": {k: v for k, v in p_own.items() if not k.startswith("_")},
                              "common": {k: v for k, v in p_com.items() if not k.startswith("_")}}
            print(f"{name:<20} {ric:>+9.4f} {p_own['gross_bps']:>+9.2f} "
                  f"{p_own['turnover']:>8.1%} {p_own['net10']:>+9.2f} {p_own['ir10']:>+8.2f} | "
                  f"{p_com['net10']:>+11.2f} {p_com['net10'] - p_own['net10']:>+8.2f} {mr:>9.3f}")
        del lab2

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"members": names, "n_common_rows": int(len(wide)),
                   "ortho_years": ortho_years,
                   "betas": betas,
                   "rho_raw_all": M_raw_all.to_dict(),
                   "rho_raw_sub": M_raw_sub.to_dict(),
                   "rho_ortho": M_ortho.to_dict(),
                   "strength": strength}, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n[json] {args.out_json}")


if __name__ == "__main__":
    main()
