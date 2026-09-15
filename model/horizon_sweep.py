"""
horizon_sweep.py — label-horizon sweep (test whether low one-day SNR is the root cause)

Diagnostic recap: validation RankIC for mse/soft_ic/lgbm is stuck near ~0.01, and even
  a non-overfitting GBDT cannot improve it, so the bottleneck is upstream. The most suspicious
  upstream factor is the **label horizon**: the current label is the one-day T+1→T+2 return,
  which has the lowest signal-to-noise ratio.

This script **reuses the existing parquet** (the 158 features are already preprocessed and
  close_adj is present, so neither is recomputed). It changes only the label holding period,
  runs one GBDT per horizon, and checks whether the ceiling rises with the horizon.

Label definition (H = holding days; avoids look-ahead: decide after the T close and enter at the T+1 close):
    label_raw(H) = close_adj.shift(-(1+H)) / close_adj.shift(-1) - 1
    label(H)     = same-day cross-sectional robust z-score (median/MAD*1.4826, clipped to ±3)
  H=1 is exactly identical to the current label (shift(-2)/shift(-1)-1).

Evaluation: average validation daily RankIC/IC across days and early-stop on RankIC, matching
  train_lgbm for comparability. Note that closes after 2023 are also in the panel (the test segment),
  so H-day forward returns for 2023 validation rows are fully available. This is not look-ahead:
  actual future returns are used only as prediction targets; features do not see the future.

Usage (from the repository root, after conda activate torch-env):
  python model/horizon_sweep.py --smoke               # Quick execution check.
  python model/horizon_sweep.py                       # Full run, H=1,5,10,20.
  python model/horizon_sweep.py --horizons 1 3 5 10 20 21
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

# Reuse shared train_lgbm / train_mlp components to keep evaluation comparable.
from train_lgbm import build_day_groups, daily_ic
from train_mlp import FULL_PATH, SMOKE_PATH, provenance
import features as F  # Importing train_mlp has already added dataset/alphaFactor to sys.path.

EPS = 1e-12
HORIZON_LOG = "model/horizon_sweep.jsonl"

# Use the same GBDT parameters as train_lgbm (low SNR → strong regularization).
PARAMS = dict(
    objective="regression", metric="None",
    learning_rate=0.05, num_leaves=31, min_child_samples=200,
    feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
    lambda_l2=1.0, num_threads=0, seed=0, verbosity=-1,
)


def robust_z(s: pd.Series) -> pd.Series:
    """Match build_dataset._robust_z: robust median/MAD standardization clipped to ±3.

    For the final (1+H) days, the entire cross-section of labels is NaN because no future
    returns are available. Taking the median of all NaNs triggers NumPy's "Mean of empty slice"
    warning. The ok mask later drops these days, so return them unchanged without computing a median.
    """
    if s.notna().sum() == 0:
        return s
    med = s.median()
    mad = (s - med).abs().median()
    return ((s - med) / (mad * 1.4826 + EPS)).clip(-3, 3)


def make_label(close: pd.Series, H: int) -> np.ndarray:
    """Recompute the label for horizon H: enter at T+1, exit at T+1+H, then robust-z by day."""
    g = close.groupby(level="instrument")
    raw = g.shift(-(1 + H)) / g.shift(-1) - 1.0
    lab = raw.groupby(level="date").transform(robust_z)
    return lab.to_numpy(dtype=np.float32)


def train_gbdt(Xtr, ytr, Xva, yva, groups, rounds, patience):
    """Run one GBDT, early-stop on validation RankIC, and return the best result."""
    def feval_ic(preds, _):
        ic, ric = daily_ic(preds, yva, groups)
        return [("rankic", ric, True), ("ic", ic, True)]

    dtrain = lgb.Dataset(Xtr, label=ytr)
    dvalid = lgb.Dataset(Xva, label=yva, reference=dtrain)
    ev: dict = {}
    lgb.train(
        PARAMS, dtrain, num_boost_round=rounds,
        valid_sets=[dvalid], valid_names=["valid"], feval=feval_ic,
        callbacks=[
            lgb.early_stopping(patience, first_metric_only=True),
            lgb.record_evaluation(ev),
            lgb.log_evaluation(period=100),
        ],
    )
    ric, ic = ev["valid"]["rankic"], ev["valid"]["ic"]
    rk = np.asarray(ric, dtype=float)
    bi = int(np.nanargmax(rk)) if not np.all(np.isnan(rk)) else 0
    return dict(best_ric=ric[bi], best_ic=ic[bi], best_round=bi + 1,
                n_rounds=len(ric), final_ric=ric[-1])


def run(args):
    path = SMOKE_PATH if args.smoke else args.data
    feat_cols = F.feature_names()
    print(f"[data] reading {path} (features + close_adj + segment only; old label is not read)")
    df = pd.read_parquet(path, columns=feat_cols + ["close_adj", "segment"])

    # Extract features once for reuse across horizons, then release the large DataFrame to save memory.
    feat = np.nan_to_num(df[feat_cols].to_numpy(dtype=np.float32), nan=0.0)
    seg = df["segment"].to_numpy()
    day_id = pd.factorize(df.index.get_level_values("date"), sort=True)[0].astype(np.int64)
    close = df["close_adj"]                              # Retain for label computation at each horizon.
    del df

    print(f"[data] N={feat.shape[0]}  features={feat.shape[1]}  horizons={args.horizons}\n")

    results = []
    for H in args.horizons:
        y = make_label(close, H)
        ok = ~np.isnan(y)                               # Drop trailing rows whose labels are NaN.
        tr = (seg == "train") & ok
        va = (seg == "valid") & ok
        Xtr, ytr = feat[tr], y[tr]
        Xva, yva = feat[va], y[va]
        groups = build_day_groups(day_id[va])
        print(f"===== horizon H={H}  (buy T+1, sell T+{1+H})  "
              f"train {tr.sum()} rows  valid {va.sum()} rows/{len(groups)} days =====")
        r = train_gbdt(Xtr, ytr, Xva, yva, groups, args.rounds, args.patience)
        r["horizon"] = H
        results.append(r)
        print(f"  → best RankIC={r['best_ric']:+.4f} @round {r['best_round']}  "
              f"IC={r['best_ic']:+.4f}  final RankIC={r['final_ric']:+.4f}\n")

    # ---- Summary table ----
    print("===== Horizon sweep summary (validation daily RankIC averaged across days; GBDT ceiling) =====")
    print(f"{'H(days)':>9} {'bestRankIC':>11} {'@round':>7} {'bestIC':>9} "
          f"{'finalRankIC':>12} {'rounds':>7}")
    for r in results:
        print(f"{r['horizon']:>9} {r['best_ric']:>+11.4f} {r['best_round']:>7} "
              f"{r['best_ic']:>+9.4f} {r['final_ric']:>+12.4f} {r['n_rounds']:>7}")
    print("=" * 62)
    best = max(results, key=lambda r: r["best_ric"])
    print(f"[conclusion] best horizon = H={best['horizon']}, best RankIC={best['best_ric']:+.4f}")

    # ---- Archive ----
    if not args.smoke:
        os.makedirs(os.path.dirname(HORIZON_LOG), exist_ok=True)
        with open(HORIZON_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                **provenance(path), "metric_tier": "A",
                                "results": results}, ensure_ascii=False) + "\n")
        print(f"[log] saved {HORIZON_LOG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=FULL_PATH)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=80)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
