"""
train_lgbm.py — LightGBM reference model (a "signal ceiling" probe)

Purpose: tree models are stable on low-SNR tabular data, include regularization, and perform
  well on factor panels, making them a strong baseline for this task. Use this model to answer
  the fork in the road:
    - If GBDT validation IC is also ≈0, the upstream signal in features/labels is absent,
      and further MLP tuning is futile.
    - If GBDT reaches 0.02–0.03, signal exists and the MLP's problem is overfitting,
      making additional regularization and early stopping worthwhile.

Directly comparable with the NN:
  - The same parquet, load_panel, and validation evaluation that averages daily IC across days.
  - A custom feval of daily RankIC drives **early stopping** (first_metric_only), equivalent
    to selecting the NN's best epoch by validation RankIC; model selection uses the same metric.
  - Results go to the same experiments.jsonl, and print_comparison automatically adds an `lgbm` row.

Tree models fit **row by row**, so training does not require one batch per day; that is needed
  only for a cross-sectional loss. **Evaluation** still computes IC by day to align with the NN.
  Using L2 regression as the objective and ranking IC for evaluation is the standard GBDT alpha
  recipe in industry (also used by Qlib).

Usage (from the repository root, after conda activate torch-env):
  python model/train_lgbm.py --smoke          # Five tickers, takes seconds; verify execution first.
  python model/train_lgbm.py                  # Full run.
"""
from __future__ import annotations
import argparse
from datetime import datetime

import numpy as np
import lightgbm as lgb

# Reuse train_mlp components to keep data, evaluation, and logging comparable.
from train_mlp import (
    FULL_PATH, SMOKE_PATH, EXP_LOG, EXP_LOG_SMOKE, provenance,
    load_panel, _pearson, _rank, log_result, print_comparison,
)


def build_day_groups(day_id: np.ndarray) -> list[np.ndarray]:
    """Group a data segment by day and return within-segment position indices for each day's IC.

    Unlike build_day_index, these are positions within this segment's array (0..len-1),
    because LightGBM predictions follow segment order.
    """
    order = np.argsort(day_id, kind="stable")
    sorted_days = day_id[order]
    cut = np.flatnonzero(np.diff(sorted_days) != 0) + 1
    return np.split(order, cut)


def _safe_mean(vals: list[float]) -> float:
    """Average across days after dropping NaNs caused by constant daily predictions.
    Return NaN if none remain. Avoid np.nanmean because it raises RuntimeWarning on all-NaN input."""
    v = [x for x in vals if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def daily_ic(preds: np.ndarray, y: np.ndarray, groups: list[np.ndarray]) -> tuple[float, float]:
    """Compute daily Pearson IC and RankIC, then average across days, matching train_mlp.eval_ic."""
    pic, ric = [], []
    for g in groups:
        if g.size < 5:                                 # Correlation is meaningless with too few stocks that day.
            continue
        p, t = preds[g], y[g]
        pic.append(_pearson(p, t))
        ric.append(_pearson(_rank(p), _rank(t)))
    return _safe_mean(pic), _safe_mean(ric)


def run(args):
    path = SMOKE_PATH if args.smoke else args.data
    print(f"[data] reading {path}")
    X, y, day_id, segment = load_panel(path)

    tr = segment == "train"
    va = segment == "valid"
    Xtr, ytr, dtr = X[tr], y[tr], day_id[tr]
    Xva, yva, dva = X[va], y[va], day_id[va]
    valid_groups = build_day_groups(dva)               # Used by feval and final evaluation.
    n_train_days = int(np.unique(dtr).size)
    n_valid_days = len(valid_groups)
    print(f"[data] train {Xtr.shape[0]} rows/{n_train_days} days  valid {Xva.shape[0]} rows/{n_valid_days} days")

    # ---- Custom evaluation: daily RankIC, used for early stopping to match NN selection. ----
    def feval_ic(preds, _eval_data):
        ic, ric = daily_ic(preds, yva, valid_groups)
        # Return a list with rankic first so first_metric_only=True uses it for early stopping.
        return [("rankic", ric, True), ("ic", ic, True)]

    dtrain = lgb.Dataset(Xtr, label=ytr)
    dvalid = lgb.Dataset(Xva, label=yva, reference=dtrain)

    # Low-SNR financial data needs strong regularization: few leaves, many samples per leaf,
    # column/row sampling, and an L2 penalty.
    params = dict(
        objective="regression",       # Trees fit pointwise L2; evaluation uses ranking IC.
        metric="None",                # Disable default L2 metric and use only the custom feval.
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,        # At least 200 samples per leaf to reduce overfitting.
        feature_fraction=0.7,         # Randomly use 70% of features per tree.
        bagging_fraction=0.7,
        bagging_freq=1,               # Randomly use 70% of samples each round.
        lambda_l2=1.0,
        num_threads=0,                # Use all CPU cores.
        seed=0,
        verbosity=-1,
    )

    evals_result: dict = {}
    print(f"\n[train] LightGBM  num_boost_round≤{args.rounds}  lr={params['learning_rate']} "
          f"num_leaves={params['num_leaves']}  early-stop patience={args.patience}\n")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.rounds,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=feval_ic,
        callbacks=[
            lgb.early_stopping(args.patience, first_metric_only=True),  # Early-stop on rankic.
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(period=50),
        ],
    )

    rankic_hist = evals_result["valid"]["rankic"]
    ic_hist = evals_result["valid"]["ic"]
    rk = np.asarray(rankic_hist, dtype=float)
    best_idx = int(np.nanargmax(rk)) if not np.all(np.isnan(rk)) else 0  # Select the best round by validation RankIC.
    best_ric, best_ic = rankic_hist[best_idx], ic_hist[best_idx]
    print(f"\n[done] training completed after {len(rankic_hist)} rounds, best_iteration={model.best_iteration}")
    print(f"[best] lgbm  best RankIC={best_ric:+.4f} @round {best_idx + 1}  (IC={best_ic:+.4f})")

    # ---- Write to the same comparison table; align schema with train_mlp by reusing epochs/lr/hidden. ----
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **provenance(path),
        "metric_tier": "A",          # best_val_* = training-history maximum on the early-stopping validation segment.
        "loss": "lgbm", "epochs": len(rankic_hist), "lr": params["learning_rate"],
        "hidden": params["num_leaves"],                # Store num_leaves in the 'hid' column.
        "device": "cpu", "n_rows": int(len(y)),
        "train_days": n_train_days, "valid_days": n_valid_days,
        "best_epoch": best_idx + 1,
        "best_val_rankic": best_ric, "best_val_ic": best_ic,
        "final_val_rankic": rankic_hist[-1], "final_val_ic": ic_hist[-1],
        "history": [{"round": i + 1, "val_ic": ic_hist[i], "val_rankic": rankic_hist[i]}
                    for i in range(len(rankic_hist))],
    }
    log_path = EXP_LOG_SMOKE if args.smoke else EXP_LOG
    log_result(record, log_path)
    print(f"[log] appended result to {log_path}")
    print_comparison(log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=FULL_PATH)
    ap.add_argument("--smoke", action="store_true", help="quick verification using a five-ticker smoke parquet")
    ap.add_argument("--rounds", type=int, default=2000, help="maximum boosting rounds with early stopping")
    ap.add_argument("--patience", type=int, default=100, help="early-stop after this many rounds without improvement")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
