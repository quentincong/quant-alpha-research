"""
tune_gbdt.py — tune LightGBM hyperparameters specifically for China A-shares

**Why**: the current GBDT parameters (num_leaves=31 / min_child_samples=200 / lr=0.05 ...)
  were originally chosen for **US equities**, then applied unchanged to A-shares, and have
  **never been tuned for A-shares**. There is also an obvious clue: `best_round` reaches
  271/235/106 in the A-share folds, and **performance is still improving after hundreds of
  boosting rounds**. This suggests **capacity limitation (underfitting)**, whereas US equities
  stop early after 8/2/5 rounds (there is nothing more to learn). Scan capacity-related parameters first.

**Discipline: select parameters on validation data and report on test data.**
  Rank each configuration by its **mean within-fold validation RankIC**. Validation is already
  the segment used for early stopping and lies inside the training window. **Predict the test
  year only once and never use it for selection.** The test column in the table below is for
  **inspection**, not **selection**. Selecting parameters on test data would leak the test set
  into model selection.

Usage:
  python model/tune_gbdt.py --data dataset/alphaFactor/dataset_alpha158_cn.parquet
"""
from __future__ import annotations
import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walkforward as W


def make_args(base, **over):
    ns = argparse.Namespace(**vars(base))
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=W.FULL_PATH)
    ap.add_argument("--seed", type=int, default=0)
    args_cli = ap.parse_args()

    base = argparse.Namespace(
        data=args_cli.data, smoke=False, rounds=2000, patience=100, val_days=120,
        embargo=2, min_history_years=2, min_train_days=250, min_test_days=60,
        seed=args_cli.seed, set=None, tag="", no_log=True, no_preds=True)

    grid = [dict(num_leaves=nl, min_child_samples=mcs, learning_rate=lr)
            for nl, mcs, lr in itertools.product([31, 63, 127], [200, 50], [0.05])] + \
           [dict(num_leaves=63, min_child_samples=mcs, learning_rate=0.02) for mcs in (200, 50)]

    print(f"[tune] {len(grid)} configurations × 6 folds, about 2 minutes each\n")
    print(f"{'num_leaves':>11} {'min_child':>10} {'lr':>6} | {'mean VAL':>9} (selection metric) | "
          f"{'mean TEST':>9} {'pos':>4} {'mean rounds':>12} {'sec':>5}")
    print("-" * 92)
    rows = []
    for g in grid:
        t0 = time.time()
        a = make_args(base, set=[f"{k}={v}" for k, v in g.items()])
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):       # Suppress noisy within-fold output.
            res = W.run(a)
        val = float(np.mean([r["val_rankic"] for r in res]))
        tst = float(np.mean([r["test_rankic"] for r in res]))
        pos = sum(r["test_rankic"] > 0 for r in res)
        rnd = float(np.mean([r["best_round"] for r in res]))
        rows.append({**g, "val": val, "test": tst, "pos": pos, "rounds": rnd})
        print(f"{g['num_leaves']:>11} {g['min_child_samples']:>10} {g['learning_rate']:>6} | "
              f"{val:>+9.4f}             | {tst:>+9.4f} {pos:>3}/6 {rnd:>12.0f} {time.time()-t0:>5.0f}")

    print("-" * 92)
    best = max(rows, key=lambda r: r["val"])
    cur = [r for r in rows if r["num_leaves"] == 31 and r["min_child_samples"] == 200
           and r["learning_rate"] == 0.05][0]
    print(f"\nCurrent parameters : leaves={cur['num_leaves']} mcs={cur['min_child_samples']} "
          f"lr={cur['learning_rate']} → val {cur['val']:+.4f} | test {cur['test']:+.4f}")
    print(f"Selected by VAL   : leaves={best['num_leaves']} mcs={best['min_child_samples']} "
          f"lr={best['learning_rate']} → val {best['val']:+.4f} | test {best['test']:+.4f}")
    print(f"\nTest change {best['test'] - cur['test']:+.4f} — **examined only after selection**; not used for selection.")


if __name__ == "__main__":
    main()
