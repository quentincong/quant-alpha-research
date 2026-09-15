"""
compare_lgbm_cn_us.py — side-by-side LightGBM IC comparison: US equities vs A-shares

Motivation: walk-forward testing confirmed the "~0.01 RankIC ceiling" for US Alpha158
  (weak features are the bottleneck; see docs/research_log.md#r11 and
  docs/research_log.md#r12). A-shares are retail-dominated, and
  price-volume/reversal factors have historically been stronger, so the ceiling may be higher.
  Apply the **exact same LGBM recipe** to both datasets and evaluate them identically to answer
  a cross-market question: "With the same factors and model, is A-share predictability
  systematically higher than US-equity predictability?"

Why the comparison is apples-to-apples:
  - The same load_panel implementation (in-house Alpha158; US uses (H+L+C)/3 as vwap,
    and CN uses post-adjusted prices).
  - The same LGBM hyperparameters (the strongly regularized train_lgbm.py recipe, unchanged).
  - The same metric: average daily RankIC across days; early stopping also uses validation RankIC.
  - The only variable is the market (dataset).

One addition beyond train_lgbm.py: **evaluate both validation and test segments**.
  Validation is the in-sample segment used to select the boosting round; test is strictly OOS.
  Only test results count for the cross-market comparison because early stopping has already
  "looked at" validation.

Usage (from the repository root, after conda activate torch-env):
  python model/compare_lgbm_cn_us.py --smoke      # Five tickers per market; verifies execution (IC is meaningless).
  python model/compare_lgbm_cn_us.py              # Full comparison (a few minutes).
"""
from __future__ import annotations
import argparse

import numpy as np
import lightgbm as lgb

from train_mlp import load_panel, _pearson, _rank
from train_lgbm import build_day_groups, daily_ic, _safe_mean  # noqa: F401  (Reuse the same daily IC.)

US_FULL   = "dataset/alphaFactor/dataset_alpha158.parquet"
US_SMOKE  = "dataset/alphaFactor/dataset_alpha158_smoke.parquet"
CN_FULL   = "dataset/alphaFactor/dataset_alpha158_cn.parquet"
CN_SMOKE  = "dataset/alphaFactor/dataset_alpha158_cn_smoke.parquet"

# Copy the strongly regularized train_lgbm.py recipe unchanged for both markets,
# ensuring that only the data varies.
PARAMS = dict(
    objective="regression",
    metric="None",
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=200,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l2=1.0,
    num_threads=0,
    seed=0,
    verbosity=-1,
)


def run_market(name: str, path: str, rounds: int, patience: int) -> dict:
    """Train LGBM on one market and return RankIC and metadata for validation and test segments."""
    print(f"\n{'='*66}\n[{name}] reading {path}\n{'='*66}")
    X, y, day_id, segment = load_panel(path)

    tr = segment == "train"
    va = segment == "valid"
    te = segment == "test"
    Xtr, ytr = X[tr], y[tr]
    Xva, yva, dva = X[va], y[va], day_id[va]
    Xte, yte, dte = X[te], y[te], day_id[te]

    va_groups = build_day_groups(dva)
    te_groups = build_day_groups(dte)
    print(f"[{name}] train {Xtr.shape[0]} rows/{np.unique(day_id[tr]).size} days  "
          f"valid {Xva.shape[0]} rows/{len(va_groups)} days  "
          f"test {Xte.shape[0]} rows/{len(te_groups)} days")

    # Early stopping uses validation daily RankIC, the same metric used to select the NN's best epoch.
    def feval_ic(preds, _d):
        ic, ric = daily_ic(preds, yva, va_groups)
        return [("rankic", ric, True), ("ic", ic, True)]

    dtrain = lgb.Dataset(Xtr, label=ytr)
    dvalid = lgb.Dataset(Xva, label=yva, reference=dtrain)

    evals: dict = {}
    model = lgb.train(
        PARAMS, dtrain,
        num_boost_round=rounds,
        valid_sets=[dvalid], valid_names=["valid"],
        feval=feval_ic,
        callbacks=[
            lgb.early_stopping(patience, first_metric_only=True),
            lgb.record_evaluation(evals),
            lgb.log_evaluation(period=100),
        ],
    )

    va_ric_hist = np.asarray(evals["valid"]["rankic"], dtype=float)
    best_idx = int(np.nanargmax(va_ric_hist)) if not np.all(np.isnan(va_ric_hist)) else 0
    best_round = best_idx + 1
    v_ic, v_ric = evals["valid"]["ic"][best_idx], evals["valid"]["rankic"][best_idx]

    # Test: predict with the round selected on validation; strictly OOS.
    pte = model.predict(Xte, num_iteration=best_round)
    t_ic, t_ric = daily_ic(pte, yte, te_groups)

    print(f"[{name}] best_round={best_round}  "
          f"valid RankIC={v_ric:+.4f} (IC={v_ic:+.4f})  "
          f"test RankIC={t_ric:+.4f} (IC={t_ic:+.4f})")

    return dict(name=name, best_round=best_round,
                train_rows=int(Xtr.shape[0]), valid_days=len(va_groups), test_days=len(te_groups),
                valid_ic=v_ic, valid_rankic=v_ric, test_ic=t_ic, test_rankic=t_ric)


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 74)
    print("  US equities vs A-shares · LightGBM daily RankIC (average across days; identical recipe)")
    print("=" * 74)
    hdr = f"{'market':<8}{'train rows':>10}{'valid days':>8}{'test days':>7}" \
          f"{'valid_IC':>10}{'valid_RIC':>11}{'test_IC':>10}{'test_RIC':>10}"
    print(hdr)
    print("-" * 74)
    for r in rows:
        print(f"{r['name']:<8}{r['train_rows']:>10}{r['valid_days']:>8}{r['test_days']:>7}"
              f"{r['valid_ic']:>+10.4f}{r['valid_rankic']:>+11.4f}"
              f"{r['test_ic']:>+10.4f}{r['test_rankic']:>+10.4f}")
    print("=" * 74)
    if len(rows) == 2:
        cn = next(r for r in rows if r["name"] == "CN")
        us = next(r for r in rows if r["name"] == "US")
        if us["test_rankic"] and not np.isnan(us["test_rankic"]) and us["test_rankic"] != 0:
            ratio = cn["test_rankic"] / us["test_rankic"]
            print(f"  test RankIC ratio  CN/US = {ratio:.2f}x   "
                  f"(CN {cn['test_rankic']:+.4f} vs US {us['test_rankic']:+.4f})")
        print("  Note: the test segments are approximately 2024-25 for both US and CN; the high A-share test result")
        print("        is driven mainly by the 2025 surge in retail participation (regime-dependent; see docs/research_log.md#r15).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="quick execution check using five-ticker smoke parquet files for each market")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--only", choices=["us", "cn"], help="run only one market")
    args = ap.parse_args()

    us_path = US_SMOKE if args.smoke else US_FULL
    cn_path = CN_SMOKE if args.smoke else CN_FULL

    rows = []
    if args.only != "cn":
        rows.append(run_market("US", us_path, args.rounds, args.patience))
    if args.only != "us":
        rows.append(run_market("CN", cn_path, args.rounds, args.patience))
    print_table(rows)


if __name__ == "__main__":
    main()
