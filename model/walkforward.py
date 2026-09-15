"""
walkforward.py — Multi-year mini walk-forward (distinguish an unlucky 2023 from weak features)

Background (continuing from 2026-07-08): after sweeping loss, recipe, and lr, MLP≈0.004
  and GBDT≈0.0105 consistently plateaued near ~0.01. But **every conclusion depended on
  the single valid=2023 year**. This script tests that premise with the same strongly
  regularized GBDT, repeatedly training on all data before year Y and testing on year Y,
  to obtain an independent OOS RankIC for every year. Interpretation:
    - Every year is ~0.01       → weak features are **confirmed** → inspect feature design.
    - 2023 is unusually low and other years are 0.03+ → the **validation window was unlucky**
      → upgrade to a formal walk-forward.

The conventions exactly match the existing scripts: the same parquet, the same daily RankIC
  evaluation, and the same strongly regularized GBDT parameters (by directly reusing functions
  from train_lgbm). Only the fixed train/valid/test split is replaced by calendar-year rolling.

Three look-ahead safeguards are fundamental to this kind of backtest and demonstrate rigor:
  1. **The early-stopping val is cut from the end of the training window** (the final
     --val_days trading days); the test year is **never used to select the round count**.
     Otherwise the model would be tuned on the answer. The test year is predicted only once.
  2. **Year-boundary embargo**: label buys at T+1 and sells at T+2, so a sample on a given day
     knows prices two days ahead. Including the final two days before the test year in training
     would peek into the test year, so drop the final --embargo days from the training window.
  3. **Fit/val-boundary embargo** (added 2026-08-30): the same logic applies inside the training
     window. Labels from the final two fit days reach the first two val days. The original
     implementation isolated boundary 2 but not the boundary between 2 and 1, letting the
     early-stopping val receive future information from fit. This **does not contaminate test
     metrics** because test is predicted only once; it affects only round selection. However,
     multi-fold best_round values were only 1~5 and the val curves were very flat, making round
     selection the script's weakest link, so this embargo was added.

The window is **expanding**: train on every year before Y. A sliding window that trains only on
  the latest N years is another option, but this mini version does not implement it.

Usage (from the repository root, after conda activate torch-env):
  python model/walkforward.py --smoke     # Five stocks; verify mechanics in seconds (ignore the numbers)
  python model/walkforward.py             # Full run
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

# Reuse shared components from existing scripts to guarantee identical conventions.
import train_mlp as TM                                  # Access the features module TM.F and path constants
from train_mlp import FULL_PATH, SMOKE_PATH, _pearson, _rank, provenance  # noqa: F401
from train_lgbm import build_day_groups, daily_ic       # Reuse daily grouping and daily IC

WF_LOG       = "model/walkforward.jsonl"
WF_LOG_SMOKE = "model/walkforward_smoke.jsonl"

# Strongly regularized GBDT recipe matching train_lgbm.run() (standard for low-SNR financial data)
LGBM_PARAMS = dict(
    objective="regression",
    metric="None",              # Disable default l2; use only our daily RankIC feval
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


def params_for(args) -> dict:
    """LGBM_PARAMS plus per-run overrides (seed / --set k=v).

    Originally, seed and all hyperparameters were hard-coded in a module constant, so both
    averaging across seeds and tuning separately for A-shares required source changes.
    Per-run overrides make both operations available from the command line.
    """
    p = dict(LGBM_PARAMS)
    # Set only seed and let LightGBM derive bagging_seed / feature_fraction_seed.
    # Setting those two explicitly changes the random stream, so seed=0 no longer reproduces
    # historical results (a pitfall found on 2026-08-30).
    p["seed"] = args.seed
    for kv in (args.set or []):
        k, v = kv.split("=", 1)
        p[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    return p


def load_panel_year(path: str, label_file: str | None = None):
    """Clean like train_mlp.load_panel, but also return each row's calendar year for walk-forward folds.

    Returns X[N,158] / y[N] / day_id[N] (global trading-day number) / year[N] (such as 2023)
    / meta[N,3].

    meta is used by `evaluate.py`: **`label` is a cross-sectional robust-z, not the return
    itself**. Calculating real decile-portfolio returns, turnover, and cost curves requires
    `label_raw` (the actual forward return) and identity columns (date, instrument). Either
    label gives the same IC because a within-day z-score is monotonic and RankIC is unchanged,
    but only label_raw can answer how many basis points the selected stocks earned.
    """
    df = pd.read_parquet(path)
    if label_file:
        # Replace labels **without touching features**: the feature table is 2.5GB, so copying it
        # just to change labels is wasteful. The external label has only four columns,
        # (date,instrument,label_raw,label), and can be joined by index.
        # Order matters: **join first**, then filter notna. The new label can cover **more** rows
        # than the old one (for example, --gate entry restores rows with failed exits); filtering
        # first would make those rows permanently unavailable.
        lab = pd.read_parquet(label_file, columns=["date", "instrument", "label_raw", "label"])
        lab["date"] = pd.to_datetime(lab["date"])
        lab["instrument"] = lab["instrument"].astype(str)
        lab = lab.set_index(["date", "instrument"])
        n_before = len(df)
        df = df.drop(columns=["label", "label_raw"]).join(lab, how="left").sort_index()
        print(f"[label] external label {label_file}: {lab.shape[0]:,} rows → "
              f"among {n_before:,} panel rows, {df['label'].notna().sum():,} have labels")
    df = df[df["label"].notna()]                         # Drop tail rows without future returns (as in load_panel)
    feat_cols = TM.F.feature_names()
    X = np.nan_to_num(df[feat_cols].to_numpy(dtype=np.float32), nan=0.0)
    y = df["label"].to_numpy(dtype=np.float32)
    dates = df.index.get_level_values("date")
    day_id = pd.factorize(dates, sort=True)[0].astype(np.int64)
    year = pd.DatetimeIndex(dates).year.to_numpy().astype(int)
    meta = pd.DataFrame({
        "date": pd.DatetimeIndex(dates),
        "instrument": df.index.get_level_values("instrument").astype(str),
        "label_raw": df["label_raw"].to_numpy(dtype=np.float64),
    })
    return X, y, day_id, year, meta


def run_fold(X, y, day_id, year, meta, test_year: int, args):
    """One fold: train on all years before test_year (tail val for early stopping) → evaluate OOS RankIC on test_year.

    Return (metrics dict, row-level prediction DataFrame for the fold's test segment).
    **Predictions must be saved**: the original implementation discarded preds after computing
    IC, forcing retraining whenever the evaluation convention changed. Once saved, evaluate.py
    can compute any number of metrics offline from a single training run.
    """
    test_mask = year == test_year
    test_days = np.unique(day_id[test_mask])
    if test_days.size < args.min_test_days:
        print(f"  [skip] {test_year}: test days {test_days.size} < {args.min_test_days}")
        return None, None
    first_test_day = int(test_days.min())

    # 1. Embargo: retain training rows only outside the embargo before the test year starts,
    # avoiding a cross-year look-ahead into T+2.
    train_pool = (year < test_year) & (day_id < first_test_day - args.embargo)
    pool_days = np.unique(day_id[train_pool])
    if pool_days.size < args.min_train_days + args.val_days:
        print(f"  [skip] {test_year}: {pool_days.size} usable training days are insufficient "
              f"(need {args.min_train_days}+{args.val_days})")
        return None, None

    # 2. Cut the final val_days trading days from the **tail** of the training window for
    #    early-stopping val; use the rest for fit. The fit segment also needs an embargo:
    #    the label on the final fit day uses T+2 prices and overlaps the first two val days.
    #    Without isolation, early-stopping val receives future information from fit. This leak
    #    affects only round selection, not test metrics themselves, but round selection is the
    #    script's weakest link because multi-fold best_round values were only 1~5.
    val_start_day = int(pool_days[-args.val_days])
    val_mask = train_pool & (day_id >= val_start_day)
    fit_mask = train_pool & (day_id < val_start_day - args.embargo)

    Xf, yf = X[fit_mask], y[fit_mask]
    Xv, yv, dv = X[val_mask], y[val_mask], day_id[val_mask]
    Xt, yt, dt = X[test_mask], y[test_mask], day_id[test_mask]
    val_groups = build_day_groups(dv)
    test_groups = build_day_groups(dt)

    # Early stopping: daily RankIC on **val** (not test); put rankic first with first_metric_only.
    def feval_ic(preds, _eval_data):
        ic, ric = daily_ic(preds, yv, val_groups)
        return [("rankic", ric, True), ("ic", ic, True)]

    dtrain = lgb.Dataset(Xf, label=yf)
    dvalid = lgb.Dataset(Xv, label=yv, reference=dtrain)
    evals_result: dict = {}
    model = lgb.train(
        params_for(args), dtrain, num_boost_round=args.rounds,
        valid_sets=[dvalid], valid_names=["val"], feval=feval_ic,
        callbacks=[
            lgb.early_stopping(args.patience, first_metric_only=True, verbose=False),
            lgb.record_evaluation(evals_result),
        ],
    )

    val_rk = evals_result["val"]["rankic"]
    best_iter = model.best_iteration or len(val_rk)      # Round count selected by early stopping on val RankIC
    best_val_ric = float(val_rk[best_iter - 1])

    # Predict the test year only once here, using best_iter selected on val, for honest OOS results.
    preds_test = model.predict(Xt, num_iteration=best_iter)
    test_ic, test_ric = daily_ic(preds_test, yt, test_groups)

    n_fit_days = int(np.unique(day_id[fit_mask]).size)
    print(f"  {test_year}: fit {n_fit_days} days + val {args.val_days} days → test {test_days.size} days "
          f"| best_round={best_iter} | val RankIC={best_val_ric:+.4f} "
          f"| **test RankIC={test_ric:+.4f}** (IC={test_ic:+.4f})")

    preds_df = meta.loc[test_mask].copy()
    preds_df["pred"] = preds_test.astype(np.float64)
    preds_df["label"] = yt.astype(np.float64)            # Cross-sectional robust-z (used for IC)
    preds_df["test_year"] = test_year

    return {
        "test_year": test_year,
        "n_fit_days": n_fit_days, "n_val_days": args.val_days,
        "n_test_days": int(test_days.size),
        "best_round": int(best_iter),
        "val_rankic": best_val_ric,
        "test_rankic": float(test_ric), "test_ic": float(test_ic),
    }, preds_df


def run(args):
    path = SMOKE_PATH if args.smoke else args.data
    print(f"[data] reading {path}")
    X, y, day_id, year, meta = load_panel_year(path, args.label_file)
    years = sorted(np.unique(year).tolist())
    print(f"[data] N={len(y)} rows, years {years[0]}–{years[-1]}, {day_id.max() + 1} days total")

    # Test every year with enough prior training history (default: start in the third data year,
    # ensuring at least two earlier years for training).
    test_years = [yr for yr in years if yr >= years[0] + args.min_history_years]
    print(f"[wf] expanding walk-forward, test years: {test_years}  "
          f"(embargo={args.embargo} days, val={args.val_days} days)\n")

    results, pred_frames = [], []
    for yr in test_years:
        r, pf = run_fold(X, y, day_id, year, meta, yr, args)
        if r is not None:
            results.append(r)
            pred_frames.append(pf)

    if results:
        rks = [r["test_rankic"] for r in results]
        print("\n===== Walk-forward summary (independent OOS by year) =====")
        print(f"{'year':>6} {'test_days':>10} {'best_round':>11} "
              f"{'val_RankIC':>11} {'test_RankIC':>12} {'test_IC':>9}")
        for r in results:
            print(f"{r['test_year']:>6} {r['n_test_days']:>10} {r['best_round']:>11} "
                  f"{r['val_rankic']:>+11.4f} {r['test_rankic']:>+12.4f} {r['test_ic']:>+9.4f}")
        print("-" * 62)
        print(f"test RankIC: mean {np.mean(rks):+.4f} | median {np.median(rks):+.4f} | "
              f"min {np.min(rks):+.4f} | max {np.max(rks):+.4f} | "
              f"positive years {sum(v > 0 for v in rks)}/{len(rks)}")
        print("=" * 62)

    # ---- Save predictions as input to evaluate.py ----
    # Train once and evaluate repeatedly. Include market in the filename so US stocks
    # (negative control) and A-shares (primary market) do not overwrite each other.
    if pred_frames and not args.no_preds:
        prov = provenance(path)
        preds = pd.concat(pred_frames, ignore_index=True)
        preds_path = (f"model/preds_walkforward_{prov['market']}"
                      f"{'_smoke' if prov['smoke'] else ''}{args.tag}.parquet")
        preds.to_parquet(preds_path, index=False)
        print(f"[preds] {len(preds):,} prediction rows saved to {preds_path} "
              f"({preds['date'].min().date()} → {preds['date'].max().date()})")

    # Persist one line per complete run, including all folds, without contaminating experiments.jsonl.
    log_path = WF_LOG_SMOKE if args.smoke else WF_LOG
    if args.no_log:
        print("[log] --no-log: skipping jsonl for this run (avoids flooding the ledger during sweeps)")
        return results
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            **provenance(path),
                            "label_file": args.label_file or "builtin",
                            # Tier C = measured: the test year is predicted once and participates in no selection.
                            "metric_tier": "C",
                            "window": "expanding", "embargo": args.embargo,
                            "val_days": args.val_days, "seed": args.seed,
                            "gbdt_params": params_for(args), "results": results},
                           ensure_ascii=False) + "\n")
    print(f"[log] results appended to {log_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=FULL_PATH)
    ap.add_argument("--smoke", action="store_true", help="Verify mechanics quickly with five stocks (ignore the numbers)")
    ap.add_argument("--rounds", type=int, default=2000, help="Maximum boosting rounds per fold (with early stopping)")
    ap.add_argument("--patience", type=int, default=100, help="Early-stopping patience (consecutive rounds without val improvement)")
    ap.add_argument("--val_days", type=int, default=120, help="Trading days cut from the training-window tail for early-stopping val")
    ap.add_argument("--embargo", type=int, default=2, help="Year-boundary embargo days (= label horizon; prevents look-ahead)")
    ap.add_argument("--min_history_years", type=int, default=2, help="Minimum prior training years required to test a year")
    ap.add_argument("--min_train_days", type=int, default=250, help="Minimum fit trading days; otherwise skip the fold")
    ap.add_argument("--min_test_days", type=int, default=60, help="Minimum test-year trading days; otherwise skip the fold")
    ap.add_argument("--seed", type=int, default=0, help="GBDT random seed (for averaging across seeds)")
    ap.add_argument("--set", action="append", metavar="KEY=VAL",
                    help="Override an LGBM hyperparameter; repeatable, e.g. --set num_leaves=63")
    ap.add_argument("--label-file", dest="label_file", default=None,
                    help="External label parquet(date,instrument,label_raw,label), replacing the built-in label; "
                         "features stay unchanged; see make_vwap_label.py")
    ap.add_argument("--tag", default="", help="Prediction-dump filename suffix to prevent runs from overwriting one another")
    ap.add_argument("--no-log", dest="no_log", action="store_true", help="Do not write walkforward.jsonl")
    ap.add_argument("--no-preds", dest="no_preds", action="store_true", help="Do not save predictions (reduces IO during sweeps)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
