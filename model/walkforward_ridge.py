"""
walkforward_ridge.py — linear floor member: Ridge / ElasticNet walk-forward

**Why include it** (planned in docs/research_log.md#r40, run in docs/research_log.md#r42):
  The current pool contains GBDT (trees) and MLP (network); both are nonlinear and consume the
  same 158-column table. Adding a **linear** member makes it **structurally the most different**:
  it cannot even express interactions, so its errors must differ from those of trees/networks.
  It is also the **floor** requested on 2026-08-29. "How far can ridge regression on 158 factors go?"
  is the first question a reader will ask; without this line, GBDT's +0.04 has no reference.

**Fold geometry matches `walkforward.py` line by line**: expanding window, year-boundary embargo,
  validation split from the training-window tail, and another embargo at the fit/validation boundary.
  The mask construction is **copied** rather than imported because `walkforward.run_fold` couples the
  model and masks inside one function and cannot separate them. `walkforward_mlp.py` established the
  same precedent. All three mask implementations must match line by line; comments below map them.
  A self-check prints n_fit_days / n_val_days / n_test_days for direct comparison with
  `walkforward.jsonl` / `walkforward_mlp.jsonl`.

**α selection must be nested**: use the **same position and data segment** as GBDT selecting
  `best_iteration` on validation and MLP selecting the epoch on validation. Select α by daily
  RankIC on the validation tail of the training window, and **predict the test year only once**.
  Never tune α on test.

**Implementation: closed form**, not an iterative solver.
  Ridge's normal equations are `(XᵀX + αI) w = Xᵀy`, a 158×158 system whose Cholesky solve takes
  under 1 millisecond. `XᵀX` / `Xᵀy` depend only on the fit segment and are **independent of α**,
  so sweeping a full α grid has approximately zero marginal cost; this is why the linear member
  is nearly free. In pandas terms, this inverts the 158-column covariance matrix once rather than
  "training." ElasticNet (L1+L2) **has no closed form** and requires coordinate descent. This file
  uses sklearn under `--model elasticnet`, but **ridge is the default** because L1 sparsification
  over 158 standardized factors is a separate selection procedure outside this round.

**Standardization**: features were standardized cross-sectionally during dataset construction.
  Standardize once more using **fit-segment** mean/standard deviation solely for numerical conditioning.
  Statistics come only from fit and are reused for validation/test, so there is no leakage.
  Center y using fit only as well, eliminating the need for an explicit intercept.

The output schema exactly matches the GBDT/MLP versions (date, instrument, label_raw, pred, label,
  test_year), so `evaluate.py` / `turnover_study.py` need no changes.

Usage:
  python model/walkforward_ridge.py \\
      --data dataset/alphaFactor/dataset_alpha158_cn.parquet \\
      --label-file dataset/alphaFactor/label_cn_vwap_entry.parquet --tag _ridge
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mlp import FULL_PATH, provenance
from train_lgbm import build_day_groups, daily_ic     # Same daily IC as the other two members.
from walkforward import load_panel_year               # Same loader → same row set.

WF_LOG = "model/walkforward_ridge.jsonl"

# α grid spans ten orders of magnitude. X is standardized, so the diagonal of XᵀX ≈ N
# (full A-share N≈2.3e6). Effective shrinkage requires α on the same scale as N, so **the grid's
# upper bound must scale with sample size**. Otherwise the sweep does nothing: α≈1e2 suffices
# for a 9.6k-row smoke set, while the full set needs 1e6~1e8. Ridge α is not dimensionless.
ALPHA_GRID = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]


def ridge_path(XtX: np.ndarray, Xty: np.ndarray, alphas) -> dict:
    """Given fit-segment normal equations, solve w over the full α grid.

    Compute `XtX`/`Xty` once (O(N·p²), the entire cost here). Each α adds only a 158×158
    Cholesky solve (O(p³), microseconds). On numerical instability, fall back to
    `np.linalg.lstsq` (pseudoinverse) and record the fallback.
    """
    p = XtX.shape[0]
    eye = np.eye(p)
    out = {}
    for a in alphas:
        A = XtX + a * eye
        try:
            w = np.linalg.solve(A, Xty)          # LAPACK gesv; equivalent to Cholesky for symmetric positive-definite input.
            deviated = False
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, Xty, rcond=None)[0]
            deviated = True
        out[a] = (w, deviated)
    return out


def enet_fit(Xf, yf, alpha, l1_ratio, max_iter, tol):
    """ElasticNet has no closed form and uses sklearn coordinate descent; the default path does not enter here.

    This path **must** materialize the full fit segment as float64 because sklearn requires one
    contiguous array. Full A-share data are 2.3M×158×8B ≈ 2.9GB, so it is not the default path;
    check available memory before using it.
    """
    from sklearn.linear_model import ElasticNet
    m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False,
                   max_iter=max_iter, tol=tol, selection="random", random_state=0)
    m.fit(Xf.astype(np.float64), yf)
    return m.coef_.astype(np.float64), int(m.n_iter_) >= max_iter


# ---------------------------------------------------------------------------
# Chunked accumulation without materializing a float64 copy
# ---------------------------------------------------------------------------
# The full A-share fit segment is 2.3M×158. `X[mask].astype(np.float64)` allocates 2.9GB at once;
# on a machine with limited RAM, the OOM risk is real.
# Ridge needs only **XᵀX (158×158) and Xᵀy (158)**, which can be accumulated in chunks:
#   XᵀX = Σ_block X_blockᵀ X_block
# Each 200k-row chunk uses only 250MB and is discarded afterward. Accumulation still uses float64
# and is numerically equivalent to the one-shot algorithm; changed summation order differs around 1e-12.
CHUNK = 200_000


def _chunks(n: int, size: int = CHUNK):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def fit_moments(X: np.ndarray, rows: np.ndarray):
    """Compute fit-segment mean/standard deviation in chunks with float64 accumulation."""
    n, p = len(rows), X.shape[1]
    s1 = np.zeros(p); s2 = np.zeros(p)
    for sl in _chunks(n):
        b = X[rows[sl]].astype(np.float64)
        s1 += b.sum(axis=0); s2 += (b * b).sum(axis=0)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0.0))
    sd[sd < 1e-12] = 1.0
    return mu, sd


def normal_equations(X, rows, y, mu, sd, y_mu):
    """Accumulate XᵀX / Xᵀy in chunks, standardizing within chunks and retaining no large copy."""
    p = X.shape[1]
    XtX = np.zeros((p, p)); Xty = np.zeros(p)
    for sl in _chunks(len(rows)):
        b = (X[rows[sl]].astype(np.float64) - mu) / sd
        yb = y[rows[sl]].astype(np.float64) - y_mu
        XtX += b.T @ b
        Xty += b.T @ yb
    return XtX, Xty


def linpred(X, rows, w, mu, sd):
    """Compute X @ w in chunks."""
    out = np.empty(len(rows))
    for sl in _chunks(len(rows)):
        out[sl] = ((X[rows[sl]].astype(np.float64) - mu) / sd) @ w
    return out


def run_fold(X, y, day_id, year, meta, test_year: int, args):
    """Run one fold. Mask construction matches steps 1–2 of `walkforward.run_fold` line by line."""
    test_mask = year == test_year
    test_days = np.unique(day_id[test_mask])
    if test_days.size < args.min_test_days:
        print(f"  [skip] {test_year}: test days {test_days.size} < {args.min_test_days}")
        return None, None
    first_test_day = int(test_days.min())

    # 1. Year-boundary embargo: the label uses T+2 prices, so drop embargo days before the test year.
    train_pool = (year < test_year) & (day_id < first_test_day - args.embargo)
    pool_days = np.unique(day_id[train_pool])
    if pool_days.size < args.min_train_days + args.val_days:
        print(f"  [skip] {test_year}: available training days {pool_days.size} are insufficient")
        return None, None

    # 2. Split validation from the training-window tail to select α; embargo fit/validation again.
    val_start_day = int(pool_days[-args.val_days])
    val_mask = train_pool & (day_id >= val_start_day)
    fit_mask = train_pool & (day_id < val_start_day - args.embargo)

    fit_rows = np.flatnonzero(fit_mask)
    val_rows = np.flatnonzero(val_mask)
    test_rows = np.flatnonzero(test_mask)
    yv, dv = y[val_rows], day_id[val_rows]
    yt, dt = y[test_rows], day_id[test_rows]

    t0 = time.time()
    # Compute standardization statistics **only from fit** and reuse for validation/test.
    mu, sd = fit_moments(X, fit_rows)
    y_mu = float(y[fit_rows].astype(np.float64).mean())     # Center y, eliminating an explicit intercept.

    val_groups = build_day_groups(dv)
    test_groups = build_day_groups(dt)

    deviations = []
    if args.model == "ridge":
        XtX, Xty = normal_equations(X, fit_rows, y, mu, sd, y_mu)
        sols = ridge_path(XtX, Xty, args.alphas)
        cand = {a: w for a, (w, dev) in sols.items()}
        deviations = [a for a, (w, dev) in sols.items() if dev]
    else:
        Xf = (X[fit_rows].astype(np.float64) - mu) / sd
        yf_c = y[fit_rows].astype(np.float64) - y_mu
        cand = {}
        for a in args.alphas:
            w, hit_max = enet_fit(Xf, yf_c, a, args.l1_ratio, args.max_iter, args.tol)
            cand[a] = w
            if hit_max:
                deviations.append(a)
        del Xf, yf_c

    # Select α by **validation daily RankIC**, analogous to GBDT best_iteration / MLP best_epoch.
    val_scores = {}
    for a, w in cand.items():
        _, ric = daily_ic(linpred(X, val_rows, w, mu, sd), yv, val_groups)
        val_scores[a] = float(ric)
    best_alpha = max(val_scores, key=val_scores.get)
    best_val_ric = val_scores[best_alpha]

    preds_test = linpred(X, test_rows, cand[best_alpha], mu, sd)   # Predict test only once.
    test_ic, test_ric = daily_ic(preds_test, yt, test_groups)
    fit_s = time.time() - t0

    n_fit_days = int(np.unique(day_id[fit_mask]).size)
    print(f"  {test_year}: fit {n_fit_days} days + val {args.val_days} days → test {test_days.size} days "
          f"| best_alpha={best_alpha:g} | val RankIC={best_val_ric:+.4f} "
          f"| **test RankIC={test_ric:+.4f}** (IC={test_ic:+.4f}) | {fit_s:.1f}s")
    print("      α grid validation RankIC: " +
          "  ".join(f"{a:g}:{v:+.4f}" for a, v in val_scores.items()))
    if deviations:
        print(f"      numerical fallback α={deviations} (ridge used lstsq pseudoinverse / enet did not converge)")

    preds_df = meta.loc[test_mask].copy()
    preds_df["pred"] = preds_test.astype(np.float64)
    preds_df["label"] = yt.astype(np.float64)
    preds_df["test_year"] = test_year
    return {
        "test_year": test_year, "n_fit_days": n_fit_days, "n_val_days": args.val_days,
        "n_test_days": int(test_days.size),
        "best_alpha": float(best_alpha),
        "val_rankic_SELECTION_SCORE": best_val_ric,
        "val_grid": {f"{a:g}": v for a, v in val_scores.items()},
        "test_rankic": float(test_ric), "test_ic": float(test_ic),
        "numeric_deviation_alphas": [float(a) for a in deviations],
        "fit_seconds": round(fit_s, 1),
    }, preds_df


def run(args):
    print(f"[data] reading {args.data}")
    X, y, day_id, year, meta = load_panel_year(args.data, args.label_file)
    years = sorted(np.unique(year).tolist())
    print(f"[data] N={len(y)} rows, years {years[0]}–{years[-1]}, {day_id.max() + 1} days total")

    test_years = [yr for yr in years if yr >= years[0] + args.min_history_years]
    print(f"[wf] model={args.model} expanding walk-forward, test years: {test_years} "
          f"(embargo={args.embargo} days, val={args.val_days} days)")
    print(f"[wf] α grid {args.alphas}"
          f"{f' | l1_ratio={args.l1_ratio}' if args.model == 'elasticnet' else ''}\n")

    results, frames = [], []
    for yr in test_years:
        r, pf = run_fold(X, y, day_id, year, meta, yr, args)
        if r is not None:
            results.append(r); frames.append(pf)

    if results:
        rks = [r["test_rankic"] for r in results]
        print("\n===== walk-forward summary (independent OOS each year) =====")
        print(f"{'year':>6} {'test_days':>10} {'best_alpha':>12} "
              f"{'val_RankIC':>11} {'test_RankIC':>12} {'test_IC':>9}")
        for r in results:
            print(f"{r['test_year']:>6} {r['n_test_days']:>10} {r['best_alpha']:>12g} "
                  f"{r['val_rankic_SELECTION_SCORE']:>+11.4f} "
                  f"{r['test_rankic']:>+12.4f} {r['test_ic']:>+9.4f}")
        print("-" * 64)
        print(f"test RankIC: mean {np.mean(rks):+.4f} | median {np.median(rks):+.4f} | "
              f"min {np.min(rks):+.4f} | max {np.max(rks):+.4f} | "
              f"positive years {sum(v > 0 for v in rks)}/{len(rks)}")
        print("=" * 64)

    if frames and not args.no_preds:
        prov = provenance(args.data)
        out = f"model/preds_wf_ridge_{prov['market']}{args.tag}.parquet"
        preds = pd.concat(frames, ignore_index=True)
        preds.to_parquet(out, index=False)
        print(f"[preds] {len(preds):,} prediction rows saved to {out}")

    if results and not args.no_log:
        with open(WF_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                **provenance(args.data),
                                "model": args.model,
                                "label_file": args.label_file or "builtin",
                                # Tier C = measured: predict each test year once; no selection involvement.
                                "metric_tier": "C",
                                "window": "expanding", "embargo": args.embargo,
                                "val_days": args.val_days,
                                "alphas": list(args.alphas),
                                "l1_ratio": args.l1_ratio,
                                "results": results}, ensure_ascii=False) + "\n")
        print(f"[log] appended to {WF_LOG}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=FULL_PATH)
    ap.add_argument("--label-file", dest="label_file", default=None)
    ap.add_argument("--model", default="ridge", choices=["ridge", "elasticnet"])
    ap.add_argument("--alphas", default=",".join(f"{a:g}" for a in ALPHA_GRID),
                    help="comma-separated α grid, selected by daily RankIC on validation (nested)")
    ap.add_argument("--l1-ratio", dest="l1_ratio", type=float, default=0.5,
                    help="elasticnet only")
    ap.add_argument("--max-iter", dest="max_iter", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--val_days", type=int, default=120)
    ap.add_argument("--embargo", type=int, default=2)
    ap.add_argument("--min_history_years", type=int, default=2)
    ap.add_argument("--min_train_days", type=int, default=250)
    ap.add_argument("--min_test_days", type=int, default=60)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-log", dest="no_log", action="store_true")
    ap.add_argument("--no-preds", dest="no_preds", action="store_true")
    args = ap.parse_args()
    args.alphas = [float(a) for a in args.alphas.split(",")]
    run(args)


if __name__ == "__main__":
    main()
