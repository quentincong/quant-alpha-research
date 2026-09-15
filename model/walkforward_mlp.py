"""
walkforward_mlp.py — MLP walk-forward using **exactly the same metric** as `walkforward.py` (GBDT)

**Why not modify `train_mlp.py`**: that script uses a single train/validation split with **no test**.
Its reported `best_val_rankic` is the **maximum over training history** on the validation segment
used for early stopping. It is a **Tier A selection score**, not performance. Moving that design
into walk-forward would directly produce invalid results. This implementation matches the GBDT version:

  · Use validation only for **selection** of the stopping epoch, equivalent to GBDT's `best_iteration`.
  · **Predict the test year only once** using weights selected on validation (best epoch's state_dict).
  · Report only test (Tier C). Record validation scores but **explicitly label them as selection scores**.

Fold geometry **matches `walkforward.py` line by line**: expanding window, year-boundary embargo,
validation split from the training-window tail, and another embargo at the fit/validation boundary.
Reuse its `load_panel_year` directly, leaving the model as the only differing variable; otherwise
the MLP-versus-GBDT comparison would not be clean.

The prediction schema exactly matches the GBDT version (date, instrument, label, label_raw, pred,
test_year), so `evaluate.py` / `turnover_study.py` run **without any changes**.

--------------------------------------------------------------------------
Two implementation details
--------------------------------------------------------------------------
1. **Keep the full table resident on the GPU**: X is float32 ≈ 2.36M×158×4B ≈ 1.5GB,
   which fits on a 12 GB GPU. Index directly by row on the GPU during training, avoiding
   per-batch CPU→GPU copies.

2. **`--batch-days k` can change the definition of soft_ic, not merely batch size.**
   `soft_ic` is a **cross-sectional** correlation whose premise is one day per batch. Combining
   k days and computing one Pearson correlation instead measures a **mixed cross-day** correlation,
   a different and incorrect objective. The correct method computes **each day's IC and then
   averages across days**. Segment scatter vectorizes this here, preserving the definition while
   processing multiple days in one forward pass. This matters for speed: one day per batch causes
   ~1,500 tiny kernel launches per epoch, making **launch overhead**, not compute, the bottleneck.

Usage:
  python model/walkforward_mlp.py --data dataset/alphaFactor/dataset_alpha158_cn.parquet \
         --only-year 2020 --no-log --no-preds       # Time one fold.
  python model/walkforward_mlp.py --data ... --tag _mlp        # Full six-fold run.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_mlp as TM
from train_mlp import AlphaMLP, provenance, _pearson, _rank
from walkforward import load_panel_year

WF_LOG = "model/walkforward_mlp.jsonl"


# ---------------------------------------------------------------------------
# Vectorized segmented soft IC with one segment per day; see header detail 2.
# ---------------------------------------------------------------------------
def soft_ic_segmented(pred, target, seg, n_seg, eps: float = 1e-8):
    """loss = 1 − mean_j(IC on day j). seg[i] identifies the day segment for row i in this batch.

    Use scatter-add for all segmented statistics with **no loop**:
      center each day separately → daily covariance/variance → one IC per day → average across days.
    At k=1 this matches `train_mlp.soft_ic_loss`; this function is its segmented generalization.
    """
    z = lambda: torch.zeros(n_seg, device=pred.device, dtype=pred.dtype)
    cnt = z().index_add_(0, seg, torch.ones_like(pred)).clamp(min=1.0)
    p_c = pred - (z().index_add_(0, seg, pred) / cnt)[seg]
    t_c = target - (z().index_add_(0, seg, target) / cnt)[seg]
    cov = z().index_add_(0, seg, p_c * t_c)
    vp = z().index_add_(0, seg, p_c * p_c)
    vt = z().index_add_(0, seg, t_c * t_c)
    return 1.0 - (cov / (torch.sqrt(vp * vt) + eps)).mean()


def mse_segmented(pred, target, seg, n_seg):
    """MSE is independent of segments; match soft_ic_segmented's signature for interchangeability."""
    return torch.mean((pred - target) ** 2)


def day_rows_of(mask, day_id):
    """Given a Boolean mask, return full-table row indices grouped by day, one element per day."""
    rows = np.flatnonzero(mask)
    rows = rows[np.argsort(day_id[rows], kind="stable")]
    d = day_id[rows]
    return np.split(rows, np.flatnonzero(np.diff(d) != 0) + 1)


def make_batches(day_rows, batch_days, device):
    """Concatenate daily row indices in groups of batch_days → [(rows, seg, n_seg), ...]."""
    out = []
    for i in range(0, len(day_rows), batch_days):
        chunk = day_rows[i:i + batch_days]
        rows = np.concatenate(chunk)
        seg = np.repeat(np.arange(len(chunk)), [len(c) for c in chunk])
        out.append((torch.from_numpy(rows).to(device),
                    torch.from_numpy(seg).to(device),
                    len(chunk)))
    return out


@torch.no_grad()
def eval_rankic(model, X_t, y_np, day_rows, device):
    """Compute daily RankIC then average across days; NumPy ranking matches the GBDT daily_ic definition."""
    model.eval()
    out = []
    for rows in day_rows:
        if len(rows) < 5:
            continue
        idx = torch.from_numpy(rows).to(device)
        p = model(X_t[idx]).float().cpu().numpy()
        out.append(_pearson(_rank(p), _rank(y_np[rows])))
    return float(np.nanmean(out)) if out else float("nan")


@torch.no_grad()
def predict_rows(model, X_t, rows, device, chunk=200_000):
    model.eval()
    parts = []
    for i in range(0, len(rows), chunk):
        idx = torch.from_numpy(rows[i:i + chunk]).to(device)
        parts.append(model(X_t[idx]).float().cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0)


# ---------------------------------------------------------------------------
def run_fold(X_t, y_t, y_np, day_id, year, meta, test_year, args, device):
    """One fold: train on all years <test_year, split tail validation for early stopping, then evaluate OOS RankIC once on test_year.

    Mask construction **exactly matches** walkforward.run_fold with the same two embargoes and validation split.
    """
    test_mask = year == test_year
    test_days = np.unique(day_id[test_mask])
    if test_days.size < args.min_test_days:
        print(f"  [skip] {test_year}: insufficient test days"); return None, None
    first_test_day = int(test_days.min())

    train_pool = (year < test_year) & (day_id < first_test_day - args.embargo)   # 1. Year-boundary embargo.
    pool_days = np.unique(day_id[train_pool])
    if pool_days.size < args.min_train_days + args.val_days:
        print(f"  [skip] {test_year}: insufficient training days"); return None, None

    val_start_day = int(pool_days[-args.val_days])
    val_mask = train_pool & (day_id >= val_start_day)
    fit_mask = train_pool & (day_id < val_start_day - args.embargo)              # ② fit/val embargo

    fit_days = day_rows_of(fit_mask, day_id)
    val_days = day_rows_of(val_mask, day_id)
    test_rows = np.flatnonzero(test_mask)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = AlphaMLP(n_features=X_t.shape[1],
                     hidden_list=tuple(int(h) for h in args.hidden.split(",")),
                     dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = soft_ic_segmented if args.loss == "soft_ic" else mse_segmented

    batches = make_batches(fit_days, args.batch_days, device)
    best_val, best_state, best_epoch, bad = -float("inf"), None, 0, 0
    # --epoch-dump is an auxiliary recorder **disabled by default**, following the `with_series`
    # precedent in turnover_study/longonly_study. When disabled, the two containers below receive
    # no writes and consume no random numbers, leaving the training trajectory elementwise unchanged.
    # When enabled, save each epoch's validation RankIC and weight snapshot for offline comparison
    # of epoch-selection rules (argmax / plateau-median / weight-avg).
    ep_curve, ep_states = [], {}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(batches))
        for bi in order:
            rows, seg, n_seg = batches[bi]
            loss = loss_fn(model(X_t[rows]), y_t[rows], seg, n_seg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        sched.step()
        v = eval_rankic(model, X_t, y_np, val_days, device)
        if args.epoch_dump:
            ep_curve.append(v)
            ep_states[ep] = {k: t.detach().to("cpu", copy=True)
                             for k, t in model.state_dict().items()}
        # Use validation only to select the epoch (= GBDT best_iteration), never as a reported result.
        if v > best_val + 1e-6:
            best_val, best_epoch, bad = v, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if args.verbose:
            print(f"    ep {ep:3d} | loss {loss.item():.4f} | val RankIC {v:+.4f}"
                  f"{'  *' if best_epoch == ep else ''}")
        if bad >= args.patience:
            break
    train_s = time.time() - t0

    model.load_state_dict(best_state)                     # Restore weights selected on validation.
    preds_test = predict_rows(model, X_t, test_rows, device)   # Predict test only once.
    test_ric = float(np.nanmean([
        _pearson(_rank(preds_test[np.searchsorted(test_rows, r)]), _rank(y_np[r]))
        for r in day_rows_of(test_mask, day_id) if len(r) >= 5]))

    print(f"  {test_year}: fit {len(fit_days)} days + val {len(val_days)} days → test {test_days.size} days "
          f"| best_epoch={best_epoch}/{ep} | val RankIC={best_val:+.4f} "
          f"| **test RankIC={test_ric:+.4f}** | {train_s:.1f}s")

    pf = meta.iloc[test_rows].copy()
    pf["pred"] = preds_test.astype(np.float64)
    pf["label"] = y_np[test_rows].astype(np.float64)
    pf["test_year"] = test_year
    res = {"test_year": test_year, "n_fit_days": len(fit_days), "n_val_days": len(val_days),
           "n_test_days": int(test_days.size), "best_epoch": best_epoch, "epochs_run": ep,
           "val_rankic_SELECTION_SCORE": best_val, "test_rankic": test_ric,
           "train_seconds": round(train_s, 1)}
    if args.epoch_dump:
        os.makedirs(args.epoch_dump, exist_ok=True)
        ck = os.path.join(args.epoch_dump,
                          f"fold{test_year}{args.tag}_seed{args.seed}.pt")
        torch.save({"test_year": test_year, "seed": args.seed, "tag": args.tag,
                    "hidden": args.hidden, "dropout": args.dropout,
                    "n_features": int(X_t.shape[1]),
                    "test_rows": test_rows,                # Store fold geometry,
                    "best_epoch": best_epoch,              # so offline scripts need not recompute masks.
                    "epochs_run": ep,
                    "val_curve": np.asarray(ep_curve, dtype=np.float64),
                    "states": ep_states}, ck)
        print(f"    [epoch-dump] {len(ep_states)} epoch snapshots → {ck}")
        res["val_curve"] = [round(float(x), 6) for x in ep_curve]
    return res, pf


def run(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[data] reading {args.data}")
    X, y, day_id, year, meta = load_panel_year(args.data, args.label_file)
    print(f"[data] N={len(y)} rows, {day_id.max() + 1} days, device={device}")

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)
    y_np = y.astype(np.float64)
    print(f"[data] feature table resident on GPU, using ~{X_t.element_size() * X_t.nelement() / 1e9:.2f}GB")
    if args.epoch_dump:
        print(f"[epoch-dump] dir={args.epoch_dump} | X_t.device={X_t.device} | y_t.device={y_t.device}")
        if args.device == "cuda":
            assert X_t.device.type == "cuda", f"required --device cuda; actual device {X_t.device}"
            assert y_t.device.type == "cuda", f"required --device cuda; actual device {y_t.device}"

    years = sorted(np.unique(year).tolist())
    test_years = [yr for yr in years if yr >= years[0] + args.min_history_years]
    if args.only_year:
        test_years = [args.only_year]
    print(f"[wf] test years: {test_years} | loss={args.loss} hidden={args.hidden} "
          f"lr={args.lr} wd={args.wd} dropout={args.dropout} batch_days={args.batch_days} "
          f"epochs<={args.epochs} patience={args.patience} seed={args.seed}\n")

    results, frames = [], []
    for yr in test_years:
        r, pf = run_fold(X_t, y_t, y_np, day_id, year, meta, yr, args, device)
        if r: results.append(r); frames.append(pf)

    if results:
        rk = [r["test_rankic"] for r in results]
        print(f"\ntest RankIC: mean {np.mean(rk):+.4f} | min {np.min(rk):+.4f} | "
              f"max {np.max(rk):+.4f} | positive {sum(v > 0 for v in rk)}/{len(rk)} | "
              f"total training {sum(r['train_seconds'] for r in results):.0f}s")

    if frames and not args.no_preds:
        prov = provenance(args.data)
        out = f"model/preds_wf_mlp_{prov['market']}{args.tag}.parquet"
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        print(f"[preds] saved {out}")

    if not args.no_log and results:
        with open(WF_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                **provenance(args.data), "model": "mlp",
                                "label_file": args.label_file or "builtin",
                                # Tier C = measured: predict the test year once with no role in selection.
                                # val_rankic_SELECTION_SCORE is Tier A; do not interpret it as performance.
                                "metric_tier": "C",
                                "loss": args.loss, "hidden": args.hidden, "lr": args.lr,
                                "wd": args.wd, "dropout": args.dropout,
                                "batch_days": args.batch_days, "seed": args.seed,
                                "epochs_max": args.epochs, "patience": args.patience,
                                "results": results}, ensure_ascii=False) + "\n")
        print(f"[log] appended to {WF_LOG}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=TM.FULL_PATH)
    ap.add_argument("--label-file", dest="label_file", default=None)
    ap.add_argument("--loss", default="soft_ic", choices=["soft_ic", "mse"])
    ap.add_argument("--hidden", default="256,128,64,32")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--batch-days", dest="batch_days", type=int, default=1,
                    help="days per batch; when >1, soft_ic segments by day automatically without changing its definition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_days", type=int, default=120)
    ap.add_argument("--embargo", type=int, default=2)
    ap.add_argument("--min_history_years", type=int, default=2)
    ap.add_argument("--min_train_days", type=int, default=250)
    ap.add_argument("--min_test_days", type=int, default=60)
    ap.add_argument("--only-year", dest="only_year", type=int, default=None,
                    help="run only this fold for timing")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epoch-dump", dest="epoch_dump", default=None, metavar="DIR",
                    help="disabled by default. Given a directory, save each fold's per-epoch validation RankIC + "
                         "per-epoch weight snapshots + test row indices as .pt for offline epoch-rule testing. "
                         "When disabled, consumes no random numbers and changes no output, following turnover_study with_series")
    ap.add_argument("--verbose", action="store_true", help="print every epoch")
    ap.add_argument("--no-log", dest="no_log", action="store_true")
    ap.add_argument("--no-preds", dest="no_preds", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
