"""
train_mlp.py — Minimal runnable MLP training script

Goal: feed the Alpha158 panel into an MLP and use an ordinary MSE loss to run the
  forward → loss → backward → optimizer.step chain, reporting cross-sectional IC
  on valid after each epoch. Once this works, switching to Soft IC loss changes
  only one loss line.

The blocks are labeled by the PyTorch concept they use:
  [Build Model]   nn.Module
  [Optimization]  the five-step training loop
  [Autograd]      loss.backward() computes gradients automatically
  [Tensors]       numpy ↔ torch and .to(device)

One key design (and a prerequisite for Soft IC later):
  **one day = one batch = one complete cross-section**. All stocks for a day enter the
  network together, contribute to one loss, and produce one parameter update. This lets
  the loss "see" the full daily cross-section, which makes cross-sectional losses such as
  Rank/Soft IC meaningful (they must compare all stocks from the same day).

Keep the full table on a 12 GB GPU: move the complete X_t/y_t tensors (float32≈1.5GB)
  once, then index rows directly on the GPU during training. This avoids a
  CPU→GPU copy for every batch (see the comments in run()).

Experiment logging: each run appends its results to model/experiments.jsonl and prints
  a comparison across losses. Switching to Soft IC next adds a row automatically for a
  direct comparison.

2026-07-08 change: upgraded the full training recipe (tapered network + Dropout, Cosine
  annealing, grad_clip=1.0, weight_decay=1e-3, and SMA-3 smoothed early stopping) to test
  whether Soft IC becoming increasingly negative was a recipe problem rather than a loss
  problem. Each change is marked below with [Change · training recipe ...].
  Note: lr must be calibrated on this data. This
  project's lr sweep (1e-5/1e-4/3e-4/1e-3) found lr=1e-3 the most stable (best≈final),
  while 1e-5 underfit to valid≈0.
  Conclusion: the current training recipe fixed soft_ic's training collapse (old: +0.005→-0.020;
  new: stable at +0.004), but the level remained ~0.004 < GBDT 0.0105. The recipe fixed
  the collapse without raising the ceiling.

Usage (from the repository root, after conda activate torch-env):
  python model/train_mlp.py --smoke                   # Five stocks; verify execution in seconds
  python model/train_mlp.py                           # Full run; default loss=soft_ic + default recipe
  python model/train_mlp.py --loss mse                # Same recipe with MSE for comparison with soft_ic
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# features.feature_names() is the authoritative source for the 158 factor names; reuse it instead of copying them.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "dataset", "alphaFactor"))
import features as F  # noqa: E402

def provenance(path: str) -> dict:
    """Identify the data used by this run and the metric's evidence tier.

    Two pitfalls found on 2026-08-30 were both impossible to reconstruct after the fact
    when they were not recorded at the time:
      1. None of the jsonl files recorded the dataset path, so US and A-share records looked
         identical and could be distinguished only by guessing from timestamps. Once A-shares
         became the primary market and US stocks the negative control, the two **must** be
         distinguishable or the negative control loses its meaning.
      2. `best_val_*` is the **maximum over training history, measured on the valid segment
         used for early stopping**. It is a **selection score (tier A)**, not measured
         performance. Taking the maximum of a noisy series can yield an attractive positive
         number even with no signal. Without an explicit label, later readers (including me)
         may mistake it for measured performance. See docs/research_log.md#r18 for tier
         definitions.
    """
    base = os.path.basename(path)
    return {"data": path,
            "market": "cn" if "_cn" in base else "us",
            "smoke": "_smoke" in base}


FULL_PATH  = "dataset/alphaFactor/dataset_alpha158.parquet"
SMOKE_PATH = "dataset/alphaFactor/dataset_alpha158_smoke.parquet"

# Append one JSON line per run here for comparing losses.
# Write smoke runs to a separate file so they do not contaminate the formal comparison.
EXP_LOG       = "model/experiments.jsonl"
EXP_LOG_SMOKE = "model/experiments_smoke.jsonl"


# ==========================================================================
# Data: read parquet → extract X / y / day number / segment (numpy, not yet in torch)
# ==========================================================================
def load_panel(path: str):
    """
    Returns:
      X        [N, 158] float32   features (cross-sectionally standardized and filled with 0 during preprocessing; no NaN)
      y        [N]      float32   label (cross-sectional robust-z of the T+1→T+2 return)
      day_id   [N]      int64     trading-day number (stocks on the same day share a day_id)
      segment  [N]      object    'train' / 'valid' / 'test'

    Note: the tail of label contains NaN (the final two days for each stock have no future
    return because of shift(-2)); drop those rows here.
    """
    feat_cols = F.feature_names()
    df = pd.read_parquet(path)

    # Rows with NaN labels cannot be used for training or evaluation; drop them.
    df = df[df["label"].notna()]

    X = df[feat_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)                     # Defensive fallback (preprocessing already filled them)
    y = df["label"].to_numpy(dtype=np.float32)

    # Convert the date index to zero-based integer day numbers. The panel is already sorted
    # by (date, instrument), so factorize produces contiguous rows for each day.
    dates = df.index.get_level_values("date")
    day_id = pd.factorize(dates, sort=True)[0].astype(np.int64)

    segment = df["segment"].to_numpy()
    return X, y, day_id, segment


def build_day_index(day_id: np.ndarray, mask: np.ndarray) -> list[torch.Tensor]:
    """
    Given a Boolean mask for one segment (train/valid), return a list of row numbers grouped by day:
      [ tensor([row numbers of all stocks for that day in the full table]), ... ]
      Each element = one day = one batch.

    Take row numbers from the segment, stably sort by day_id, and split whenever day_id changes.
    """
    rows = np.flatnonzero(mask)
    order = np.argsort(day_id[rows], kind="stable")
    rows = rows[order]
    days = day_id[rows]
    cut = np.flatnonzero(np.diff(days) != 0) + 1      # Boundaries between days
    groups = np.split(rows, cut)
    return [torch.from_numpy(g.copy()) for g in groups]


# ==========================================================================
# [Build Model] nn.Module — 158 factors → one cross-sectional score
#
# [Change · training recipe: network geometry]
#   Old: 158→64→64→1 (two flat layers with bare Linear+ReLU).
#   New: 158→[256→128→64→32]→1, a tapered network that narrows layer by layer, with
#        each hidden layer structured as Linear→ReLU→Dropout.
#   Why tapered + Dropout:
#     - Tapering: a wide input layer (256/512) first combines 158 weak factors in a
#       high-dimensional nonlinear space, then gradually compresses them into a score,
#       matching the intuition of an information funnel.
#     - Dropout(0.1): overfitting is the main enemy at low signal-to-noise ratio (the reason
#       soft_ic became increasingly negative in the previous run). Randomly dropping 10% of
#       neurons during training discourages memorizing noise in a few factors.
# ==========================================================================
class AlphaMLP(nn.Module):
    def __init__(self, n_features: int = 158,
                 hidden_list: tuple[int, ...] = (256, 128, 64, 32),
                 dropout: float = 0.1):
        super().__init__()                            # The parent initializer must run first
        layers: list[nn.Module] = []
        d = n_features
        for h in hidden_list:                         # Each hidden layer: Linear→ReLU→Dropout
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))                # Final regression score; no activation or Dropout
        self.net = nn.Sequential(*layers)

    def forward(self, x):                             # x: [N_stocks, 158]
        return self.net(x).squeeze(-1)                # [N, 1] -> [N], one score per stock


# ==========================================================================
# Evaluation: cross-sectional IC — compute corr(pred, label) daily, then average across days.
# This is the standard alpha-model score and exactly what Soft IC aims to optimize directly.
# ==========================================================================
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else np.nan


def _rank(x: np.ndarray) -> np.ndarray:
    return x.argsort().argsort().astype(np.float64)   # Ranks from 0..n-1


# ==========================================================================
# Soft IC loss — differentiable IC that directly optimizes cross-sectional IC
# ==========================================================================
def soft_ic_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Loss = 1 - IC, where IC is the Pearson correlation between pred and target for the
    day's stocks.

    IC uses only arithmetic operations (demean → multiply and average for covariance →
    divide by two standard deviations), and **every step is differentiable**, so autograd
    can propagate gradients automatically. That is all "differentiable IC" means.

    The essential difference from MSE (and why it can help here):
      - MSE forces the model to match return magnitudes, while returns are 99% noise, so the
        model retreats toward the mean and IC≈0.
      - Soft IC cares only whether pred and target move together (correlation). Arbitrary
        scaling or shifting of pred leaves the loss unchanged because Pearson correlation is
        affine-invariant, so the model only needs to learn the ranking direction.
      - It also inherently rejects constant predictions: equal pred values give σ_p=0,
        IC=0, and loss=1 (worst), so gradients push the model away from the trap MSE falls into.

    Because one day is one batch, the provided pred/target already form the complete daily
    cross-section. Compute the loss directly without grouping by date_idx.
    """
    pred = pred.float()
    target = target.float()
    p_c = pred - pred.mean()
    t_c = target - target.mean()
    cov = (p_c * t_c).mean()
    ic = cov / (p_c.std(unbiased=False) * t_c.std(unbiased=False) + eps)
    return 1.0 - ic


# ==========================================================================
# Loss selector
# ==========================================================================
def make_loss(name: str):
    if name == "mse":
        return nn.MSELoss()
    if name == "soft_ic":
        return soft_ic_loss                            # Callable with the MSELoss signature: fn(pred, target)
    raise ValueError(f"Unknown loss: {name!r} (supported: 'mse' / 'soft_ic')")


# ==========================================================================
# Experiment log and loss comparison table
# ==========================================================================
def log_result(record: dict, path: str) -> None:
    """Append one experiment result as a JSON line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_comparison(path: str) -> None:
    """Read all experiments from the log and print the loss comparison table."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return
    print("\n===== Experiment comparison (valid: daily IC averaged across days; best selected by SMA-3 smoothing) =====")
    print(f"{'time':<19} {'loss':<8} {'ep':>3} {'lr':>7} {'hidden':>15} "
          f"{'bestRankIC':>11} {'@ep':>4} {'bestIC':>9} {'finalRankIC':>12}")
    for r in rows:
        print(f"{r['time']:<19} {r['loss']:<8} {r['epochs']:>3} {r['lr']:>7.0e} "
              f"{str(r['hidden']):>15} "
              f"{r['best_val_rankic']:>+11.4f} {r['best_epoch']:>4} {r['best_val_ic']:>+9.4f} "
              f"{r['final_val_rankic']:>+12.4f}")
    print("=" * 90)


# ==========================================================================
# [Optimization] Main training loop
# ==========================================================================
def run(args):
    torch.manual_seed(0)
    np.random.seed(0)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    path = SMOKE_PATH if args.smoke else args.data
    print(f"[data] reading {path}")
    X, y, day_id, segment = load_panel(path)
    print(f"[data] N={len(y)}  features={X.shape[1]}  days={day_id.max() + 1}  device={device}")

    # [Tensors] numpy → torch, moving the entire table to the GPU at once.
    #   X is float32 ≈ 2.39M×158×4B ≈ 1.5GB, which fits on a 12 GB GPU.
    #   Once resident on the GPU, the training loop indexes rows there directly instead of
    #   copying every batch from CPU→GPU, eliminating half of the round-trip overhead.
    X_t = torch.from_numpy(X).to(device)               # [N, 158], resident on GPU
    y_t = torch.from_numpy(y).to(device)               # [N], resident on GPU

    # Move each day's row-index tensor to the same device as well: PyTorch advanced indexing
    # requires the index and indexed tensors on the same device, or X_t[idx] reports a mismatch.
    train_days = [d.to(device) for d in build_day_index(day_id, segment == "train")]
    valid_days = [d.to(device) for d in build_day_index(day_id, segment == "valid")]
    print(f"[data] train days={len(train_days)}  valid days={len(valid_days)}")
    print(f"[data] X_t.device={X_t.device}  full feature table resident on GPU, "
          f"using ~{X_t.element_size() * X_t.nelement() / 1e9:.2f}GB")

    # [Build Model] + move to GPU
    hidden_list = tuple(int(h) for h in args.hidden.split(","))   # "256,128,64,32" → (256,128,64,32)
    model = AlphaMLP(n_features=X.shape[1], hidden_list=hidden_list,
                     dropout=args.dropout).to(device)

    # [Change · training recipe: weight_decay 1e-5 → 1e-3]
    #   weight_decay is the L2 regularization strength. Increasing it 100× at low SNR further
    #   suppresses overfitting. It complements Dropout: Dropout removes connections, while
    #   weight_decay reduces weight magnitudes.
    # The optimizer receives all model parameters (the W and b values in each nn.Linear).
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # [Change · training recipe: add Cosine LR annealing]
    #   Without a scheduler at low SNR, val IC variance across seeds increases noticeably.
    #   Cosine smoothly lowers lr from its initial value to ~0 along half a cosine curve:
    #   large exploratory steps early and small refinement steps late. This addresses the prior
    #   pattern where lr=1e-3 without annealing bounced through noise and fit the validation set.
    #   T_max is the total epoch count, so one complete decay cycle spans all training.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = make_loss(args.loss)                     # mse / soft_ic

    def eval_ic(day_batches):
        """Compute daily IC, then average across days."""
        model.eval()
        pic, ric = [], []
        with torch.no_grad():
            for idx in day_batches:
                if idx.numel() < 5:                    # Too few stocks that day for a meaningful correlation
                    continue
                pred = model(X_t[idx]).cpu().numpy()    # X_t is on GPU; index it directly
                yb = y_t[idx].cpu().numpy()             # y_t is also on GPU; return values to CPU for corr
                pic.append(_pearson(pred, yb))
                ric.append(_pearson(_rank(pred), _rank(yb)))
        return float(np.nanmean(pic)), float(np.nanmean(ric))

    print(f"\n[train] loss={args.loss} epochs={args.epochs} lr={args.lr} "
          f"hidden={args.hidden} dropout={args.dropout} wd={args.wd} "
          f"grad_clip={args.grad_clip} sched=cosine patience={args.patience}\n")
    history = []                                        # One record per epoch for logging and comparison
    best_smooth = -float("inf")                         # Best smoothed value tracked for early stopping
    bad = 0                                             # Consecutive epochs without improvement
    stopped_epoch = args.epochs
    for epoch in range(1, args.epochs + 1):
        model.train()                                  # Enter training mode (enables Dropout)
        np.random.shuffle(train_days)                  # Shuffle day order each epoch
        running = 0.0
        for idx in train_days:
            xb = X_t[idx]                              # Full daily cross-section [n_stocks, 158], already on GPU (zero-copy)
            yb = y_t[idx]                              # [n_stocks], already on GPU

            # ---- The standard five steps (step 4.5 adds the new gradient clipping) ----
            pred = model(xb)                           # 1. Forward pass
            loss = loss_fn(pred, yb)                   # 2. Compute loss (mse / soft_ic)
            optimizer.zero_grad()                      # 3. Clear prior gradients (they accumulate by default)
            loss.backward()                            # 4. [Autograd] Backward pass fills every W.grad automatically
            # [Change · training recipe: grad_clip=1.0]
            #   Clip the total norm of all parameter gradients to ≤1.0. If an occasional batch
            #   produces exploding gradients (common at low SNR), this keeps one step from
            #   throwing the parameters far off course. It is a safety belt for stable training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()                           # 5. Update parameters with the clipped gradients
            running += loss.item()

        scheduler.step()                               # Advance one cosine step per epoch to lower lr smoothly
        train_loss = running / len(train_days)
        v_ic, v_ric = eval_ic(valid_days)              # Dropout turns off automatically in eval mode
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_ic": v_ic, "val_rankic": v_ric,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"  epoch {epoch:3d} | lr {optimizer.param_groups[0]['lr']:.2e} "
              f"| train {args.loss} {train_loss:.4f} "
              f"| valid IC {v_ic:+.4f} | valid RankIC {v_ric:+.4f}")

        # [Change · training recipe: SMA-3 smoothed early stopping]
        #   Do not use a single epoch, which can select a noisy false peak as happened with
        #   MSE's best 0.0125→final 0.0035. Instead, use mean RankIC over the latest 3 epochs
        #   and stop after patience consecutive epochs without a new high.
        sma3 = float(np.mean([h["val_rankic"] for h in history[-3:]]))
        if sma3 > best_smooth + 1e-5:
            best_smooth = sma3
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                stopped_epoch = epoch
                print(f"  [early-stop] SMA-3 did not improve for {args.patience} rounds; "
                      f"stopping at epoch {epoch}.")
                break

    print("\n[done] The full forward→loss→backward→step chain completed on this data.")

    # ---- Log this experiment and print the comparison across losses ----
    # [Change · training recipe: select best by SMA-3 smoothing, not the luckiest single epoch]
    #   Compute an SMA-3 for every epoch (including the first two) and take the epoch with the
    #   highest smoothed value. The reported best is then a stable plateau rather than a noisy spike.
    rankics = [h["val_rankic"] for h in history]
    smoothed = [float(np.mean(rankics[max(0, i - 2):i + 1])) for i in range(len(rankics))]
    best = history[int(np.argmax(smoothed))]
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **provenance(path),
        # Tier A = selection score: best_val_* is the maximum training-history value measured
        # on the valid segment used for early stopping. It is not measured performance; see
        # docs/research_log.md#r18.
        "metric_tier": "A",
        "loss": args.loss, "epochs": args.epochs, "lr": args.lr, "hidden": args.hidden,
        "dropout": args.dropout, "wd": args.wd, "grad_clip": args.grad_clip,
        "sched": "cosine", "patience": args.patience, "stopped_epoch": stopped_epoch,
        "device": device, "n_rows": int(len(y)),
        "train_days": len(train_days), "valid_days": len(valid_days),
        "best_epoch": best["epoch"],
        "best_val_rankic": best["val_rankic"], "best_val_ic": best["val_ic"],
        "final_val_rankic": history[-1]["val_rankic"], "final_val_ic": history[-1]["val_ic"],
        "history": history,                              # Complete per-epoch history for plotting later
    }
    log_path = EXP_LOG_SMOKE if args.smoke else EXP_LOG
    log_result(record, log_path)
    print(f"[log] results appended to {log_path}")
    print(f"[best] loss={args.loss}  best RankIC={best['val_rankic']:+.4f} "
          f"@epoch {best['epoch']}  (IC={best['val_ic']:+.4f})")
    print_comparison(log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=FULL_PATH)
    ap.add_argument("--smoke", action="store_true", help="Quick verification with a five-stock smoke parquet")
    ap.add_argument("--loss", default="soft_ic", help="Loss function: mse / soft_ic (default: soft_ic)")
    # Defaults below are the training recipe established on 2026-07-08 (old values in parentheses).
    ap.add_argument("--epochs", type=int, default=100)          # Old: 30; allow enough for cosine + early stopping
    ap.add_argument("--lr", type=float, default=1e-3)           # 2026-07-08 lr sweep: 1e-5 underfit to 0; 1e-3 was most stable (best≈final). Calibrate lr on the actual data rather than copying it.
    ap.add_argument("--hidden", default="256,128,64,32",        # Old: "64" (flat); now tapered
                    help="Comma-separated hidden-layer widths, e.g. 256,128,64,32")
    ap.add_argument("--dropout", type=float, default=0.1)       # Old: 0 (none)
    ap.add_argument("--wd", type=float, default=1e-3)           # Old: 1e-5; weight_decay increased 100×
    ap.add_argument("--grad_clip", type=float, default=1.0)     # Old: none
    ap.add_argument("--patience", type=int, default=15,         # SMA-3 early-stopping patience
                    help="Stop when SMA-3 has not improved for this many epochs")
    ap.add_argument("--device", default=None, help="cuda / cpu; selected automatically by default")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
