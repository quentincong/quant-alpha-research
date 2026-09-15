"""
meta_mlp.py — second layer (meta / stacking): use a small MLP to learn how to combine GBDT and MLP

**Why this layer may be worthwhile** (following the 2026-08-31 combination-layer result):
  GBDT and MLP are strongest at **opposite tails**. MLP has the better D10 (+6.17 vs +5.31),
  while GBDT has the better D1 (−20.45 vs −22.61). Their daily cross-sectional rank correlation
  is ρ≈0.57–0.62, so **they are genuinely different models**. Yet equal-weighted rank averaging
  **does not beat GBDT** on the long-short definition because a linear blend **cannot structurally
  apply different weights by quantile and leg**. This layer is justified only because it can do so:
  a **nonlinear** meta-model can learn to "trust MLP more at the top and GBDT more at the bottom"
  **from the data**, rather than manually specifying the rule after observing results. The latter
  fits a rule to one observation, exactly the error this project aims to prevent.

**The input has only two features** and is deliberately kept minimal:
  each base model's daily **cross-sectional rank percentile**, centered to [-0.5, 0.5].
  · Use ranks rather than raw scores because the models' score scales differ completely;
    ranks are comparable and match the equal-weight baseline's definition.
  · Rank itself encodes where a stock falls that day, allowing the network to learn
    quantile-dependent weights.

**Nesting determines whether this layer is valid**:
  for test year k, train the meta-model **only on years before k**, whose base-model
  predictions are themselves out of sample, then apply it to year k. The meta-model's score
  is therefore also out of sample with respect to the combination rule.
  · 2020 is the first test year and has no preceding year, so **meta coverage is 2021–2025**,
    exactly matching `selection_null.py --mode nested` for direct comparison.
  · Leave a **purge gap** between train/validation and validation/test. The label is forward-looking,
    so without a gap adjacent-day labels cross split boundaries, as in the walkforward.py embargo fix.

**Why the network must be small**: it only needs to learn how to combine two ranks, while the
  effect size is only ~0.2–0.9 bps and seed standard deviation alone is 0.5 bps. With too many
  parameters, the network learns fold-specific luck. The default has two layers, (16, 8).

**Discipline**: report error bars across three seeds; always run the US negative control;
  reuse the predetermined `ema5/exit10` (long-short) and `ema5/exit30` (long-only) combination
  variants and **do not reselect them for the meta-model**.

Usage (from the repository root, after conda activate torch-env):
  python model/meta_mlp.py --market cn --seed 0
  python model/meta_mlp.py --market us --seed 0
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mlp import AlphaMLP, soft_ic_loss, _pearson

BASES = {
    "cn": ("model/preds_walkforward_cn_vwapentry.parquet",
           "model/preds_wf_mlp_cn_vwapentry_s{mlp_seed}.parquet"),
    "us": ("model/preds_walkforward_us_hlc3.parquet",
           "model/preds_wf_mlp_us_hlc3.parquet"),
}
PURGE = 5          # Days removed at split boundaries; a five-day buffer is ample for a label that reaches T+2.


def load_pair(market: str, mlp_seed: int) -> pd.DataFrame:
    g_path, m_path = BASES[market]
    m_path = m_path.format(mlp_seed=mlp_seed)
    g = pd.read_parquet(g_path); g["date"] = pd.to_datetime(g["date"])
    m = pd.read_parquet(m_path); m["date"] = pd.to_datetime(m["date"])
    key = ["date", "instrument"]
    j = g.merge(m[key + ["pred"]], on=key, suffixes=("_g", "_m"), how="inner")
    assert len(j) == len(g) == len(m), f"row counts do not match: {len(j)} / {len(g)} / {len(m)}"
    j = j.sort_values(key).reset_index(drop=True)
    # Features are centered daily cross-sectional rank percentiles, one per model.
    j["f_g"] = j.groupby("date")["pred_g"].rank(pct=True) - 0.5
    j["f_m"] = j.groupby("date")["pred_m"].rank(pct=True) - 0.5
    return j


def day_slices(dates: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) intervals for contiguous rows on the same day; data are date-sorted."""
    chg = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    edges = np.concatenate([[0], chg, [len(dates)]])
    return list(zip(edges[:-1], edges[1:]))


def train_fold(df, tr_mask, va_mask, seed, hidden, lr, wd, dropout,
               epochs, patience, device):
    torch.manual_seed(seed); np.random.seed(seed)
    X = torch.tensor(df[["f_g", "f_m"]].to_numpy(np.float32), device=device)
    y = torch.tensor(df["label"].to_numpy(np.float32), device=device)
    d = df["date"].to_numpy()
    tr = [(a, b) for a, b in day_slices(d) if tr_mask[a]]
    va = [(a, b) for a, b in day_slices(d) if va_mask[a]]

    net = AlphaMLP(n_features=2, hidden_list=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, bad, hist = -np.inf, None, 0, []
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        for i in rng.permutation(len(tr)):
            a, b = tr[i]
            if b - a < 30:                       # Too few stocks for reliable cross-sectional IC.
                continue
            opt.zero_grad()
            loss = soft_ic_loss(net(X[a:b]), y[a:b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            ics = [_pearson(net(X[a:b]).cpu().numpy(), y[a:b].cpu().numpy())
                   for a, b in va if b - a >= 30]
        ic = float(np.nanmean(ics))
        hist.append(ic)
        sma = float(np.mean(hist[-3:]))          # SMA-3 early stopping, matching train_mlp.py.
        if sma > best:
            best, bad = sma, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, best, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["cn", "us"], default="cn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp-seed", dest="mlp_seed", type=int, default=0,
                    help="base-model MLP prediction seed to use (CN has s0/s1/s2)")
    ap.add_argument("--hidden", default="16,8", help="deliberately small; it only combines two ranks")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    hidden = tuple(int(x) for x in args.hidden.split(","))
    df = load_pair(args.market, args.mlp_seed)
    yr = df["date"].dt.year.to_numpy()
    years = sorted(df["test_year"].unique())
    print(f"[data] market={args.market} mlp_seed={args.mlp_seed} {len(df):,} rows "
          f"| meta hidden={hidden} seed={args.seed} device={dev}")
    print(f"[nest] test years {years[1]}–{years[-1]} ({years[0]} has no preceding training year and is skipped)")

    out = []
    for k in years[1:]:
        t0 = time.time()
        prior = np.flatnonzero(yr < k)
        # Validation is the final year of the preceding window; training is the remainder,
        # with a purge gap between them.
        va_year = k - 1
        va_idx = prior[yr[prior] == va_year]
        tr_idx = prior[yr[prior] < va_year]
        if tr_idx.size == 0:                     # For k=2021 only one year exists; split it internally 80/20.
            cut = int(len(prior) * 0.8)
            tr_idx, va_idx = prior[:cut], prior[cut:]
        udates = np.unique(df["date"].to_numpy())
        # Purge the final PURGE days of training and validation so forward labels do not cross boundaries.
        tr_last = df["date"].to_numpy()[tr_idx].max()
        keep_tr = df["date"].to_numpy()[tr_idx] <= udates[np.searchsorted(udates, tr_last) - PURGE]
        tr_idx = tr_idx[keep_tr]
        va_last = df["date"].to_numpy()[va_idx].max()
        keep_va = df["date"].to_numpy()[va_idx] <= udates[np.searchsorted(udates, va_last) - PURGE]
        va_idx = va_idx[keep_va]

        tr_mask = np.zeros(len(df), bool); tr_mask[tr_idx] = True
        va_mask = np.zeros(len(df), bool); va_mask[va_idx] = True
        net, best, eps = train_fold(df, tr_mask, va_mask, args.seed, hidden, args.lr,
                                    args.wd, args.dropout, args.epochs, args.patience, dev)
        te = np.flatnonzero(yr == k)
        X = torch.tensor(df.loc[te, ["f_g", "f_m"]].to_numpy(np.float32), device=dev)
        net.eval()
        with torch.no_grad():
            p = net(X).cpu().numpy()
        sub = df.loc[te, ["date", "instrument", "label_raw", "label", "test_year"]].copy()
        sub["pred"] = p
        out.append(sub)
        ic = np.nanmean([_pearson(p[a:b], df["label"].to_numpy()[te][a:b])
                         for a, b in day_slices(df["date"].to_numpy()[te]) if b - a >= 30])
        print(f"  {k}: train {tr_mask.sum():>7,} rows / val {va_mask.sum():>7,} rows "
              f"| best_val_IC {best:+.4f} | {eps} epoch | **test RankIC-ish {ic:+.4f}** "
              f"| {time.time() - t0:.0f}s", flush=True)

    res = pd.concat(out, ignore_index=True)
    name = args.out or (f"model/preds_meta_{args.market}_"
                        f"{'vwapentry' if args.market == 'cn' else 'hlc3'}_s{args.seed}.parquet")
    res.to_parquet(name, index=False)
    print(f"\n→ {name}  {len(res):,} rows  {res['date'].min().date()}–{res['date'].max().date()}")


if __name__ == "__main__":
    main()
