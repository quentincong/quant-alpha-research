"""
longonly_null.py — add a **permutation null** for portfolio net returns/IR (supports long-short and long-only)

**Question to answer** (2026-08-30): after removing the short leg, net@10 ≈ +3.11 bps
and IR ≈ 1.19 remain. **Is this real, or was noise selected?**

**Constructing the null (the key is to disrupt only one thing)**:
  **Shuffle `label_raw` within each trading day**, then run the same long-only portfolio as usual.
  · Constituents, number of holdings, turnover, EMA smoothing, and the no-trade buffer all remain
    **exactly unchanged each day** because they depend only on `pred`, which is unchanged byte-for-byte.
  · The only disrupted relationship is between **the selected stocks and how much they actually earned**.
  · The benchmark (equal-weighted universe mean) is **identically unchanged** by within-day shuffling,
    so the active-return null is naturally centered at zero.

This is much cleaner than "shuffling pred": shuffling pred destroys the prediction's **time-series
persistence**, causing turnover to jump from 10% to ~100% and making costs incomparable. The resulting
portfolios would no longer represent the same object.

**Implementation speedup**: run the holdings logic only once. Each subsequent permutation
**samples n_held values without replacement from that day's return vector and takes the mean**.
This is distributionally identical to "shuffle, then average by holdings," but dozens of times faster.

Usage:
  python model/longonly_null.py --preds model/preds_walkforward_cn_vwapentry.parquet \
         --span 5 --exit 0.30 --n 2000
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turnover_study import smooth
from evaluate import apply_cap


def holdings(df: pd.DataFrame, q_enter: float, q_exit: float, legs: str):
    """Run the holdings logic once and record each day's return vector, leg sizes, turnover, and realized return.

    legs="long"      → long-only; return = D10 mean − universe mean (**active return**, implementable in A-shares)
    legs="longshort" → long-short; return = D10 mean − D1 mean (**assumes short selling is available**)
    Turnover is computed from the actual holdings under each definition.
    """
    rets, nL, nS, turn, obs = [], [], [], [], []
    prev = {1: {}, -1: {}}
    for _, day in df.groupby("date", sort=True):
        pct = day["pred"].rank(pct=True, method="first").to_numpy()
        names = day["instrument"].to_numpy()
        r = day["label_raw"].to_numpy()
        sides = (1, -1) if legs == "longshort" else (1,)
        held, ok, t_day, val = {}, True, 0.0, 0.0
        for side in sides:
            p = pct if side == 1 else 1.0 - pct
            h = set(names[p >= 1.0 - q_enter]) | (set(names[p >= 1.0 - q_exit]) & set(prev[side]))
            if not h:
                ok = False; break
            idx = np.isin(names, list(h))
            w = {nm: 1.0 / idx.sum() for nm in names[idx]}
            allnm = set(w) | set(prev[side])
            t_day += sum(abs(w.get(k, 0.0) - prev[side].get(k, 0.0)) for k in allnm) / 2
            prev[side] = w
            held[side] = idx
            val += side * float(r[idx].mean())
        if not ok:
            continue
        if legs == "long":
            val -= float(r.mean())                 # Subtract the benchmark to obtain active return.
        rets.append(r); turn.append(t_day); obs.append(val)
        nL.append(int(held[1].sum()))
        nS.append(int(held[-1].sum()) if legs == "longshort" else 0)
    t = np.asarray(turn, dtype=float); t[0] = np.nan
    return rets, np.asarray(nL), np.asarray(nS), t, np.asarray(obs)


def stats(active, turn, cost_bps):
    net = active - np.nan_to_num(turn, nan=0.0) * cost_bps / 1e4
    sd = net.std(ddof=1)
    return net.mean() * 1e4, (float(net.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--span", type=int, default=5)
    ap.add_argument("--exit", dest="q_exit", type=float, default=0.30)
    ap.add_argument("--cost", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--legs", choices=["long", "longshort"], default="long",
                    help="long=long-only (subtract universe benchmark; implementable in A-shares) / "
                         "longshort=long-short (assumes short selling is available)")
    ap.add_argument("--cap", type=float, default=None,
                    help="|label_raw| outlier safeguard; market default (us=0.8, cn=None); 0 disables it")
    args = ap.parse_args()

    df = pd.read_parquet(args.preds); df["date"] = pd.to_datetime(df["date"])
    # Share market defaults with evaluate/turnover/longonly so the tools stay consistent.
    market = "cn" if "_cn" in os.path.basename(args.preds) else "us"
    df = apply_cap(df, market, args.cap)
    sm = smooth(df, args.span)
    rets, nL, nS, turn, act = holdings(sm, 0.10, args.q_exit, args.legs)
    obs_net, obs_ir = stats(act, turn, args.cost)
    lbl = "active (after benchmark)" if args.legs == "long" else "gross long-short"
    print(f"[data] {os.path.basename(args.preds)}  legs={args.legs}  "
          f"ema{args.span}/exit{int(args.q_exit*100)}  {len(act)} days  cost {args.cost:.0f}bps")
    print(f"[observed] {lbl} {act.mean()*1e4:+.2f} bps | net@{args.cost:.0f} {obs_net:+.2f} bps "
          f"| IR {obs_ir:+.2f}\n")

    rng = np.random.default_rng(args.seed)
    null_net = np.empty(args.n); null_ir = np.empty(args.n)
    for b in range(args.n):
        a = np.empty(len(rets))
        for i, r in enumerate(rets):
            # Sampling nL(+nS) without replacement is distributionally equivalent to shuffling
            # within the day and averaging the held names.
            pick = rng.choice(len(r), nL[i] + nS[i], replace=False)
            if args.legs == "longshort":
                a[i] = r[pick[:nL[i]]].mean() - r[pick[nL[i]:]].mean()
            else:
                a[i] = r[pick].mean() - r.mean()
        null_net[b], null_ir[b] = stats(a, turn, args.cost)
    p_net = float((null_net >= obs_net).mean())
    p_ir = float((null_ir >= obs_ir).mean())
    print(f"[null] {args.n} within-day permutations (constituents/turnover/smoothing **unchanged**; only return assignment is shuffled)")
    print(f"  net@{args.cost:.0f}: null mean {null_net.mean():+.3f} | std {null_net.std():.3f} | "
          f"95th pct {np.quantile(null_net,0.95):+.3f} | **observed {obs_net:+.2f}** | "
          f"p={p_net:.4f} | z={(obs_net-null_net.mean())/null_net.std():+.1f}")
    print(f"  IR      : null mean {null_ir.mean():+.3f} | std {null_ir.std():.3f} | "
          f"95th pct {np.quantile(null_ir,0.95):+.3f} | **observed {obs_ir:+.2f}** | "
          f"p={p_ir:.4f} | z={(obs_ir-null_ir.mean())/null_ir.std():+.1f}")
    print(f"\nInterpretation: p is the frequency with which pure noise reaches or exceeds the observed value."
          f"\nThis tests **only** return assignment, not the fact that the variant was selected from 24 candidates; "
          f"that requires permuting the entire selection procedure.")


if __name__ == "__main__":
    main()
