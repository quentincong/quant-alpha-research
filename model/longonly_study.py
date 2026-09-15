"""
longonly_study.py — how much remains if **short selling is unavailable**?

**Why this file is necessary** (the largest concern identified on 2026-08-30):
Under the vwap definition, D10−D1 = +23.63 bps, of which **D1 (the short leg)
contributes −16.29, about 70%**. In practice, **shorting individual A-shares is
extremely difficult**: eligible securities are limited, borrow is scarce, costs are high,
and positions may be forcibly closed. Therefore, **that "70%" may be entirely unattainable
in the A-share market.** Reporting only the long-short spread assumes shorting is available,
an assumption that is **not defensible in this market**.

**Use a long-only definition that can actually be implemented in A-shares instead**:

    active return = equal-weighted D10 return − **equal-weighted full-universe return**

Interpretation: "how much more did a fully invested position in the **model's top 10%**
earn than a fully invested position in the **entire market**?"
  · No securities lending or leverage is required; **both retail and mutual-fund A-share
    investors can implement it**.
  · Subtracting the universe mean is necessary; otherwise, gains may simply reflect
    **a rising market** (beta), not alpha.
  · Costs include only **long-leg turnover** (buy commission + 5 bps sell stamp duty + slippage).

**Also report raw D10 returns** to show how much comes from beta versus stock selection.

Usage:
  python model/longonly_study.py --preds model/preds_walkforward_cn_vwapfull.parquet
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

N_DEC = 10
COSTS = [0, 5, 10, 20]


def long_only(df: pd.DataFrame, q_enter: float = 0.10, q_exit: float | None = None,
              with_series: bool = False) -> dict:
    """Hold only the top q_enter (optionally with a no-trade buffer), benchmarked against the equal-weighted full universe.

    When with_series=True, also return daily `_days` / `_act` / `_turn` series for
    resampling in block_bootstrap.py. It is disabled by default, leaving existing callers unchanged.
    """
    if q_exit is None:
        q_exit = q_enter
    act, gross, bench, turn, days = [], [], [], [], []
    prev: dict[str, float] = {}
    for the_date, day in df.groupby("date", sort=True):
        pct = day["pred"].rank(pct=True, method="first").to_numpy()
        names = day["instrument"].to_numpy()
        rets = day["label_raw"].to_numpy()
        enter = set(names[pct >= 1.0 - q_enter])
        keep = set(names[pct >= 1.0 - q_exit]) & set(prev)
        held = enter | keep
        if not held:
            continue
        idx = np.isin(names, list(held))
        w = {nm: 1.0 / idx.sum() for nm in names[idx]}
        g = float(rets[idx].mean())
        b = float(rets.mean())                      # Equal-weighted full universe = benchmark.
        allnm = set(w) | set(prev)
        turn.append(sum(abs(w.get(k, 0.0) - prev.get(k, 0.0)) for k in allnm) / 2)
        prev = w
        gross.append(g); bench.append(b); act.append(g - b); days.append(the_date)
    if not act:
        return {}
    act = np.asarray(act); gross = np.asarray(gross); bench = np.asarray(bench)
    turn = np.asarray(turn, dtype=float); turn[0] = np.nan
    out = {"n_days": len(act), "gross_bps": gross.mean() * 1e4,
           "bench_bps": bench.mean() * 1e4, "active_bps": act.mean() * 1e4,
           "turnover": float(np.nanmean(turn))}
    if with_series:
        out["_days"] = pd.DatetimeIndex(days)
        out["_act"] = act
        out["_turn"] = np.nan_to_num(turn, nan=0.0)
    for c in COSTS:
        net = act - np.nan_to_num(turn, nan=0.0) * c / 1e4
        sd = net.std(ddof=1)
        out[f"net{c}"] = net.mean() * 1e4
        out[f"ir{c}"] = float(net.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--spans", default="1,3,5,10")
    ap.add_argument("--exits", default="0.10,0.20,0.30")
    ap.add_argument("--cap", type=float, default=None,
                    help="|label_raw| outlier safeguard; market default (us=0.8, cn=None). "
                         "Required for US equities because the original label has no safeguard and contains "
                         "bankruptcy-reorganization price splices (see evaluate.apply_cap)")
    args = ap.parse_args()

    df = pd.read_parquet(args.preds)
    df["date"] = pd.to_datetime(df["date"])
    market = "cn" if "_cn" in os.path.basename(args.preds) else "us"
    df = apply_cap(df, market, args.cap)
    print(f"[data] {args.preds}  {len(df):,} rows  {df['date'].nunique()} days")
    print("Definition: equal-weighted long D10; benchmark = equal-weighted full universe; **no short selling**\n")
    print("=" * 100)
    print(f"{'variant':>16} {'D10 gross':>8} {'benchmark':>8} {'active':>8} {'turnover':>7} "
          f"{'net@5':>8} {'net@10':>8} {'net@20':>8} {'IR@10':>7}")
    print("=" * 100)
    n = 0
    for span in [int(x) for x in args.spans.split(",")]:
        sm = smooth(df, span)
        for qe in [float(x) for x in args.exits.split(",")]:
            r = long_only(sm, 0.10, qe)
            if not r:
                continue
            n += 1
            print(f"{f'ema{span}/exit{int(qe*100)}':>16} {r['gross_bps']:>+8.2f} "
                  f"{r['bench_bps']:>+8.2f} {r['active_bps']:>+8.2f} {r['turnover']:>7.1%} "
                  f"{r['net5']:>+8.2f} {r['net10']:>+8.2f} {r['net20']:>+8.2f} {r['ir10']:>+7.2f}")
    print("=" * 100)
    print(f"A total of {n} variants (a selection procedure recorded in the trial ledger).")
    print("**Active return is the claimable quantity** — gross D10 return includes market beta, which is not attributable to the model.")


if __name__ == "__main__":
    main()
