"""
selection_null.py — test the **entire selection procedure** by permutation

**Why this file is required**: `selection_null.py` tests the full selection procedure:
  The reported headline variants (long-short `ema5/exit10`, long-only `ema5/exit30`) were **not
  prespecified**. They were **the best selected** from grids of 24 long-short and 12 long-only cells.
  The reported score is therefore the **maximum over correlated draws** and is naturally biased
  upward; even with no underlying signal, the "best" candidate will look good. `longonly_null.py`
  tests return assignment for **this variant**, but **does not test that the variant was selected**.
  This file adds the latter test.

**Run both formulations because they answer different questions:**

  1. `--mode nested` (**honest reselection**): put variant selection inside walk-forward.
     For each test year k, select the variant **using only years before k**, then apply it to year k.
     The reported number is then **out of sample with respect to variant selection**.
     Its difference from the "global best variant" is the **free benefit purchased by selection**,
     directly interpretable without a null. This is easiest to explain in 60 seconds and closest
     to live deployment.

  2. `--mode permute` (**permutation null for the selection procedure**): under H0 that predictions
     and returns are unrelated, **scan the full grid and take the best result**, repeating N times.
     This produces the distribution of best-of-K with no signal. The null side includes **the same
     selection bias**, making the two sides comparable. It asks: **"Can the current headline level
     be obtained solely by selecting the best candidate from noise?"**

     A pitfall appeared on the first run on 2026-08-31 and must not be repeated. The original plan
     was to use `best − fixed baseline` to isolate the contribution from selecting well. **It fails.**
     The observed difference is +5.22, while the null mean is an even larger **+9.40**. This does not
     mean selection contributed nothing; the difference is **dominated by the turnover-cost mechanism**.
     Permuting returns does not change holdings or turnover. High-span variants inherently have lower
     turnover and save more costs, so even in a world with **no signal**, "best-of-24" reliably beats
     `ema1/exit10` by about 9 bps. That is **deterministic cost arithmetic**, not selection skill,
     making the differences on the two sides incomparable.
     → **Use `--mode nested` to quantify selection bias**; it compares two selection procedures on the
       same real data and avoids this confound. Permutation mode examines only the **level**, not the gap.

**Why many permutations are inexpensive**: holdings depend only on `pred`, so permuting `label_raw`
  **does not change holdings, turnover, or smoothing**. Compute and cache holdings once; each later
  permutation only substitutes a return vector and reaggregates, as in `longonly_null.py`.

**Discipline**: every conclusion must also run on the **US negative control**. If the negative control
  also passes, the test has no discriminatory power (the lesson from docs/research_log.md#r31: the within-day permutation
  null also produced p<0.001 for US equities).

Usage (from the repository root, after conda activate torch-env):
  python model/selection_null.py --preds model/preds_walkforward_cn_vwapentry.parquet \
         --legs longshort --mode nested
  python model/selection_null.py --preds model/preds_walkforward_cn_vwapentry.parquet \
         --legs long --mode permute --n 200
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

COSTS = [0, 5, 10, 20]
# Grid defaults **must** match the grid originally swept; otherwise this tests a different selection procedure.
GRID = {"longshort": ("1,2,3,5,10,20", "0.10,0.20,0.30,0.40"),   # turnover_study default.
        "long":      ("1,3,5,10",      "0.10,0.20,0.30")}        # longonly_study default.


def build_holdings(df: pd.DataFrame, span: int, q_enter: float, q_exit: float, legs: str):
    """Run holdings logic once and cache daily holding indices + turnover + return vector.

    This **depends only on pred**, so permuting label_raw requires no recomputation at this layer,
    which makes the script tractable. Returned `rets` are full-universe daily return vectors
    on which permutations operate.
    """
    sm = smooth(df, span)          # EMA uses current and prior days only; causal, and no annual reset better matches deployment.
    dates, rets, longs, shorts, turn = [], [], [], [], []
    prev = {1: {}, -1: {}}
    sides = (1, -1) if legs == "longshort" else (1,)
    for dt, day in sm.groupby("date", sort=True):
        pct = day["pred"].rank(pct=True, method="first").to_numpy()
        names = day["instrument"].to_numpy()
        held, ok, t_day = {}, True, 0.0
        for side in sides:
            p = pct if side == 1 else 1.0 - pct
            h = set(names[p >= 1.0 - q_enter]) | (set(names[p >= 1.0 - q_exit]) & set(prev[side]))
            if not h:
                ok = False; break
            idx = np.flatnonzero(np.isin(names, list(h)))
            w = {nm: 1.0 / idx.size for nm in names[idx]}
            allnm = set(w) | set(prev[side])
            t_day += sum(abs(w.get(k, 0.0) - prev[side].get(k, 0.0)) for k in allnm) / 2
            prev[side] = w
            held[side] = idx
        if not ok:
            continue
        dates.append(dt)
        rets.append(day["label_raw"].to_numpy())
        longs.append(held[1])
        shorts.append(held[-1] if legs == "longshort" else None)
        turn.append(t_day)
    t = np.asarray(turn, dtype=float)
    if t.size:
        t[0] = np.nan
    return {"dates": np.asarray(dates), "rets": rets, "longs": longs,
            "shorts": shorts, "turn": t}


def daily_active(h: dict, rets: list[np.ndarray], legs: str) -> np.ndarray:
    """Given holdings and returns, compute daily active/long-short returns; rerun only this step for permutations."""
    out = np.empty(len(rets))
    for i, r in enumerate(rets):
        v = r[h["longs"][i]].mean()
        v -= r[h["shorts"][i]].mean() if legs == "longshort" else r.mean()
        out[i] = v
    return out


def stats(active: np.ndarray, turn: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Net return (bps/day) and naive annualized IR. mask selects specific years."""
    if mask is not None:
        active, turn = active[mask], turn[mask]
    out = {}
    for c in COSTS:
        net = active - np.nan_to_num(turn, nan=0.0) * c / 1e4
        sd = net.std(ddof=1)
        out[f"net{c}"] = net.mean() * 1e4
        out[f"ir{c}"] = float(net.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
    return out


def scan(df: pd.DataFrame, legs: str, spans: list[int], exits: list[float]):
    """Compute and cache holdings for the full grid. **This is the only expensive step; run it once.**"""
    variants, H = [], {}
    for span in spans:
        for q_exit in exits:
            tag = f"ema{span}/exit{int(q_exit * 100)}"
            H[tag] = build_holdings(df, span, 0.10, q_exit, legs)
            variants.append(tag)
            print(f"  [scan] {tag:>14}  {len(H[tag]['dates'])} days", flush=True)
    n = len(H[variants[0]]["dates"])
    assert all(len(H[v]["dates"]) == n for v in variants), "variant day counts differ; permutations cannot be shared"
    return variants, H


def mode_nested(variants, H, rets, legs, years):
    """Honest reselection: select year k's variant using only years before k."""
    dates = H[variants[0]]["dates"]
    yr = pd.DatetimeIndex(dates).year.to_numpy()
    A = {v: daily_active(H[v], rets, legs) for v in variants}

    ys = sorted(years)
    picks, chosen_act, chosen_turn = [], [], []
    for k in ys[1:]:
        prior, cur = yr < k, yr == k
        best = max(variants, key=lambda v: stats(A[v], H[v]["turn"], prior)["net10"])
        s = stats(A[best], H[best]["turn"], cur)
        picks.append((k, best, s["net10"], s["ir10"]))
        chosen_act.append(A[best][cur]); chosen_turn.append(H[best]["turn"][cur])

    print(f"\n{'=' * 78}\n[nested] select each year's variant using **past years only**\n{'=' * 78}")
    print(f"  {'test year':>8} {'selected variant':>16} {'year net@10':>13} {'IR@10':>8}")
    for k, v, n10, ir in picks:
        print(f"  {k:>8} {v:>16} {n10:>+13.2f} {ir:>+8.2f}")

    later = yr >= ys[1]
    nested = stats(np.concatenate(chosen_act), np.concatenate(chosen_turn))
    glob = max(variants, key=lambda v: stats(A[v], H[v]["turn"])["net10"])
    oracle = stats(A[glob], H[glob]["turn"], later)
    base = stats(A["ema1/exit10"], H["ema1/exit10"]["turn"], later)
    print(f"\n  Three definitions over the same years ({ys[1]}–{ys[-1]}):")
    print(f"    {'definition':<34} {'net@10':>9} {'IR@10':>8}")
    print(f"    {'1. ex-post global best ' + glob + ' (current headline)':<34} "
          f"{oracle['net10']:>+9.2f} {oracle['ir10']:>+8.2f}")
    print(f"    {'2. honest reselection (past years only)':<34} {nested['net10']:>+9.2f} {nested['ir10']:>+8.2f}")
    print(f"    {'3. fixed baseline ema1/exit10':<34} {base['net10']:>+9.2f} {base['ir10']:>+8.2f}")
    print(f"\n  Free benefit purchased by selection = 1 − 2 = {oracle['net10'] - nested['net10']:+.2f} bps "
          f"(IR {oracle['ir10'] - nested['ir10']:+.2f})")
    print(f"  Claimable turnover-reduction benefit = 2 − 3 = {nested['net10'] - base['net10']:+.2f} bps "
          f"(IR {nested['ir10'] - base['ir10']:+.2f})")


def mode_permute(variants, H, rets, legs, n_iter, seed):
    """Selection-procedure permutation null: scan the **entire grid** and take the best result each time."""
    A = {v: daily_active(H[v], rets, legs) for v in variants}
    real = {v: stats(A[v], H[v]["turn"])["net10"] for v in variants}
    best_v = max(real, key=real.get)
    real_best, real_base = real[best_v], real["ema1/exit10"]

    rng = np.random.default_rng(seed)
    null_best, null_base, null_gap = [], [], []
    for it in range(n_iter):
        perm = [rng.permutation(r) for r in rets]          # Share permutations across variants to preserve their correlations.
        s = {v: stats(daily_active(H[v], perm, legs), H[v]["turn"])["net10"] for v in variants}
        b = max(s.values())
        null_best.append(b); null_base.append(s["ema1/exit10"]); null_gap.append(b - s["ema1/exit10"])
        if (it + 1) % 25 == 0:
            print(f"  [permute] {it + 1}/{n_iter}", flush=True)

    def report(name, obs, null):
        null = np.asarray(null)
        p = (1 + (null >= obs).sum()) / (1 + len(null))
        z = (obs - null.mean()) / null.std(ddof=1) if null.std(ddof=1) > 0 else np.nan
        print(f"  {name:<26} observed {obs:>+8.2f} | null mean {null.mean():>+7.2f} "
              f"sd {null.std(ddof=1):>5.2f} | 95th pct {np.quantile(null, .95):>+7.2f} "
              f"| p={p:.4f} z={z:+.1f}")

    print(f"\n{'=' * 78}\n[permute] {n_iter} within-day permutations; each reruns the full "
          f"{len(variants)}-cell grid and takes the best\n{'=' * 78}")
    print(f"  observed best variant: {best_v}")
    report("best-of-K (net@10)", real_best, null_best)
    report("fixed baseline ema1/exit10", real_base, null_base)
    report("(reference) best−baseline; not selection evidence", real_best - real_base, null_gap)
    print("\nInterpretation: the first two rows are this mode's conclusions: how high best-of-K reaches without signal.")
    print("     Observed far above null means **the headline level was not created by selecting the best noise**.")
    print("     The third row is reference only and **cannot** measure selection bias: turnover differences between")
    print("     variants are deterministic, so high span saves costs without signal and null best−baseline is inherently large (see header).")
    print("     Use --mode nested to quantify selection bias.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--legs", choices=["long", "longshort"], default="longshort")
    ap.add_argument("--mode", choices=["nested", "permute"], default="nested")
    ap.add_argument("--spans", default=None, help="default to the grid originally swept for this definition")
    ap.add_argument("--exits", default=None)
    ap.add_argument("--n", type=int, default=200, help="number of permutations under --mode permute")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=float, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.preds)
    df["date"] = pd.to_datetime(df["date"])
    market = "cn" if "_cn" in os.path.basename(args.preds) else "us"
    df = apply_cap(df, market, args.cap)
    # Sort globally by (date, instrument) so daily return-vector order matches across variants,
    # allowing permutations to be shared.
    df = df.sort_values(["date", "instrument"]).reset_index(drop=True)
    gs, ge = GRID[args.legs]
    spans = [int(x) for x in (args.spans or gs).split(",")]
    exits = [float(x) for x in (args.exits or ge).split(",")]
    print(f"[data] {args.preds}  market={market}  {len(df):,} rows  "
          f"{df['date'].nunique()} days  legs={args.legs}  grid {len(spans) * len(exits)} cells")

    variants, H = scan(df, args.legs, spans, exits)
    rets = H[variants[0]]["rets"]
    if args.mode == "nested":
        mode_nested(variants, H, rets, args.legs, sorted(df["test_year"].unique()))
    else:
        mode_permute(variants, H, rets, args.legs, args.n, args.seed)


if __name__ == "__main__":
    main()
