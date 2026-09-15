"""
turnover_study.py — turnover reduction study

**Why run this study**: the cost curve in evaluate.py identifies a clear bottleneck.
  A-shares earn +11.14 bps/day gross but have **110% average daily turnover**, so the strategy
  breaks even at **~10 bps of cost**. A-share stamp duty alone is 5 bps one-way (0.05% on sales);
  after commission and slippage, actual cost is likely ≥10 bps. **Turnover, not the model, is the
  bottleneck.** Turnover multiplies the cost term, so reducing it may be easier than increasing IC.

**Why retraining is unnecessary**: every operation is **post-processing** of predictions saved
  by `walkforward.py`. Train once, then sweep any number of variants here.

**Three methods**:
  A. **EMA prediction smoothing**: take an exponential moving average of each stock's daily score.
     Turnover comes from daily rank noise; smoothing trades some IC for stability. It is **causal**,
     using only current and prior predictions.
  B. **Hysteresis / no-trade band**: enter at top q_enter but **exit only after falling outside q_exit**.
     Stocks near the boundary no longer churn in and out.
  C. Combine both methods.

**This is itself a selection procedure**: the more variants are swept, the more likely the
  selected "best" result is luck. Therefore, this script **prints the total variant count**
  for selection_null.py and the trial ledger.

Usage:
  python model/turnover_study.py --preds model/preds_walkforward_cn.parquet
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import ic_panel, decile_panel, COST_BPS, apply_cap


def smooth(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """Apply an EMA to pred over time within each instrument. span=1 means no smoothing.

    **Causality**: `ewm(...).mean()` uses only current and past values. The prediction for
    day t is available on day t, so smoothing introduces no future information.
    """
    if span <= 1:
        return df
    d = df.sort_values(["instrument", "date"]).copy()
    d["pred"] = (d.groupby("instrument", sort=False)["pred"]
                   .transform(lambda s: s.ewm(span=span, adjust=False).mean()))
    return d.sort_values(["date", "instrument"]).reset_index(drop=True)


def portfolio(df: pd.DataFrame, q_enter: float = 0.10, q_exit: float | None = None,
              with_series: bool = False) -> dict:
    """Long-short portfolio, turnover, and cost curve.

    q_exit=None → pure decile reconstruction (rebuild completely from each day's ranking,
    matching evaluate.py).
    q_exit>q_enter → buffer rule: **holdings = (prior holdings ∩ still in top q_exit) ∪ (top q_enter)**.

    When with_series=True, also return daily `_days` / `_gross` / `_turn` series for resampling
    in `block_bootstrap.py`. It is **disabled by default**, so existing callers remain unchanged.
    Error bars are not a new metric; they resample the same metric. Metric drift has caused
    problems three times in this project.
    """
    if q_exit is None:
        q_exit = q_enter
    gross, turn, days = [], [], []
    prev = {1: {}, -1: {}}
    for the_date, day in df.groupby("date", sort=True):
        pct = day["pred"].rank(pct=True, method="first").to_numpy()
        names = day["instrument"].to_numpy()
        rets = day["label_raw"].to_numpy()
        g = t_day = 0.0
        ok = True
        for side in (1, -1):
            # side=+1 selects the high-score tail; side=-1 selects the low-score tail,
            # handled symmetrically with 1-pct.
            p = pct if side == 1 else 1.0 - pct
            enter = set(names[p >= 1.0 - q_enter])
            keep = set(names[p >= 1.0 - q_exit]) & set(prev[side])
            held = enter | keep
            if not held:
                ok = False
                break
            idx = np.isin(names, list(held))
            w = {nm: 1.0 / idx.sum() for nm in names[idx]}
            g += side * float(rets[idx].mean())
            allnm = set(w) | set(prev[side])
            t_day += sum(abs(w.get(k, 0.0) - prev[side].get(k, 0.0)) for k in allnm) / 2
            prev[side] = w
        if ok:
            gross.append(g); turn.append(t_day); days.append(the_date)
    gross, turn = np.asarray(gross), np.asarray(turn)
    if gross.size == 0:
        return {}
    turn[0] = np.nan
    out = {"gross_bps": gross.mean() * 1e4, "turnover": float(np.nanmean(turn))}
    if with_series:
        out["_days"] = pd.DatetimeIndex(days)
        out["_gross"] = gross
        out["_turn"] = np.nan_to_num(turn, nan=0.0)   # Exclude initial-day entry turnover, matching the cost definition below.
    for bps in COST_BPS:
        net = gross - np.nan_to_num(turn, nan=0.0) * bps / 1e4
        out[f"net{bps}"] = net.mean() * 1e4
        out[f"ir{bps}"] = net.mean() / net.std(ddof=1) * np.sqrt(252) if net.std(ddof=1) > 0 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--spans", default="1,2,3,5,10,20")
    ap.add_argument("--exits", default="0.10,0.20,0.30,0.40")
    ap.add_argument("--cap", type=float, default=None,
                    help="|label_raw| outlier safeguard; market default (us=0.8, cn=None). "
                         "Required for US equities because the original label has no safeguard and contains "
                         "bankruptcy-reorganization price splices (see evaluate.apply_cap)")
    args = ap.parse_args()

    df = pd.read_parquet(args.preds)
    df["date"] = pd.to_datetime(df["date"])
    market = "cn" if "_cn" in os.path.basename(args.preds) else "us"
    df = apply_cap(df, market, args.cap)
    print(f"[data] {args.preds}  {len(df):,} rows  {df['date'].nunique()} days\n")

    spans = [int(x) for x in args.spans.split(",")]
    exits = [float(x) for x in args.exits.split(",")]
    rows, n_variants = [], 0

    print("=" * 104)
    print(f"{'variant':>22} {'RankIC':>8} {'mono':>7} {'gross':>8} {'turn':>7} "
          f"{'net@5':>8} {'net@10':>8} {'net@20':>8} {'IR@10':>7}")
    print("=" * 104)
    for span in spans:
        sm = smooth(df, span)
        ic = ic_panel(sm)["rankic_mean"]
        mono = decile_panel(sm)["monotonicity_spearman"]
        for q_exit in exits:
            n_variants += 1
            p = portfolio(sm, q_enter=0.10, q_exit=q_exit)
            if not p:
                continue
            tag = f"ema{span}/exit{int(q_exit*100)}"
            rows.append({"variant": tag, "span": span, "q_exit": q_exit,
                         "rankic": ic, "mono": mono, **p})
            print(f"{tag:>22} {ic:>+8.4f} {mono:>+7.3f} {p['gross_bps']:>+8.2f} "
                  f"{p['turnover']:>7.1%} {p['net5']:>+8.2f} {p['net10']:>+8.2f} "
                  f"{p['net20']:>+8.2f} {p['ir10']:>+7.2f}")
    print("=" * 104)
    r = pd.DataFrame(rows)
    best = r.loc[r["net10"].idxmax()]
    base = r[(r["span"] == 1) & (r["q_exit"] == 0.10)].iloc[0]
    print(f"\nBaseline (ema1/exit10, = evaluate.py definition): "
          f"RankIC {base['rankic']:+.4f} | turnover {base['turnover']:.1%} | net@10 {base['net10']:+.2f} bps")
    print(f"Best net@10       : {best['variant']}: "
          f"RankIC {best['rankic']:+.4f} | turnover {best['turnover']:.1%} | net@10 {best['net10']:+.2f} bps")
    print(f"\nEvaluated {n_variants} variants in this run. This is a **selection procedure**, "
          f"so the winner requires selection_null.py to count. Recorded in the trial ledger.")


if __name__ == "__main__":
    main()
