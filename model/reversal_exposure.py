"""
reversal_exposure.py — reversal exposure audit

**Why run this audit** (planned in docs/research_log.md#r36, run in docs/research_log.md#r38):
  The headline in docs/research_log.md#r35 is CN long-short net@10 **+21.76 bps** after honest reselection.
  However, this project's signal has **long been suspected of leaning heavily toward reversal**:
  docs/research_log.md#r20 measured that "ROC5 alone obtains 76% of the model's RankIC," and
  docs/research_log.md#r33 recorded concerns
  about bid-ask bounce. **But the share of model predictions that is simply reversal has never
  been measured.** "How much is reversal worth by itself?" differs from "how much of the alpha
  is reversal in disguise?" **They are distinct questions, and only the first was answered.**

**Reuse evaluate.py's constant as the signal definition**: `BASELINE_FACTOR = "ROC5"`,
  where `ROC5 = close.shift(5)/close`; the **unchanged positive sign is the reversal signal**
  (the derivation is fixed in evaluate.py's header). It is **ranking-equivalent** to `−ret5`
  because both are monotone decreasing transforms of the prior five-day return.

**Three stages, from cheapest to most expensive**:
  1. **Exposure**: run daily cross-sectional OLS `pred_z = α + β·rev_z`, and report mean β,
     mean R², and daily Spearman. This is **purely descriptive, involves no model selection,
     and does not leak when computed on the full sample**.
  2. **Orthogonalization**: compute `adj = pred_z − β·rev_z`, then apply the **same metric**
     (smooth + portfolio from turnover_study) and measure remaining net@10.
     **Estimate β with an expanding window**: β for year k uses only years before k, following
     the same discipline as selection_null.py `--mode nested`. Also print the full-sample β result
     side by side, **label it as leaking**, and use it only as an upper-bound reference.
  3. **Reference**: treat ROC5 itself as pred under the same variant to show what pure reversal
     is worth under this definition.

**No retraining and no changes to existing artifacts**: all operations post-process saved predictions.

Usage:
  python model/reversal_exposure.py --preds model/preds_cmp_gbdt_cn_vwapentry.parquet \\
         --data dataset/alphaFactor/dataset_alpha158_cn.parquet
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import BASELINE_FACTOR, apply_cap, ic_panel
from turnover_study import portfolio, smooth


def _csnorm(s: pd.Series) -> pd.Series:
    """Daily cross-sectional z-score. A **daily monotone affine transform does not change ranks**,
    so decile portfolios and turnover are completely insensitive to it; the script self-checks this."""
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / sd if sd > 0 else s * 0.0


def load_reversal(preds: pd.DataFrame, data_path: str) -> pd.DataFrame:
    """Merge ROC5 into the prediction table and standardize cross-sectionally by day.

    Read only ROC5 from the 2.4G dataset; parquet column storage keeps the cost low.
    This matches the make_baselines read path at evaluate.py:229.
    """
    raw = pd.read_parquet(data_path, columns=[BASELINE_FACTOR]).reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    m = preds.merge(raw, on=["date", "instrument"], how="left")
    n_miss = m[BASELINE_FACTOR].isna().sum()
    if n_miss:
        print(f"  [merge] {n_miss:,}/{len(m):,} rows lack ROC5 ({n_miss/len(m):.2%}) → dropping")
    m = m[m[BASELINE_FACTOR].notna()].copy()
    m["rev_z"] = m.groupby("date", sort=False)[BASELINE_FACTOR].transform(_csnorm)
    m["pred_z"] = m.groupby("date", sort=False)["pred"].transform(_csnorm)
    return m


def daily_beta(df: pd.DataFrame) -> pd.DataFrame:
    """Run daily cross-sectional regression pred_z ~ rev_z and return daily (beta, r2, spearman, n).

    Both columns are z-scored by day, so **beta is the correlation coefficient** and R² = beta².
    Also compute Spearman because this project's metric is RankIC, making rank correlation comparable.
    """
    rows = []
    for the_date, day in df.groupby("date", sort=True):
        if len(day) < 20:
            continue
        p, r = day["pred_z"].to_numpy(), day["rev_z"].to_numpy()
        if p.std() < 1e-12 or r.std() < 1e-12:
            continue
        beta = float(np.cov(p, r, ddof=0)[0, 1] / r.var())
        pear = float(np.corrcoef(p, r)[0, 1])
        sp = float(pd.Series(p).corr(pd.Series(r), method="spearman"))
        rows.append({"date": the_date, "beta": beta, "r2": pear ** 2,
                     "spearman": sp, "n": len(day)})
    return pd.DataFrame(rows)


def expanding_beta(bt: pd.DataFrame) -> dict:
    """For year k, average daily β over all years before k using an expanding window.

    This is the script's only choice about which data to use, so it must be nested. Following
    selection_null.py --mode nested, **estimating β on the current year to clean current-year
    predictions is leakage** and makes results look better than reality. The first year has no
    preceding year and therefore no β, so the caller drops it.
    """
    bt = bt.copy()
    bt["year"] = bt["date"].dt.year
    years = sorted(bt["year"].unique())
    return {y: float(bt.loc[bt["year"] < y, "beta"].mean()) for y in years[1:]}


def run_variant(df: pd.DataFrame, span: int, q_exit: float, col: str) -> dict:
    """Treat one column as pred and apply the fixed smooth(span) + portfolio(q_exit) metric."""
    d = df[["date", "instrument", "label", "label_raw"]].copy()
    d["pred"] = df[col].to_numpy()
    d = smooth(d, span)
    out = portfolio(d, q_enter=0.10, q_exit=q_exit)
    out["rankic"] = ic_panel(d)["rankic_mean"]
    return out


def _fmt(tag: str, r: dict) -> str:
    return (f"  {tag:<34} net@10 {r.get('net10', float('nan')):+7.2f} bps | "
            f"IR {r.get('ir10', float('nan')):+6.2f} | gross {r.get('gross_bps', float('nan')):+7.2f} | "
            f"turnover {r.get('turnover', float('nan')):6.1%} | RankIC {r.get('rankic', float('nan')):+7.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--data", required=True, help="source dataset from which to read ROC5")
    ap.add_argument("--span", type=int, default=5, help="EMA span (headline variant = 5)")
    ap.add_argument("--exit", dest="q_exit", type=float, default=0.10,
                    help="q_exit (headline variant = 0.10)")
    ap.add_argument("--cap", type=float, default=None)
    args = ap.parse_args()

    preds = pd.read_parquet(args.preds)
    preds["date"] = pd.to_datetime(preds["date"])
    market = "cn" if "_cn" in os.path.basename(args.preds) else "us"
    preds = apply_cap(preds, market, args.cap)
    print(f"[data] {args.preds}  market={market}  {len(preds):,} rows  "
          f"{preds['date'].min().date()} → {preds['date'].max().date()}")
    print(f"[variant] ema{args.span} / exit{int(args.q_exit*100)}")

    df = load_reversal(preds, args.data)

    # ---------------- 1. Exposure ----------------
    bt = daily_beta(df)
    print(f"\n{'='*78}\n[1] Reversal exposure (daily cross-sectional regression pred_z ~ rev_z; rev = positive {BASELINE_FACTOR})\n{'='*78}")
    print(f"  trading days {len(bt):,}")
    print(f"  mean β        {bt['beta'].mean():+.4f}   (sd {bt['beta'].std(ddof=1):.4f})")
    print(f"  mean R²       {bt['r2'].mean():.4f}      ← share of cross-sectional prediction variance explained by reversal")
    print(f"  mean Spearman {bt['spearman'].mean():+.4f}   ← comparable with the project's RankIC")
    print(f"  share of days with β>0  {(bt['beta'] > 0).mean():.1%}")
    bt["year"] = bt["date"].dt.year
    print("\n  By year:")
    print("    " + f"{'year':>6} {'mean β':>9} {'mean R²':>9} {'Spearman':>10} {'days':>6}")
    for y, g in bt.groupby("year"):
        print(f"    {y:>6} {g['beta'].mean():>+9.4f} {g['r2'].mean():>9.4f} "
              f"{g['spearman'].mean():>+10.4f} {len(g):>6}")

    # ---------------- 2. Orthogonalization ----------------
    ebeta = expanding_beta(bt)
    df["year"] = df["date"].dt.year
    keep = df["year"].isin(ebeta.keys())
    n_drop_year = sorted(set(df["year"]) - set(ebeta.keys()))
    sub = df[keep].copy()
    sub["beta_used"] = sub["year"].map(ebeta)
    sub["pred_ortho_nested"] = sub["pred_z"] - sub["beta_used"] * sub["rev_z"]
    sub["pred_ortho_full"] = sub["pred_z"] - bt["beta"].mean() * sub["rev_z"]

    print(f"\n{'='*78}\n[2] Apply the same metric after orthogonalization ({n_drop_year} year dropped because no preceding year estimates β)\n{'='*78}")
    print("  expanding-window β: " + "  ".join(f"{y}:{b:+.4f}" for y, b in ebeta.items()))

    raw_full = run_variant(df, args.span, args.q_exit, "pred")
    raw_z = run_variant(df, args.span, args.q_exit, "pred_z")
    print("\n  [self-check] pred and pred_z should match elementwise (daily z-score does not change ranks):")
    print(_fmt("raw pred (full period)", raw_full))
    print(_fmt("raw pred_z (full period)", raw_z))

    print(f"\n  [main result] comparison over the same year range ({min(ebeta)}–{max(ebeta)}):")
    base = run_variant(sub, args.span, args.q_exit, "pred_z")
    nest = run_variant(sub, args.span, args.q_exit, "pred_ortho_nested")
    full = run_variant(sub, args.span, args.q_exit, "pred_ortho_full")
    rev = run_variant(sub, args.span, args.q_exit, "rev_z")
    print(_fmt("1. raw pred", base))
    print(_fmt("2. orthogonalized (nested β; honest)", nest))
    print(_fmt("3. orthogonalized (full-sample β; leaking upper bound only)", full))
    print(_fmt("4. pure reversal rev_z", rev))
    d_nest = nest.get("net10", np.nan) - base.get("net10", np.nan)
    print(f"\n  Cost of removing reversal (2−1): {d_nest:+.2f} bps"
          f" ({abs(d_nest)/abs(base.get('net10', np.nan)):.1%} of the original)")


if __name__ == "__main__":
    main()
