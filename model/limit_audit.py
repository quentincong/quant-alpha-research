"""
limit_audit.py — price-limit gates cover only the long leg; the short leg has never had gates

**What this script audits**
The gate in `build_dataset.add_label_cn` is:
    can_buy@T+1  AND  can_sell@T+2
This is a **round trip for the long leg** (buy at T+1 and sell at T+2). However,
`evaluate.py` / `turnover_study.py` construct a **decile long-short portfolio** (long D10,
short D1). A round trip for the short leg is:
    T+1 **sell to open** → T+1 must not be limit-down (`can_sell@T+1`)
    T+2 **buy to cover** → T+2 must not be limit-up (`can_buy@T+2`)
**Neither condition has ever been checked anywhere in the project.**

**Why this matters**: D1 is the model's lowest-predicted decile, precisely the group
**most likely to hit limit-down**. The short leg therefore records some returns from positions
that **could not have been opened that day**. After the 2026-08-30 vwap experiment, about
**70%** of D10−D1 comes from the short leg (D1 = −16.29 bps), making this gap more important than before.

This script **does not change any label**. It does only three things:
  1. Report the share of "untradeable short-leg" observations in each decile and check whether they concentrate in D1.
  2. Remove untradeable short-leg rows and recompute D10−D1 and the long-short portfolio.
  3. Also report long-leg gates; they should be 0% because the label already removes them. This is a **self-check**.

**This script does not address the broader assumption that borrowing individual A-shares
is inherently very difficult.** That belongs in the README Limitations section and cannot be
fixed in code. It answers the narrower question: **assuming short selling is available, how much
short-leg return is lost to price limits?**

Usage:
  python model/limit_audit.py --preds model/preds_walkforward_cn_vwapfull.parquet
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASET = "dataset/alphaFactor/dataset_alpha158_cn.parquet"
N_DEC = 10


def load_flags() -> pd.DataFrame:
    """Read authoritative can_buy/can_sell flags from the dataset and derive tradability for all four actions."""
    g = pd.read_parquet(DATASET, columns=["can_buy", "can_sell"]).reset_index()
    g["date"] = pd.to_datetime(g["date"]); g["instrument"] = g["instrument"].astype(str)
    g = g.sort_values(["date", "instrument"]).reset_index(drop=True)
    gb = g.groupby("instrument", sort=False)
    out = pd.DataFrame({"date": g["date"], "instrument": g["instrument"]})
    # Long leg (already enforced in the label; checked here as a self-test).
    out["long_ok"] = gb["can_buy"].shift(-1).eq(True) & gb["can_sell"].shift(-2).eq(True)
    # Short leg (**never previously checked**): opening requires selling at T+1;
    # covering requires buying at T+2.
    out["short_open_ok"] = gb["can_sell"].shift(-1).eq(True)    # T+1 is not limit-down.
    out["short_cover_ok"] = gb["can_buy"].shift(-2).eq(True)    # T+2 is not limit-up.
    out["short_ok"] = out["short_open_ok"] & out["short_cover_ok"]
    return out


def deciles(df: pd.DataFrame) -> pd.Series:
    r = df.groupby("date")["pred"].rank(method="first")
    cnt = df.groupby("date")["pred"].transform("size")
    return np.floor((r - 1) * N_DEC / cnt).astype(int).clip(0, N_DEC - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.preds)
    df["date"] = pd.to_datetime(df["date"]); df["instrument"] = df["instrument"].astype(str)
    df = df.merge(load_flags(), on=["date", "instrument"], how="left")
    df["decile"] = deciles(df)
    print(f"[data] {args.preds}  {len(df):,} rows\n")

    print("Untradeable share by decile (D1 = lowest prediction = short leg; D10 = highest prediction = long leg)")
    print(f"{'decile':>7} {'long untradeable':>16} {'short untradeable':>16} "
          f"{'T+1 limit-down/open':>20} {'T+2 limit-up/cover':>19}")
    for d, g in df.groupby("decile"):
        print(f"{'D%d' % (d + 1):>7} {1 - g['long_ok'].mean():>15.3%} "
              f"{1 - g['short_ok'].mean():>15.3%} {1 - g['short_open_ok'].mean():>19.3%} "
              f"{1 - g['short_cover_ok'].mean():>18.3%}")

    # ------------------------------------------------------------------
    # Entry versus exit: they are fundamentally different and must not be handled together.
    #   · A price limit blocking **entry** means the position **was never established**;
    #     removing that row is correct.
    #   · A price limit blocking **exit** means the position is **already held** and the
    #     limit event occurs at T+2. The T+1 trade cannot be undone, so **the row must remain**;
    #     the loss is real. Removing failed exits truncates only the loss tail and inflates returns.
    #     The current label's can_sell@T+2 makes this mistake; the first version of this script
    #     made the same mistake on the short leg and was corrected.
    #
    #   Current status of the four actions:
    #     Long entry  can_buy@T+1   correctly gated
    #     Long exit   can_sell@T+2  **incorrectly gated**; real losses are removed
    #     Short entry can_sell@T+1  **not gated at all**; impossible positions are recorded
    #     Short exit  can_buy@T+2   not gated, which is correct
    # ------------------------------------------------------------------
    per = df.groupby(["date", "decile"])["label_raw"].mean().unstack()
    d1_raw, d10_raw = per[0].mean(), per[N_DEC - 1].mean()

    # Filter only on **entry**: limit-down at T+1 means the short cannot be opened,
    # so the row must not enter the short leg.
    ent = df[(df["decile"] != 0) | (df["short_open_ok"] == True)]      # noqa: E712
    d1_entry = ent.groupby(["date", "decile"])["label_raw"].mean().unstack()[0].mean()

    # Comparison: also remove **failed exits**. This is incorrect and is shown only to
    # measure how much it artificially inflates performance.
    both = df[(df["decile"] != 0) | (df["short_ok"] == True)]          # noqa: E712
    d1_both = both.groupby(["date", "decile"])["label_raw"].mean().unstack()[0].mean()

    print(f"\n{'=' * 74}")
    print(f"  D10                                : {d10_raw * 1e4:+8.2f} bps")
    print(f"  D1  raw (no short-leg gates)        : {d1_raw * 1e4:+8.2f} bps")
    print(f"  D1  entry-only filter (T+1 limit-down) : {d1_entry * 1e4:+8.2f} bps  ← correct definition")
    print(f"  D1  also filter failed exits (**wrong**) : {d1_both * 1e4:+8.2f} bps  "
          f"← comparison only: artificial gain {(d1_raw - d1_both) * 1e4:+.2f} bps")
    print("-" * 74)
    print(f"  D10−D1  raw                        : {(d10_raw - d1_raw) * 1e4:+8.2f} bps")
    print(f"  D10−D1  short-entry gate (correct) : {(d10_raw - d1_entry) * 1e4:+8.2f} bps")
    print(f"      → cost of the short-entry gate : "
          f"{((d10_raw - d1_entry) - (d10_raw - d1_raw)) * 1e4:+8.2f} bps")
    print(f"  D10−D1  wrongly filter exits too   : {(d10_raw - d1_both) * 1e4:+8.2f} bps "
          f"← this value is **not credible**; it only shows the inflation")
    print("=" * 74)
    print("\nNote: even under the correct definition, failed short exits still realize the label loss. In reality,"
          "\n    an uncovered position remains trapped and losses can **only grow**. This is still an **optimistic** lower bound.")


if __name__ == "__main__":
    main()
