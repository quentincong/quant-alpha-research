"""
make_vwap_label.py — rebuild the A-share label to quantify how much IC comes from close pricing and the current gate

This script generates labels along the two **price × gate** dimensions. It outputs labels only,
without copying 2.5GB of features, and `walkforward.py --label-file` substitutes them.

--------------------------------------------------------------------------
Dimension 1: `--price {close,vwap}` — execution price
--------------------------------------------------------------------------
The current label is `close_adj(T+2)/close_adj(T+1)-1` (buy and sell at the close).
It deliberately matches the US formula for cross-market comparison (docs/research_log.md#r24 corrected
the mistaken claim that vwap was used).

**Why test vwap** (not to improve the score, but for **credibility**):
  1. **The closing price is an instantaneous execution price.** This strategy trades about 240
     positions each day, which cannot all execute at the final close print. vwap = `amount/volume`
     is the day's volume-weighted average price, an **attainable price** when orders are spread
     through the day and the industry execution benchmark.
  2. **Bias matters more than optimism**: the close contains **bid-ask bounce**. If T+1 closes
     at the bid and T+2 at the ask, a positive return appears from microstructure alone and has
     no economic meaning. This bounce is **naturally mean-reverting** and looks exactly like a
     **reversal factor**. This project's signal is strongly reversal-oriented (**ROC5 alone obtains
     76% of the model's RankIC**), so part of **+0.0236 may be only bounce**. Averaging over the day
     largely removes bounce from vwap.

  **Expectation: RankIC will decline. The decline measures the part borrowed from close pricing.**

  Measured scale (sample of 60 tickers): the standard deviation of `vwap/close-1` is **1.07%**,
  while the label's own standard deviation is ≈ 1.99%. Across both legs, changing the price
  definition perturbs the label by about ±1.5% versus a ~2% signal. This is **not** a marginal
  repricing, so a large change is expected.

**Adjustment pitfall**: `amount/volume` produces an **unadjusted** average price, while the label
  must use adjusted prices. Multiply by the same-day adjustment factor `adj = AdjClose/Close`;
  otherwise ex-dividend dates create spurious large gaps.

**Effect of price limits on vwap (raised on 2026-08-30)**:
  A-shares have price limits, so **vwap is naturally milder than the close**; a daily average price
  cannot be pinned to the ±10%/±20% extreme as the close can. Therefore, the vwap label's **tails
  are mechanically compressed** and the standard deviation of `label_raw` declines. RankIC is
  dimensionless and unaffected, but **decile bps and the cost curve change scale**. Inspect the
  standard deviation before comparing net@10 across labels. The script prints both definitions.

--------------------------------------------------------------------------
Dimension 2: `--gate {full,entry,none}` — scope of the price-limit gate
--------------------------------------------------------------------------
The current gate (`build_dataset.add_label_cn`) is `can_buy@T+1 AND can_sell@T+2`.
If either condition fails, the entire label becomes NaN and the row disappears from the panel.
**These conditions are fundamentally different**:

  · `can_buy@T+1` is an **entry filter**. A limit-up close at T+1 means buying is impossible
    and the position is never established, so **removing the row is correct** and intentional.

  · `can_sell@T+2` is a **failed exit**, which is entirely different. The position was **already
    bought at the T+1 close**, and limit-down occurs at **T+2**. The T+1 purchase cannot be undone.
    Whether the position can be sold does not change the fact that the T+1→T+2 loss occurred.
    Removing the row pretends the position never existed.

  **The consequence is asymmetric**: the gate removes **large negative labels** (T+2 limit-down
  ≈ −10%, the largest losses in the dataset), while leaving corresponding large positive labels
  (`can_sell` checks only limit-down; T+2 **limit-up** is never removed). This one-sided truncation
  of the label distribution's **left tail** biases returns upward.

  Therefore, `--gate entry` keeps only `can_buy@T+1` (the entry filter) and **restores** losses
  from failed exits. `--gate full` is the current behavior. Their difference is the contribution
  borrowed from left-tail truncation.

--------------------------------------------------------------------------
Self-check (`--verify`)
--------------------------------------------------------------------------
`--price close --gate full` must be **elementwise identical** to the existing `label_raw`/`label`
in `dataset_alpha158_cn.parquet`. A match proves that the gate, shift alignment, |rtn| safeguard,
and cross-sectional robust z-score exactly reproduce the original implementation. **Only then can
later label differences be attributed uniquely to the changed variable.** Investigate any mismatch
here before interpreting other numbers. This follows the same idea as evaluate.py's self_check.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
  python dataset/alphaFactor/make_vwap_label.py --verify              # Run the self-check first.
  python dataset/alphaFactor/make_vwap_label.py --price vwap --gate full
  python dataset/alphaFactor/make_vwap_label.py --price close --gate entry
  python dataset/alphaFactor/make_vwap_label.py --price vwap --gate entry
"""
from __future__ import annotations
import argparse
import glob
import os

import numpy as np
import pandas as pd

OUT_TPL = "dataset/alphaFactor/label_{market}_{price}_{gate}.parquet"
EPS     = 1e-12

# ---------------------------------------------------------------------------
# Keep all market differences in this table; the remaining code is shared for comparability.
# ---------------------------------------------------------------------------
MARKETS = {
    "cn": dict(raw="dataset/rawdata_cn/raw",
               ds="dataset/alphaFactor/dataset_alpha158_cn.parquet",
               has_gate=True,       # Has can_buy/can_sell price-limit gates.
               rtn_cap=0.8,         # |rtn|>0.8 safeguard for suspension/reopen gaps.
               id_from="code"),     # instrument comes from the code column.
    "us": dict(raw="dataset/rawdata/raw",
               ds="dataset/alphaFactor/dataset_alpha158.parquet",
               has_gate=False,      # US equities have no price limits, so build_dataset.add_label has no gate.
               rtn_cap=None,        # **The original US implementation has no |rtn| safeguard**; adding one would break the match.
               id_from="filename"), # instrument = filename because US raw data have no code column.
}

# ---------------------------------------------------------------------------
# Why US equities require a vwap proxy, and why that proxy must first be calibrated on A-shares
# ---------------------------------------------------------------------------
# The US source (yfinance) has **only OHLCV and no turnover value**, so true `amount/volume`
# vwap is **unavailable**. The only option is the **typical-price** proxy
# `hlc3 = (high+low+close)/3`, also used by the `vwap` feature in features.py.
#
# If US equities then show "no improvement," there are **two** mutually exclusive explanations:
#   (a) US equities genuinely have no signal, so the negative control holds;
#   (b) hlc3 is a poor proxy that erased the effect itself.
# **US results alone cannot distinguish them.**
#
# Therefore, hlc3 must also be run on **A-shares**, the only market with both true vwap and hlc3.
#   · If CN-hlc3 reproduces most of the CN true-vwap improvement, hlc3 is a **valid proxy** and
#     US results are interpretable.
#   · If CN-hlc3 reproduces nothing, it is a **bad proxy** and the US test has **no explanatory power**.
# This **proxy calibration** is required for the US experiment to be meaningful.
#
# True VWAP is an **executable industry benchmark** that a VWAP algorithm can approximate.
# hlc3 is only a **statistical proxy and cannot be traded directly**. The US test therefore checks
# whether the **mechanism replicates**, not tradability.


def _robust_z(s: pd.Series) -> pd.Series:
    """Exactly match build_dataset.py: robust median/MAD standardization clipped to ±3."""
    med = s.median()
    mad = (s - med).abs().median()
    return ((s - med) / (mad * 1.4826 + EPS)).clip(-3, 3)


def read_raw_prices(market: str) -> pd.DataFrame:
    """Read 1313 raw parquets and return (date,instrument) → vwap_adj / adj_raw / oneword.

    **Do not filter anything here.** A previous draft filtered `vwap>0` before shifting, causing
    `groupby.shift(-1)` to jump across a deleted day and pair nonadjacent days, a silent error that
    does not produce NaN. Pass the row grid unchanged to the caller, align it to the dataset index,
    and only then shift.
    """
    cfg = MARKETS[market]
    files = sorted(glob.glob(os.path.join(cfg["raw"], "*.parquet")))
    print(f"[1/5] reading {len(files)} raw parquet files (market={market}) ...")
    base = ["Date", "Close", "Adj Close", "High", "Low", "Volume"]
    cols = base + (["code", "amount"] if market == "cn" else [])
    rows = []
    for i, f in enumerate(files):
        d = pd.read_parquet(f, columns=cols)
        # Match features.load_adjusted cleaning to keep the row grid aligned.
        d = d.dropna(subset=["Close", "Adj Close"]).drop_duplicates("Date")
        if d.empty:
            continue
        close = d["Close"].to_numpy(dtype=float)
        hi    = d["High"].to_numpy(dtype=float)
        lo    = d["Low"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            adj = np.where(close > 0, d["Adj Close"].to_numpy(dtype=float) / close, np.nan)
        # hlc3 (typical-price proxy): exactly match features.load_adjusted.
        #   mean of high*f, low*f, close*f(=AdjClose) = f*(H+L+C)/3
        hlc3_adj = adj * (hi + lo + close) / 3.0
        if market == "cn":
            vol = d["Volume"].to_numpy(dtype=float)
            amt = d["amount"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                vwap_raw = np.where(vol > 0, amt / vol, np.nan)   # Unadjusted average execution price.
            vwap_adj = vwap_raw * adj
            inst = d["code"].astype(str)
        else:
            vwap_adj = np.full(len(d), np.nan)      # US data have no turnover value, so true vwap is unavailable.
            inst = os.path.splitext(os.path.basename(f))[0]
        rows.append(pd.DataFrame({
            "date": pd.to_datetime(d["Date"]),
            "instrument": inst,
            "vwap_adj": vwap_adj,     # Adjusted true average execution price (A-shares only).
            "hlc3_adj": hlc3_adj,     # Adjusted typical-price proxy (both markets).
            "adj_raw": adj,           # Used to match the dataset's adjustFactor.
            "oneword": hi == lo,      # One-price limit: daily high=low, so execution is effectively impossible.
        }))
        if (i + 1) % 400 == 0:
            print(f"      {i + 1}/{len(files)}")
    out = pd.concat(rows, ignore_index=True)
    out["instrument"] = out["instrument"].astype(str)
    return out


def build(market: str, price: str, gate: str) -> pd.DataFrame:
    cfg = MARKETS[market]
    # ---- Base = dataset's own row grid and gate columns; **do not rederive**, avoiding drift. ----
    print(f"[2/5] reading dataset index ({cfg['ds']}) ...")
    cols = ["close_adj", "adjustFactor", "label_raw", "label"]
    if cfg["has_gate"]:
        cols += ["can_buy", "can_sell"]
    ds = pd.read_parquet(cfg["ds"], columns=cols).reset_index()
    ds["date"] = pd.to_datetime(ds["date"])
    ds["instrument"] = ds["instrument"].astype(str)
    print(f"      dataset {len(ds):,} rows, {ds['instrument'].nunique()} tickers")

    raw = read_raw_prices(market)
    print(f"[3/5] aligning to dataset index ({len(raw):,} raw rows) ...")
    m = ds.merge(raw, on=["date", "instrument"], how="left")
    cov = m["vwap_adj"].notna().mean()
    print(f"      vwap coverage {cov:.4%}"
          f" | adjustment-factor match max|adj_raw-adjustFactor| = "
          f"{(m['adj_raw'] - m['adjustFactor']).abs().max():.3e}")

    # ---- Perform every shift on the dataset row grid, matching build_dataset.add_label_cn. ----
    m = m.sort_values(["date", "instrument"]).reset_index(drop=True)
    g = m.groupby("instrument", sort=False)
    px = {"close": "close_adj", "vwap": "vwap_adj", "hlc3": "hlc3_adj"}[price]
    if m[px].notna().sum() == 0:
        raise SystemExit(f"[abort] market={market} has no {px}; US data have no turnover value and no true vwap, so use --price hlc3")
    rtn = g[px].shift(-2) / g[px].shift(-1) - 1.0

    if cfg["has_gate"]:
        can_buy_t1  = g["can_buy"].shift(-1)   # Can buy at T+1 (entry filter).
        can_sell_t2 = g["can_sell"].shift(-2)  # Can sell at T+2 (failed exit; see file header).
        if gate == "full":
            keep = can_buy_t1.eq(True) & can_sell_t2.eq(True)
        elif gate == "entry":
            keep = can_buy_t1.eq(True)
        elif gate == "none":
            keep = pd.Series(True, index=m.index)
        else:
            raise ValueError(gate)
        rtn = rtn.where(keep)
    else:
        # US equities have no price limits and build_dataset.add_label has no gate; do not add one here.
        can_buy_t1 = can_sell_t2 = pd.Series(np.nan, index=m.index)
        if gate != "none":
            raise SystemExit(f"[abort] market={market} has no gate columns; --gate must be none")
    if cfg["rtn_cap"] is not None:
        # With price limits, a normal T+1→T+2 return **cannot reach** 0.8; this safeguard catches
        # only suspension/reopen gaps and data errors.
        rtn = rtn.where(rtn.abs() <= cfg["rtn_cap"])
    m["label_new"] = rtn

    print("[4/5] cross-sectional robust z-score (same denominator rows as the original implementation) ...")
    m["label_z"] = m.groupby("date")["label_new"].transform(_robust_z)

    # ---- Price-limit diagnostic flags for decline-attribution decomposition in evaluate. ----
    if cfg["has_gate"]:
        m["lim_up_t1"] = can_buy_t1.eq(False)     # T+1 limit-up (cannot buy).
        m["lim_dn_t2"] = can_sell_t2.eq(False)    # T+2 limit-down (cannot sell → failed exit).
        m["lim_up_t2"] = g["can_buy"].shift(-2).eq(False)   # T+2 limit-up (**never removed by the gate**).
    else:
        for c in ("lim_up_t1", "lim_dn_t2", "lim_up_t2"):
            m[c] = False                           # US equities have no price-limit mechanism.
    m["oneword_t1"] = g["oneword"].shift(-1).eq(True)
    return m


def verify(m: pd.DataFrame) -> bool:
    """Require --price close --gate full to be elementwise identical to the existing dataset label."""
    print("\n" + "=" * 72)
    print("[verify] close × full must be elementwise identical to dataset label_raw / label")
    ok = True
    for new, old in [("label_new", "label_raw"), ("label_z", "label")]:
        a, b = m[new], m[old]
        same_nan = a.isna().eq(b.isna())
        both = a.notna() & b.notna()
        dmax = float((a[both] - b[both]).abs().max()) if both.any() else 0.0
        good = bool(same_nan.all()) and dmax == 0.0
        ok &= good
        print(f"  {new:>10} vs {old:<10} NaN pattern matches={bool(same_nan.all())}  "
              f"max|diff|={dmax:.3e}  n={int(both.sum()):,}  "
              f"{'OK' if good else '★MISMATCH★'}")
        if not same_nan.all():
            print(f"      rows with mismatched NaN patterns: {int((~same_nan).sum()):,}")
    print(f"[verify] {'passed — implementation exactly matches the original; later differences are uniquely attributable' if ok else 'failed — investigate here first'}")
    print("=" * 72)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["cn", "us"], default="cn")
    ap.add_argument("--price", choices=["close", "vwap", "hlc3"], default="vwap",
                    help="close=closing price / vwap=amount÷volume true average execution price (A-shares only) / "
                         "hlc3=(H+L+C)/3 typical-price proxy (both markets)")
    ap.add_argument("--gate", choices=["full", "entry", "none"], default="full")
    ap.add_argument("--verify", action="store_true",
                    help="run close×full and compare with the existing dataset label without saving")
    args = ap.parse_args()

    dflt_gate = "full" if MARKETS[args.market]["has_gate"] else "none"
    price, gate = ((("close", dflt_gate)) if args.verify else (args.price, args.gate))
    if not MARKETS[args.market]["has_gate"]:
        gate = "none"
    m = build(args.market, price, gate)

    if args.verify:
        raise SystemExit(0 if verify(m) else 1)

    out = m[["date", "instrument", "label_new", "label_z",
             "lim_up_t1", "lim_dn_t2", "lim_up_t2", "oneword_t1"]].copy()
    out = out.rename(columns={"label_new": "label_raw", "label_z": "label"})
    out = out[out["label"].notna()]
    path = OUT_TPL.format(market=args.market, price=price, gate=gate)
    print(f"[5/5] saving {path}")
    out.to_parquet(path, index=False)

    base = pd.read_parquet(MARKETS[args.market]["ds"], columns=["label_raw"]).dropna()
    print(f"\n[done] {path}  {len(out):,} rows"
          f" (comparison: existing built-in label {len(base):,} rows, difference {len(out) - len(base):+,})")
    print(f"  new label_raw      mean {out['label_raw'].mean():+.6f}  std {out['label_raw'].std():.6f}")
    print(f"  existing label_raw mean {base['label_raw'].mean():+.6f}  std {base['label_raw'].std():.6f}"
          f"   ← std change is **mechanical** (vwap tails compress under price limits); inspect before comparing bps across labels")
    print(f"  price-limit flag shares: T+1 limit-up {out['lim_up_t1'].mean():.3%} | "
          f"T+2 limit-down {out['lim_dn_t2'].mean():.3%} | T+2 limit-up {out['lim_up_t2'].mean():.3%} | "
          f"T+1 one-price limit {out['oneword_t1'].mean():.3%}")


if __name__ == "__main__":
    main()
