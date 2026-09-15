"""
build_dataset.py — assemble raw data into a model-ready Alpha158 dataset

Pipeline (each stage is an independent function; one stage's output is the next stage's input):
  1. build_panel      iterate over tickers → features.compute_ticker → assemble a MultiIndex(date,instrument) panel
  2. add_label        add the default label: T+1→T+2 return + cross-sectional robust z-score (same as features; replaceable)
  3. preprocess       apply cross-sectional robust z-score + clip + fillna to features (leave label unchanged)
  4. split_segments   split train/valid/test by time and write the 'segment' column
  5. save             save parquet

Usage:
  python alphaFactor/build_dataset.py --smoke         # Five tickers end to end + model smoke fit.
  python alphaFactor/build_dataset.py                 # Full run.
  python alphaFactor/build_dataset.py --limit 50      # First 50 tickers.

Depends only on numpy/pandas (the smoke model uses the already-installed sklearn).
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

EPS = 1e-12
RAW_DIR = "dataset/rawdata/raw"
OUT_PATH = "dataset/alphaFactor/dataset_alpha158.parquet"

# A-shares (CN) use a different raw directory/output name and the tradability-gated add_label_cn.
RAW_DIR_CN = "dataset/rawdata_cn/raw"
OUT_PATH_CN = "dataset/alphaFactor/dataset_alpha158_cn.parquet"
GEM_20PCT_DATE = pd.Timestamp("2020-08-24")  # ChiNext registration reform changed price limits from ±10 to ±20.

# Time-split boundaries, inclusive on the left and exclusive on the right.
TRAIN_END = "2023-01-01"
VALID_END = "2024-01-01"   # valid = [TRAIN_END, VALID_END), test = [VALID_END, ∞)


# ----------------------------------------------------------------------------
# 1. Assemble the panel
# ----------------------------------------------------------------------------
def build_panel(paths: list[str]) -> pd.DataFrame:
    """Compute Alpha158 for each ticker and assemble a long MultiIndex(date, instrument) table."""
    frames = []
    for p in paths:
        ticker = os.path.splitext(os.path.basename(p))[0]
        try:
            f = F.compute_ticker(p)
        except Exception as e:
            print(f"  [skip] {ticker}: {type(e).__name__} {e}")
            continue
        f.insert(0, "instrument", ticker)
        f = f.set_index("instrument", append=True)        # index = (date, instrument)
        frames.append(f)
    if not frames:
        raise RuntimeError("no ticker was computed successfully")
    panel = pd.concat(frames)
    panel.index = panel.index.set_names(["date", "instrument"])
    panel = panel.sort_index()
    return panel


# ----------------------------------------------------------------------------
# 2. Label (default: T+1→T+2 return + cross-sectional rank normalization; replaceable)
# ----------------------------------------------------------------------------
def add_label(panel: pd.DataFrame) -> pd.DataFrame:
    """
    label_raw = Ref(close,-2)/Ref(close,-1) - 1   (buy at T+1 close, sell at T+2 close; avoids look-ahead)
    label     = same-day cross-sectional robust z-score: (x - median) / (MAD*1.4826 + EPS), clipped to ±3

    Why use this definition (option C, replacing the original rank):
      - Preserve return magnitude: without ranking, "large gain/small gain" remains a large/small
        positive value, suitable for Pearson IC / MSE.
      - Robust standardization: median/MAD is not distorted by fat-tailed extremes such as earnings
        shocks, so an inflated standard deviation does not flatten ordinary stocks. It exactly matches
        feature preprocessing in _robust_z, keeping the project consistent and interpretable.
      - Clipping at ±3 is tunable: after robust standardization, ±3 means three robust standard
        deviations and has a scale. It can be relaxed to ±5 if values pile up, or removed for an IC loss.
    """
    g = panel.groupby(level="instrument")["close_adj"]
    panel["label_raw"] = g.shift(-2) / g.shift(-1) - 1.0
    panel["label"] = (
        panel.groupby(level="date")["label_raw"].transform(_robust_z)
    )
    return panel


# ----------------------------------------------------------------------------
# 2b. A-share label (same close_adj return as US + tradability gate; market differences live only here)
# ----------------------------------------------------------------------------
# Design (decision finalized on 2026-07-09):
#   · Use exactly the US return formula, close_adj(T+2)/close_adj(T+1)-1, to preserve cross-market
#     ceiling comparability. A-shares add only a tradability gate; the label recipe does not change.
#   · If buying is impossible (T+1 limit-up) or selling is impossible (T+2 limit-down), set the
#     entire label to NaN and remove it (option B, the selected policy).
#     The downloader already omits suspension bars, turning them into reopen gaps caught by |rtn|>0.8.
#   · Rolling to the next tradable day and using actual vwap exit prices were discussed but deliberately
#     not implemented; they can be explained conceptually without implementation.
# Price limits use closing pctChg near the threshold as a proxy: a close pinned to the limit represents
# a genuinely impossible buy/sell. Rules are board-, ST-, and date-aware: Main Board ±10;
# STAR 688 and ChiNext 300 after 2020-08-24 ±20; ST ±5.

def _cn_tradable_flags(raw_path: str) -> pd.DataFrame:
    """Read one raw A-share parquet, compute daily can_buy/can_sell, and return a (date,instrument)-indexed table."""
    df = pd.read_parquet(raw_path, columns=["Date", "pctChg", "isST", "code"])
    code = str(df["code"].iloc[0])
    date = pd.to_datetime(df["Date"])
    num = code.split(".")[1]                       # sz.300750 → 300750
    is_star = num.startswith("688")                # STAR Market ±20.
    is_gem = num.startswith(("300", "301"))         # ChiNext: ±20 after 2020-08-24, ±10 before.
    gem20 = is_gem & (date >= GEM_20PCT_DATE).to_numpy()
    base = np.where(is_star | gem20, 19.8, 9.8)     # Main Board / SME Board ±10.
    thr = np.where(df["isST"].to_numpy() == 1, 4.8, base)  # ST overrides the threshold to ±5.
    pct = df["pctChg"].to_numpy()
    out = pd.DataFrame({
        "date": date.values,
        "instrument": code,
        "can_buy": pct < thr,                       # Not limit-up → can buy.
        "can_sell": pct > -thr,                     # Not limit-down → can sell (NaN pctChg → False conservatively).
    }).set_index(["date", "instrument"])
    return out


def attach_cn_tradable(panel: pd.DataFrame, paths: list[str]) -> pd.DataFrame:
    """Concatenate per-ticker can_buy/can_sell flags and join them to the exactly aligned panel index."""
    flags = pd.concat([_cn_tradable_flags(p) for p in paths])
    panel = panel.join(flags, how="left")
    return panel


def add_label_cn(panel: pd.DataFrame) -> pd.DataFrame:
    """A-share label: close_adj T+1→T+2 return, tradability gate, |rtn|>0.8 safeguard, then cross-sectional robust z."""
    g = panel.groupby(level="instrument")
    rtn = g["close_adj"].shift(-2) / g["close_adj"].shift(-1) - 1.0
    can_buy_t1 = g["can_buy"].shift(-1)             # Can buy at T+1.
    can_sell_t2 = g["can_sell"].shift(-2)           # Can sell at T+2.
    gate = can_buy_t1.eq(True) & can_sell_t2.eq(True)  # NaN → False; naturally removes tails/gaps.
    rtn = rtn.where(gate)                            # Impossible buy/sell → NaN.
    rtn = rtn.where(rtn.abs() <= 0.8)               # Extreme return (reopen gap/data error) → NaN.
    panel["label_raw"] = rtn
    panel["label"] = panel.groupby(level="date")["label_raw"].transform(_robust_z)
    return panel


# ----------------------------------------------------------------------------
# 3. Feature preprocessing (cross-sectional robust z-score → clip → fillna)
# ----------------------------------------------------------------------------
def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    return ((s - med) / (mad * 1.4826 + EPS)).clip(-3, 3)


def preprocess(panel: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Apply a cross-sectional robust z-score and ±3 clip each trading day; fill remaining NaNs with zero."""
    panel[feat_cols] = (
        panel.groupby(level="date")[feat_cols].transform(_robust_z)
    )
    panel[feat_cols] = panel[feat_cols].fillna(0.0)
    return panel


# ----------------------------------------------------------------------------
# 4. Time split
# ----------------------------------------------------------------------------
def split_segments(panel: pd.DataFrame) -> pd.DataFrame:
    dates = panel.index.get_level_values("date")
    seg = np.where(dates < pd.Timestamp(TRAIN_END), "train",
          np.where(dates < pd.Timestamp(VALID_END), "valid", "test"))
    panel["segment"] = seg
    return panel


# ----------------------------------------------------------------------------
# 5. Save
# ----------------------------------------------------------------------------
def save(panel: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    panel.to_parquet(out_path)
    print(f"  saved {out_path}  shape={panel.shape}")


# ----------------------------------------------------------------------------
# End-to-end orchestration
# ----------------------------------------------------------------------------
def run(paths: list[str], out_path: str, do_smoke: bool, market: str = "us") -> pd.DataFrame:
    feat_cols = F.feature_names()
    print(f"[1/5] build_panel  ({len(paths)} tickers, market={market}) ...")
    panel = build_panel(paths)
    print(f"      panel {panel.shape}, dates {panel.index.get_level_values('date').min().date()}"
          f" → {panel.index.get_level_values('date').max().date()}")

    print("[2/5] add_label ...")
    if market == "cn":
        panel = attach_cn_tradable(panel, paths)   # Attach can_buy/can_sell.
        panel = add_label_cn(panel)                # Return + tradability gate.
    else:
        panel = add_label(panel)
    print("[3/5] preprocess (cross-sectional robust z) ...")
    panel = preprocess(panel, feat_cols)
    print("[4/5] split_segments ...")
    panel = split_segments(panel)
    seg_counts = panel["segment"].value_counts().to_dict()
    print(f"      segment counts {seg_counts}")

    print("[5/5] save ...")
    save(panel, out_path)

    if do_smoke:
        smoke_fit(panel, feat_cols)
    return panel


# Representative smoke-test tickers for each market (long history and good liquidity).
SMOKE_PREFER = {
    "us": ["AAPL", "MSFT", "AMZN", "JPM", "XOM"],
    "cn": ["sh.600000", "sh.600519", "sz.000001", "sz.300750", "sh.601318"],
}


def smoke_fit(panel: pd.DataFrame, feat_cols: list[str]) -> None:
    """Verify that the dataset feeds a model by fitting sklearn histogram GBDT on train and scoring validation."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from scipy.stats import spearmanr

    tr = panel[(panel["segment"] == "train") & panel["label"].notna()]
    va = panel[(panel["segment"] == "valid") & panel["label"].notna()]
    print(f"\n[smoke] model integration check: train={tr.shape[0]} valid={va.shape[0]}")
    if tr.shape[0] < 50 or va.shape[0] < 10:
        print("[smoke] too few samples (normal for a small ticker smoke set); checking only X/y shape alignment:")
        print(f"        X_train={tr[feat_cols].shape}  y_train={tr['label'].shape}  no NaN="
              f"{not tr[feat_cols].isna().any().any()}")
        return
    Xtr, ytr = tr[feat_cols].to_numpy(), tr["label"].to_numpy()
    Xva, yva = va[feat_cols].to_numpy(), va["label"].to_numpy()
    model = HistGradientBoostingRegressor(max_iter=50, max_depth=4, learning_rate=0.05)
    model.fit(Xtr, ytr)
    pred = model.predict(Xva)
    ic, _ = spearmanr(pred, yva)
    print(f"[smoke] model fit/predict succeeded; valid Rank-IC={ic:.4f} (smoke data; value is not meaningful)")
    print("[smoke] → full feature→label→preprocess→split→model pipeline is connected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "cn"], default="us",
                    help="us=Russell1000 (close_adj label) / cn=CSI 800 (with tradability gate)")
    ap.add_argument("--raw-dir", default=None, help="select directory from --market by default")
    ap.add_argument("--out", default=None, help="select output name from --market by default")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N tickers")
    ap.add_argument("--smoke", action="store_true", help="run five tickers end to end + model smoke fit")
    args = ap.parse_args()

    raw_dir = args.raw_dir or (RAW_DIR_CN if args.market == "cn" else RAW_DIR)
    out_path = args.out or (OUT_PATH_CN if args.market == "cn" else OUT_PATH)

    paths = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    if not paths:
        raise SystemExit(f"{raw_dir} has no parquet files")

    if args.smoke:
        prefer = SMOKE_PREFER[args.market]
        chosen = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in prefer]
        paths = chosen or paths[:5]
        out = out_path.replace(".parquet", "_smoke.parquet")
    else:
        if args.limit:
            paths = paths[:args.limit]
        out = out_path

    run(paths, out, do_smoke=args.smoke, market=args.market)


if __name__ == "__main__":
    main()
