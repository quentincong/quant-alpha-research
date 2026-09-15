"""
features.py — price adjustment + Alpha158 feature computation

Convert one stock's raw daily parquet (OHLCV + Adj Close) into 158 Alpha158 feature columns.
- Price adjustment: factor = AdjClose/Close, multiplied back into OHLC.
- vwap: proxy with (high+low+close)/3 because Yahoo provides no turnover value for true vwap.
- Alpha158 = 9 KBAR + 4 price ratios + 29 rolling operators × 5 windows [5,10,20,30,60].

Each formula was checked against Qlib's qlib/contrib/data/loader.py. All rolling/shift operations
are performed **within one stock** and never across stocks, which would splice windows together.

Depends only on numpy / pandas.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WINDOWS = [5, 10, 20, 30, 60]
EPS = 1e-12

# Complete Alpha158 feature-column list, used to verify that all 158 are present.
KBAR_NAMES = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
PRICE_NAMES = ["OPEN0", "HIGH0", "LOW0", "VWAP0"]
ROLL_OPS = ["ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD",
            "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN", "CNTD",
            "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD"]


def feature_names() -> list[str]:
    """Return the 158 feature-column names in fixed order."""
    names = list(KBAR_NAMES) + list(PRICE_NAMES)
    for op in ROLL_OPS:
        for d in WINDOWS:
            names.append(f"{op}{d}")
    return names


# ----------------------------------------------------------------------------
# 1. Price adjustment
# ----------------------------------------------------------------------------
def load_adjusted(path: str) -> pd.DataFrame:
    """
    Read one parquet, apply post-adjustment, and return adjusted OHLCV+vwap in ascending Date order.
    Also retain raw_close and adjustFactor as accompanying metadata.
    """
    df = pd.read_parquet(path)
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adjclose", "Volume": "volume",
    })
    df = df.dropna(subset=["close", "adjclose"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")

    factor = df["adjclose"] / df["close"]          # Post-adjustment factor.
    raw_close = df["close"].copy()
    out = pd.DataFrame(index=df.index)
    out["open"] = df["open"] * factor
    out["high"] = df["high"] * factor
    out["low"] = df["low"] * factor
    out["close"] = df["adjclose"]                  # = close * factor
    out["volume"] = df["volume"].astype("float64")
    out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3.0   # Proxy.
    out["adjustFactor"] = factor                   # Metadata.
    out["raw_close"] = raw_close                   # Metadata (unadjusted close).
    return out


# ----------------------------------------------------------------------------
# 2. Vectorized complex operator: rolling linear regression of price on time index 0..d-1
# ----------------------------------------------------------------------------
def _rolling_regress(y: pd.Series, d: int):
    """
    Run a rolling linear regression of the past d values y on x=[0,1,...,d-1].
    Return (slope, rsquare, resid_last) as ndarrays aligned with y and NaN for the first d-1 rows.
      slope      = regression slope
      rsquare    = regression R^2
      resid_last = residual at the final point (actual - fitted)
    Compute in one pass with a sliding-window matrix and analytical solution, avoiding slow rolling.apply.
    """
    arr = y.to_numpy(dtype="float64")
    n = arr.shape[0]
    slope = np.full(n, np.nan)
    rsq = np.full(n, np.nan)
    resi = np.full(n, np.nan)
    if n < d:
        return slope, rsq, resi

    # Sliding-window matrix W: shape (n-d+1, d); each row is a length-d window in time order.
    W = np.lib.stride_tricks.sliding_window_view(arr, d)
    x = np.arange(d, dtype="float64")
    Sx = x.sum()
    Sxx = (x * x).sum()
    denom = d * Sxx - Sx * Sx                       # Constant.

    Sy = W.sum(axis=1)
    Sxy = W @ x
    Syy = (W * W).sum(axis=1)

    b = (d * Sxy - Sx * Sy) / denom                 # slope
    a = (Sy - b * Sx) / d                            # intercept
    pred_last = a + b * (d - 1)
    resid_last = W[:, -1] - pred_last

    # R^2 = (cov(x,y))^2 / (var(x)*var(y))
    cov_xy = Sxy / d - (Sx / d) * (Sy / d)
    var_x = Sxx / d - (Sx / d) ** 2
    var_y = Syy / d - (Sy / d) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = (cov_xy * cov_xy) / (var_x * var_y)
    r2 = np.where(var_y <= EPS, 0.0, r2)            # Define R^2 as zero for flat prices.

    slope[d - 1:] = b
    rsq[d - 1:] = r2
    resi[d - 1:] = resid_last
    return slope, rsq, resi


def _rolling_idxmax(s: pd.Series, d: int) -> pd.Series:
    """Position of the maximum within the window (0=oldest ... d-1=newest)."""
    return s.rolling(d).apply(np.argmax, raw=True)


def _rolling_idxmin(s: pd.Series, d: int) -> pd.Series:
    return s.rolling(d).apply(np.argmin, raw=True)


# ----------------------------------------------------------------------------
# 3. Alpha158
# ----------------------------------------------------------------------------
def alpha158(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the adjusted DataFrame from load_adjusted and return 158 features with the same index.
    """
    o, h, l, c, v, vwap = (df["open"], df["high"], df["low"], df["close"],
                           df["volume"], df["vwap"])
    feat = {}

    # --- (a) KBAR shapes (9) ---
    hl = (h - l) + EPS
    feat["KMID"] = (c - o) / o
    feat["KLEN"] = (h - l) / o
    feat["KMID2"] = (c - o) / hl
    greater_oc = np.maximum(o, c)
    less_oc = np.minimum(o, c)
    feat["KUP"] = (h - greater_oc) / o
    feat["KUP2"] = (h - greater_oc) / hl
    feat["KLOW"] = (less_oc - l) / o
    feat["KLOW2"] = (less_oc - l) / hl
    feat["KSFT"] = (2 * c - h - l) / o
    feat["KSFT2"] = (2 * c - h - l) / hl

    # --- (b) Price ratios (4, window=0) ---
    feat["OPEN0"] = o / c
    feat["HIGH0"] = h / c
    feat["LOW0"] = l / c
    feat["VWAP0"] = vwap / c

    # Preparation: day-to-day changes.
    c_ref1 = c.shift(1)
    v_ref1 = v.shift(1)
    dc = c - c_ref1                 # close - Ref(close,1)
    dv = v - v_ref1
    abs_dc = dc.abs()
    abs_dv = dv.abs()
    up_c = dc.clip(lower=0)         # Greater(close-Ref(close,1), 0)
    dn_c = (-dc).clip(lower=0)      # Greater(Ref(close,1)-close, 0)
    up_v = dv.clip(lower=0)
    dn_v = (-dv).clip(lower=0)
    log_v = np.log(v + 1)
    ret = c / c_ref1                # close/Ref(close,1)
    log_vr = np.log(v / v_ref1 + 1)
    wv = (c / c_ref1 - 1).abs() * v   # |close/Ref(close,1)-1| * volume (used by WVMA).

    # --- (c) Rolling operators (29 × 5 windows) ---
    for d in WINDOWS:
        feat[f"ROC{d}"] = c.shift(d) / c
        feat[f"MA{d}"] = c.rolling(d).mean() / c
        feat[f"STD{d}"] = c.rolling(d).std() / c

        slope, rsq, resi = _rolling_regress(c, d)
        idx = c.index
        feat[f"BETA{d}"] = pd.Series(slope, index=idx) / c
        feat[f"RSQR{d}"] = pd.Series(rsq, index=idx)
        feat[f"RESI{d}"] = pd.Series(resi, index=idx) / c

        feat[f"MAX{d}"] = h.rolling(d).max() / c
        feat[f"MIN{d}"] = l.rolling(d).min() / c
        feat[f"QTLU{d}"] = c.rolling(d).quantile(0.8) / c
        feat[f"QTLD{d}"] = c.rolling(d).quantile(0.2) / c
        feat[f"RANK{d}"] = c.rolling(d).rank(pct=True)

        low_min = l.rolling(d).min()
        high_max = h.rolling(d).max()
        feat[f"RSV{d}"] = (c - low_min) / (high_max - low_min + EPS)

        feat[f"IMAX{d}"] = _rolling_idxmax(h, d) / d
        feat[f"IMIN{d}"] = _rolling_idxmin(l, d) / d
        feat[f"IMXD{d}"] = (_rolling_idxmax(h, d) - _rolling_idxmin(l, d)) / d

        feat[f"CORR{d}"] = c.rolling(d).corr(log_v)
        feat[f"CORD{d}"] = ret.rolling(d).corr(log_vr)

        cntp = (dc > 0).rolling(d).mean()
        cntn = (dc < 0).rolling(d).mean()
        feat[f"CNTP{d}"] = cntp
        feat[f"CNTN{d}"] = cntn
        feat[f"CNTD{d}"] = cntp - cntn

        sum_absdc = abs_dc.rolling(d).sum() + EPS
        sump = up_c.rolling(d).sum()
        sumn = dn_c.rolling(d).sum()
        feat[f"SUMP{d}"] = sump / sum_absdc
        feat[f"SUMN{d}"] = sumn / sum_absdc
        feat[f"SUMD{d}"] = (sump - sumn) / sum_absdc

        feat[f"VMA{d}"] = v.rolling(d).mean() / (v + EPS)
        feat[f"VSTD{d}"] = v.rolling(d).std() / (v + EPS)
        feat[f"WVMA{d}"] = wv.rolling(d).std() / (wv.rolling(d).mean() + EPS)

        sum_absdv = abs_dv.rolling(d).sum() + EPS
        vsump = up_v.rolling(d).sum()
        vsumn = dn_v.rolling(d).sum()
        feat[f"VSUMP{d}"] = vsump / sum_absdv
        feat[f"VSUMN{d}"] = vsumn / sum_absdv
        feat[f"VSUMD{d}"] = (vsump - vsumn) / sum_absdv

    out = pd.DataFrame(feat, index=df.index)
    # Fix column order.
    out = out[feature_names()]
    return out


def compute_ticker(path: str) -> pd.DataFrame:
    """
    End to end: one parquet → price adjustment → Alpha158.
    Return a DataFrame containing 158 feature columns plus raw_close / adjustFactor metadata.
    """
    adj = load_adjusted(path)
    feat = alpha158(adj)
    feat["close_adj"] = adj["close"]          # Adjusted close used by the label.
    feat["raw_close"] = adj["raw_close"]       # Unadjusted close metadata.
    feat["adjustFactor"] = adj["adjustFactor"]
    return feat


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Alpha158 features for one parquet file.")
    parser.add_argument("path", nargs="?", default="rawdata/raw/AAPL.parquet")
    args = parser.parse_args()
    p = args.path
    f = compute_ticker(p)
    names = feature_names()
    print(f"path={p}")
    print(f"shape={f.shape}  expected feature columns=158, actual={len(names)}")
    miss = [n for n in names if n not in f.columns]
    print(f"missing columns: {miss if miss else 'none'}")
    allnan = [n for n in names if f[n].isna().all()]
    print(f"all-NaN columns: {allnan if allnan else 'none'}")
    print(f.tail(2)[["KMID", "MA5", "STD20", "BETA10", "RSQR10", "RANK5",
                     "IMAX5", "CORR60", "VSUMD60"]].to_string())
