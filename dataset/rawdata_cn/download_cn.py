#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_cn.py — A-share daily raw-market-data downloader (baostock version), matching US rawdata/download.py.

DOWNLOAD ONLY. Do not compute factors, standardize, or create labels. This script only saves
baostock's raw daily bars with a schema identical to the US data, allowing the frozen upstream
features.py (Alpha158) to be reused unchanged.

Why baostock: it is free, needs no token, covers 2018–2025, and provides raw
tradestatus / isST / pctChg / preclose / amount fields for upstream handling of price limits,
suspensions, and ST status.

Universe (CSI 800 proxy):
    baostock does not provide the CSI 800 directly. Take historical CSI 300 and CSI 500
    constituents quarterly and form their union, yielding every stock that entered either index
    during these years. This reduces survivorship bias and is more robust than the static current
    Russell 1000 snapshot for US equities. Membership changes over time, but the downloader pulls
    each stock's full history. Write the result to universe_full_cn.txt, one code per line
    (sh.600000 / sz.000001).

Per-ticker storage (merge two queries → the same seven US columns + A-share metadata):
    Date, Open, High, Low, Close, Adj Close, Volume        ← the only seven columns features.py reads
    code, amount, preclose, turn, tradestatus, pctChg, isST  ← used upstream to determine tradability
  · Close      = raw unadjusted close (adjustflag=3)
  · Adj Close  = post-adjusted close (adjustflag=1); features.py multiplies OHLC by factor=AdjClose/Close
  · Volume     = raw unadjusted share volume, required by price-volume features
  · Drop suspension rows (tradestatus != 1) here because they have no real execution price and
    would contaminate factors. An impossible trade during suspension becomes a reopen-gap bar,
    caught upstream by the limit-up gate + |rtn|>0.8 safeguard.

Operational safeguards (matching the US version, but simpler because baostock uses one session
and has no IP blocking):
    · Resume from manifest.csv: skip ok; retry fail/empty; --force restarts everything.
    · Atomic writes with temp + os.replace leave no partial file if interrupted.
    · Wrap each ticker in try so one bad ticker does not stop the batch.
    · Retry transient network failures --retries times per code.

Usage:
    python download_cn.py                      # Build the universe, then download all data to ./raw.
    python download_cn.py --limit 20           # Download only the first 20 tickers for debugging.
    python download_cn.py --codes sh.600000,sz.300750   # Download specified tickers.
    python download_cn.py --universe-only      # Generate universe_full_cn.txt without downloading.
    python download_cn.py --force              # Ignore the manifest and redownload everything.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import akshare as ak
except ImportError:
    sys.exit("akshare is not installed. Run pip install -r requirements.txt from the repository root")

# Use baostock only to build the universe from CSI 300/500 constituents;
# market-data downloads use akshare, which is ~180x faster.
import baostock as bs

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
UNIVERSE_FILE = HERE / "universe_full_cn.txt"
MANIFEST = HERE / "manifest_cn.csv"

START = "2018-01-01"
END = "2025-12-31"

# Raw baostock market-data fields; values are strings and are converted to float before saving.
KFIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"


# --------------------------------------------------------------------------- #
# baostock session (log in once, log out once)
# --------------------------------------------------------------------------- #
def _login():
    lg = bs.login()
    if lg.error_code != "0":
        sys.exit(f"baostock login failed: {lg.error_code} {lg.error_msg}")


def _rs_to_df(rs) -> pd.DataFrame:
    """Consume baostock ResultData into a DataFrame."""
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


# --------------------------------------------------------------------------- #
# 1. Universe = quarterly union of historical CSI 300 and CSI 500 constituents
# --------------------------------------------------------------------------- #
def _quarter_ends(start: str, end: str) -> list[str]:
    """Generate quarter-end date strings within [start, end]."""
    q = pd.date_range(start=start, end=end, freq="QE")  # quarter-end
    return [d.strftime("%Y-%m-%d") for d in q]


def build_universe(start: str, end: str) -> list[str]:
    """Union of hs300 and zz500 constituent codes at all quarter ends."""
    codes: set[str] = set()
    for date in _quarter_ends(start, end):
        for fn, name in [(bs.query_hs300_stocks, "hs300"),
                         (bs.query_zz500_stocks, "zz500")]:
            df = _rs_to_df(fn(date))
            if not df.empty and "code" in df.columns:
                codes.update(df["code"].tolist())
        print(f"  [universe] {date}: cumulative {len(codes)} tickers")
    out = sorted(codes)
    UNIVERSE_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  wrote {UNIVERSE_FILE}  ({len(out)} tickers)")
    return out


def read_universe() -> list[str]:
    out = []
    for line in UNIVERSE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


# --------------------------------------------------------------------------- #
# 2. Per-ticker download: merge raw (unadjusted) and adj (post-adjusted) queries
# --------------------------------------------------------------------------- #
def _to_symbol(code: str) -> str:
    """Convert sh.600000 / sz.000001 to akshare's six-digit symbol 600000 / 000001."""
    return code.split(".")[-1]


def _fetch_akshare(code: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Merge two akshare calls (unadjusted + post-adjusted) into the US schema; return None if empty.
    It is fast (~0.2s/ticker) but depends on eastmoney over HTTP and may fail under blocking,
    rate limits, or proxy instability. akshare automatically excludes suspended days with no bar,
    so the panel naturally follows trading days, matching the baostock version.
      · Close     = unadjusted close; Adj Close = post-adjusted close
      · pctChg    = unadjusted percentage change, needed to identify price limits
      · isST      = this akshare endpoint has no daily ST flag, so approximate as 0
                    (ST names are rare in the CSI 800)
    """
    sym = _to_symbol(code)
    s, e = start.replace("-", ""), end.replace("-", "")
    # (a) Unadjusted: raw OHLCV + percentage change + turnover value/rate.
    raw = ak.stock_zh_a_hist(symbol=sym, period="daily",
                             start_date=s, end_date=e, adjust="")
    if raw is None or raw.empty:
        return None
    # (b) Post-adjusted: only the adjusted close is needed.
    hfq = ak.stock_zh_a_hist(symbol=sym, period="daily",
                             start_date=s, end_date=e, adjust="hfq")
    if hfq is None or hfq.empty:
        return None
    hfq = hfq[["日期", "收盘"]].rename(columns={"收盘": "adjclose"})

    df = raw.merge(hfq, on="日期", how="inner")
    if df.empty:
        return None

    out = pd.DataFrame({
        "Date": pd.to_datetime(df["日期"]),
        "Open": df["开盘"].astype(float),
        "High": df["最高"].astype(float),
        "Low": df["最低"].astype(float),
        "Close": df["收盘"].astype(float),          # Unadjusted.
        "Adj Close": df["adjclose"].astype(float),   # Post-adjusted.
        "Volume": df["成交量"].astype(float),
        "code": code,
        "amount": df["成交额"].astype(float),
        "turn": df["换手率"].astype(float),
        "pctChg": df["涨跌幅"].astype(float),         # Unadjusted percentage change → identify price limits.
        "isST": 0,                                    # Approximation; see docstring.
    })
    out = out.dropna(subset=["Close", "Adj Close"])
    out = out.sort_values("Date").reset_index(drop=True)
    return out if not out.empty else None


def _fetch_sina(code: str, start: str, end: str) -> pd.DataFrame | None:
    """
    akshare stock_zh_a_daily (**Sina source**): bypass unreliable or blocked eastmoney at similar speed.
    Sina provides date/open/high/low/close/volume/amount/outstanding_share/turnover but **no percentage
    change**, so compute pctChg from the unadjusted close to identify price limits. Merge two calls
    (unadjusted + post-adjusted). Symbol format is sh600000 without a dot. Sina also omits suspension bars.
    """
    sym = code.replace(".", "")           # sh.600000 → sh600000
    s, e = start.replace("-", ""), end.replace("-", "")
    raw = ak.stock_zh_a_daily(symbol=sym, start_date=s, end_date=e, adjust="")
    if raw is None or raw.empty:
        return None
    hfq = ak.stock_zh_a_daily(symbol=sym, start_date=s, end_date=e, adjust="hfq")
    if hfq is None or hfq.empty:
        return None
    hfq = hfq[["date", "close"]].rename(columns={"close": "adjclose"})

    df = raw.merge(hfq, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if df.empty:
        return None
    pct = df["close"].astype(float).pct_change() * 100.0   # Computed unadjusted percentage change → identify price limits.

    out = pd.DataFrame({
        "Date": pd.to_datetime(df["date"]),
        "Open": df["open"].astype(float),
        "High": df["high"].astype(float),
        "Low": df["low"].astype(float),
        "Close": df["close"].astype(float),          # Unadjusted.
        "Adj Close": df["adjclose"].astype(float),    # Post-adjusted.
        "Volume": df["volume"].astype(float),
        "code": code,
        "amount": df["amount"].astype(float),
        "turn": df["turnover"].astype(float),
        "pctChg": pct.values,
        "isST": 0,                                    # This Sina endpoint has no ST flag; approximate as 0.
    })
    out = out.dropna(subset=["Close", "Adj Close"])
    return out.reset_index(drop=True) if not out.empty else None


def _fetch_baostock(code: str, start: str, end: str) -> pd.DataFrame | None:
    """
    baostock version: merge two queries (unadjusted adjustflag=3 + post-adjusted adjustflag=1).
    Slow (~35s/ticker) but stable because it uses baostock's own servers rather than eastmoney.
    The caller must already have run bs.login(). Output columns exactly match _fetch_akshare,
    so downstream build_dataset sees no difference.
    """
    raw = _rs_to_df(bs.query_history_k_data_plus(
        code, KFIELDS, start_date=start, end_date=end, frequency="d", adjustflag="3"))
    if raw.empty:
        return None
    adj = _rs_to_df(bs.query_history_k_data_plus(
        code, "date,close", start_date=start, end_date=end, frequency="d", adjustflag="1"))
    if adj.empty:
        return None
    adj = adj.rename(columns={"close": "adjclose"})
    df = raw.merge(adj, on="date", how="inner")

    for c in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg", "adjclose"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce")
    df["isST"] = pd.to_numeric(df["isST"], errors="coerce").fillna(0).astype(int)
    df = df[df["tradestatus"] == 1].dropna(subset=["close", "adjclose"])   # Drop suspension rows.
    if df.empty:
        return None
    out = pd.DataFrame({
        "Date": pd.to_datetime(df["date"]),
        "Open": df["open"], "High": df["high"], "Low": df["low"],
        "Close": df["close"], "Adj Close": df["adjclose"], "Volume": df["volume"],
        "code": code, "amount": df["amount"], "turn": df["turn"],
        "pctChg": df["pctChg"], "isST": df["isST"],
    })
    return out.sort_values("Date").reset_index(drop=True)


def fetch_one(code: str, start: str, end: str, source: str = "sina") -> pd.DataFrame | None:
    """Dispatch by source with an identical output schema:
       sina (fast, Sina) / akshare (fast, eastmoney, may be blocked) / baostock (slow, stable)."""
    if source == "baostock":
        return _fetch_baostock(code, start, end)
    if source == "akshare":
        return _fetch_akshare(code, start, end)
    return _fetch_sina(code, start, end)


# --------------------------------------------------------------------------- #
# 3. Save + manifest (atomic writes and resumability)
# --------------------------------------------------------------------------- #
def write_parquet_atomic(df: pd.DataFrame, fpath: Path) -> None:
    tmp = fpath.with_suffix(fpath.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, fpath)


def load_manifest() -> dict[str, list]:
    if not MANIFEST.exists():
        return {}
    try:
        m = pd.read_csv(MANIFEST)
        return {str(r["code"]): [str(r["code"]), str(r["status"]),
                                 r.get("rows"), r.get("last_date")]
                for _, r in m.iterrows()}
    except Exception:
        return {}


def save_manifest(rows: dict[str, list]) -> None:
    df = pd.DataFrame(rows.values(),
                      columns=["code", "status", "rows", "last_date"])
    tmp = MANIFEST.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, MANIFEST)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="A-share daily raw-market-data downloader (akshare/baostock sources)")
    ap.add_argument("--source", choices=["sina", "akshare", "baostock"], default="sina",
                    help="sina=fast (Sina, avoids eastmoney blocks) / akshare=fast (eastmoney, may be blocked) / "
                         "baostock=slow but stable (~35s/ticker, own servers)")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--codes", help="comma-separated codes; skip the universe and download only these")
    ap.add_argument("--limit", type=int, default=None, help="download only the first N tickers for debugging")
    ap.add_argument("--retries", type=int, default=3, help="network retry count per ticker")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between tickers")
    ap.add_argument("--force", action="store_true", help="ignore the manifest and redownload everything")
    ap.add_argument("--universe-only", action="store_true",
                    help="generate universe_full_cn.txt without downloading")
    ap.add_argument("--proxy", default=None,
                    help="explicit proxy URL (for example http://HOST:PORT); by default inherit the system proxy settings")
    args = ap.parse_args()

    # sina/eastmoney are domestic HTTP sources; routing them through a system HTTP proxy can raise
    # ProxyError, so NO_PROXY is set for these domains by default. baostock uses its own socket and
    # ignores HTTP proxies. --proxy overrides.
    _NOPROXY = {
        "sina": "sina.com.cn,sinajs.cn",
        "akshare": "eastmoney.com,push2his.eastmoney.com",
    }
    if args.proxy:
        os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["http_proxy"] = os.environ["https_proxy"] = args.proxy
    elif args.source in _NOPROXY:
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = _NOPROXY[args.source]

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Universe: log in to baostock only when rebuilding constituents; market-data downloads do not need it. ----
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
        print(f"[universe] specified {len(codes)} tickers; skipping constituent query")
    elif not UNIVERSE_FILE.exists():
        print("[universe] building quarterly union of CSI 300 ∪ CSI 500 (baostock) ...")
        _login()
        try:
            codes = build_universe(args.start, args.end)
        finally:
            bs.logout()
    else:
        codes = read_universe()
        print(f"[universe] reusing {UNIVERSE_FILE}  ({len(codes)} tickers)")

    if args.universe_only:
        print("[done] --universe-only; exiting")
        return 0

    if args.limit:
        codes = codes[:args.limit]

    # ---- Resume ----
    rows = load_manifest()
    prior = {c: row[1] for c, row in rows.items()}
    todo = [c for c in codes if args.force or prior.get(c) != "ok"]
    skipped = len(codes) - len(todo)
    print(f"[plan] {len(codes)} total, skipping {skipped} completed, downloading {len(todo)} now, source={args.source}")

    # The baostock source keeps one login session for the full download; akshare uses HTTP and needs no login.
    if args.source == "baostock":
        _login()

    t0 = time.time()
    ok = fail = 0
    for i, code in enumerate(todo, 1):
        fpath = RAW_DIR / f"{code}.parquet"
        df = None
        for attempt in range(1, args.retries + 1):
            try:
                df = fetch_one(code, args.start, args.end, args.source)
                break
            except Exception as exc:
                print(f"  [{i}/{len(todo)}] {code} attempt {attempt} failed: {exc}")
                time.sleep(1.0 * attempt)
        if df is None or df.empty:
            rows[code] = [code, "fail", 0, None]
            fail += 1
        else:
            try:
                write_parquet_atomic(df, fpath)
                last = str(pd.to_datetime(df["Date"].iloc[-1]).date())
                rows[code] = [code, "ok", len(df), last]
                ok += 1
            except Exception as exc:
                print(f"  [{i}/{len(todo)}] {code} save failed: {exc}")
                rows[code] = [code, "fail", 0, None]
                fail += 1

        if i % 50 == 0 or i == len(todo):
            save_manifest(rows)
            el = time.time() - t0
            print(f"  [{i}/{len(todo)}] ok={ok} fail={fail} elapsed {el:.0f}s")
        if args.sleep:
            time.sleep(args.sleep)

    if args.source == "baostock":
        bs.logout()

    save_manifest(rows)
    print(f"[done] ok={ok} fail={fail} elapsed {time.time()-t0:.0f}s → {MANIFEST}")
    if fail:
        bad = [c for c, r in rows.items() if r[1] == "fail"]
        print(f"  {len(bad)} tickers failed; rerunning this script retries them automatically: {bad[:20]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
