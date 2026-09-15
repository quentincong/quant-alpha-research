#!/usr/bin/env python3
"""
download.py — Raw daily OHLCV downloader for the quant dataset (yfinance).

DOWNLOAD ONLY. No factor computation, no winsorize/zscore, no merging.
The job of this script is to pull raw daily bars from Yahoo Finance and
park them on disk in a resumable way for an unattended remote run.

Universe : pass a one-symbol-per-line file with --tickers-file. The checked-in
           dataset/rawdata/universe_full.txt is ordered with larger names first.
           No live web dependency is required for the universe.
Window   : 2018-01-01 .. 2025-12-31 inclusive (full years 2018-2025).
Priority : if a ticker's full-window pull fails, salvage the most recent
           window so newer data still lands ("prioritize complete recent data").

What it stores (per ticker, auto_adjust=False -> keep BOTH raw + adjusted):
    Date, Open, High, Low, Close, Adj Close, Volume

Rate-limit defenses (Yahoo bans by IP, globally — not per ticker):
    * Tickers are pulled in BATCHES (one request per ~50 tickers via the
      multi-symbol form of yf.download). ~1000 tickers => ~20 requests
      instead of 1000, which is the single biggest 429 reducer.
    * Empty responses are interpreted, not blindly retried: an ENTIRELY
      empty batch is treated as a suspected rate-limit (long "ban" sleep +
      retry), while a ticker that is empty *inside an otherwise populated
      batch* is treated as genuinely no-data (delisted) and not hammered.
    * A global circuit breaker aborts the run after N consecutive empty
      batches so we stop refreshing Yahoo's ban timer and can resume later.
    * yfinance 1.x routes through curl_cffi (browser impersonation), which is
      far more 429-resistant than bare requests — pin it in requirements.txt.

Unattended-run hardening (so it survives a multi-hour remote run):
    * Per-request --timeout so a stuck socket errors out and retries instead
      of hanging the whole run forever.
    * Atomic writes (temp file + os.replace) for every parquet AND the
      manifest -> a kill mid-write never corrupts data or the resume index.
    * The per-ticker write body is guarded: a bad write (for example, a full
      disk) marks that ticker fail and continues.
    * Rotating log (5 x 10MB) so a retry/limit storm can't fill the disk.
    * No external alerting — the EXIT CODE is the signal: 0 = clean,
      1 = circuit-breaker abort, 2 = fail rate past --max-fail-pct. Wire it to
      a systemd OnFailure= / wrapper. Run under tmux or systemd so an SSH
      disconnect (SIGHUP) doesn't kill it.

Output layout:
    <out>/raw/<TICKER>.parquet     one file per ticker (raw bars)
    <out>/manifest.csv             per-ticker status (resume / audit)
    <out>/download.log             run log

Resume is driven by manifest.csv:
    status 'ok'              -> skipped on re-run
    status 'partial'/'fail'  -> retried on re-run
    Records for tickers NOT in the current run are preserved (merged), so
    using --limit or a different list never erases prior results.
Use --force to redo everything.

Examples:
    python download.py --out ./data --tickers-file dataset/rawdata/universe_full.txt
    python download.py --out ./data --tickers-file my.txt  # custom universe
    python download.py --out ./data --force
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not installed. Run: pip install -r requirements.txt")


# --------------------------------------------------------------------------- #
# Universe — one ticker per line, optionally supplied with --tickers-file
# --------------------------------------------------------------------------- #
# Provider holdings pages and Wikipedia both block server-side fetches. Pass the
# checked-in dataset/rawdata/universe_full.txt with --tickers-file.
DEFAULT_UNIVERSE = Path(__file__).resolve().parent / "russell1000.txt"


def read_tickers_file(path: Path) -> list[str]:
    """One ticker per line; blank lines and '#' comments ignored."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def yahoo_symbol(ticker: str) -> str:
    """Yahoo uses '-' where index lists use '.' (e.g. BRK.B -> BRK-B)."""
    return ticker.strip().upper().replace(".", "-")


def exclusive_end(end: str) -> str:
    """yfinance treats `end` as exclusive; bump by 1 day to make it inclusive."""
    d = datetime.strptime(end, "%Y-%m-%d").date() + timedelta(days=1)
    return d.isoformat()


# --------------------------------------------------------------------------- #
# Download core
# --------------------------------------------------------------------------- #
def is_rate_limit(exc: BaseException) -> bool:
    """True if an exception looks like a Yahoo 429 / rate-limit.

    yfinance's behaviour here is version-dependent: 1.x usually swallows the
    HTTPError and returns an empty frame (handled by the empty-batch path),
    but some paths/versions raise YFRateLimitError or a Too-Many-Requests
    error. Detect by type name + message so we don't depend on importing a
    class that may not exist in every yfinance version.
    """
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "toomanyrequests" in name:
        return True
    msg = str(exc).lower()
    return "too many requests" in msg or "rate limit" in msg or "429" in msg


def _split_batch(df: pd.DataFrame | None, ysyms: list[str]) -> dict[str, pd.DataFrame]:
    """Split a multi-symbol yf.download frame into {ysym: Date-column frame}.

    Only non-empty per-ticker frames are returned. yfinance fills tickers it
    could not fetch with all-NaN columns, so we drop all-NaN rows and skip the
    ones that come back empty (delisted / no data in window).
    """
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out

    if isinstance(df.columns, pd.MultiIndex):
        # group_by='ticker' => level 0 is the ticker symbol.
        present = set(df.columns.get_level_values(0))
        for ysym in ysyms:
            if ysym not in present:
                continue
            sub = df[ysym].dropna(how="all")
            if not sub.empty:
                out[ysym] = sub.reset_index()
    else:
        # Single-ticker batch -> flat columns, no ticker level.
        sub = df.dropna(how="all")
        if not sub.empty:
            out[ysyms[0]] = sub.reset_index()
    return out


def fetch_batch(
    ysyms: list[str],
    start: str,
    end_excl: str,
    retries: int,
    base_sleep: float,
    ban_sleep: float,
    timeout: float = 30.0,
    treat_empty_as_ban: bool = True,
) -> tuple[dict[str, pd.DataFrame], bool]:
    """Download a batch of tickers in ONE request, with rate-limit awareness.

    Returns (per_ticker_frames, banned):
        per_ticker_frames : {ysym: Date-column DataFrame} for non-empty results
        banned            : True if every attempt came back empty / rate-limited
                            (i.e. a suspected global IP ban), so the caller can
                            trip the circuit breaker instead of marching on.

    A ticker missing from the returned dict despite a populated batch is
    genuinely empty (delisted) — NOT a rate-limit — and must not be retried.

    treat_empty_as_ban : when True (default, full-window pulls) an entirely
        empty batch is read as a suspected global ban -> long sleep + retry.
        Pass False for the SALVAGE pass: those tickers already failed the full
        window, so an empty batch just means they're genuinely dead — return
        immediately instead of burning ban_sleep retries on dead tickers. A
        real rate-limit *exception* is still honoured in both modes.
    """
    for attempt in range(1, retries + 1):
        rate_limited = False
        try:
            df = yf.download(
                ysyms,
                start=start,
                end=end_excl,
                interval="1d",
                auto_adjust=False,   # keep raw Close AND Adj Close
                actions=False,
                progress=False,
                threads=False,       # one connection at a time -> gentler on Yahoo
                group_by="ticker",
                timeout=timeout,     # per-request cap -> a stuck socket can't hang the run
            )
            got = _split_batch(df, ysyms)
            if got:
                return got, False
            if not treat_empty_as_ban:
                # Salvage of already-failed tickers: empty == genuinely dead,
                # not a ban. Don't waste ban_sleep retries on dead tickers.
                logging.info("batch[%d syms]: empty in salvage -> treated as no-data",
                             len(ysyms))
                return {}, False
            # Empty for the WHOLE batch: with ~50 symbols this is the signature
            # of a global rate-limit, not 50 simultaneously-delisted tickers.
            rate_limited = True
            logging.warning(
                "batch[%d syms]: entirely empty (attempt %d/%d) -> suspected rate-limit",
                len(ysyms), attempt, retries,
            )
        except Exception as exc:  # noqa: BLE001 - log and retry anything
            rate_limited = is_rate_limit(exc)
            logging.warning(
                "batch[%d syms]: error '%s' (attempt %d/%d)%s",
                len(ysyms), exc, attempt, retries,
                " [rate-limit]" if rate_limited else "",
            )

        if attempt < retries:
            if rate_limited:
                # Yahoo's window is tens of seconds to minutes; a 1/2/4s backoff
                # cannot clear a real ban. Wait out the window before retrying.
                time.sleep(ban_sleep)
            else:
                time.sleep(base_sleep * (2 ** (attempt - 1)) + random.uniform(0, base_sleep))

    return {}, True


def write_parquet_atomic(df: pd.DataFrame, fpath: Path) -> None:
    """Write parquet to a temp file then os.replace -> never leave a half file.

    A host shutdown mid-write would otherwise corrupt the parquet while the manifest
    already says 'ok', and resume would skip it -> silent bad data.
    """
    tmp = fpath.with_suffix(fpath.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, fpath)  # atomic on the same filesystem


def load_prior_manifest(manifest_path: Path) -> dict[str, list]:
    """Resume map: ticker -> full row [ticker, status, rows, last_date]."""
    if not manifest_path.exists():
        return {}
    try:
        m = pd.read_csv(manifest_path)
        out: dict[str, list] = {}
        for _, r in m.iterrows():
            t = str(r["ticker"])
            out[t] = [t, str(r["status"]),
                      r.get("rows"), r.get("last_date")]
        return out
    except Exception:  # noqa: BLE001
        return {}


def save_manifest(rows: dict[str, list], manifest_path: Path) -> None:
    """Atomically (re)write the manifest from the in-memory row map."""
    df = pd.DataFrame(rows.values(),
                      columns=["ticker", "status", "rows", "last_date"])
    tmp = manifest_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, manifest_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Raw daily OHLCV downloader (yfinance).")
    ap.add_argument("--out", default="./data", help="output root dir (default ./data)")
    ap.add_argument("--tickers-file", help="file with one ticker per line; "
                                           "use dataset/rawdata/universe_full.txt")
    ap.add_argument("--start", default="2018-01-01", help="start date YYYY-MM-DD")
    ap.add_argument("--end", default="2025-12-31",
                    help="end date YYYY-MM-DD, inclusive (default 2025-12-31)")
    ap.add_argument("--recent-start", default="2024-01-01",
                    help="salvage-window start when full pull fails")
    ap.add_argument("--retries", type=int, default=4, help="retries per batch")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="base seconds between batches / non-ban backoff unit")
    ap.add_argument("--chunk-size", type=int, default=50,
                    help="tickers per yf.download request (fewer requests = fewer 429)")
    ap.add_argument("--ban-sleep", type=float, default=90.0,
                    help="seconds to wait out a suspected rate-limit window")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="per-request network timeout (s) -> a stuck socket "
                         "errors out and retries instead of hanging the run")
    ap.add_argument("--max-empty-chunks", type=int, default=3,
                    help="abort (circuit breaker) after this many consecutive "
                         "entirely-empty batches -> likely IP ban")
    ap.add_argument("--max-fail-pct", type=float, default=50.0,
                    help="exit code 2 if the fail rate over tickers attempted "
                         "this run exceeds this %% (supervisor alert; 100=off)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if previously marked ok")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process first N tickers (debug)")
    args = ap.parse_args()

    out_root = Path(args.out)
    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            # Rotate so a retry/limit storm over a long unattended run can't
            # grow the log without bound: 5 x 10MB.
            logging.handlers.RotatingFileHandler(
                out_root / "download.log", maxBytes=10 * 1024 * 1024,
                backupCount=5, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    end_excl = exclusive_end(args.end)
    recent_excl = end_excl  # salvage shares the same (inclusive) end

    # ---- universe (already big-first if from the pre-baked list) ----
    universe_file = Path(args.tickers_file) if args.tickers_file else DEFAULT_UNIVERSE
    if not universe_file.exists():
        sys.exit(f"Universe file not found: {universe_file}. "
                 f"Pass --tickers-file dataset/rawdata/universe_full.txt.")
    tickers = read_tickers_file(universe_file)
    logging.info("Loaded %d tickers from %s", len(tickers), universe_file)

    if args.limit:
        tickers = tickers[: args.limit]

    # Merge with prior manifest so records for tickers NOT in this run survive.
    rows: dict[str, list] = load_prior_manifest(manifest_path)
    prior = {t: row[1] for t, row in rows.items()}

    logging.info(
        "Window: %s..%s (inclusive) | salvage from %s | out=%s | force=%s | "
        "chunk=%d | ban_sleep=%.0fs",
        args.start, args.end, args.recent_start, out_root, args.force,
        args.chunk_size, args.ban_sleep,
    )

    # Tickers that still need work (skip ones already 'ok' unless --force).
    todo = [t for t in tickers if args.force or prior.get(t) != "ok"]
    skipped = len(tickers) - len(todo)
    if skipped:
        logging.info("Resume: skipping %d tickers already marked ok", skipped)

    chunks = [todo[i:i + args.chunk_size]
              for i in range(0, len(todo), args.chunk_size)]
    n_chunks = len(chunks)

    t0 = time.time()
    consecutive_empty = 0
    aborted = False

    for ci, chunk in enumerate(chunks, 1):
        ymap = {yahoo_symbol(t): t for t in chunk}     # ysym -> original ticker
        ysyms = list(ymap)

        # 1) full window, one request for the whole chunk
        got, banned = fetch_batch(ysyms, args.start, end_excl,
                                  args.retries, args.sleep, args.ban_sleep,
                                  args.timeout)

        if banned and not got:
            # Whole chunk came back empty after all retries -> suspected ban.
            consecutive_empty += 1
            logging.error(
                "[chunk %d/%d] entirely empty after retries "
                "(consecutive empty chunks=%d/%d)",
                ci, n_chunks, consecutive_empty, args.max_empty_chunks,
            )
            for t in chunk:
                rows[t] = [t, "fail", 0, None]
            save_manifest(rows, manifest_path)

            if consecutive_empty >= args.max_empty_chunks:
                logging.error(
                    "CIRCUIT BREAKER: %d consecutive empty chunks -> likely a "
                    "global IP ban. Aborting so we stop refreshing Yahoo's ban "
                    "timer. Re-run later to resume from the manifest.",
                    consecutive_empty,
                )
                aborted = True
                break

            time.sleep(args.ban_sleep)   # wait out the window before next chunk
            continue

        consecutive_empty = 0

        # 2) salvage: tickers empty in the full window may be delisted, but
        #    might just lack the older history -> one batched recent-window pull.
        missing = [ymap[y] for y in ysyms if y not in got]
        salvage: dict[str, pd.DataFrame] = {}
        if missing:
            msyms = [yahoo_symbol(t) for t in missing]
            logging.info("[chunk %d/%d] %d missing -> batched salvage from %s",
                         ci, n_chunks, len(missing), args.recent_start)
            salvage, _ = fetch_batch(msyms, args.recent_start, recent_excl,
                                     args.retries, args.sleep, args.ban_sleep,
                                     args.timeout, treat_empty_as_ban=False)

        # 3) record + write each ticker in the chunk
        ok = part = fail = 0
        for t in chunk:
            ysym = yahoo_symbol(t)
            fpath = raw_dir / f"{ysym}.parquet"
            if ysym in got:
                df, st = got[ysym], "ok"
            elif ysym in salvage:
                df, st = salvage[ysym], "partial"
            else:
                df, st = None, "fail"

            if df is None or df.empty:
                rows[t] = [t, "fail", 0, None]
                fail += 1
                continue

            # Guard the write body: a bad parquet write (for example, a full
            # disk) must mark this ticker and move on, not
            # crash the whole unattended run. Marked 'fail' so resume retries it.
            try:
                write_parquet_atomic(df, fpath)
                last = str(pd.to_datetime(df["Date"].iloc[-1]).date())
                rows[t] = [t, st, len(df), last]
                ok += st == "ok"
                part += st == "partial"
            except Exception as exc:  # noqa: BLE001
                logging.error("%s: write failed '%s' -> marked fail", t, exc)
                rows[t] = [t, "fail", 0, None]
                fail += 1

        logging.info("[chunk %d/%d] ok=%d partial=%d fail=%d",
                     ci, n_chunks, ok, part, fail)

        # persist manifest every chunk -> crash-safe resume on the remote host
        save_manifest(rows, manifest_path)

        if ci < n_chunks:
            time.sleep(args.sleep + random.uniform(0, args.sleep))

    save_manifest(rows, manifest_path)

    manifest = pd.DataFrame(rows.values(),
                            columns=["ticker", "status", "rows", "last_date"])
    ok = (manifest["status"] == "ok").sum()
    part = (manifest["status"] == "partial").sum()
    fail = (manifest["status"] == "fail").sum()
    logging.info("%s in %.1fs | ok=%d partial=%d fail=%d | manifest -> %s",
                 "ABORTED" if aborted else "DONE",
                 time.time() - t0, ok, part, fail, manifest_path)
    if part or fail:
        bad = manifest.loc[manifest["status"] != "ok", "ticker"].tolist()
        logging.warning("Re-run to retry partial/failed (%d): %s",
                        len(bad), ", ".join(bad[:50]) + (" ..." if len(bad) > 50 else ""))

    # Exit code is the "alert" (no external channel): non-zero on a circuit-
    # breaker abort OR when the fail rate over tickers attempted this run blows
    # past the threshold -> a systemd OnFailure= / wrapper can catch a bad night.
    attempted = len(todo)
    fail_this_run = sum(1 for t in todo if rows.get(t, [None, None])[1] == "fail")
    fail_pct = (fail_this_run / attempted * 100) if attempted else 0.0
    if aborted:
        return 1
    if attempted and fail_pct >= args.max_fail_pct:
        logging.error("FAIL RATE %.1f%% (%d/%d attempted) >= %.0f%% threshold "
                      "-> exiting non-zero for the supervisor to catch.",
                      fail_pct, fail_this_run, attempted, args.max_fail_pct)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
