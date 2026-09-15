"""
block_bootstrap.py — block-bootstrap confidence intervals for reported metrics

**Why this is required**: the naive t-stat reported by `evaluate.py` is **inflated** because of
  **autocorrelation in the IC series**:
    · the exit band keeps most holdings from one day to the next (label_raw itself is a one-day
      T+1→T+2 return, so daily returns do not overlap);
    · ema5 smoothing blends today's score into yesterday's history;
    · market regimes persist in contiguous stretches.
  The naive t assumes 1,453 days are 1,453 **independent** observations. They are not, so the
  denominator (standard error) is too small and t is too large.

**Why block bootstrap instead of Newey-West / HAC**: the methodological boundary is the
  **60-second rule**; any method committed to the repository must be explainable without prompts.
  Block bootstrap needs only two sentences: "Resampling individual days destroys autocorrelation,
  so **resample contiguous L-day blocks with replacement**, recompute the metric, and read its
  distribution." HAC is a closed-form formula that becomes hard to defend when asked about kernel
  bandwidth. **Simulation is preferable to formula.**

**Why differences are valid here but not under the permutation null** (following the discipline
  from docs/research_log.md#r35): the permutation null shuffles returns while **holding positions and
  turnover fixed**, so even without signal, high-span variants reliably beat low-span variants
  by about 9 bps. That is **deterministic cost arithmetic**, making the two sides incomparable.
  **Bootstrap differs** because it resamples **days**. In a paired comparison, both models receive
  **the same days** through the same idx matrix, and each side's positions, turnover, and costs
  **move together with those days**. The cost term naturally cancels in the difference and creates
  no offset. In short: **permutation tests only levels; paired bootstrap can test differences.**

**Method (circular moving-block bootstrap)**
  For n days and block length L, draw ceil(n/L) **uniformly random starting points** and take
  L days after each start, **wrapping** to the beginning at the end. Concatenate and truncate to n days.
  Circular wrapping gives every day equal selection probability; a noncircular version systematically
  undersamples boundary days. Use a **percentile CI**, reading the 2.5% and 97.5% quantiles directly
  from the bootstrap distribution. Do not use BCa because its bias-correction acceleration cannot
  be derived independently and violates the 60-second rule.

**An approximation that must be stated explicitly**: turnover is **path-dependent** because
  turn_t depends on holdings at t−1. Resampling changes block boundaries, so one day in each L-day
  block has turn_t whose "yesterday" is not yesterday in the resampled path. With L=20 this affects
  5% of days. Therefore, the script **must print bootstrap mean versus point estimate**. Agreement
  shows that the boundary approximation does not shift the center and supports the metric's credibility.
  The L∈{5,10,20,40} block-length sweep serves the same purpose: if intervals are insensitive to L,
  the conclusion does not depend on choosing L.

**The two error bars are not interchangeable** and must be reported separately:
  · **seed sd**: retrain with different random seeds to measure **model-training** uncertainty.
  · **block bootstrap CI**: expose the same model to a different historical sample to measure
    **sample-period** uncertainty.
  Previously this project had only the former. The latter asks whether the alpha works only in these six years.

Usage (from the repository root, after conda activate torch-env):
  # Levels: all metrics for one model + block-length sensitivity.
  OMP_NUM_THREADS=6 python model/block_bootstrap.py \
      --preds model/preds_walkforward_cn_vwapentry.parquet

  # Paired: use the first as reference and compare each other model on the same days and idx matrix.
  OMP_NUM_THREADS=6 python model/block_bootstrap.py \
      --preds model/preds_walkforward_cn_vwapentry.parquet:GBDT_s0 \
      --preds model/preds_wf_mlp_cn_vwapentry_s0.parquet:MLP_s0 --paired
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import apply_cap, ic_panel
from train_lgbm import daily_ic
from turnover_study import smooth, portfolio
from longonly_study import long_only

# Frozen portfolio variants (set in docs/research_log.md#r30 and reused in docs/research_log.md#r35; never reselect for a new model,
# because reselection is another selection procedure).
LS_SPAN, LS_EXIT = 5, 0.10      # Long-short: ema5 / exit10.
LO_SPAN, LO_EXIT = 5, 0.30      # Long-only: ema5 / exit30.
COST = 10                       # Reporting cost tier (bps), matching the README headline.
ANN = np.sqrt(252)


# ---------------------------------------------------------------------------
# Daily series: use existing functions throughout and **do not create another metric**
# ---------------------------------------------------------------------------
def build_series(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Compress one prediction parquet into a one-row-per-day time-series table.

    Columns: ic (daily RankIC), ls_gross/ls_turn (long-short ema5/exit10), and lo_act/lo_turn
    (long-only ema5/exit30). Use the **intersection** of their day sets because portfolio() skips
    days with an empty leg, and report days outside the intersection.
    """
    # 1. Daily RankIC: same daily_ic as evaluate.ic_panel, with dates retained.
    days, ics = [], []
    for the_date, g in df.groupby("date", sort=True):
        ics.append(daily_ic(g["pred"].to_numpy(), g["label"].to_numpy(), [np.arange(len(g))])[1])
        days.append(the_date)
    ic = pd.Series(ics, index=pd.DatetimeIndex(days), name="ic").dropna()
    # Match evaluate.ic_panel's series elementwise; otherwise this has created a different metric.
    ref = ic_panel(df)["series"]
    assert len(ref) == len(ic) and np.allclose(ref, ic.to_numpy()), \
        f"{name}: daily IC differs from evaluate.ic_panel; metric definition drifted"

    # 2. Long-short / long-only legs: apply EMA smoothing, then existing portfolio functions.
    ls = portfolio(smooth(df, LS_SPAN), q_enter=0.10, q_exit=LS_EXIT, with_series=True)
    lo = long_only(smooth(df, LO_SPAN), q_enter=0.10, q_exit=LO_EXIT, with_series=True)

    out = pd.DataFrame({"ic": ic}).join(
        pd.DataFrame({"ls_gross": ls["_gross"], "ls_turn": ls["_turn"]}, index=ls["_days"])).join(
        pd.DataFrame({"lo_act": lo["_act"], "lo_turn": lo["_turn"]}, index=lo["_days"]))
    n_drop = out.isna().any(axis=1).sum()
    if n_drop:
        print(f"    [{name}] {n_drop} days are missing under one definition (empty portfolio leg) and are dropped after intersection")
    return out.dropna()


# ---------------------------------------------------------------------------
# Statistics: compute only from resampled days, with uniform signature (S, idx) -> array
# ---------------------------------------------------------------------------
def _stats(S: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """idx shape (B, n): B resamples of n days. Return a length-B array for each metric.

    B=1 with idx=arange(n) is the point estimate. The point estimate and bootstrap use
    **the same code**, preventing one definition for the point estimate and another for the interval.
    """
    ic = S["ic"][idx]
    lsn = (S["ls_gross"][idx] - S["ls_turn"][idx] * COST / 1e4) * 1e4
    lon = (S["lo_act"][idx] - S["lo_turn"][idx] * COST / 1e4) * 1e4
    def _ir(a):
        sd = a.std(axis=1, ddof=1)
        return np.where(sd > 0, a.mean(axis=1) / np.where(sd > 0, sd, 1) * ANN, np.nan)
    icsd = ic.std(axis=1, ddof=1)
    return {
        "RankIC":    ic.mean(axis=1),
        "ICIR":      np.where(icsd > 0, ic.mean(axis=1) / np.where(icsd > 0, icsd, 1), np.nan),
        "LS net@10": lsn.mean(axis=1),
        "LS IR@10":  _ir(lsn),
        "LO net@10": lon.mean(axis=1),
        "LO IR@10":  _ir(lon),
    }


def make_idx(n: int, L: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block: ceil(n/L) random starts × block length L, wrapping and truncating to n."""
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n
    return idx.reshape(B, -1)[:, :n]


def point(S: dict[str, np.ndarray]) -> dict[str, float]:
    n = len(S["ic"])
    return {k: float(v[0]) for k, v in _stats(S, np.arange(n)[None, :]).items()}


def boot(S: dict[str, np.ndarray], L: int, B: int, rng, chunk: int = 500) -> dict[str, np.ndarray]:
    n = len(S["ic"])
    acc: dict[str, list] = {}
    for lo_ in range(0, B, chunk):
        idx = make_idx(n, L, min(chunk, B - lo_), rng)
        for k, v in _stats(S, idx).items():
            acc.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in acc.items()}


def boot_paired(SA, SB, L, B, rng, chunk: int = 500) -> dict[str, np.ndarray]:
    """Paired: feed the same idx to both models so cost terms move with days and do not shift the difference."""
    n = len(SA["ic"])
    acc: dict[str, list] = {}
    for lo_ in range(0, B, chunk):
        idx = make_idx(n, L, min(chunk, B - lo_), rng)
        a, b = _stats(SA, idx), _stats(SB, idx)
        for k in a:
            acc.setdefault(k, []).append(a[k] - b[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


def ci_row(name: str, pt: float, dist: np.ndarray, show_sign: bool = False) -> str:
    lo_, hi = np.nanpercentile(dist, [2.5, 97.5])
    se = float(np.nanstd(dist, ddof=1))
    bias = float(np.nanmean(dist)) - pt
    excl = "  " if (lo_ <= 0 <= hi) else " ★"      # ★ = 95% interval excludes zero.
    s = (f"  {name:>10} {pt:>+9.4f}  [{lo_:>+8.4f}, {hi:>+8.4f}]  SE {se:>7.4f}  "
         f"bias {bias:>+8.4f}{excl}")
    if show_sign:
        s += f"  sign flip {float(np.nanmean(np.sign(dist) != np.sign(pt))):>6.1%}"
    return s


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", required=True,
                    help="prediction parquet, optionally path:display-name; repeatable; first is reference under --paired")
    ap.add_argument("--paired", action="store_true",
                    help="in addition to levels, compare each remaining model with the first on the same days and idx")
    ap.add_argument("--blocks", default="5,10,20,40", help="block-length sweep in days")
    ap.add_argument("--main-block", type=int, default=20, help="main reporting block length (README promises 20)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=float, default=None,
                    help="|label_raw| outlier safeguard; market default (us=0.8, cn=None)")
    args = ap.parse_args()

    Ls = [int(x) for x in args.blocks.split(",")]
    if args.main_block not in Ls:
        Ls.append(args.main_block)

    # ---- Load ----
    panels: dict[str, pd.DataFrame] = {}
    for spec in args.preds:
        path, _, nm = spec.partition(":")
        nm = nm or os.path.basename(path).replace("preds_", "").replace(".parquet", "")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        market = "cn" if "_cn" in os.path.basename(path) else "us"
        df = apply_cap(df, market, args.cap)
        print(f"[data] {nm}: {path}  market={market}  {len(df):,} rows  {df['date'].nunique()} days")
        panels[nm] = build_series(df, nm)

    names = list(panels)
    if args.paired and len(names) > 1:
        common = panels[names[0]].index
        for nm in names[1:]:
            common = common.intersection(panels[nm].index)
        for nm in names:
            if len(panels[nm]) != len(common):
                print(f"[align] {nm}: {len(panels[nm])} days → intersection {len(common)} days")
            panels[nm] = panels[nm].loc[common]

    S = {nm: {c: p[c].to_numpy(dtype=np.float64) for c in p.columns} for nm, p in panels.items()}

    print(f"\nDefinition: long-short ema{LS_SPAN}/exit{int(LS_EXIT*100)}, long-only ema{LO_SPAN}/exit{int(LO_EXIT*100)}"
          f" (**frozen variants, never reselected for any model**); cost {COST} bps;"
          f" circular moving-block, B={args.n_boot}, seed={args.seed}")

    # ---- Levels ----
    for nm in names:
        rng = np.random.default_rng(args.seed)
        pt = point(S[nm])
        print(f"\n{'=' * 92}\n[level] {nm}  (n={len(S[nm]['ic'])} days, main block length L={args.main_block})\n{'=' * 92}")
        print(f"  {'metric':>10} {'estimate':>9}  {'95% percentile CI':^22}  {'SE':>10}  {'bootstrap mean−estimate':>10}")
        d = boot(S[nm], args.main_block, args.n_boot, rng)
        for k in pt:
            print(ci_row(k, pt[k], d[k]))
        # Block-length sensitivity: interval credibility depends on sensitivity to L.
        print(f"\n  [block-length sensitivity] change in 95% CI with L (★ = excludes zero)")
        print(f"  {'L':>4} " + " ".join(f"{k:^26}" for k in ("RankIC", "LS net@10", "LO net@10")))
        for L in sorted(Ls):
            rng = np.random.default_rng(args.seed)
            dl = boot(S[nm], L, args.n_boot, rng)
            cells = []
            for k in ("RankIC", "LS net@10", "LO net@10"):
                a, b = np.nanpercentile(dl[k], [2.5, 97.5])
                cells.append(f"[{a:>+8.4f},{b:>+8.4f}]{'★' if not (a <= 0 <= b) else ' '}")
            print(f"  {L:>4} " + " ".join(f"{c:^26}" for c in cells))

    # ---- Paired differences ----
    if args.paired and len(names) > 1:
        ref = names[0]
        for nm in names[1:]:
            rng = np.random.default_rng(args.seed)
            pa, pb = point(S[nm]), point(S[ref])
            print(f"\n{'=' * 92}\n[paired difference] {nm} − {ref}  (same {len(S[ref]['ic'])} days, L={args.main_block})"
                  f"\n{'=' * 92}")
            print("  Same idx matrix for both sides → turnover/cost move with days, excluding the cost offset from docs/research_log.md#r35")
            d = boot_paired(S[nm], S[ref], args.main_block, args.n_boot, rng)
            for k in pa:
                print(ci_row(k, pa[k] - pb[k], d[k], show_sign=True))

    print(f"\n{'=' * 92}")
    print("Interpretation: ★ = 95% percentile CI excludes zero. Bias = bootstrap mean − point estimate,"
          " which should be near zero;\n      a large bias means block boundaries damaged turnover path dependence (see the header approximation).")
    print("**This CI measures sample-period uncertainty, distinct from the standard deviation across retraining seeds; do not combine them.**")


if __name__ == "__main__":
    main()
