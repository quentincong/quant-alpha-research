"""
epoch_rule.py — comparison of rules for selecting an epoch (robust epoch selection)

**Problem** (planned in docs/research_log.md#r40, run in docs/research_log.md#r41):
  The MLP walk-forward uses the **argmax** of validation RankIC to select the stopping epoch.
  But best_epoch across three seeds in `walkforward_mlp.jsonl` is:
      2020: 17 / 17 / 31     2021: 3 / 2 / 14     2022: 9 / 41 / 5
      2023: 16 / 6 / 6       2024: 23 / 1 / 1     2025: 1 / 4 / 1
  With **the same data and only a different random seed, selected epochs differ by 40x**.
  Taking the maximum of a noisy curve is itself a **selection procedure** that transfers noise
  into the weights. **The goal is variance reduction, not a higher score.** Success is determined
  by the remeasured seed standard deviation, not by whether the headline increases.

**Design constraint** (added in docs/research_log.md#r37): the 2020 fold remains at epoch 17 across learning-rate
  changes and GPU nondeterminism, so **it is a real peak**. The rule therefore **cannot assume
  every fold is noise**. It must be nearly harmless on peaked folds, falling back near argmax,
  and smooth only flat folds. Both rules below have this property because plateau width is derived
  from the validation curve's own noise scale rather than chosen arbitrarily.

--------------------------------------------------------------------------
Three rule families
--------------------------------------------------------------------------
`argmax` (current behavior / baseline)
    ep* = argmax_ep val(ep).

`plateau_median` (plateau median)
    First estimate the validation curve's **per-epoch measurement noise** σ from adjacent differences:
        σ ≈ std(val[t] − val[t−1]) / √2
    Difference variance is twice single-point variance. This standard conversion from curve
    jitter to point jitter is explainable in 60 seconds.
    Plateau = { ep : val(ep) ≥ max(val) − z·σ }; select the **median** epoch number in the plateau.
    · Real peak: the peak exceeds noise substantially, so the plateau contains only nearby points
      and the median ≈ argmax, causing little harm.
    · Pure noise: the whole curve lies within the z·σ band, so the median ≈ curve midpoint and
      no longer selects the luckiest point.

`wavg_*` (weight averaging, the simplest form of SWA)
    **Average each parameter** across several epoch state_dict values to create new weights.
    The network has only Linear/ReLU/Dropout and **no BatchNorm**, so no running statistics need
    re-estimation and the operation is a plain arithmetic mean. Three variants:
      wavg_best_k  : final k epochs ending at argmax ([ep*−k+1, ep*])
      wavg_run_k   : final k epochs of the full run, including the post-peak patience segment
      wavg_top_k   : k epochs with the highest validation values, not necessarily contiguous
    Weight averaging **does not equal** prediction averaging because the network is nonlinear:
    `f(mean(w)) ≠ mean(f(w))`. Weights must actually be averaged before a forward pass; averaging
    predictions is not a substitute.

--------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------
Input = per-fold .pt files written by `walkforward_mlp.py --epoch-dump DIR`, containing
  per-epoch validation RankIC, per-epoch weight snapshots, and **that fold's test row indices**.
  Saving row indices means this script **recomputes no masks**. Fold geometry is identical by
  construction to training, eliminating the risk of drift from a second definition.

Every rule uses only the validation curve; test is predicted only once at the end. The rule itself
is also fitted on data, but it fits **validation**, exactly like argmax, keeping the comparison clean.

Reuse the established metrics: daily RankIC from `evaluate.ic_panel` and
`turnover_study.portfolio` at **ema5 / exit10**, the headline variant without reselection.

Usage:
  python model/epoch_rule.py --ckpt-dir model/checkpoints/epoch_rule --tag-glob '_epsel_s*' \\
      --data dataset/alphaFactor/dataset_alpha158_cn.parquet \\
      --label-file dataset/alphaFactor/label_cn_vwap_entry.parquet
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mlp import AlphaMLP, _pearson, _rank
from walkforward import load_panel_year
from turnover_study import portfolio, smooth
from evaluate import ic_panel, apply_cap

HEADLINE_SPAN, HEADLINE_EXIT = 5, 0.10     # Do not reselect for this experiment; reuse the GBDT headline.


# ---------------------------------------------------------------------------
# Rules use only the validation curve and return which epochs to use + whether to average weights.
# ---------------------------------------------------------------------------
def noise_sigma(curve: np.ndarray) -> float:
    """Estimate single-point measurement noise σ from first differences; use full-curve std if too short."""
    if len(curve) < 3:
        return float(np.std(curve)) if len(curve) > 1 else 0.0
    return float(np.std(np.diff(curve), ddof=1) / np.sqrt(2.0))


def rule_argmax(curve: np.ndarray) -> tuple[list[int], str]:
    return [int(np.argmax(curve)) + 1], "single"


def rule_plateau_median(curve: np.ndarray, z: float = 1.0) -> tuple[list[int], str]:
    sig = noise_sigma(curve)
    thr = curve.max() - z * sig
    plateau = np.flatnonzero(curve >= thr) + 1
    return [int(np.median(plateau))], "single"        # If the median falls between two points, take the lower integer.


def rule_wavg_best_k(curve: np.ndarray, k: int) -> tuple[list[int], str]:
    best = int(np.argmax(curve)) + 1
    return list(range(max(1, best - k + 1), best + 1)), "wavg"


def rule_wavg_run_k(curve: np.ndarray, k: int) -> tuple[list[int], str]:
    n = len(curve)
    return list(range(max(1, n - k + 1), n + 1)), "wavg"


def rule_wavg_top_k(curve: np.ndarray, k: int) -> tuple[list[int], str]:
    order = np.argsort(-curve, kind="stable")[:k] + 1
    return sorted(int(e) for e in order), "wavg"


def build_rules(ks) -> dict:
    r = {"argmax": rule_argmax,
         "plateau_med_z1": lambda c: rule_plateau_median(c, 1.0),
         "plateau_med_z0.5": lambda c: rule_plateau_median(c, 0.5),
         "plateau_med_z2": lambda c: rule_plateau_median(c, 2.0)}
    for k in ks:
        r[f"wavg_best_k{k}"] = (lambda k: lambda c: rule_wavg_best_k(c, k))(k)
        r[f"wavg_run_k{k}"] = (lambda k: lambda c: rule_wavg_run_k(c, k))(k)
        r[f"wavg_top_k{k}"] = (lambda k: lambda c: rule_wavg_top_k(c, k))(k)
    return r


def state_for(ck: dict, epochs: list[int], mode: str) -> dict:
    """Get weights for the rule's epochs: one snapshot for single, arithmetic parameter mean for wavg."""
    states = ck["states"]
    eps = [e for e in epochs if e in states]
    if not eps:
        raise KeyError(f"epoch {epochs} absent from snapshots (available {min(states)}–{max(states)})")
    if mode == "single" or len(eps) == 1:
        return states[eps[0]], eps
    acc = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in states[eps[0]].items()}
    for e in eps:
        for k, v in states[e].items():
            acc[k] += v.to(torch.float64)
    return {k: (v / len(eps)).to(states[eps[0]][k].dtype) for k, v in acc.items()}, eps


@torch.no_grad()
def predict(model, X_t, rows, chunk=200_000):
    model.eval()
    parts = []
    for i in range(0, len(rows), chunk):
        idx = torch.from_numpy(np.asarray(rows[i:i + chunk])).to(X_t.device)
        parts.append(model(X_t[idx]).float().cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0)


def daily_rankic(pred: np.ndarray, y: np.ndarray, day: np.ndarray) -> float:
    out = []
    order = np.argsort(day, kind="stable")
    d = day[order]
    for blk in np.split(order, np.flatnonzero(np.diff(d) != 0) + 1):
        if len(blk) < 5:
            continue
        out.append(_pearson(_rank(pred[blk]), _rank(y[blk])))
    return float(np.nanmean(out)) if out else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", default="model/checkpoints/epoch_rule",
                    help="per-epoch snapshot directory (generated content, gitignored by default)")
    ap.add_argument("--tag-glob", dest="tag_glob", default="_epsel_s*")
    ap.add_argument("--data", required=True)
    ap.add_argument("--label-file", dest="label_file", default=None)
    ap.add_argument("--ks", default="3,5,10")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", dest="out_json", default="model/epoch_rule_results.json")
    ap.add_argument("--dump-rules", dest="dump_rules", default="argmax,plateau_med_z1,wavg_best_k5",
                    help="save predictions for these rules as parquet for later combination/pool stages")
    ap.add_argument("--dump-prefix", dest="dump_prefix", default="model/preds_wf_mlp_cn_eprule")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.ckpt_dir, f"fold*{args.tag_glob}.pt")))
    if not files:
        sys.exit(f"no snapshots found: {args.ckpt_dir}/fold*{args.tag_glob}.pt")
    print(f"[ckpt] {len(files)} fold snapshots")

    print(f"[data] reading {args.data}")
    X, y, day_id, year, meta = load_panel_year(args.data, args.label_file)
    device = args.device
    X_t = torch.from_numpy(X).to(device)
    y_np = y.astype(np.float64)
    print(f"[data] N={len(y)} rows | X_t.device={X_t.device}")
    assert (args.device != "cuda") or X_t.device.type == "cuda"

    ks = [int(k) for k in args.ks.split(",")]
    rules = build_rules(ks)
    print(f"[rules] {list(rules)}\n")

    # Memory discipline: DataFrames for 13 rules × 3 seeds × 2.3M rows exceed 10GB.
    # All rules/seeds share **the same test rows**, so store one metadata copy and only one
    # float32 prediction vector per rule (13×2.3M×4B ≈ 120MB/seed). Process one seed at a time.
    by_seed = defaultdict(list)
    for f in files:
        m = re.match(r"fold(\d+)(.*)_seed(\d+)\.pt$", os.path.basename(f))
        by_seed[int(m.group(3))].append((int(m.group(1)), f))
    seeds = sorted(by_seed)

    per_fold, chosen, curve_info = {}, {}, []
    port = {}                      # port[(rule, seed)] = portfolio dict
    market = "cn" if "_cn" in os.path.basename(args.data) else "us"

    for seed in seeds:
        folds = sorted(by_seed[seed])
        base_parts, preds_acc = [], defaultdict(list)
        for test_year, f in folds:
            ck = torch.load(f, weights_only=False, map_location="cpu")
            curve = np.asarray(ck["val_curve"], dtype=np.float64)
            rows = np.asarray(ck["test_rows"])
            sig = noise_sigma(curve)
            peak_z = (curve.max() - np.median(curve)) / sig if sig > 0 else np.nan
            curve_info.append({"seed": seed, "year": test_year, "n_ep": len(curve),
                               "argmax": int(np.argmax(curve)) + 1, "sigma": sig,
                               "max": float(curve.max()), "median": float(np.median(curve)),
                               "peak_z": float(peak_z)})
            model = AlphaMLP(n_features=ck["n_features"],
                             hidden_list=tuple(int(h) for h in ck["hidden"].split(",")),
                             dropout=ck["dropout"]).to(device)
            b = meta.iloc[rows][["date", "instrument", "label_raw"]].copy()
            b["label"] = y_np[rows]
            b["test_year"] = test_year
            base_parts.append(b)
            for rname, rfn in rules.items():
                eps, mode = rfn(curve)
                st, used = state_for(ck, eps, mode)
                model.load_state_dict(st)
                p = predict(model, X_t, rows)
                per_fold[(rname, seed, test_year)] = daily_rankic(p, y_np[rows], day_id[rows])
                chosen[(rname, seed, test_year)] = used
                preds_acc[rname].append(p.astype(np.float32))
            del ck
            print(f"  seed{seed} fold{test_year}: {len(curve)} epochs, "
                  f"argmax={int(np.argmax(curve))+1}, sigma={sig:.4f}, peak_z={peak_z:.2f}")

        base = pd.concat(base_parts, ignore_index=True)
        base["date"] = pd.to_datetime(base["date"])
        dump_set = {r.strip() for r in args.dump_rules.split(",") if r.strip()}
        for rname in rules:
            d = base.copy()
            d["pred"] = np.concatenate(preds_acc[rname]).astype(np.float64)
            d = apply_cap(d, market, None, quiet=True)
            port[(rname, seed)] = portfolio(smooth(d, HEADLINE_SPAN),
                                            q_enter=0.10, q_exit=HEADLINE_EXIT)
            if rname in dump_set:
                out = f"{args.dump_prefix}_{rname}_s{seed}.parquet"
                d.to_parquet(out, index=False)
                print(f"  [dump] {out}")
            del d
        del base, base_parts, preds_acc
        print(f"  seed{seed} complete\n")

    years = sorted({yy for (_, _, yy) in per_fold})

    # ---------------- Validation-curve shape (distinguish real peak from noise) ----------------
    ci = pd.DataFrame(curve_info).sort_values(["year", "seed"])
    print(f"\n{'='*90}\n[A] validation-curve shape: peak sigma above median (large peak_z = real peak; small = flat/noisy)\n{'='*90}")
    print(ci.to_string(index=False,
                       formatters={"sigma": "{:.4f}".format, "max": "{:+.4f}".format,
                                   "median": "{:+.4f}".format, "peak_z": "{:.2f}".format}))

    # ---------------- Each rule: all-fold mean RankIC by seed + seed standard deviation ----------------
    print(f"\n{'='*104}\n[B] test RankIC by rule (one value per fold → fold mean → standard deviation across seeds)\n{'='*104}")
    hdr = (f"{'rule':>18} " + " ".join(f"{('s'+str(s)):>9}" for s in seeds) +
           f" {'mean':>9} {'seed_sd':>9} {'sd/argmax':>10}")
    print(hdr); print("-" * len(hdr))
    summary, base_sd = {}, None
    for rname in rules:
        per_seed = [float(np.mean([per_fold[(rname, s, yy)] for yy in years])) for s in seeds]
        mu, sd = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1))
        if rname == "argmax":
            base_sd = sd
        summary[rname] = {"per_seed_rankic": per_seed, "mean_rankic": mu,
                          "seed_sd_rankic": sd,
                          "per_fold": {f"s{s}_{yy}": per_fold[(rname, s, yy)]
                                       for s in seeds for yy in years},
                          "chosen_epochs": {f"s{s}_{yy}": chosen[(rname, s, yy)]
                                            for s in seeds for yy in years}}
        print(f"{rname:>18} " + " ".join(f"{v:>+9.4f}" for v in per_seed) +
              f" {mu:>+9.4f} {sd:>9.5f} {sd / base_sd if base_sd else float('nan'):>10.2f}")

    # ---------------- Portfolio layer: same ema5/exit10 metric ----------------
    print(f"\n{'='*104}\n[C] portfolio layer (same metric: ema{HEADLINE_SPAN}/exit{int(HEADLINE_EXIT*100)},"
          f" variant not reselected for this experiment)\n{'='*104}")
    hdr = (f"{'rule':>18} " + " ".join(f"{('net@10 s'+str(s)):>10}" for s in seeds) +
           f" {'mean':>9} {'seed_sd':>9} {'IR@10':>8} {'turn':>7}")
    print(hdr); print("-" * len(hdr))
    port_base_sd = None
    for rname in rules:
        nets = [port[(rname, s)]["net10"] for s in seeds]
        irs = [port[(rname, s)]["ir10"] for s in seeds]
        turns = [port[(rname, s)]["turnover"] for s in seeds]
        mu, sd = float(np.mean(nets)), float(np.std(nets, ddof=1))
        if rname == "argmax":
            port_base_sd = sd
        summary[rname].update({"net10_per_seed": nets, "net10_mean": mu,
                               "net10_seed_sd": sd, "ir10_mean": float(np.mean(irs)),
                               "turnover_mean": float(np.mean(turns))})
        print(f"{rname:>18} " + " ".join(f"{v:>+10.2f}" for v in nets) +
              f" {mu:>+9.2f} {sd:>9.3f} {np.mean(irs):>+8.2f} {np.mean(turns):>7.1%}")

    print(f"\n[argmax baseline] seed sd: RankIC {base_sd:.5f} | net@10 {port_base_sd:.3f} bps "
          f"— these two values are the empirical resolution baseline")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"ckpt_dir": args.ckpt_dir, "data": args.data,
                   "label_file": args.label_file, "seeds": seeds, "years": years,
                   "headline_variant": f"ema{HEADLINE_SPAN}/exit{int(HEADLINE_EXIT*100)}",
                   "curve_info": curve_info, "rules": summary}, f,
                  ensure_ascii=False, indent=1, default=float)
    print(f"[json] {args.out_json}")


if __name__ == "__main__":
    main()
