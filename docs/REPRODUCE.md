# Reproducing the research

I run every command below from the repository root. The repository contains code,
configuration, ticker lists, manifests, and compact result ledgers. It does not contain raw
market data, built datasets, labels, prediction parquets, or model checkpoints.

The canonical file-production commands are also indexed in
[`results_provenance.md`](results_provenance.md), section 0. That sheet is the authority for
every reported research number and for the exact command behind it.

## Environment

I use the single installation path documented at the top of `requirements.txt`:

1. Create and activate a clean conda environment using the Python version named there.
2. Install PyTorch first with the exact command and CUDA wheel index shown there.
3. Install the remaining pinned packages and check the environment:

   ```bash
   pip install -r requirements.txt
   pip check
   ```

The data-source packages are needed only when rebuilding raw data. Training from existing
parquets does not contact a data provider.

For comparable CPU results, I pin the thread count before running downloads, builds, models,
or evaluations:

```bash
export OMP_NUM_THREADS=6
```

LightGBM otherwise uses all available cores. A different core count can change reduction order,
the selected early-stopping round, and therefore the stored result. The six-thread setting is
also the safe choice for the mixed performance/efficiency-core CPU on which I developed the
pipeline.

### GPU requirements

The GBDT and ridge walk-forward scripts run on CPU. The MLP script selects CUDA when it is
available and otherwise falls back to CPU; I recommend a CUDA-capable GPU for a full MLP
walk-forward run. `--device cuda` makes that requirement explicit. The PyTorch import check is
not sufficient: I run the actual CUDA kernel probe printed at the top of `requirements.txt`
before a long job. The remaining evaluation tools are CPU post-processing jobs.

## Raw data

Both downloaders are resumable. They write one parquet per symbol under their market's `raw/`
directory and atomically update a manifest. Re-running skips successful symbols; `--force`
requests a fresh download. Provider availability and licensing can change, so I review the
provider's terms before downloading or retaining data.

### US universe and download

I use the checked-in `dataset/rawdata/universe_full.txt` as the exact universe input:

```bash
python dataset/rawdata/download.py \
  --tickers-file dataset/rawdata/universe_full.txt \
  --out dataset/rawdata
```

The explicit `--out dataset/rawdata` is important: it places the files under
`dataset/rawdata/raw/`, where the dataset builder looks by default (provenance row I1).

I produced the checked-in ticker list by extracting and normalizing symbols from a
current broad-market holdings snapshot, merging the pre-existing large-cap list, and
ordering the result with larger names first. I do not redistribute the source holdings export.
The derived one-symbol-per-line list is therefore the reproducible input shipped here; an
identical-source regeneration script is not included. For a new universe, I supply my own
lawfully obtained point-in-time (historical) constituents with `--tickers-file`. The downloader normalizes
dots in class-share symbols to Yahoo's hyphen convention.

This US list is a current snapshot rather than point-in-time historical membership. It therefore
has survivorship bias and is not a clean historical index-constituent panel.

The line-count mismatch between `universe_full.txt` and `manifest.csv` is explained by resume
semantics, not by hidden universe members. The manifest preserves records from symbols that are
not in the current run; the manifest-only records in this checkout are failed attempts with no
downloaded rows. I use the ticker file as the current universe and the manifest as an audit and
resume history. Provenance row D1 reports the manifest totals.

### China A-share universe and download

The CN downloader builds `universe_full_cn.txt` as the quarterly union of historical CSI 300 and
CSI 500 constituents and then downloads each symbol's full window:

```bash
python dataset/rawdata_cn/download_cn.py --universe-only
python dataset/rawdata_cn/download_cn.py
```

Without the first command, the downloader reuses the checked-in universe; it rebuilds the file
automatically only when the file is missing. It writes raw files under `dataset/rawdata_cn/raw/`
and resume state to `manifest_cn.csv`. The source can be selected with
`--source`; `--limit` is useful for a connectivity check. The downloaded coverage and known
failure bias are recorded in provenance row D2.

## Build the feature datasets

After both raw-data directories exist, I build the market panels with the commands in provenance
rows I1 and I2:

```bash
python dataset/alphaFactor/build_dataset.py --market us
python dataset/alphaFactor/build_dataset.py --market cn
```

The outputs are:

- `dataset/alphaFactor/dataset_alpha158.parquet`
- `dataset/alphaFactor/dataset_alpha158_cn.parquet`

For a quick wiring check, I use `--smoke`. Smoke output is only a mechanical test; I do not treat
its score as a research result.

## Build and verify labels

I first prove that the label builder reproduces each market's built-in close label on the same row
grid:

```bash
python dataset/alphaFactor/make_vwap_label.py --market cn --verify
python dataset/alphaFactor/make_vwap_label.py --market us --verify
```

I stop if either comparison fails. I then build the current CN label and the comparable US proxy
label (provenance row I3):

```bash
python dataset/alphaFactor/make_vwap_label.py \
  --market cn --price vwap --gate entry

python dataset/alphaFactor/make_vwap_label.py \
  --market us --price hlc3 --gate none
```

True turnover-value VWAP is available in the CN source. The US source has OHLCV but no turnover
value, so `hlc3` is a statistical proxy rather than an executable VWAP. I do not present the two
markets as perfectly matched.

## Walk-forward models

The stored design uses an expanding walk-forward with the fold, validation, and embargo settings
in provenance row D4. I keep the command tags because downstream filenames depend on them.

### GBDT

```bash
python model/walkforward.py \
  --data dataset/alphaFactor/dataset_alpha158_cn.parquet \
  --label-file dataset/alphaFactor/label_cn_vwap_entry.parquet \
  --seed 0 --tag _vwapentry

python model/walkforward.py \
  --data dataset/alphaFactor/dataset_alpha158.parquet \
  --label-file dataset/alphaFactor/label_us_hlc3_none.parquet \
  --tag _hlc3
```

These commands produce the prediction files named in provenance rows I4 and I5 and append run
metadata to `model/walkforward.jsonl` unless `--no-log` is passed.

### MLP and ridge

```bash
python model/walkforward_mlp.py \
  --data dataset/alphaFactor/dataset_alpha158_cn.parquet \
  --label-file dataset/alphaFactor/label_cn_vwap_entry.parquet \
  --seed 0 --tag _vwapentry_s0 --device cuda

python model/walkforward_ridge.py \
  --data dataset/alphaFactor/dataset_alpha158_cn.parquet \
  --label-file dataset/alphaFactor/label_cn_vwap_entry.parquet \
  --tag _vwapentry
```

The corresponding US MLP and the per-epoch MLP workflow are specified in provenance rows I7 and
I8. The ridge output is specified in I9. I use `--no-preds` only for experiments whose downstream
portfolio evaluation is intentionally skipped.

## Evaluation and audits

The prediction parquets are the stable boundary between training and evaluation. Common commands
are:

```bash
# RankIC, ICIR, deciles, turnover, and the unsmoothed portfolio
python model/evaluate.py \
  --preds model/preds_walkforward_cn_vwapentry.parquet

# Frozen turnover-reduction grid
python model/turnover_study.py \
  --preds model/preds_walkforward_cn_vwapentry.parquet

# Long-only grid
python model/longonly_study.py \
  --preds model/preds_walkforward_cn_vwapentry.parquet

# Nested portfolio-variant reselection
python model/selection_null.py \
  --preds model/preds_walkforward_cn_vwapentry.parquet \
  --mode nested --legs longshort

# Sample-period uncertainty
python model/block_bootstrap.py \
  --preds model/preds_walkforward_cn_vwapentry.parquet

# Price-limit and reversal audits
python model/limit_audit.py \
  --preds model/preds_walkforward_cn_vwapfull.parquet

python model/reversal_exposure.py \
  --preds model/preds_walkforward_cn_vwapentry.parquet \
  --data dataset/alphaFactor/dataset_alpha158_cn.parquet
```

The exact commands for the permutation null, long-only null, paired model bootstrap, negative
control, pool correlation, and epoch-rule audit are in provenance rows T6, T8–T10, N3–N9, I10,
and I11. I use each script's `--help` before changing a default or output tag.

### Result-record limitation

Walk-forward RankICs and fold metadata are stored in the shipped JSONL result files. Pool and
epoch-rule summaries are stored in JSON. Decile, portfolio, null, and bootstrap tools currently
print their results only to the terminal. The numbers in `results_provenance.md` were transcribed
from those runs and checked against the named command; a machine-readable archive of that output
is planned. Reproducing a reported portfolio number therefore means regenerating the unpublished data
artifacts, running the exact command in the provenance row, and comparing terminal output with the
sheet.

## Runtime expectations

I do not publish wall-clock estimates because the provenance sheet does not contain a controlled
hardware benchmark, and network providers dominate download time. The downloaders print elapsed
time and are resumable; the dataset builds are storage- and memory-bound; GBDT and ridge are CPU
jobs; the full MLP walk-forward is the GPU-sensitive stage; and bootstrap/permutation runs scale
with the requested resample count. For a new machine, I time a limited download or smoke build and
one model fold before starting a full run. I retain the command, hardware description, thread
setting, and printed elapsed time with any new result.

## Minimal end-to-end checklist

- I install the documented environment and confirm the CUDA kernel when I plan to run the MLP.
- I export `OMP_NUM_THREADS=6` and keep it fixed across comparable runs.
- I download both markets and review the two manifests rather than assuming every ticker succeeded.
- I build both feature panels, verify the close labels, and build the current CN and US labels.
- I run walk-forward models with the exact provenance tags so downstream filenames match.
- I run the evaluation command named by each result row and compare its terminal output with
  `docs/results_provenance.md`.
- I keep all generated parquets, raw bars, checkpoints, and run logs out of version control.
