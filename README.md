# MRSCN Reproducibility Package

This reproducibility package accompanies the revised MRSCN manuscript and
provides the model implementation, downstream analysis code, archived
predictions, processed data, experimental splits, locked configurations, and
other supporting artifacts used in the reported experiments. The public
materials are split across two persistent, citable deposits:

1. **This repository** (GitHub, archived to Zenodo via GitHub's Zenodo
   integration on release) — `code/`, `scripts/downstream_analysis/`, this
   README, `requirements.txt`, and `LICENSE`. It contains the model
   implementation and the original downstream analysis and supporting scripts.
2. **A second, separate Zenodo deposit** (data only, uploaded directly since
   Zenodo's GitHub integration is not intended for multi-gigabyte data) —
   `results/`, `data/`, `splits/`, `locked_configs/`, `cache/`, plus a small
   `mrscn_reproducibility_patch.zip` (see "Reproducibility patch" below). It
   contains the archived model predictions, processed dataset, train/val/test
   splits, locked configuration, cached structured/character-sequence features,
   and other intermediate artifacts used by the released analysis code.

Together, the released artifacts support direct reproduction of the downstream
analyses for workflow items 21-25 and 27-28 from the archived predictions and
supporting data. The original scripts for Items 20 and 26 are also provided,
but exact rerunning of their model-level profiling procedures requires trained
model checkpoints that are not included in the public deposits, as documented
below.

Unzip both into the same parent folder, then unzip
`mrscn_reproducibility_patch.zip` on top, so the final layout looks like:

```
MRSCN_reproducibility_package/
  code/                        <- this repository
  scripts/downstream_analysis/ <- this repository
  README.md                    <- this repository
  requirements.txt             <- this repository
  LICENSE                      <- this repository
  results/                     <- data deposit
  data/                        <- data deposit (includes data/processed/TB/TB_training_ready.csv, from the patch)
  splits/                      <- data deposit
  locked_configs/              <- data deposit
  cache/                       <- data deposit
  folds/                       <- from the patch, required by code/utils/data_utils.py
```

**Data deposit DOI:** `10.5281/zenodo.22204730`
**Code deposit DOI (this repository, via GitHub-Zenodo integration):** `10.5281/zenodo.22214147`

## What's in `code/`

A from-scratch, leakage-checked implementation of the proposed MRSCN model
(frozen DistilBERT projection + character-CNN stream + structured-feature
stream + Structural Consistency Module + fusion classifier — see
`code/models/mrscn.py`), with strict data-usage separation:

- `code/train.py` — trains one fold at a time (`python train.py --fold 1`),
  using the training set for parameter updates and the validation set ONLY
  for early stopping and decision-threshold search. Supports `--resume`.
- `code/test.py` — loads the fold's best checkpoint and frozen
  validation-derived threshold, and runs test-set inference **exactly once**
  (`python test.py --fold 1`). The test set is never touched anywhere else.
- `code/evaluation.py` — shared metrics module (accuracy, precision, recall,
  specificity, F1, macro-F1, ROC-AUC, Average Precision, MCC, balanced
  accuracy, FPR/FNR, confusion matrix).
- `code/configs/config.py` — central configuration (paths, DistilBERT model
  name, character vocabulary, training hyperparameters, 80/10/10 split
  ratios, 5-fold seeds file, threshold search grid).
- `code/utils/` — data loading, scaling, fold-index loading, seeding helpers.
- `code/seed_registry.json` — the actual 5 fold seeds used
  (`global_seed: 42`, plus one seed per fold).

**Note on terminology:** `evaluation.py` originally stored Average Precision
(`sklearn.metrics.average_precision_score`) under the key `"pr_auc"`. That
name is misleading — it's not a trapezoidal `auc(recall, precision)` — and is
exactly the terminology issue Editor Comment 29 flags for the manuscript
(Equation 34). It has been renamed to `"average_precision"` here (in both
`evaluation.py` and `test.py`'s output) so the deposited code and the
corrected manuscript wording agree. No computation changed, only the label.

**Environment note:** this repo trains against a pre-extracted parquet
(`configs.config.DATA_PATH`, expected at `datasets/TB_extracted_features.parquet`
relative to the repo root) holding the URL, numeric label, and the 25
structured feature columns. Build it after assembling the full package
(code, data deposit, and patch, per the layout above) with:

```
python code/utils/csv_to_parquet.py --csv data/processed/TB/TB_training_ready.csv
```

`TB_training_ready.csv` comes from the reproducibility patch (see below), not
directly from the data deposit's `data/processed/TB/step2_deduplicated_dataset.csv`
— the patch's notes explain exactly what was changed and why. The DistilBERT embeddings and character sequences are handled separately from
the parquet input. Precomputed DistilBERT embeddings are not included in this
deposit and must be regenerated before training or inference that requires
them, as described below.

## Reproducibility patch

`mrscn_reproducibility_patch.zip` (in the data deposit) contains two small
fixes discovered when this package was verified end to end against
`code/configs/config.py` and `code/utils/data_utils.py`:

- `data/processed/TB/TB_training_ready.csv` — the same 640,845 rows and 25
  structured features as the data deposit's `step2_deduplicated_dataset.csv`,
  with the numeric `binary_label` column renamed to `label` (verified
  identical to the string `label` column for every row) and the non-feature
  `regex_valid` flag dropped. Without this, `get_feature_cols()` would treat
  `binary_label` and `regex_valid` as model features rather than the label
  and a QA flag respectively.
- `folds/fold_1_indices.json` … `folds/fold_5_indices.json` — renamed copies
  of the data deposit's `splits/TB/run_1_indices.json` … `run_5_indices.json`,
  at the path `load_fold_indices()` actually reads. Seeds verified against
  `code/seed_registry.json`.

See `PATCH_NOTES.md` inside the zip for the full verification trail.

### DistilBERT embeddings

Precomputed DistilBERT embedding files are not included in this repository or
the archived data deposit because the original cached embedding files are no
longer available. Accordingly, `datasets/bert_features.npy` is not distributed
with the reproducibility package.

The DistilBERT representations must be regenerated from the corresponding URL
data using the DistilBERT preprocessing and representation-generation
procedure before model training or inference that requires these embeddings.
For the default downstream reproduction workflow, the archived per-sample
model predictions and experimental outputs are used directly, so regeneration
of the original DistilBERT embedding cache is not required.

## What's in `scripts/downstream_analysis/`

The directory contains the original downstream analysis and supporting scripts
for workflow items 20-28:

| Item | What it produces |
|---|---|
| 20 | Peak inference memory profiling (GPU models, CPU/tree models, consolidated) |
| 21 | Confusion matrix + phishing-subtype breakdown + run metadata |
| 22 | McNemar paired comparisons vs. top baselines |
| 23 | McNemar paired comparisons for ablations, extended with the partially fine-tuned DistilBERT variant (Holm-Bonferroni recomputed across the enlarged family) |
| 24 | Quantitative false-positive / false-negative error characterization |
| 25 | Calibration analysis (ECE, Brier score) + reliability diagram, extended to 13 models |
| 26 | Consolidated computational-complexity table (storage/latency/FLOPs), extended with the partially fine-tuned DistilBERT variant, plus a supplementary fair (true end-to-end) FLOPs comparison |
| 27 | ROC and Precision-Recall curves, MRSCN vs. Table 7 and Table 12 baseline groups (Average Precision via `average_precision_score`, not a trapezoidal PR-AUC) |
| 28 | Normalized confusion-matrix heatmap, representative run |

`run_all.py` is the default reproduction driver for items 21-25 and 27-28.
These analyses operate on the archived predictions, processed data, locked
configuration files, and other released intermediate artifacts in the
companion data deposit.

Items 20 and 26 are not included in the default driver because their original
inference-memory, latency, storage, and FLOPs profiling procedures require
trained `.pt` model checkpoints that are not distributed with the public
reproducibility package. The original Item 20 and Item 26 scripts are retained
in this directory for methodological transparency and provenance.

Files prefixed `support_` include utilities used during the original
experimental workflow, including prediction regeneration, baseline refitting,
and checkpoint-reload inference. Some of these utilities require the original
cached DistilBERT embeddings and/or trained model checkpoints and therefore
are not part of the default public reproduction workflow.

Within the default driver, execution order matters. In particular, `item23b`
and `item25c` extend outputs generated by their corresponding preceding scripts
and should be run once after those scripts. See the header of `run_all.py` for
details.

**Scope:** the default workflow reproduces the downstream analyses for items
21-25 and 27-28 from the released archived model predictions and supporting
artifacts. It does not retrain the 13 models from scratch. Model training and
testing code is provided separately in `code/train.py` and `code/test.py`;
retraining requires the corresponding model inputs, regenerated DistilBERT
representations where applicable, and the locked seeds, and requires
substantially greater computational resources.

**Environment:** every script was written for and run in Google Colab with
this project's Google Drive folder mounted at
`/content/drive/MyDrive/MRSCN_Revision` (each script calls
`google.colab.drive.mount()` and defines `ROOT` itself). To run outside
Colab, mount or symlink this package's `results/`, `data/`, `cache/`, and
`locked_configs/` folders at that same path, or edit each script's `ROOT`
constant directly — the scripts were left exactly as verified against the
already-reported numbers, rather than generalized, to avoid introducing
untested changes.

## Reference feature-extraction script

The original Phase 1 structured-feature extraction script is included at
`code/reference/phase1_feature_extraction.py` as reference-only documentation
of the domain-age, IP-blacklist, SFH, and web-traffic extraction logic used
during feature generation.

## What's excluded from the archived data, and why

- **`glove/`** (a single ~990MB file under the project's `cache/` folder on
  Drive) — a standard, publicly downloadable pretrained GloVe embeddings
  file (Stanford NLP). Not re-hosted here; if a baseline model needs it,
  download the matching file from https://nlp.stanford.edu/projects/glove/
  and place it at `cache/glove/<filename>`.
- **`checkpoints/step1_TB/` and `checkpoints/step3_SG/`** — confirmed to be
  resumable structured-feature-extraction progress state (chunked parquet +
  a `state.json` tracking `completed_chunks`), fully superseded by the final
  processed dataset in `data/`. Not archived.
- **`checkpoints/tldextract_cache/`** — a public-suffix-list cache,
  trivially re-fetched from the internet. Not archived.
- **Trained model checkpoints** (`.pt` files, ~5GB across all 13 models,
  dominated by the partially fine-tuned DistilBERT variant at ~3.2GB and
  TrackPhish at ~865MB) — not included in this deposit round. The archived
  per-sample predictions in `results/` are sufficient for the default
  downstream reproduction workflow covering items 21-25 and 27-28. The
  original Item 20 and Item 26 profiling scripts, as well as prediction-
  regeneration and checkpoint-reload utilities, require the trained
  checkpoints and therefore cannot be rerun exactly from the released files
  alone. The checkpoints may be deposited separately in the future if needed.

## Citation / DOI

This repository has not yet been pushed to GitHub or archived to Zenodo, and
the data deposit has not yet been created. Suggested order of operations,
since the manuscript's Code Availability statement needs a DOI that resolves
before the paper is published:

1. Create the data deposit on Zenodo (New Upload), attach the five archive
   files plus `mrscn_reproducibility_patch.zip`, and use "Reserve DOI" to get
   a DOI before publishing it.
2. Push this repository to GitHub, enable it in Zenodo's GitHub integration,
   then cut a GitHub release — Zenodo mints a DOI for that release
   automatically and archives the repository's exact state.
3. Add both DOIs to the two placeholders near the top of this README, to the
   manuscript's Code Availability statement, and to the response letter's two
   remaining `[DOI]` placeholders (Response 31, paragraphs 459 and 461).

## License

MIT — see `LICENSE`.
