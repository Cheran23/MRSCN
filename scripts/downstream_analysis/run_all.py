#!/usr/bin/env python3
# ============================================================
# run_all.py - driver for downstream analyses that can be
# reproduced directly from the archived predictions and other
# released artifacts in the companion data deposit.
#
# SCOPE: this driver runs the reproducible downstream analyses
# for workflow items 21-25 and 27-28 using archived model
# predictions, processed data, locked configuration files, and
# other released intermediate artifacts.
#
# Items 20 and 26 are intentionally not included in this default
# driver because their original profiling procedures require
# trained .pt model checkpoints that are not distributed with
# the public reproducibility package. Their original scripts are
# retained in this directory for methodological transparency and
# provenance, but they cannot be rerun exactly from the released
# files alone.
#
# Prediction-regeneration support scripts are also not included
# in this default driver because the original cached DistilBERT
# embeddings and trained model checkpoints are not distributed.
# Instead, the archived per-sample prediction files are used
# directly by the downstream statistical analyses.
#
# ENVIRONMENT: the downstream scripts were originally written
# and run in Google Colab with the project folder mounted at
#   /content/drive/MyDrive/MRSCN_Revision
# Each script defines ROOT using that location. To run elsewhere,
# either reproduce that directory layout or edit each script's
# ROOT constant to point to the extracted companion data deposit.
#
# ORDER MATTERS for scripts within the same analysis item.
# In particular, item23b and item25c extend outputs produced by
# their corresponding preceding scripts and should be run once
# after those scripts.
# ============================================================
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (script filename, one-line description)
ORDER = [
    # ---- Prerequisites: regenerate/refit predictions consumed below ----
    # ---- Item 20: peak inference memory profiling ----
    # ---- Item 21: confusion matrix + subtype breakdown + run metadata ----
    ("item21_confusion_matrix_subtype_breakdown.py",
     "Item 21: confusion matrix, phishing-subtype breakdown, run metadata"),

    # ---- Item 22: paired comparisons vs. top baselines ----
    ("item22_paired_comparisons_vs_baselines.py",
     "Item 22: McNemar paired comparisons, MRSCN vs. top baselines"),

    # ---- Item 23: paired comparisons for ablations (a then b - b extends a's CSV) ----
    ("item23a_paired_comparisons_ablations.py",
     "Item 23a: McNemar paired comparisons, MRSCN vs. ablation variants"),
    ("item23b_extend_with_distilbert_partial_ft.py",
     "Item 23b: extend item 23's table with the partially fine-tuned DistilBERT comparison "
     "(recomputes Holm-Bonferroni across the enlarged family - must run AFTER item23a, exactly once)"),

    # ---- Item 24: FP/FN error characterization ----
    ("support_item24_build_derived_features.py",
     "Item 24 support: build derived URL features used in the FP/FN breakdown"),
    ("item24_fp_fn_error_characterization.py",
     "Item 24: quantitative false-positive / false-negative error characterization"),

    # ---- Item 25: calibration (a, b, then c - c extends a's CSV) ----
    ("item25a_calibration_analysis.py", "Item 25a: calibration analysis (ECE, Brier score)"),
    ("item25b_reliability_diagram.py",  "Item 25b: reliability diagram"),
    ("item25c_extend_with_distilbert_partial_ft.py",
     "Item 25c: extend item 25's calibration table with the partially fine-tuned DistilBERT row "
     "(must run AFTER item25a, exactly once)"),

    # ---- Item 26: computational complexity table (strict a->b->c->d->e->f order) ----
    # ---- Item 27: ROC / PR curves (a then b) ----
    ("item27a_representative_run_selection_and_inventory.py",
     "Item 27a: re-derive the representative run, verify prediction availability/alignment"),
    ("item27b_roc_pr_curves.py",
     "Item 27b: ROC and Precision-Recall curves, MRSCN vs. Table 7 and Table 12 baseline groups"),

    # ---- Item 28: normalized confusion-matrix heatmap ----
    ("item28_confusion_matrix_heatmap.py",
     "Item 28: normalized confusion-matrix heatmap, representative run"),
]


def main():
    print("=" * 100)
    print(f"[run_all] {len(ORDER)} scripts scheduled, in dependency order.")
    print("=" * 100)

    for i, (fname, desc) in enumerate(ORDER, start=1):
        path = os.path.join(HERE, fname)
        print(f"\n{'#' * 100}")
        print(f"# [{i}/{len(ORDER)}] {fname}")
        print(f"# {desc}")
        print(f"{'#' * 100}\n")

        if not os.path.exists(path):
            print(f"[run_all] **MISSING** {path} - skipping. This script was expected to be part of "
                  f"this package; if you deleted or moved it, restore it before re-running.")
            continue

        result = subprocess.run([sys.executable, path])
        if result.returncode != 0:
            print(f"\n[run_all] **FAILED** at step {i}/{len(ORDER)}: {fname} "
                  f"(exit code {result.returncode}). Stopping - fix the error above before continuing; "
		  f"re-running this driver from the start is safe for most steps, but item23b/25c "
                  f"are NOT idempotent (they extend an existing CSV) - see each script's own header "
                  f"comment before re-running those specifically.")
            sys.exit(result.returncode)

    print("\n" + "=" * 100)
    print("[run_all] All scheduled downstream analyses for items 21-25 and 27-28 have "
          "been regenerated into this package's results/ folder (or wherever ROOT points).")
    print("=" * 100)


if __name__ == "__main__":
    main()
