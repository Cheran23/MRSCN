# ============================================================
# PHASE 6 - Cell 3: Item 21 - Confusion matrix + subtype breakdown
# + explicit run metadata (Editor Comments 23, 24, 25)
#
# Uses the representative run confirmed by Cell 2: run 4
# (seed_used=774201, best_threshold=0.63, test F1=0.9767), and the real
# subtype column confirmed by Cell 2 to be 'label' (values: phishing,
# malware, defacement - the 3-class label that binary_label was derived
# from). test_idx values in predictions.npz are POSITIONAL row indices
# into step2_deduplicated_dataset.csv - confirmed directly from Step 13
# Cell 3 (binary_label = df['binary_label'].values, and struct_full/
# char_full/bert_mm/train_idx/val_idx/test_idx all index that same array
# in that same row order), so df.iloc[test_idx] recovers the correct
# subtype per test sample.
#
# [VERIFIED DATA NOTE]
# An earlier workflow specification referred to 5 malicious URLs lacking
# subtype labels. Direct verification of the locked
# step2_deduplicated_dataset.csv used for training found 0 malicious rows
# with a missing 'label' value. All 212,951 malicious rows
# (95,308 defacement + 93,998 phishing + 23,645 malware) have a subtype.
# This analysis therefore uses the verified dataset value of 0 missing
# subtype labels.
# ============================================================
import json
import math
import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'
REPRESENTATIVE_RUN = 4  # median test F1 across the 5 C1_full_mrscn runs, confirmed by Cell 2

# ------------------------------------------------------------
# Load representative run's results + predictions + subtype labels
# ------------------------------------------------------------
with open(f'{ROOT}/results/step13/C1_full_mrscn/run_{REPRESENTATIVE_RUN}_results.json') as f:
    rep_results = json.load(f)

pred = np.load(f'{ROOT}/results/step13/C1_full_mrscn/run_{REPRESENTATIVE_RUN}_predictions.npz')
test_idx = pred['test_idx']
y_true = pred['y_true']
y_pred = pred['y_pred']

df = pd.read_csv(f'{ROOT}/data/processed/TB/step2_deduplicated_dataset.csv')
subtype = df.iloc[test_idx]['label'].values

# Sanity check (assumption-free - does not hardcode the benign label string):
# every malicious test row (y_true==1) must have a subtype in the 3 known malicious
# categories confirmed by Cell 2, with none missing; and every one of those rows'
# subtype must be non-null, matching Cell 2's global 0-missing finding.
MALICIOUS_SUBTYPES = ['phishing', 'malware', 'defacement']
malicious_subtype_vals = pd.Series(subtype[y_true == 1])
bad_vals = malicious_subtype_vals[~malicious_subtype_vals.isin(MALICIOUS_SUBTYPES)]
assert bad_vals.empty, (
    f"[FATAL] {len(bad_vals)} malicious test rows have an unexpected/missing subtype value "
    f"(row alignment may be broken, or df.iloc(test_idx) is not positional as assumed): "
    f"{bad_vals.value_counts(dropna=False).to_dict()}"
)
print("[OK] df.iloc[test_idx]['label'] alignment verified: every malicious test row's subtype "
      f"is one of {MALICIOUS_SUBTYPES}, none missing, across all {len(test_idx):,} test rows.")

benign_subtype_vals = pd.Series(subtype[y_true == 0]).value_counts(dropna=False)
print(f"[FYI] Benign test rows' 'label' value(s) (not used for filtering below): "
      f"{benign_subtype_vals.to_dict()}")

RUN_METADATA = {
    "representative_run_selection_rule": "median test F1 across 5 runs, ties broken by lowest run_id",
    "run_id": REPRESENTATIVE_RUN,
    "seed_used": rep_results["seed_used"],
    "best_threshold": rep_results["best_threshold"],
    "test_f1": rep_results["test_metrics"]["f1"],
    "n_test_samples": int(len(test_idx)),
}
print("\n[RUN METADATA]")
for k, v in RUN_METADATA.items():
    print(f"  {k}: {v}")

# ------------------------------------------------------------
# Wilson score confidence interval (no scipy dependency)
# ------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

# ------------------------------------------------------------
# Overall confusion matrix (cross-checked against results.json)
# ------------------------------------------------------------
tp = int(np.sum((y_true == 1) & (y_pred == 1)))
tn = int(np.sum((y_true == 0) & (y_pred == 0)))
fp = int(np.sum((y_true == 0) & (y_pred == 1)))
fn = int(np.sum((y_true == 1) & (y_pred == 0)))

json_tp, json_tn = rep_results["test_metrics"]["tp"], rep_results["test_metrics"]["tn"]
json_fp, json_fn = rep_results["test_metrics"]["fp"], rep_results["test_metrics"]["fn"]
cm_match = (tp, tn, fp, fn) == (json_tp, json_tn, json_fp, json_fn)
print(f"\n[CONFUSION MATRIX] run {REPRESENTATIVE_RUN} "
      f"(seed={RUN_METADATA['seed_used']}, threshold={RUN_METADATA['best_threshold']:.2f})")
print(f"  TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
print(f"  Cross-check vs results.json: {'MATCH' if cm_match else 'MISMATCH - investigate'}")

overall_recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
overall_fnr = fn / (tp + fn) if (tp + fn) > 0 else float('nan')
overall_recall_ci = wilson_ci(tp, tp + fn)
overall_fnr_ci = wilson_ci(fn, tp + fn)
print(f"  Overall recall (sensitivity) = {overall_recall:.4f} "
      f"(95% CI {overall_recall_ci[0]:.4f}-{overall_recall_ci[1]:.4f})")
print(f"  Overall FNR = {overall_fnr:.4f} "
      f"(95% CI {overall_fnr_ci[0]:.4f}-{overall_fnr_ci[1]:.4f})")

# ------------------------------------------------------------
# Subtype-level breakdown (malicious class only: phishing/malware/defacement)
# (MALICIOUS_SUBTYPES already defined above during the alignment sanity check)
# ------------------------------------------------------------
total_fn = fn

rows = []
for st in MALICIOUS_SUBTYPES:
    mask = (subtype == st)
    n_st = int(mask.sum())
    tp_st = int(np.sum(mask & (y_true == 1) & (y_pred == 1)))
    fn_st = int(np.sum(mask & (y_true == 1) & (y_pred == 0)))
    denom = tp_st + fn_st  # should equal n_st, since all rows of this subtype are malicious
    assert denom == n_st, f"[FATAL] {st}: TP+FN ({denom}) != subtype count ({n_st})"

    recall_st = tp_st / denom if denom > 0 else float('nan')
    fnr_st = fn_st / denom if denom > 0 else float('nan')
    recall_ci = wilson_ci(tp_st, denom)
    fnr_ci = wilson_ci(fn_st, denom)
    share_of_total_fn = fn_st / total_fn if total_fn > 0 else float('nan')

    rows.append({
        "subtype": st,
        "n_test_samples": n_st,
        "correctly_detected_tp": tp_st,
        "missed_fn": fn_st,
        "recall": recall_st,
        "recall_ci_low": recall_ci[0],
        "recall_ci_high": recall_ci[1],
        "fnr": fnr_st,
        "fnr_ci_low": fnr_ci[0],
        "fnr_ci_high": fnr_ci[1],
        "share_of_total_false_negatives": share_of_total_fn,
    })

df_subtype = pd.DataFrame(rows)
print("\n[SUBTYPE-LEVEL BREAKDOWN] (malicious class only, representative run)")
print(df_subtype.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

subtype_total_n = df_subtype['n_test_samples'].sum()
subtype_total_fn = df_subtype['missed_fn'].sum()
print(f"\n[CHECK] Sum of subtype sample counts = {subtype_total_n:,} "
      f"(should equal total malicious test samples TP+FN = {tp + fn:,}): "
      f"{'MATCH' if subtype_total_n == (tp + fn) else 'MISMATCH'}")
print(f"[CHECK] Sum of subtype FN = {subtype_total_fn:,} (should equal overall FN = {total_fn:,}): "
      f"{'MATCH' if subtype_total_fn == total_fn else 'MISMATCH'}")

# ------------------------------------------------------------
# Missing-subtype-label accounting: verified locked dataset contains 0 missing subtype labels
# ------------------------------------------------------------
n_missing_subtype_in_test = int(pd.isna(subtype[y_true == 1]).sum())

print(
    f"\n[VERIFY] Malicious test-set rows with missing subtype label: "
    f"{n_missing_subtype_in_test}. The locked deduplicated dataset used for "
    f"training contains 0 malicious rows with missing subtype labels."
)

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
out_dir = f'{ROOT}/results/step21_item21'
import os
os.makedirs(out_dir, exist_ok=True)

with open(f'{out_dir}/run_metadata.json', 'w') as f:
    json.dump(RUN_METADATA, f, indent=2)

confusion_summary = {
    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    "overall_recall": overall_recall, "overall_recall_ci": list(overall_recall_ci),
    "overall_fnr": overall_fnr, "overall_fnr_ci": list(overall_fnr_ci),
    "cross_check_vs_results_json": "MATCH" if cm_match else "MISMATCH",
    "n_malicious_test_rows_missing_subtype": n_missing_subtype_in_test,
}
with open(f'{out_dir}/confusion_matrix_summary.json', 'w') as f:
    json.dump(confusion_summary, f, indent=2)

df_subtype.to_csv(f'{out_dir}/subtype_breakdown.csv', index=False)

print(f"\nSaved: {out_dir}/run_metadata.json")
print(f"Saved: {out_dir}/confusion_matrix_summary.json")
print(f"Saved: {out_dir}/subtype_breakdown.csv")

df_subtype
