# ============================================================
# PHASE 7 - Cell 1: Setup - re-derive the representative run, inventory
# Table 7 / Table 12 baseline predictions for it.
# (READ-ONLY)
#
# "Representative run" is re-derived here with the exact same logic Phase
# 6 Cell 2 used (median test F1 across the 5 C1_full_mrscn runs, ties
# broken by lowest run_id) rather than hardcoded from memory, so this
# notebook is guaranteed consistent with every number already reported
# for "the representative run" in items 21/25.
#
# Table 7 baseline group (workflow item 16): Logistic Regression, Random
# Forest, XGBoost, BERT-CNN, TCN-Attention, BiLSTM - confirmed from the
# workflow spec text directly (he_bertcnn/do_tcnmhsa are Table 7 members
# even though their files live under step17/, which also holds the
# Table 12 group - a storage-folder detail, not a table-membership one).
#
# Table 12 baseline group (workflow item 17): Khalife-RF, Mohanty-GBoost,
# Omolara-AdaBoost, ResMLP (Remya et al.), TrackPhish (Kondaiah et al.).
# ============================================================
import os, json
import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'
RESULTS_DIR = f'{ROOT}/results'

# ------------------------------------------------------------
# 1. Representative run - identical logic to Phase 6 Cell 2
# ------------------------------------------------------------
f1_per_run = {}
for run_id in range(1, 6):
    with open(f'{RESULTS_DIR}/step13/C1_full_mrscn/run_{run_id}_results.json') as f:
        r = json.load(f)
    f1_per_run[run_id] = r['test_metrics']['f1']

sorted_runs = sorted(f1_per_run.items(), key=lambda kv: (kv[1], kv[0]))
median_idx = len(sorted_runs) // 2
REPRESENTATIVE_RUN = sorted_runs[median_idx][0]
print("C1_full_mrscn F1 per run:")
for run_id, f1 in f1_per_run.items():
    print(f"  run {run_id}: F1={f1:.4f}")
print(f"\n[REPRESENTATIVE RUN] run {REPRESENTATIVE_RUN} (F1={f1_per_run[REPRESENTATIVE_RUN]:.4f}), "
      "median-F1, ties broken by lowest run_id.")

# ------------------------------------------------------------
# 2. Model group definitions (from the workflow spec, items 16/17)
# ------------------------------------------------------------
MRSCN = ('step13', 'C1_full_mrscn', 'MRSCN (proposed)')

TABLE7_GROUP = [
    ('step16', 'LogisticRegression', 'Logistic Regression'),
    ('step16', 'RandomForest', 'Random Forest'),
    ('step16', 'XGBoost', 'XGBoost'),
    ('step17', 'he_bertcnn', 'BERT-CNN'),
    ('step17', 'do_tcnmhsa', 'TCN-Attention'),
    ('step16', 'BiLSTM', 'BiLSTM'),
]

TABLE12_GROUP = [
    ('step17', 'khalife_rf', 'Khalife et al. (RF)'),
    ('step17', 'mohanty_gboost_fst', 'Mohanty & Acharya (GBoost)'),
    ('step17', 'omolara_adaboost', 'Omolara & Alawida (AdaBoost)'),
    ('step17', 'remya_resmlp', 'ResMLP (Remya et al.)'),
    ('step17', 'trackphish', 'TrackPhish (Kondaiah et al.)'),
]

# ------------------------------------------------------------
# 3. Inventory: does every model have predictions for the representative
#    run specifically (not just "5/5 somewhere" - the exact run_id)?
# ------------------------------------------------------------
def check(step, config_key, label, run_id):
    p = f'{RESULTS_DIR}/{step}/{config_key}/run_{run_id}_predictions.npz'
    exists = os.path.exists(p)
    n = None
    if exists:
        d = np.load(p)
        n = len(d['test_idx'])
    return exists, n, p

print("\n" + "=" * 100)
print(f"[INVENTORY] Predictions for run {REPRESENTATIVE_RUN} specifically:")
print("=" * 100)
all_ok = True
for step, key, label in [MRSCN] + TABLE7_GROUP + TABLE12_GROUP:
    exists, n, p = check(step, key, label, REPRESENTATIVE_RUN)
    group = "MRSCN" if (step, key) == MRSCN[:2] else ("Table7" if (step, key, label) in TABLE7_GROUP else "Table12")
    print(f"  [{group:7s}] {label:30s} ({step}/{key}): {'OK n=' + str(n) if exists else '**MISSING** ' + p}")
    if not exists:
        all_ok = False

print(f"\n[RESULT] {'All models have predictions for the representative run - ready for item 27.' if all_ok else '**GAPS FOUND - resolve before building ROC/PR curves.**'}")

# ------------------------------------------------------------
# 4. test_idx alignment check within each group (needed since ROC/PR
#    curves plot multiple models on the same axes - must be scored on the
#    exact same test set to be visually/statistically comparable)
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("[ALIGNMENT] test_idx SET match against MRSCN, within each group, for the representative run:")
print("=" * 100)
mrscn_pred = np.load(f'{RESULTS_DIR}/{MRSCN[0]}/{MRSCN[1]}/run_{REPRESENTATIVE_RUN}_predictions.npz')
mrscn_test_set = set(mrscn_pred['test_idx'].tolist())

for group_name, group in [("Table 7", TABLE7_GROUP), ("Table 12", TABLE12_GROUP)]:
    print(f"\n-- {group_name} --")
    for step, key, label in group:
        p = f'{RESULTS_DIR}/{step}/{key}/run_{REPRESENTATIVE_RUN}_predictions.npz'
        if not os.path.exists(p):
            print(f"  {label}: SKIPPED (missing file)")
            continue
        d = np.load(p)
        same_set = set(d['test_idx'].tolist()) == mrscn_test_set
        print(f"  {label}: {'OK (same test set as MRSCN)' if same_set else '**MISMATCH - different test set**'}")

print("\n[NEXT] If everything above is OK, Cell 2 builds the Table 7 ROC+PR figure pair, "
      "Cell 3 builds the Table 12 pair, Cell 4 builds the normalized confusion-matrix heatmap.")
