# ============================================================
# Phase 6 addendum notebook - Cell 3: Extend item 25's calibration table
# with every model that now has real predictions (previously 9; LR/RF/
# XGBoost and step14_finetuned_distilbert were all missing/blocked then).
#
# Brier score and ECE formulas match the original 9-model table's stated
# methodology (mean squared prob error; 10 equal-width bins, weighted
# |accuracy-confidence|). For the NEW rows only, "confidence" per bin =
# mean predicted probability of the positive/phishing class in that bin,
# "accuracy" = empirical fraction of true positives in that bin - the
# standard binary calibration-curve convention (equivalent to sklearn's
# calibration_curve), documented explicitly here since the exact original
# Cell 12 source wasn't available to copy verbatim in this notebook. The
# original 9 rows are NOT recomputed or touched - only appended to.
# ============================================================
import os, shutil, time
import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'
RESULTS_DIR = f'{ROOT}/results'
SUMMARY_PATH = f'{RESULTS_DIR}/step21_item25/calibration_summary.csv'
PERRUN_PATH = f'{RESULTS_DIR}/step21_item25/calibration_per_run.csv'

NEW_MODELS = [
    ('step16', 'LogisticRegression', 'Logistic Regression'),
    ('step16', 'RandomForest', 'Random Forest'),
    ('step16', 'XGBoost', 'XGBoost'),
    ('step14', 'step14_finetuned_distilbert', 'MRSCN (partial FT DistilBERT)'),
]

def load_predictions(step, config_key, run_id):
    p = f'{RESULTS_DIR}/{step}/{config_key}/run_{run_id}_predictions.npz'
    d = np.load(p)
    return {k: d[k] for k in d.files}

def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))

def ece_10bin(y_true, y_prob):
    n = len(y_true)
    bin_edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < 9:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)  # include 1.0 in the last bin
        n_bin = int(np.sum(mask))
        if n_bin == 0:
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += (n_bin / n) * abs(accuracy - confidence)
    return float(ece)

# ------------------------------------------------------------
# Compute per-run rows for the 4 newly-available models
# ------------------------------------------------------------
new_perrun_rows = []
for step, config_key, label in NEW_MODELS:
    for run_id in range(1, 6):
        pred = load_predictions(step, config_key, run_id)
        y_true, y_prob = pred['y_true'], pred['y_prob']
        new_perrun_rows.append({
            "model": label, "run_id": run_id, "n_test": len(y_true),
            "brier_score": brier_score(y_true, y_prob),
            "ece_10bin": ece_10bin(y_true, y_prob),
        })

df_perrun_new = pd.DataFrame(new_perrun_rows)

# ------------------------------------------------------------
# Aggregate to summary rows (mean/std across 5 runs), matching the
# existing summary table's schema exactly.
# ------------------------------------------------------------
new_summary_rows = []
for step, config_key, label in NEW_MODELS:
    sub = df_perrun_new[df_perrun_new['model'] == label]
    new_summary_rows.append({
        "model": label,
        "brier_mean": float(sub['brier_score'].mean()),
        "brier_std": float(sub['brier_score'].std()),
        "ece_mean": float(sub['ece_10bin'].mean()),
        "ece_std": float(sub['ece_10bin'].std()),
        "n_runs": 5,
    })
df_summary_new = pd.DataFrame(new_summary_rows)

# ------------------------------------------------------------
# Load existing tables, append (don't touch existing rows), back up, save
# ------------------------------------------------------------
df_perrun_old = pd.read_csv(PERRUN_PATH)
df_summary_old = pd.read_csv(SUMMARY_PATH)

df_perrun_final = pd.concat([df_perrun_old, df_perrun_new[df_perrun_old.columns.tolist()]], ignore_index=True)
df_summary_final = pd.concat([df_summary_old, df_summary_new[df_summary_old.columns.tolist()]], ignore_index=True)
df_summary_final = df_summary_final.sort_values('brier_mean').reset_index(drop=True)

ts = int(time.time())
for path, df_old in [(SUMMARY_PATH, df_summary_old), (PERRUN_PATH, df_perrun_old)]:
    backup_path = path.replace('.csv', f'_before_distilbert_addendum_{ts}.csv')
    shutil.copyfile(path, backup_path)
    print(f"[BACKUP] {path} -> {backup_path}")

df_perrun_final.to_csv(PERRUN_PATH, index=False)
df_summary_final.to_csv(SUMMARY_PATH, index=False)
print(f"[SAVED] {PERRUN_PATH} ({df_perrun_final.shape[0]} rows, was {df_perrun_old.shape[0]})")
print(f"[SAVED] {SUMMARY_PATH} ({df_summary_final.shape[0]} rows, was {df_summary_old.shape[0]})")

print("\n" + "=" * 100)
print("[UPDATED] item 25 calibration summary - all 13 models, sorted by Brier score (lower = better):")
print("=" * 100)
print(df_summary_final.to_string(index=False))

print("\n" + "=" * 100)
print("[HEADLINE] Frozen vs. partially fine-tuned DistilBERT calibration:")
print("=" * 100)
frozen = df_summary_final[df_summary_final['model'] == 'MRSCN (C1_full_mrscn)'].iloc[0]
partial = df_summary_final[df_summary_final['model'] == 'MRSCN (partial FT DistilBERT)'].iloc[0]
print(f"  Frozen  : Brier={frozen['brier_mean']:.4f}+-{frozen['brier_std']:.4f} | ECE={frozen['ece_mean']:.4f}+-{frozen['ece_std']:.4f}")
print(f"  Partial : Brier={partial['brier_mean']:.4f}+-{partial['brier_std']:.4f} | ECE={partial['ece_mean']:.4f}+-{partial['ece_std']:.4f}")

print("\n[NOTE] The 4 new rows use the standard positive-class-probability calibration-curve "
      "convention (confidence=mean(y_prob) per bin, accuracy=mean(y_true) per bin) - documented "
      "here since the original Cell 12 source wasn't available to copy verbatim this session. "
      "The original 9 rows above are untouched. If exact-formula parity with the original 9 "
      "matters for the manuscript text, flag it and I'll cross-check against Cell 12 directly.")
