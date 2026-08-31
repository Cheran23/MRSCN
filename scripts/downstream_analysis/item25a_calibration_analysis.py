# ============================================================
# PHASE 6 - Cell 12: Item 25 - Calibration analysis
# Brier score + Expected Calibration Error (10 equal-width bins) from raw
# pre-threshold probabilities, per run, mean+-SD across the 5 runs.
# Addresses Editor Comment 27.
#
# Covers every model with real saved y_prob: MRSCN (step13), BiLSTM
# (step16), and all 7 Step 17 baselines (khalife_rf, mohanty_gboost_fst,
# omolara_adaboost, trackphish, remya_resmlp, he_bertcnn, do_tcnmhsa) -
# confirmed by directly reading Step 17's save_predictions() source
# (test_idx, y_true, y_prob, y_pred - same schema Cell 1 used to
# regenerate MRSCN/BiLSTM's). LogisticRegression/RandomForest/XGBoost are
# excluded, same reason as every other Phase 6 item: no checkpoint, no
# saved probabilities, deferred to Phase 7 item 27's refit.
# ============================================================
import json
import os
import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'

MODELS = {
    'MRSCN (C1_full_mrscn)': ('step13', 'C1_full_mrscn'),
    'BiLSTM': ('step16', 'BiLSTM'),
    'BERT-CNN': ('step17', 'he_bertcnn'),
    'TCN-Attention': ('step17', 'do_tcnmhsa'),
    'TrackPhish': ('step17', 'trackphish'),
    'ResMLP': ('step17', 'remya_resmlp'),
    'Khalife-RF': ('step17', 'khalife_rf'),
    'Mohanty-GBoost': ('step17', 'mohanty_gboost_fst'),
    'Omolara-AdaBoost': ('step17', 'omolara_adaboost'),
}

def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))

def ece_score(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    bin_details = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            bin_details.append({'bin_lo': lo, 'bin_hi': hi, 'n': 0,
                                 'confidence': float('nan'), 'accuracy': float('nan')})
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (n_bin / n) * abs(accuracy - confidence)
        bin_details.append({'bin_lo': lo, 'bin_hi': hi, 'n': n_bin,
                             'confidence': confidence, 'accuracy': accuracy})
    return float(ece), bin_details

per_run_rows = []
skipped = []
for model_label, (step, key) in MODELS.items():
    for run_id in range(1, 6):
        fp = f'{ROOT}/results/{step}/{key}/run_{run_id}_predictions.npz'
        if not os.path.exists(fp):
            skipped.append(f'{model_label} run {run_id}: file not found ({fp})')
            continue
        pred = np.load(fp)
        if 'y_prob' not in pred.files:
            skipped.append(f'{model_label} run {run_id}: no y_prob key in npz - cannot compute calibration.')
            continue
        y_true = pred['y_true'].astype(np.float64)
        y_prob = pred['y_prob'].astype(np.float64)
        assert y_prob.min() >= 0.0 and y_prob.max() <= 1.0, \
            f"[FATAL] {model_label} run {run_id}: y_prob outside [0,1] - not a valid probability."

        brier = brier_score(y_true, y_prob)
        ece, _ = ece_score(y_true, y_prob, n_bins=10)
        per_run_rows.append({'model': model_label, 'run_id': run_id, 'n_test': len(y_true),
                              'brier_score': brier, 'ece_10bin': ece})

df_calib_per_run = pd.DataFrame(per_run_rows)
print("[ITEM 25] Per-run Brier score + ECE (10 equal-width bins):")
print(df_calib_per_run.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if skipped:
    print(f"\n[SKIPPED] {len(skipped)} (model, run) pairs had no usable y_prob:")
    for s in skipped:
        print(f"  - {s}")

df_calib_summary = (df_calib_per_run.groupby('model')
                     .agg(brier_mean=('brier_score', 'mean'), brier_std=('brier_score', 'std'),
                          ece_mean=('ece_10bin', 'mean'), ece_std=('ece_10bin', 'std'),
                          n_runs=('run_id', 'count'))
                     .reset_index()
                     .sort_values('brier_mean'))
print("\n[SUMMARY] Mean +- SD across runs, sorted by Brier score (lower = better calibrated):")
print(df_calib_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
out_dir = f'{ROOT}/results/step21_item25'
os.makedirs(out_dir, exist_ok=True)
df_calib_per_run.to_csv(f'{out_dir}/calibration_per_run.csv', index=False)
df_calib_summary.to_csv(f'{out_dir}/calibration_summary.csv', index=False)
print(f"\nSaved: {out_dir}/calibration_per_run.csv")
print(f"Saved: {out_dir}/calibration_summary.csv")

df_calib_summary
