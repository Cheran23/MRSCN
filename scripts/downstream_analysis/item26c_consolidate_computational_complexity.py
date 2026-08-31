# ============================================================
# PHASE 6 - Cell 18: Item 26 - Consolidate the computational-complexity
# table. Joins: peak inference memory (already computed + saved in Step 20
# - reused as-is, NOT recomputed), trainable params + training time (mean
# +- SD across the 5 runs, straight from each model's results.json - Cell
# 15 confirmed these values are real and plausible), and this Phase's new
# Cells 16/17 measurements (model storage size, TRUE batch=1 forward-pass
# latency, FLOPs/MACs for the 6 GPU models, total_tree_nodes for the 5
# tree-ensemble classical models). Addresses Editor Comments 20, 21,
# Reviewer 4 Comment 2.
#
# Run this AFTER Cells 16 and 17 have both saved their CSVs.
#
# Column-by-column honesty notes (so nothing in the final table implies
# more comparability than actually exists):
#  - trainable_params: real neural-net parameter count for the 6 GPU
#    models and for LogisticRegression (coef_+intercept_); NaN for the 5
#    tree ensembles (see total_tree_nodes instead - NOT the same concept).
#  - flops_2x_macs / macs: only computed for the 6 GPU models (thop). NaN
#    for all 6 classical models - FLOPs isn't a meaningful concept for
#    non-differentiable tree/linear-with-no-forward-matmul-chain models,
#    scoped out deliberately rather than reported as a misleading 0.
#  - latency_ms_batch1_TRUE: this Cell's own fresh single-sample timing
#    (Cells 16/17), NOT the same as results.json's latency_ms_per_url,
#    which was a batch_size=64-averaged, THROUGHPUT-derived number. Both
#    are kept in the table, explicitly labeled, so the mislabeling
#    Reviewer 4 flagged can't recur - the batched figure is now clearly
#    named as such rather than presented as "latency."
# ============================================================
import json
import os
import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'

MODELS = {
    'C1_full_mrscn': ('step13', 'C1_full_mrscn', 'MRSCN (full)'),
    'BiLSTM': ('step16', 'BiLSTM', 'BiLSTM'),
    'trackphish': ('step17', 'trackphish', 'TrackPhish (Kondaiah)'),
    'remya_resmlp': ('step17', 'remya_resmlp', 'ResMLP (Remya)'),
    'he_bertcnn': ('step17', 'he_bertcnn', 'BERT-CNN (He)'),
    'do_tcnmhsa': ('step17', 'do_tcnmhsa', 'TCN-Attention (Do)'),
    'LogisticRegression': ('step16', 'LogisticRegression', 'Logistic Regression'),
    'RandomForest': ('step16', 'RandomForest', 'Random Forest'),
    'XGBoost': ('step16', 'XGBoost', 'XGBoost'),
    'khalife_rf': ('step17', 'khalife_rf', 'Khalife et al. (RF, TLD+dots)'),
    'mohanty_gboost_fst': ('step17', 'mohanty_gboost_fst', 'Mohanty & Acharya (GBoost+FST)'),
    'omolara_adaboost': ('step17', 'omolara_adaboost', 'Omolara & Alawida (AdaBoost)'),
}

# ------------------------------------------------------------
# 1. trainable_params + training_time_hours (mean+-SD across 5 runs) +
#    the OLD batched-throughput latency_ms_per_url, straight from
#    results.json - kept and clearly relabeled, not discarded.
# ------------------------------------------------------------
per_model_json_stats = []
for config_key, (step, key, label) in MODELS.items():
    params_list, time_list, batched_latency_list = [], [], []
    for run_id in range(1, 6):
        fp = f'{ROOT}/results/{step}/{key}/run_{run_id}_results.json'
        with open(fp) as f:
            r = json.load(f)
        if r.get('n_trainable_params') is not None:
            params_list.append(r['n_trainable_params'])
        time_list.append(r['training_time_hours'])
        batched_latency_list.append(r['latency_ms_per_url'])
    per_model_json_stats.append({
        'config_key': config_key, 'label': label,
        'trainable_params_results_json': int(np.mean(params_list)) if params_list else np.nan,
        'training_time_hours_mean': float(np.mean(time_list)),
        'training_time_hours_std': float(np.std(time_list)),
        'latency_ms_per_url_BATCHED_throughput_derived_mean': float(np.mean(batched_latency_list)),
    })
df_json_stats = pd.DataFrame(per_model_json_stats)

# ------------------------------------------------------------
# 2. Peak inference memory - reused as-is from Step 20 (NOT recomputed)
# ------------------------------------------------------------
df_memory = pd.read_csv(f'{ROOT}/results/step20/step20_master_summary.csv')
df_memory = df_memory[['config_key', 'mean_peak_mb', 'std_peak_mb']].rename(
    columns={'mean_peak_mb': 'peak_inference_memory_mb_mean', 'std_peak_mb': 'peak_inference_memory_mb_std'})

# ------------------------------------------------------------
# 3. This Phase's new measurements (Cells 16 + 17)
# ------------------------------------------------------------
df_gpu = pd.read_csv(f'{ROOT}/results/step21_item26/gpu_storage_latency_flops.csv')
df_gpu = df_gpu.rename(columns={'latency_ms_batch1_mean': 'latency_ms_batch1_TRUE_mean',
                                 'latency_ms_batch1_std': 'latency_ms_batch1_TRUE_std'})
df_gpu['total_tree_nodes'] = np.nan
df_gpu = df_gpu[['config_key', 'storage_mb', 'latency_ms_batch1_TRUE_mean', 'latency_ms_batch1_TRUE_std',
                  'macs', 'flops_2x_macs', 'total_tree_nodes']]

df_cpu = pd.read_csv(f'{ROOT}/results/step21_item26/cpu_storage_latency_complexity.csv')
df_cpu = df_cpu.rename(columns={'latency_ms_batch1_mean': 'latency_ms_batch1_TRUE_mean',
                                 'latency_ms_batch1_std': 'latency_ms_batch1_TRUE_std'})
df_cpu['macs'] = np.nan
df_cpu['flops_2x_macs'] = np.nan
# Cell 17's own trainable_params (only non-null for LogisticRegression) takes
# priority over results.json's (which is null for all 6 classical models
# anyway, per Cell 15) - reconciled in the merge step below.
df_cpu_trainable = df_cpu[['config_key', 'trainable_params']].rename(
    columns={'trainable_params': 'trainable_params_cell17'})
df_cpu = df_cpu[['config_key', 'storage_mb', 'latency_ms_batch1_TRUE_mean', 'latency_ms_batch1_TRUE_std',
                  'macs', 'flops_2x_macs', 'total_tree_nodes']]

df_new_measurements = pd.concat([df_gpu, df_cpu], ignore_index=True)

# ------------------------------------------------------------
# Merge everything
# ------------------------------------------------------------
df_final = (df_json_stats
            .merge(df_memory, on='config_key', how='left')
            .merge(df_new_measurements, on='config_key', how='left')
            .merge(df_cpu_trainable, on='config_key', how='left'))

df_final['trainable_params'] = df_final['trainable_params_cell17'].combine_first(
    df_final['trainable_params_results_json'])
df_final = df_final.drop(columns=['trainable_params_cell17', 'trainable_params_results_json'])

FINAL_COLS = [
    'config_key', 'label', 'trainable_params', 'total_tree_nodes',
    'training_time_hours_mean', 'training_time_hours_std',
    'storage_mb', 'macs', 'flops_2x_macs',
    'latency_ms_batch1_TRUE_mean', 'latency_ms_batch1_TRUE_std',
    'latency_ms_per_url_BATCHED_throughput_derived_mean',
    'peak_inference_memory_mb_mean', 'peak_inference_memory_mb_std',
]
df_final = df_final[FINAL_COLS]

print("[ITEM 26] Consolidated computational-complexity table (all 12 MRSCN + baseline models):")
print(df_final.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

out_dir = f'{ROOT}/results/step21_item26'
os.makedirs(out_dir, exist_ok=True)
df_final.to_csv(f'{out_dir}/computational_complexity_consolidated.csv', index=False)
print(f"\nSaved: {out_dir}/computational_complexity_consolidated.csv")

print("\n[NOTE] latency_ms_batch1_TRUE_* is the correctly-labeled 'forward-pass, not end-to-end' "
      "latency (Reviewer 4 Comment 2) - true single-sample timing. "
      "latency_ms_per_url_BATCHED_throughput_derived_mean is kept alongside it, explicitly "
      "relabeled, for transparency about what the original per-run measurement actually was.")

df_final
