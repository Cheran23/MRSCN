# ============================================================
# Phase 6 addendum notebook - Cell 5: Add step14_finetuned_distilbert's
# row to the item 26 consolidated computational-complexity table.
#
# Same column set/semantics as the original Cell 18 consolidation:
#  - trainable_params: real value from results.json (15,997,666 - same
#    every run, same architecture).
#  - total_tree_nodes: N/A (GPU model, not a tree ensemble).
#  - training_time_hours: mean+-SD across the 5 runs' results.json.
#  - storage_mb / macs / flops_2x_macs / latency_ms_batch1_TRUE:
#    from Cell 4's fresh measurement (this addendum notebook).
#  - latency_ms_per_url_BATCHED_throughput_derived: mean across the 5
#    runs' results.json (same "not true single-request latency" caveat
#    Reviewer 4 Comment 2 flagged for every other model in this table).
#  - peak_inference_memory_mb: NOT AVAILABLE - Step 20 never profiled
#    step14 (it didn't exist yet at that point in the revision). Left as
#    NaN rather than fabricated; flagged explicitly below.
#
# Only APPENDS a new row - does not touch or recompute the existing 12.
# ============================================================
import os, json, shutil, time
import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'
RESULTS_DIR = f'{ROOT}/results'
CONSOLIDATED_PATH = f'{RESULTS_DIR}/step21_item26/computational_complexity_consolidated.csv'
GPU_CSV_PATH = f'{RESULTS_DIR}/step21_item26/gpu_storage_latency_flops.csv'

df_final_old = pd.read_csv(CONSOLIDATED_PATH)
if 'step14_finetuned_distilbert' in df_final_old['config_key'].values:
    print("[SKIP] step14_finetuned_distilbert already present in computational_complexity_consolidated.csv.")
else:
    # ------------------------------------------------------------
    # 1. results.json stats across the 5 runs
    # ------------------------------------------------------------
    params_list, time_list, batched_latency_list = [], [], []
    for run_id in range(1, 6):
        with open(f'{RESULTS_DIR}/step14/run_{run_id}_results.json') as f:
            r = json.load(f)
        params_list.append(r['n_trainable_params'])
        time_list.append(r['training_time_hours'])
        batched_latency_list.append(r['latency_ms_per_url'])

    # ------------------------------------------------------------
    # 2. Cell 4's fresh storage/latency/FLOPs measurement
    # ------------------------------------------------------------
    df_gpu = pd.read_csv(GPU_CSV_PATH)
    gpu_row = df_gpu[df_gpu['config_key'] == 'step14_finetuned_distilbert']
    assert len(gpu_row) == 1, "step14 row not found in gpu_storage_latency_flops.csv - run Cell 4 first"
    gpu_row = gpu_row.iloc[0]

    new_row = {
        'config_key': 'step14_finetuned_distilbert',
        'label': 'MRSCN (partial FT DistilBERT)',
        'trainable_params': int(np.mean(params_list)),
        'total_tree_nodes': np.nan,
        'training_time_hours_mean': float(np.mean(time_list)),
        'training_time_hours_std': float(np.std(time_list)),
        'storage_mb': gpu_row['storage_mb'],
        'macs': gpu_row['macs'],
        'flops_2x_macs': gpu_row['flops_2x_macs'],
        'latency_ms_batch1_TRUE_mean': gpu_row['latency_ms_batch1_mean'],
        'latency_ms_batch1_TRUE_std': gpu_row['latency_ms_batch1_std'],
        'latency_ms_per_url_BATCHED_throughput_derived_mean': float(np.mean(batched_latency_list)),
        'peak_inference_memory_mb_mean': np.nan,
        'peak_inference_memory_mb_std': np.nan,
    }

    ts = int(time.time())
    backup_path = CONSOLIDATED_PATH.replace('.csv', f'_before_step14_addendum_{ts}.csv')
    shutil.copyfile(CONSOLIDATED_PATH, backup_path)
    print(f"[BACKUP] {CONSOLIDATED_PATH} -> {backup_path}")

    df_final = pd.concat([df_final_old, pd.DataFrame([new_row])[df_final_old.columns.tolist()]], ignore_index=True)
    df_final.to_csv(CONSOLIDATED_PATH, index=False)
    print(f"[SAVED] {CONSOLIDATED_PATH} ({df_final.shape[0]} rows, was {df_final_old.shape[0]})")
    df_final_old = df_final

print("\n" + "=" * 100)
print("[UPDATED] item 26 consolidated computational-complexity table (13 models):")
print("=" * 100)
print(df_final_old.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

# ------------------------------------------------------------
# Headline: frozen vs. partial FT, side by side
# ------------------------------------------------------------
frozen = df_final_old[df_final_old['config_key'] == 'C1_full_mrscn'].iloc[0]
partial = df_final_old[df_final_old['config_key'] == 'step14_finetuned_distilbert'].iloc[0]
print("\n" + "=" * 100)
print("[HEADLINE] Frozen vs. partially fine-tuned DistilBERT - efficiency trade-off:")
print("=" * 100)
print(f"  Trainable params : frozen={frozen['trainable_params']:,.0f}  |  partial={partial['trainable_params']:,.0f}")
print(f"  Training time (h): frozen={frozen['training_time_hours_mean']:.3f}+-{frozen['training_time_hours_std']:.3f}  |  "
      f"partial={partial['training_time_hours_mean']:.3f}+-{partial['training_time_hours_std']:.3f}")
print(f"  Storage (MB)     : frozen={frozen['storage_mb']:.2f}  |  partial={partial['storage_mb']:.2f}")
print(f"  True batch-1 latency (ms): frozen={frozen['latency_ms_batch1_TRUE_mean']:.4f}  |  "
      f"partial={partial['latency_ms_batch1_TRUE_mean']:.4f}")
print(f"  FLOPs            : frozen={frozen['flops_2x_macs']:,.0f}  |  partial={partial['flops_2x_macs']:,.0f}")

print("\n[NOTE] peak_inference_memory_mb is NaN for step14 - Step 20's memory profiling ran before "
      "this variant existed in this revision cycle, so it was never measured. Flagging rather than "
      "estimating; can be added with a dedicated Step-20-style profiling pass if the manuscript "
      "table needs that column filled in too.")
