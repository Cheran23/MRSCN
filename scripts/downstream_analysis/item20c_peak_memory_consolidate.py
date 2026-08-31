# ============================================================
# STEP 20 - Cell 4: Combined summary + download
# Scans results/step20/*_memory.json directly from disk (rather than
# trusting in-memory gpu_results/cpu_results), so this works correctly
# even if Cells 2/3 were run in separate sessions or re-run out of order.
# ============================================================
import glob, json, os, zipfile
import pandas as pd
from google.colab import files

STEP20_RESULTS_DIR = f'{ROOT}/results/step20'
json_files = sorted(glob.glob(f'{STEP20_RESULTS_DIR}/*_memory.json'))
print(f"Found {len(json_files)} result files under {STEP20_RESULTS_DIR}")

records = []
for fp in json_files:
    with open(fp) as f:
        d = json.load(f)
    records.append({
        'config_key': d.get('config_key'),
        'label': d.get('label'),
        'framework': d.get('framework'),
        'device': d.get('device'),
        'has_ckpt': d.get('has_ckpt'),
        'n_runs': d.get('n_runs'),
        'mean_peak_mb': d.get('mean_mb'),
        'std_peak_mb': d.get('std_mb'),
    })

df_step20 = pd.DataFrame(records)

EXPECTED_GPU = ['C1_full_mrscn', 'BiLSTM', 'trackphish', 'remya_resmlp', 'he_bertcnn', 'do_tcnmhsa']
EXPECTED_CPU = ['LogisticRegression', 'RandomForest', 'XGBoost', 'khalife_rf', 'mohanty_gboost_fst', 'omolara_adaboost']
missing = [k for k in EXPECTED_GPU + EXPECTED_CPU if k not in df_step20['config_key'].unique()]
if missing:
    print(f"[WARNING] Missing results for: {missing}")
else:
    print(f"[OK] All {len(EXPECTED_GPU) + len(EXPECTED_CPU)} models present.")

print("\n=== GPU models (torch.cuda peak allocated, MB) ===")
print(df_step20[df_step20['device'] == 'cuda'][['config_key', 'label', 'mean_peak_mb', 'std_peak_mb', 'n_runs']]
      .sort_values('mean_peak_mb').to_string(index=False))

print("\n=== CPU models (process RSS delta, MB) ===")
print(df_step20[df_step20['device'] == 'cpu'][['config_key', 'label', 'mean_peak_mb', 'std_peak_mb', 'n_runs']]
      .sort_values('mean_peak_mb').to_string(index=False))

print("\n[NOTE] GPU figures are torch.cuda peak-allocated-memory readings (weights + single-sample "
      "activations); CPU figures are process RSS deltas measured via psutil (fitted model + "
      "single-sample inference, above a clean per-run baseline). These are two different "
      "measurement instruments on two different scales - for Table 26, present them as two "
      "sub-tables (GPU baselines vs. CPU baselines), not as one directly-sortable combined ranking.")

summary_path = f'{STEP20_RESULTS_DIR}/step20_master_summary.csv'
df_step20.to_csv(summary_path, index=False)
print(f"\nSaved: {summary_path}")

zip_path = f'{STEP20_RESULTS_DIR}/step20_all_results.zip'
with zipfile.ZipFile(zip_path, 'w') as zf:
    for fp in json_files:
        zf.write(fp, os.path.relpath(fp, ROOT))
    zf.write(summary_path, 'step20_master_summary.csv')
print(f"Saved: {zip_path}")

files.download(summary_path)
files.download(zip_path)

df_step20
