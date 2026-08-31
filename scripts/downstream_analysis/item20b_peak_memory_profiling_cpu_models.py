# ============================================================
# STEP 20 - Cell 3: CPU/sklearn peak inference memory profiling
#
# None of these 6 models (LogisticRegression, RandomForest, XGBoost,
# khalife_rf, mohanty_gboost_fst, omolara_adaboost) have a saved checkpoint
# (confirmed by Cell 0), and unlike the GPU neural nets, tree-ensemble memory
# is NOT weight-value-independent - it depends on the actual fitted tree
# structure, which depends on the real training data. So every model here is
# refit on the REAL locked training split with the ORIGINAL hyperparameters
# and feature engineering, copied verbatim from Step 16 Cell 17
# (LR/RF/XGBoost) and Step 17 Cells 21/22/23 (khalife/mohanty/omolara).
#
# Measures process RSS (psutil), not CUDA stats, since these run on CPU.
# Same "clean baseline" fix as the GPU cell: RSS delta is measured relative
# to a per-run baseline taken BEFORE that run's model/data is built, and
# training-only arrays are dropped before the inference measurement so
# training memory isn't counted as inference memory.
#
# Self-contained: loads TB data/splits/seeds directly from Drive rather than
# assuming Cell 1's globals use the same names (Cell 1 was written for the
# GPU/checkpoint pipeline, not this CPU one).
# ============================================================
import subprocess
try:
    import psutil
except ImportError:
    subprocess.run(['pip', 'install', '-q', 'psutil'])
    import psutil
try:
    import tldextract
except ImportError:
    subprocess.run(['pip', 'install', '-q', 'tldextract'])
    import tldextract
try:
    import xgboost as xgb
except ImportError:
    subprocess.run(['pip', 'install', '-q', 'xgboost'])
    import xgboost as xgb

import os, json, time, gc
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------
# Self-contained data load (TB) - independent of Cell 1's globals
# ------------------------------------------------------------
df_cpu = pd.read_csv(f'{ROOT}/data/processed/TB/step2_deduplicated_dataset.csv')
url_col_cpu = 'url' if 'url' in df_cpu.columns else [c for c in df_cpu.columns if 'url' in c.lower()][0]
urls_all_cpu = df_cpu[url_col_cpu].astype(str).values
labels_all_cpu = df_cpu['binary_label'].values.astype(np.int64)

struct_full_cpu = np.load(f'{ROOT}/cache/TB/structured_features.npy')
with open(f'{ROOT}/cache/TB/structured_features_meta.json') as f:
    struct_meta_cpu = json.load(f)

def _find_feature_names(obj, target_len):
    if isinstance(obj, list) and len(obj) == target_len and all(isinstance(x, str) for x in obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_feature_names(v, target_len)
            if found is not None:
                return found
    return None

FEATURE_NAMES_CPU = _find_feature_names(struct_meta_cpu, struct_full_cpu.shape[1])
assert FEATURE_NAMES_CPU is not None, "Could not find a feature-name list in structured_features_meta.json"
struct_df_all_cpu = pd.DataFrame(struct_full_cpu, columns=FEATURE_NAMES_CPU)

with open(f'{ROOT}/locked_configs/run_seeds_locked.json') as f:
    RUN_SEEDS_CPU = json.load(f)['run_seeds']

def load_split_cpu(run_id):
    with open(f'{ROOT}/splits/TB/run_{run_id}_indices.json') as f:
        split = json.load(f)
    return np.array(split['train_indices']), np.array(split['val_indices']), np.array(split['test_indices'])

# Precompute TLD/dot-count features once for the whole dataset (Khalife et
# al.) - matches the original Step 17 Cell 21 methodology, which also
# computed these once and reused them across all 5 runs.
print("Precomputing TLD/dot-count features for Khalife et al. (one-time)...")
_tlds_all_cpu = []
for u in urls_all_cpu:
    ext = tldextract.extract(u)
    _tlds_all_cpu.append(ext.suffix.lower() if ext.suffix else '')
tlds_all_cpu = np.array(_tlds_all_cpu)
dots_all_cpu = np.array([str(u).count('.') for u in urls_all_cpu], dtype=int)

print(f"[OK] CPU profiling data loaded: {len(df_cpu):,} rows, struct={struct_full_cpu.shape}, "
      f"{len(FEATURE_NAMES_CPU)} feature names.")

# ------------------------------------------------------------
# RSS profiling helper
# ------------------------------------------------------------
_process = psutil.Process(os.getpid())

def get_rss_mb():
    return _process.memory_info().rss / (1024 ** 2)

def profile_cpu_inference_memory(predict_fn, n_trials=30, n_warmup=5):
    gc.collect()
    for _ in range(n_warmup):
        _ = predict_fn()
    gc.collect()
    readings_mb = []
    for _ in range(n_trials):
        _ = predict_fn()
        readings_mb.append(get_rss_mb())
    return float(np.mean(readings_mb)), float(np.std(readings_mb))

# ------------------------------------------------------------
# Feature-selection helpers (Mohanty & Acharya, verbatim from Step 17 Cell 22)
# ------------------------------------------------------------
def paper_cfs_filter(X, y, d=0.1):
    selected = []
    for feature in X.columns:
        r = np.corrcoef(X[feature], y)[0, 1]
        if not np.isnan(r) and abs(r) > d:
            selected.append(feature)
    return selected

def t_test_feature_selection(X, y, alpha=0.05):
    n = len(y)
    dof = n - 2
    t_critical = tdist.ppf(1 - alpha / 2, dof)
    selected = []
    for feature in X.columns:
        r = np.corrcoef(X[feature], y)[0, 1]
        if np.isnan(r):
            continue
        t_stat = r * np.sqrt(dof / (1 - r ** 2 + 1e-12))
        if abs(t_stat) > t_critical:
            selected.append(feature)
    return selected

# ------------------------------------------------------------
# Per-model fit + single-sample predict_fn builders. Each returns a
# zero-arg predict_fn closure (mirrors the GPU cell's fwd(m) pattern) that
# performs exactly one single-sample predict_proba() call when invoked.
# All hyperparameters/feature engineering copied verbatim from the original
# training cells (Step 16 Cell 17 for LR/RF/XGBoost; Step 17 Cells 21-23 for
# khalife/mohanty/omolara).
# ------------------------------------------------------------
def fit_logreg(run_id, seed):
    train_idx, val_idx, test_idx = load_split_cpu(run_id)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(struct_full_cpu[train_idx]).astype(np.float32)
    y_train = labels_all_cpu[train_idx]
    X_test_row = scaler.transform(struct_full_cpu[test_idx[:1]]).astype(np.float32)
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)
    model.fit(X_train, y_train)
    del X_train, y_train
    def predict_fn():
        return model.predict_proba(X_test_row)
    return predict_fn

def fit_rf(run_id, seed):
    train_idx, val_idx, test_idx = load_split_cpu(run_id)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(struct_full_cpu[train_idx]).astype(np.float32)
    y_train = labels_all_cpu[train_idx]
    X_test_row = scaler.transform(struct_full_cpu[test_idx[:1]]).astype(np.float32)
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    del X_train, y_train
    def predict_fn():
        return model.predict_proba(X_test_row)
    return predict_fn

def fit_xgb(run_id, seed):
    train_idx, val_idx, test_idx = load_split_cpu(run_id)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(struct_full_cpu[train_idx]).astype(np.float32)
    y_train = labels_all_cpu[train_idx]
    X_test_row = scaler.transform(struct_full_cpu[test_idx[:1]]).astype(np.float32)
    neg, pos = np.bincount(y_train)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               scale_pos_weight=neg / pos, random_state=seed,
                               eval_metric='logloss', n_jobs=-1, tree_method='hist')
    model.fit(X_train, y_train)
    del X_train, y_train
    def predict_fn():
        return model.predict_proba(X_test_row)
    return predict_fn

def fit_khalife_rf(run_id, seed):
    train_idx, val_idx, test_idx = load_split_cpu(run_id)
    train_tlds = set(tlds_all_cpu[train_idx])
    tld_to_id = {tld: i + 1 for i, tld in enumerate(sorted(train_tlds))}
    tld_ids_all = np.array([tld_to_id.get(t, 0) for t in tlds_all_cpu])
    X_all = np.column_stack([tld_ids_all, dots_all_cpu]).astype(float)
    X_train = X_all[train_idx]
    y_train = labels_all_cpu[train_idx]
    X_test_row = X_all[test_idx[:1]]
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    del X_train, y_train, X_all
    def predict_fn():
        return model.predict_proba(X_test_row)
    return predict_fn

def fit_mohanty(run_id, seed):
    train_idx, val_idx, test_idx = load_split_cpu(run_id)
    X_train_df = struct_df_all_cpu.iloc[train_idx].reset_index(drop=True)
    y_train = labels_all_cpu[train_idx]
    cfs_cols = paper_cfs_filter(X_train_df, y_train, d=0.1)
    if not cfs_cols:
        cfs_cols = list(struct_df_all_cpu.columns)
    tt_cols = t_test_feature_selection(X_train_df[cfs_cols], y_train, alpha=0.05)
    final_cols = tt_cols if tt_cols else cfs_cols
    X_train = struct_df_all_cpu.iloc[train_idx][final_cols].values
    X_test_row = struct_df_all_cpu.iloc[test_idx[:1]][final_cols].values
    model = GradientBoostingClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    del X_train, X_train_df
    def predict_fn():
        return model.predict_proba(X_test_row)
    return predict_fn

def fit_omolara(run_id, seed):
    train_idx, val_idx, test_idx = load_split_cpu(run_id)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(struct_full_cpu[train_idx])
    y_train = labels_all_cpu[train_idx]
    X_test_row = scaler.transform(struct_full_cpu[test_idx[:1]])
    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
                                n_estimators=100, algorithm="SAMME", random_state=seed)
    model.fit(X_train, y_train)
    del X_train, y_train
    def predict_fn():
        return model.predict_proba(X_test_row)
    return predict_fn

CPU_MODELS = [
    {'config_key': 'LogisticRegression', 'label': 'Logistic Regression',                'fit_fn': fit_logreg},
    {'config_key': 'RandomForest',       'label': 'Random Forest',                      'fit_fn': fit_rf},
    {'config_key': 'XGBoost',            'label': 'XGBoost',                            'fit_fn': fit_xgb},
    {'config_key': 'khalife_rf',         'label': 'Khalife et al. (RF, TLD+dots)',      'fit_fn': fit_khalife_rf},
    {'config_key': 'mohanty_gboost_fst', 'label': 'Mohanty & Acharya (GBoost+FST)',     'fit_fn': fit_mohanty},
    {'config_key': 'omolara_adaboost',   'label': 'Omolara & Alawida (AdaBoost)',       'fit_fn': fit_omolara},
]

CPU_RESULTS_DIR = f'{ROOT}/results/step20'
os.makedirs(CPU_RESULTS_DIR, exist_ok=True)

cpu_results = []
for m in CPU_MODELS:
    config_key = m['config_key']
    out_path = f'{CPU_RESULTS_DIR}/{config_key}_memory.json'

    print(f"\n=== CPU profiling: {config_key} ({m['label']}) ===")
    per_run_peaks = []
    failed = False
    for run_id_int in range(1, 6):
        run_id = str(run_id_int)
        seed = RUN_SEEDS_CPU[run_id] if run_id in RUN_SEEDS_CPU else RUN_SEEDS_CPU[run_id_int]
        try:
            gc.collect()
            baseline_mb = get_rss_mb()
            predict_fn = m['fit_fn'](run_id_int, seed)
            mean_mb, std_mb = profile_cpu_inference_memory(predict_fn, n_trials=30, n_warmup=5)
            peak_above_baseline = max(mean_mb - baseline_mb, 0.0)
            per_run_peaks.append(peak_above_baseline)
            print(f"  run {run_id_int}: peak={peak_above_baseline:.2f} MB above pre-run baseline "
                  f"(raw RSS={mean_mb:.2f} MB, std over 30 trials={std_mb:.4f})")
            del predict_fn
            gc.collect()
        except Exception as e:
            print(f"  [ERROR] run {run_id_int} failed: {type(e).__name__}: {e}")
            failed = True
            break

    if failed or not per_run_peaks:
        print(f"[FAILED] {config_key} - skipping, will retry on next Cell run.")
        continue

    rec = {
        'config_key': config_key, 'label': m['label'], 'framework': 'sklearn' if config_key != 'XGBoost' else 'xgboost',
        'device': 'cpu', 'has_ckpt': False, 'n_runs': len(per_run_peaks),
        'mean_mb': float(np.mean(per_run_peaks)), 'std_mb': float(np.std(per_run_peaks)),
        'per_run_mb': per_run_peaks,
    }
    with open(out_path, 'w') as f:
        json.dump(rec, f, indent=2)
    cpu_results.append(rec)
    print(f"[DONE] {config_key}: {rec['mean_mb']:.2f} +/- {rec['std_mb']:.2f} MB")

print("\nCPU profiling pass complete.")
print(f"{len(cpu_results)}/{len(CPU_MODELS)} models profiled successfully.")
