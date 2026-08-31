# ============================================================
# PHASE 6 - Cell 17: Item 26 (classical/CPU models) - model storage size,
# TRUE single-sample latency, and a tree/coefficient complexity metric
# (FLOPs doesn't apply to non-differentiable tree ensembles the way it
# does to neural nets - see Cell 14's scoping note).
#
# Fit logic copied VERBATIM from Step 20 Cell 3 (same real hyperparameters
# and feature engineering already used for the confirmed CPU memory
# numbers), refit here since none of these 6 models have a persisted
# model file (same reason Step 20 Cell 3 had to refit for memory).
#
# "trainable_params" is only a meaningful concept for LogisticRegression
# (coef_ + intercept_, a real parameter count). For the 5 tree ensembles
# (RandomForest, XGBoost, khalife_rf, mohanty_gboost_fst,
# omolara_adaboost), that column is left NaN and a separate
# 'total_tree_nodes' column is reported instead - explicitly NOT
# comparable to a neural net's parameter count or to LogisticRegression's,
# just the closest real structural-complexity analogue tree ensembles have.
# ============================================================
import subprocess
try:
    import xgboost as xgb
except ImportError:
    subprocess.run(['pip', 'install', '-q', 'xgboost'])
    import xgboost as xgb
try:
    import tldextract
except ImportError:
    subprocess.run(['pip', 'install', '-q', 'tldextract'])
    import tldextract

import os, json, pickle, time, gc
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'

# ------------------------------------------------------------
# Self-contained data load (same as Step 20 Cell 3)
# ------------------------------------------------------------
df_cpu = pd.read_csv(f'{ROOT}/data/processed/TB/step2_deduplicated_dataset.csv')
urls_all_cpu = df_cpu['url'].astype(str).values
labels_all_cpu = df_cpu['binary_label'].values.astype(np.int64)
struct_full_cpu = np.load(f'{ROOT}/cache/TB/structured_features.npy')
with open(f'{ROOT}/cache/TB/structured_features_meta.json') as f:
    struct_meta_cpu = json.load(f)
FEATURE_NAMES_CPU = struct_meta_cpu['feature_order']
struct_df_all_cpu = pd.DataFrame(struct_full_cpu, columns=FEATURE_NAMES_CPU)

with open(f'{ROOT}/locked_configs/run_seeds_locked.json') as f:
    RUN_SEEDS_CPU = json.load(f)['run_seeds']

def load_split_cpu(run_id):
    with open(f'{ROOT}/splits/TB/run_{run_id}_indices.json') as f:
        split = json.load(f)
    return np.array(split['train_indices']), np.array(split['val_indices']), np.array(split['test_indices'])

print("Precomputing TLD/dot-count features for Khalife et al. (one-time)...")
_tlds_all_cpu = [tldextract.extract(u).suffix.lower() if tldextract.extract(u).suffix else '' for u in urls_all_cpu]
tlds_all_cpu = np.array(_tlds_all_cpu)
dots_all_cpu = np.array([str(u).count('.') for u in urls_all_cpu], dtype=int)
print(f"[OK] CPU data loaded: {len(df_cpu):,} rows.")

def paper_cfs_filter(X, y, d=0.1):
    selected = []
    for feature in X.columns:
        r = np.corrcoef(X[feature], y)[0, 1]
        if not np.isnan(r) and abs(r) > d:
            selected.append(feature)
    return selected

def t_test_feature_selection(X, y, alpha=0.05):
    n = len(y); dof = n - 2
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
# Fit functions -> (model, X_test_row)  [run 1, seed 1 - same as GPU cell]
# ------------------------------------------------------------
def fit_logreg(run_id, seed):
    train_idx, val_idx, test_idx = load_split_cpu(run_id)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(struct_full_cpu[train_idx]).astype(np.float32)
    y_train = labels_all_cpu[train_idx]
    X_test_row = scaler.transform(struct_full_cpu[test_idx[:1]]).astype(np.float32)
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)
    model.fit(X_train, y_train)
    return model, X_test_row

def fit_rf(run_id, seed):
    train_idx, val_idx, test_idx = load_split_cpu(run_id)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(struct_full_cpu[train_idx]).astype(np.float32)
    y_train = labels_all_cpu[train_idx]
    X_test_row = scaler.transform(struct_full_cpu[test_idx[:1]]).astype(np.float32)
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_test_row

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
    return model, X_test_row

def fit_khalife_rf(run_id, seed):
    train_idx, val_idx, test_idx = load_split_cpu(run_id)
    train_tlds = set(tlds_all_cpu[train_idx])
    tld_to_id = {tld: i + 1 for i, tld in enumerate(sorted(train_tlds))}
    tld_ids_all = np.array([tld_to_id.get(t, 0) for t in tlds_all_cpu])
    X_all = np.column_stack([tld_ids_all, dots_all_cpu]).astype(float)
    X_train = X_all[train_idx]; y_train = labels_all_cpu[train_idx]
    X_test_row = X_all[test_idx[:1]]
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    return model, X_test_row

def fit_mohanty(run_id, seed):
    train_idx, val_idx, test_idx = load_split_cpu(run_id)
    X_train_df = struct_df_all_cpu.iloc[train_idx].reset_index(drop=True)
    y_train = labels_all_cpu[train_idx]
    cfs_cols = paper_cfs_filter(X_train_df, y_train, d=0.1) or list(struct_df_all_cpu.columns)
    tt_cols = t_test_feature_selection(X_train_df[cfs_cols], y_train, alpha=0.05)
    final_cols = tt_cols if tt_cols else cfs_cols
    X_train = struct_df_all_cpu.iloc[train_idx][final_cols].values
    X_test_row = struct_df_all_cpu.iloc[test_idx[:1]][final_cols].values
    model = GradientBoostingClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    return model, X_test_row

def fit_omolara(run_id, seed):
    train_idx, val_idx, test_idx = load_split_cpu(run_id)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(struct_full_cpu[train_idx])
    y_train = labels_all_cpu[train_idx]
    X_test_row = scaler.transform(struct_full_cpu[test_idx[:1]])
    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
                                n_estimators=100, algorithm="SAMME", random_state=seed)
    model.fit(X_train, y_train)
    return model, X_test_row

CPU_MODELS = [
    {'config_key': 'LogisticRegression', 'label': 'Logistic Regression',            'fit_fn': fit_logreg},
    {'config_key': 'RandomForest',       'label': 'Random Forest',                  'fit_fn': fit_rf},
    {'config_key': 'XGBoost',            'label': 'XGBoost',                        'fit_fn': fit_xgb},
    {'config_key': 'khalife_rf',         'label': 'Khalife et al. (RF, TLD+dots)',  'fit_fn': fit_khalife_rf},
    {'config_key': 'mohanty_gboost_fst', 'label': 'Mohanty & Acharya (GBoost+FST)', 'fit_fn': fit_mohanty},
    {'config_key': 'omolara_adaboost',   'label': 'Omolara & Alawida (AdaBoost)',   'fit_fn': fit_omolara},
]

def total_tree_nodes(model, config_key):
    if config_key in ('RandomForest', 'khalife_rf'):
        return int(sum(t.tree_.node_count for t in model.estimators_))
    if config_key == 'XGBoost':
        return int(len(model.get_booster().trees_to_dataframe()))
    if config_key == 'mohanty_gboost_fst':
        return int(sum(t.tree_.node_count for stage in model.estimators_ for t in stage))
    if config_key == 'omolara_adaboost':
        return int(sum(t.tree_.node_count for t in model.estimators_))
    return None

def measure_latency_ms(predict_fn, n_trials=50, n_warmup=10):
    for _ in range(n_warmup):
        _ = predict_fn()
    times_ms = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = predict_fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return float(np.mean(times_ms)), float(np.std(times_ms))

rows = []
for m in CPU_MODELS:
    config_key = m['config_key']
    print(f"\n=== {config_key} ({m['label']}) ===")
    gc.collect()
    seed = RUN_SEEDS_CPU['1'] if '1' in RUN_SEEDS_CPU else RUN_SEEDS_CPU[1]
    model, X_test_row = m['fit_fn'](1, seed)

    storage_mb = len(pickle.dumps(model)) / (1024 ** 2)
    predict_fn = lambda: model.predict_proba(X_test_row)
    latency_mean_ms, latency_std_ms = measure_latency_ms(predict_fn)

    trainable_params = None
    if config_key == 'LogisticRegression':
        trainable_params = int(model.coef_.size + model.intercept_.size)
    node_count = total_tree_nodes(model, config_key)

    print(f"  storage_mb={storage_mb:.4f}  latency_ms={latency_mean_ms:.4f}+-{latency_std_ms:.4f}  "
          f"trainable_params={trainable_params}  total_tree_nodes={node_count}")

    rows.append({
        'config_key': config_key, 'label': m['label'], 'device': 'cpu',
        'storage_mb': storage_mb, 'latency_ms_batch1_mean': latency_mean_ms,
        'latency_ms_batch1_std': latency_std_ms, 'trainable_params': trainable_params,
        'total_tree_nodes': node_count,
    })
    del model
    gc.collect()

df_cpu_complexity = pd.DataFrame(rows)
print("\n[SUMMARY]")
print(df_cpu_complexity.to_string(index=False))

out_dir = f'{ROOT}/results/step21_item26'
os.makedirs(out_dir, exist_ok=True)
df_cpu_complexity.to_csv(f'{out_dir}/cpu_storage_latency_complexity.csv', index=False)
print(f"\nSaved: {out_dir}/cpu_storage_latency_complexity.csv")

df_cpu_complexity
