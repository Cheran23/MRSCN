# ============================================================
# LR/RF/XGBoost refit notebook - Cell 2: Refit + regenerate predictions
#
# These 3 models were the ones Phase 6 Cell 0 classified [BLOCKED] - no
# checkpoint, no saved predictions - deferred until Phase 7's ROC/PR curves
# actually needed them. Hyperparameters and feature engineering copied
# VERBATIM from Step 16 Cell 17 (the same source used for Step 20 Cell 3's
# CPU memory profiling, and re-verified there against the real
# results.json values). Unlike a checkpoint-based regeneration, sklearn
# has no saved weights to reload - refitting on the exact same locked
# train split + seed IS the reproduction method here, validated the same
# way as every other regenerated-prediction cell this phase: an F1
# cross-check against the original locked results.json, loud on mismatch,
# not silently trusted.
# ============================================================
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import subprocess
try:
    import xgboost as xgb
except ImportError:
    subprocess.run(['pip', 'install', '-q', 'xgboost'])
    import xgboost as xgb

def save_predictions_npz(step, config_key, run_id, test_idx, y_true, y_prob, y_pred):
    out_dir = f'{ROOT}/results/{step}/{config_key}'
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f'{out_dir}/run_{run_id}_predictions.npz',
                         test_idx=test_idx, y_true=y_true, y_prob=y_prob, y_pred=y_pred)

def fit_predict_logreg(train_idx, test_idx, seed):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(struct_full[train_idx]).astype(np.float32)
    y_train = binary_label[train_idx]
    X_test = scaler.transform(struct_full[test_idx]).astype(np.float32)
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]

def fit_predict_rf(train_idx, test_idx, seed):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(struct_full[train_idx]).astype(np.float32)
    y_train = binary_label[train_idx]
    X_test = scaler.transform(struct_full[test_idx]).astype(np.float32)
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]

def fit_predict_xgb(train_idx, test_idx, seed):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(struct_full[train_idx]).astype(np.float32)
    y_train = binary_label[train_idx]
    X_test = scaler.transform(struct_full[test_idx]).astype(np.float32)
    neg, pos = np.bincount(y_train)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               scale_pos_weight=neg / pos, random_state=seed,
                               eval_metric='logloss', n_jobs=-1, tree_method='hist')
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]

MODELS = {
    'LogisticRegression': fit_predict_logreg,
    'RandomForest': fit_predict_rf,
    'XGBoost': fit_predict_xgb,
}

mismatch_warnings = []
for config_key, fit_predict_fn in MODELS.items():
    print(f"\n=== Refitting: {config_key} (step16) ===")
    for run_id in range(1, 6):
        result_path = f'{ROOT}/results/step16/{config_key}/run_{run_id}_results.json'
        import json
        with open(result_path) as f:
            original = json.load(f)
        locked_threshold = original['best_threshold']
        original_f1 = original['test_metrics']['f1']

        split = RUN_SPLITS[str(run_id)]
        train_idx = np.array(split['train_indices'])
        test_idx = np.array(split['test_indices'])
        seed = RUN_SEEDS[str(run_id)] if str(run_id) in RUN_SEEDS else RUN_SEEDS[run_id]

        y_test = binary_label[test_idx]
        y_prob = fit_predict_fn(train_idx, test_idx, seed)
        y_pred = (y_prob >= locked_threshold).astype(int)
        regenerated_f1 = f1_score(y_test, y_pred, zero_division=0)

        match = abs(regenerated_f1 - original_f1) < 1e-3
        flag = "OK" if match else "MISMATCH"
        if not match:
            mismatch_warnings.append(f"{config_key} run {run_id}: original F1={original_f1:.4f}, "
                                      f"regenerated F1={regenerated_f1:.4f}")

        save_predictions_npz('step16', config_key, run_id, test_idx, y_test, y_prob, y_pred)
        print(f"  run {run_id}: original F1={original_f1:.4f} | regenerated F1={regenerated_f1:.4f} [{flag}]")

print("\n" + "=" * 80)
if mismatch_warnings:
    print(f"[WARNING] {len(mismatch_warnings)} run(s) did NOT reproduce the original F1 within tolerance:")
    for w in mismatch_warnings:
        print(f"  - {w}")
    print("Investigate before trusting these regenerated predictions.")
else:
    print("[OK] All 15 regenerated F1 scores (3 configs x 5 runs) match the original locked "
          "results.json within 1e-3. Predictions are trustworthy.")
print("=" * 80)
