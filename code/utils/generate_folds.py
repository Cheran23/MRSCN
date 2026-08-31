#!/usr/bin/env python3
# utils/generate_folds.py
# =============================================================
# ONE-TIME SCRIPT: Generate and persist fold indices.
# Run ONCE, commit the outputs to version control.
# Never re-run — doing so would change indices and break
# reproducibility across Colab sessions / accounts.
#
# Usage:
#   python utils/generate_folds.py
# =============================================================

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from configs.config import (
    DATA_PATH, FOLDS_DIR, SEED_REGISTRY,
    N_FOLDS, TEST_SIZE_OUTER, TEST_SIZE_INNER,
)
from utils.seed_utils import load_seed_registry, apply_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to raw CSV (if Parquet not yet created). "
             "Will be converted to Parquet automatically."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing fold files (dangerous — only if you want to reset)."
    )
    args = parser.parse_args()

    os.makedirs(FOLDS_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # Guard: refuse to overwrite unless --force
    # ----------------------------------------------------------
    existing = [f for f in os.listdir(FOLDS_DIR) if f.endswith("_indices.json")]
    if existing and not args.force:
        print(
            f"[GUARD] Fold files already exist in {FOLDS_DIR}.\n"
            "To regenerate (breaks reproducibility), use --force."
        )
        return

    # ----------------------------------------------------------
    # Load global seed
    # ----------------------------------------------------------
    registry = load_seed_registry(SEED_REGISTRY)
    global_seed = registry["global_seed"]
    apply_seed(global_seed)

    # ----------------------------------------------------------
    # Load labels only (memory-safe)
    # ----------------------------------------------------------
    print(f"[INFO] Loading data from {DATA_PATH} ...")
    if DATA_PATH.endswith(".parquet"):
        df = pd.read_parquet(DATA_PATH, columns=["url", "label"])
    else:
        raw = args.csv or DATA_PATH.replace(".parquet", ".csv")
        df = pd.read_csv(raw, usecols=["url", "type"])
        df["label"] = df["type"].apply(
            lambda x: 0 if str(x).lower() == "legitimate" else 1
        )
        df = df[["url", "label"]]

    labels = df["label"].values
    indices = np.arange(len(labels))

    # ----------------------------------------------------------
    # Outer split: 80 % train+val, 20 % temp
    # Then split temp 50/50 → 10 % val, 10 % test
    # This is the HOLDOUT test set — fixed for the entire study.
    # ----------------------------------------------------------
    train_val_idx, temp_idx, _, temp_y = train_test_split(
        indices, labels,
        test_size=TEST_SIZE_OUTER,
        stratify=labels,
        random_state=global_seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=TEST_SIZE_INNER,
        stratify=temp_y,
        random_state=global_seed,
    )

    # Persist holdout (val + test) — same for every fold
    holdout = {
        "val_indices":  val_idx.tolist(),
        "test_indices": test_idx.tolist(),
    }
    holdout_path = os.path.join(FOLDS_DIR, "holdout_indices.json")
    with open(holdout_path, "w") as f:
        json.dump(holdout, f, indent=2)
    print(f"[INFO] Holdout indices saved → {holdout_path}")
    print(f"       Val : {len(val_idx):,}  |  Test: {len(test_idx):,}")

    # ----------------------------------------------------------
    # 5-Fold CV on the 80 % train_val pool
    # ----------------------------------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=global_seed)
    train_val_labels = labels[train_val_idx]

    for fold_id, (rel_train_idx, rel_val_fold_idx) in enumerate(
        skf.split(train_val_idx, train_val_labels), start=1
    ):
        fold_train_idx = train_val_idx[rel_train_idx]
        fold_val_idx   = train_val_idx[rel_val_fold_idx]

        fold_data = {
            "fold_id":       fold_id,
            "train_indices": fold_train_idx.tolist(),
            "val_indices":   fold_val_idx.tolist(),
            # test indices are always the same holdout set
            "test_indices":  test_idx.tolist(),
        }
        path = os.path.join(FOLDS_DIR, f"fold_{fold_id}_indices.json")
        with open(path, "w") as f:
            json.dump(fold_data, f, indent=2)

        print(
            f"[INFO] Fold {fold_id} saved → {path} "
            f"| Train: {len(fold_train_idx):,}  Val: {len(fold_val_idx):,}"
        )

    print("\n[DONE] Fold indices generated and persisted. Commit these files to version control.")


if __name__ == "__main__":
    main()
