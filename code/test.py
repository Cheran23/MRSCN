#!/usr/bin/env python3
# test.py
# =============================================================
# MRSCN Framework — Test Evaluation Script
# Usage:
#   python test.py --fold 1
#
# This script:
#   1. Loads the best model for the specified fold
#   2. Loads the frozen validation-derived threshold
#   3. Runs inference on the test set EXACTLY ONCE
#   4. Computes and saves metrics
#
# CRITICAL: No threshold search, no model selection,
#            no hyperparameter tuning occurs here.
#            The test set is touched ONLY in this script.
# =============================================================

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from configs.config import (
    BATCH_SIZE, CHECKPOINTS_DIR, DATA_PATH,
    FOLDS_DIR, N_FOLDS, RESULTS_DIR, SEED_REGISTRY,
)
from evaluation import collect_predictions, evaluate_binary
from models.mrscn import MRSCNModel
from utils.data_utils import (
    build_dataloaders,
    get_feature_cols,
    load_fold_indices,
    prepare_scaler,
)
from utils.seed_utils import apply_seed, get_fold_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_fold(fold_id: int) -> dict:
    # ---- Seed ----
    seed = get_fold_seed(fold_id, SEED_REGISTRY)
    apply_seed(seed)

    # ---- Load training summary for frozen threshold ----
    summary_path = os.path.join(RESULTS_DIR, f"fold_{fold_id}_train_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Train summary not found: {summary_path}\n"
            f"Run train.py --fold {fold_id} first."
        )
    with open(summary_path, "r") as f:
        train_summary = json.load(f)

    # This threshold was derived from the validation set in train.py.
    # It is applied here AS-IS — no re-optimisation on the test set.
    frozen_threshold = train_summary["best_threshold"]
    logger.info(
        f"[Fold {fold_id}] Frozen threshold (from validation): {frozen_threshold:.4f}"
    )

    # ---- Load fold indices ----
    fold_indices = load_fold_indices(FOLDS_DIR, fold_id)

    # ---- Feature columns & scaler (SAME scaler as training) ----
    feature_cols = get_feature_cols(DATA_PATH)
    scaler = prepare_scaler(DATA_PATH, fold_indices["train_indices"], feature_cols)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # ---- Dataloaders ----
    _train_loader_unused, _val_loader_unused, test_loader = build_dataloaders(
        parquet_path  = DATA_PATH,
        fold_indices  = fold_indices,
        feature_cols  = feature_cols,
        tokenizer     = tokenizer,
        scaler        = scaler,
        batch_size    = BATCH_SIZE,
        seed          = seed,
    )

    # ---- Load best model ----
    model = MRSCNModel(
        structured_dim = len(feature_cols),
        num_classes    = 2,
    ).to(DEVICE)

    best_model_path = os.path.join(CHECKPOINTS_DIR, f"fold_{fold_id}_best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model not found: {best_model_path}\n"
            f"Run train.py --fold {fold_id} first."
        )
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()
    logger.info(f"[Fold {fold_id}] Best model loaded.")

    # ==========================================================
    # TEST INFERENCE — happens EXACTLY ONCE per fold
    # ==========================================================
    inference_start = time.time()
    y_test, test_probs = collect_predictions(model, test_loader, DEVICE)
    inference_time_s = time.time() - inference_start

    n_test_samples = len(y_test)
    latency_ms = (inference_time_s / n_test_samples) * 1000.0

    # ---- Apply frozen threshold ----
    metrics = evaluate_binary(y_test, test_probs, frozen_threshold)

    # ---- Full results payload ----
    results = {
        "fold_id":                     fold_id,
        "seed_used":                   seed,
        "best_threshold":              frozen_threshold,
        "n_test_samples":              n_test_samples,
        "latency_ms_per_url":          float(latency_ms),
        "validation_best_f1":          train_summary["best_validation_f1"],
        "training_time_hours":         train_summary["training_time_hours"],
        "test_metrics": {
            "accuracy":          metrics["accuracy"],
            "precision":         metrics["precision"],
            "recall":            metrics["recall"],
            "sensitivity":       metrics["sensitivity"],
            "specificity":       metrics["specificity"],
            "f1":                metrics["f1"],
            "macro_f1":          metrics["macro_f1"],
            "roc_auc":           metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "mcc":               metrics["mcc"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "fpr":               metrics["fpr"],
            "fnr":               metrics["fnr"],
            "tp":                metrics["tp"],
            "tn":                metrics["tn"],
            "fp":                metrics["fp"],
            "fn":                metrics["fn"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
    }

    # ---- Save JSON ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f"fold_{fold_id}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"[Fold {fold_id}] Results (JSON) → {json_path}")

    # ---- Save CSV (publication-ready) ----
    flat = {
        "fold_id":              fold_id,
        "seed_used":            seed,
        "best_threshold":       frozen_threshold,
        "n_test_samples":       n_test_samples,
        "training_time_hours":  train_summary["training_time_hours"],
        "latency_ms_per_url":   float(latency_ms),
        "validation_f1":        train_summary["best_validation_f1"],
    }
    flat.update({f"test_{k}": v for k, v in results["test_metrics"].items()})
    csv_path = os.path.join(RESULTS_DIR, f"fold_{fold_id}_results.csv")
    pd.DataFrame([flat]).to_csv(csv_path, index=False)
    logger.info(f"[Fold {fold_id}] Results (CSV) → {csv_path}")

    return results


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MRSCN on the held-out test set."
    )
    parser.add_argument(
        "--fold", type=int, required=True,
        choices=list(range(1, N_FOLDS + 1)),
        help="Fold index (1–5).",
    )
    args = parser.parse_args()

    results = evaluate_fold(fold_id=args.fold)

    print("\n" + "="*60)
    print(f"FOLD {args.fold} TEST EVALUATION RESULTS")
    m = results["test_metrics"]
    print(f"  Accuracy         : {m['accuracy']:.4f}")
    print(f"  Precision        : {m['precision']:.4f}")
    print(f"  Recall/Sens.     : {m['recall']:.4f}")
    print(f"  Specificity      : {m['specificity']:.4f}")
    print(f"  F1-Score         : {m['f1']:.4f}")
    print(f"  Macro F1         : {m['macro_f1']:.4f}")
    print(f"  ROC-AUC          : {m['roc_auc']:.4f}")
    print(f"  Average Precision: {m['average_precision']:.4f}")
    print(f"  MCC              : {m['mcc']:.4f}")
    print(f"  Balanced Acc.    : {m['balanced_accuracy']:.4f}")
    print(f"  FPR              : {m['fpr']:.4f}")
    print(f"  FNR              : {m['fnr']:.4f}")
    print(f"  TP/TN/FP/FN      : {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")
    print(f"  Latency          : {results['latency_ms_per_url']:.3f} ms/URL")
    print("="*60)
