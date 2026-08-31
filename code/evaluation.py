# evaluation.py
# =============================================================
# MRSCN Framework — Evaluation Module
# Supports: binary and multiclass classification
# ZERO leakage: threshold is always passed in from validation;
# it is NEVER optimised inside this module.
# =============================================================

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)
from typing import Tuple, Optional


# ----------------------------------------------------------
# Raw probability extraction (no threshold logic here)
# ----------------------------------------------------------
def collect_predictions(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model over loader and return (true_labels, predicted_probs).
    For binary:    probs shape = (N,)   [P(class=1)]
    For multiclass: probs shape = (N, C) [softmax over C classes]
    """
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in loader:
            bert_feat = batch["bert_features"].to(device)
            char_ids  = batch["char_ids"].to(device)
            features  = batch["features"].to(device)
            labels    = batch["label"]

            logits = model(bert_features=bert_feat, char_ids=char_ids, features=features)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

            all_labels.append(labels.numpy())
            all_probs.append(probs)

    true  = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    num_classes = probs.shape[1]
    if num_classes == 2:
        return true, probs[:, 1]          # return P(malicious)
    return true, probs                     # multiclass: full softmax


# ----------------------------------------------------------
# Threshold optimisation — VALIDATION SET ONLY
# ----------------------------------------------------------
def optimise_threshold_on_validation(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    thresh_min: float = 0.10,
    thresh_max: float = 0.91,
    thresh_step: float = 0.01,
) -> Tuple[float, float]:
    """
    Search thresholds on VALIDATION probabilities maximising F1.
    Returns (best_threshold, best_val_f1).

    This function must NEVER be called with test probabilities.
    """
    best_thresh = 0.5
    best_f1     = 0.0

    for t in np.arange(thresh_min, thresh_max, thresh_step):
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = float(round(t, 4))

    return best_thresh, best_f1


# ----------------------------------------------------------
# Binary evaluation
# ----------------------------------------------------------
def evaluate_binary(
    y_true:    np.ndarray,
    probs:     np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute full binary classification metrics.
    `threshold` must be the frozen validation-derived threshold.
    """
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    total = tn + fp + fn + tp
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # sensitivity / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # TNR
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    f1        = f1_score(y_true, preds, zero_division=0)
    macro_f1  = f1_score(y_true, preds, average="macro", zero_division=0)

    try:
        auc_roc = roc_auc_score(y_true, probs)
    except ValueError:
        auc_roc = float("nan")

    try:
        average_precision = average_precision_score(y_true, probs)
    except ValueError:
        average_precision = float("nan")

    mcc = matthews_corrcoef(y_true, preds)
    bal_acc = balanced_accuracy_score(y_true, preds)

    return {
        "accuracy":          float(accuracy),
        "precision":         float(precision),
        "recall":            float(recall),        # sensitivity
        "sensitivity":       float(recall),
        "specificity":       float(specificity),
        "f1":                float(f1),
        "macro_f1":          float(macro_f1),
        "roc_auc":           float(auc_roc),
        # NOTE: this is Average Precision (sklearn's average_precision_score,
        # a step-function summary of the PR curve), NOT a trapezoidal
        # auc(recall, precision). Named "average_precision" rather than the
        # earlier "pr_auc" to match the corrected terminology used throughout
        # the manuscript (Editor Comment 29 / workflow item 27).
        "average_precision": float(average_precision),
        "mcc":               float(mcc),
        "balanced_accuracy": float(bal_acc),
        "fpr":               float(fpr),
        "fnr":               float(fnr),
        "tp":                int(tp),
        "tn":                int(tn),
        "fp":                int(fp),
        "fn":                int(fn),
        "confusion_matrix":  cm.tolist(),
        "threshold":         float(threshold),
    }


# ----------------------------------------------------------
# Multiclass evaluation
# ----------------------------------------------------------
def evaluate_multiclass(
    y_true:     np.ndarray,
    probs:      np.ndarray,
    class_names: Optional[list] = None,
) -> dict:
    """
    Compute multiclass metrics.
    `probs` shape: (N, C) softmax probabilities.
    No threshold needed — argmax is used for predictions.
    """
    preds = np.argmax(probs, axis=1)
    num_classes = probs.shape[1]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    cm       = confusion_matrix(y_true, preds)
    accuracy = accuracy_score(y_true, preds)

    per_class_precision = precision_score(
        y_true, preds, average=None, zero_division=0
    ).tolist()
    per_class_recall = recall_score(
        y_true, preds, average=None, zero_division=0
    ).tolist()
    per_class_f1 = f1_score(
        y_true, preds, average=None, zero_division=0
    ).tolist()

    weighted_precision = precision_score(
        y_true, preds, average="weighted", zero_division=0
    )
    weighted_recall = recall_score(
        y_true, preds, average="weighted", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, preds, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)

    try:
        auc_roc = roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        auc_roc = float("nan")

    mcc     = matthews_corrcoef(y_true, preds)
    bal_acc = balanced_accuracy_score(y_true, preds)

    per_class = {
        class_names[i]: {
            "precision": per_class_precision[i],
            "recall":    per_class_recall[i],
            "f1":        per_class_f1[i],
        }
        for i in range(num_classes)
    }

    return {
        "accuracy":             float(accuracy),
        "macro_f1":             float(macro_f1),
        "weighted_precision":   float(weighted_precision),
        "weighted_recall":      float(weighted_recall),
        "weighted_f1":          float(weighted_f1),
        "roc_auc":              float(auc_roc),
        "mcc":                  float(mcc),
        "balanced_accuracy":    float(bal_acc),
        "confusion_matrix":     cm.tolist(),
        "per_class_metrics":    per_class,
    }
