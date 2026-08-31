#!/usr/bin/env python3
# train.py
# =============================================================
# MRSCN Framework — Training Script
# Usage:
#   python train.py --fold 1
#   python train.py --fold 2 --resume
#
# Data usage policy (STRICT):
#   Training set  → model parameter updates ONLY
#   Validation set → early stopping, threshold optimisation
#   Test set       → NEVER touched here; only in test.py
# =============================================================

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoTokenizer, DistilBertModel, DistilBertModel, DistilBertModel, DistilBertModel, DistilBertModel, DistilBertModel, DistilBertModel

from configs.config import (
    AMP_ENABLED, BATCH_SIZE, CHECKPOINTS_DIR, DATA_PATH,
    EPOCHS, FOLDS_DIR, GRAD_ACCUM_STEPS, LR, MIN_DELTA,
    MODEL_NAME, N_FOLDS, PATIENCE, RESULTS_DIR, SEED_REGISTRY,
    THRESH_MAX, THRESH_MIN, THRESH_STEP,
)
from evaluation import (
    collect_predictions,
    evaluate_binary,
    optimise_threshold_on_validation,
)
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


# ----------------------------------------------------------
# Checkpoint helpers
# ----------------------------------------------------------
def checkpoint_path(fold_id: int) -> str:
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    return os.path.join(CHECKPOINTS_DIR, f"fold_{fold_id}_checkpoint.pt")


def save_checkpoint(state: dict, fold_id: int) -> None:
    path = checkpoint_path(fold_id)
    torch.save(state, path)
    logger.info(f"[Checkpoint] Saved → {path}")


def load_checkpoint(fold_id: int) -> dict | None:
    path = checkpoint_path(fold_id)
    if os.path.exists(path):
        logger.info(f"[Resume] Loading checkpoint from {path}")
        return torch.load(path, map_location=DEVICE)
    return None


# ----------------------------------------------------------
# Training loop
# ----------------------------------------------------------
def train_one_fold(fold_id: int, resume: bool = False) -> dict:
    # ---- Seed (fold-specific, loaded from registry) ----
    seed = get_fold_seed(fold_id, SEED_REGISTRY)
    apply_seed(seed)
    logger.info(f"[Fold {fold_id}] Seed={seed}  Device={DEVICE}")

    # ---- Load fold indices (pre-generated, immutable) ----
    fold_indices = load_fold_indices(FOLDS_DIR, fold_id)

    # ---- Feature columns & scaler (fit on train ONLY) ----
    feature_cols = get_feature_cols(DATA_PATH)
    scaler = prepare_scaler(DATA_PATH, fold_indices["train_indices"], feature_cols)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ---- Dataloaders ----
    # NOTE: test_loader is returned but intentionally unused in this script.
    train_loader, val_loader, _test_loader_unused = build_dataloaders(
        parquet_path  = DATA_PATH,
        fold_indices  = fold_indices,
        feature_cols  = feature_cols,
        tokenizer     = tokenizer,
        scaler        = scaler,
        batch_size    = BATCH_SIZE,
        seed          = seed,
    )

    # ---- Model ----
    model = MRSCNModel(
        structured_dim = len(feature_cols),
        num_classes    = 2,
    ).to(DEVICE)

    # ---- Class-weighted loss ----
    # Compute class weights from training labels only
    train_labels = np.array(
        [fold_indices["train_indices"][i] for i in range(len(fold_indices["train_indices"]))]
    )
    import pandas as pd
    df_labels = pd.read_parquet(DATA_PATH, columns=["label"])
    y_train = df_labels["label"].iloc[fold_indices["train_indices"]].values
    class_counts = np.bincount(y_train)
    weights = torch.tensor(
        (1.0 / class_counts) / (1.0 / class_counts).sum(),
        dtype=torch.float,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ---- Optimiser & scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    scaler_amp = GradScaler(enabled=AMP_ENABLED)

    # ---- State ----
    start_epoch      = 0
    best_val_f1      = 0.0
    best_threshold   = 0.5
    patience_counter = 0
    training_start   = time.time()

    # ---- Resume if requested ----
    if resume:
        ckpt = load_checkpoint(fold_id)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scaler_amp.load_state_dict(ckpt["amp_scaler_state_dict"])
            start_epoch      = ckpt["epoch"]
            best_val_f1      = ckpt["best_validation_score"]
            best_threshold   = ckpt["best_threshold"]
            patience_counter = ckpt["patience_counter"]
            logger.info(
                f"[Resume] Resumed from epoch {start_epoch}, "
                f"best_val_f1={best_val_f1:.4f}, threshold={best_threshold:.4f}"
            )

    # ==========================================================
    # TRAINING LOOP
    # ==========================================================
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Fold {fold_id} | Epoch {epoch+1}")):
            bert_feat = batch["bert_features"].to(DEVICE)
            char_ids  = batch["char_ids"].to(DEVICE)
            features  = batch["features"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            with autocast(enabled=AMP_ENABLED):
                logits = model(bert_features=bert_feat, char_ids=char_ids, features=features)
                loss   = criterion(logits, labels) / GRAD_ACCUM_STEPS

            scaler_amp.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS

        avg_loss = total_loss / len(train_loader)

        # ==========================================================
        # VALIDATION — used for early stopping & threshold search
        # NEVER use test set here.
        # ==========================================================
        y_val, val_probs = collect_predictions(model, val_loader, DEVICE)

        # Threshold optimised on VALIDATION set only
        epoch_thresh, epoch_val_f1 = optimise_threshold_on_validation(
            y_val, val_probs,
            thresh_min  = THRESH_MIN,
            thresh_max  = THRESH_MAX,
            thresh_step = THRESH_STEP,
        )

        val_metrics = evaluate_binary(y_val, val_probs, epoch_thresh)
        scheduler.step(epoch_val_f1)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"[Fold {fold_id}] Epoch {epoch+1}/{EPOCHS} | "
            f"Loss={avg_loss:.4f} | Val-F1={epoch_val_f1:.4f} | "
            f"Thresh={epoch_thresh:.3f} | AUC={val_metrics['roc_auc']:.4f} | "
            f"Elapsed={epoch_time:.0f}s"
        )

        # ---- Model selection (validation metric only) ----
        if epoch_val_f1 > best_val_f1 + MIN_DELTA:
            best_val_f1    = epoch_val_f1
            best_threshold = epoch_thresh
            patience_counter = 0

            # Save best model weights
            best_model_path = os.path.join(
                CHECKPOINTS_DIR, f"fold_{fold_id}_best_model.pt"
            )
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"[Fold {fold_id}] New best model saved (val_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(
                f"[Fold {fold_id}] No improvement ({patience_counter}/{PATIENCE})"
            )

        # ---- Checkpoint (safe resume) ----
        save_checkpoint(
            {
                "epoch":                epoch + 1,
                "fold_id":              fold_id,
                "seed":                 seed,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "amp_scaler_state_dict": scaler_amp.state_dict(),
                "best_threshold":       best_threshold,
                "best_validation_score": best_val_f1,
                "patience_counter":     patience_counter,
            },
            fold_id,
        )

        # ---- Early stopping ----
        if patience_counter >= PATIENCE:
            logger.info(f"[Fold {fold_id}] Early stopping at epoch {epoch+1}")
            break

    training_hours = (time.time() - training_start) / 3600.0
    logger.info(
        f"[Fold {fold_id}] Training complete | "
        f"Best Val-F1={best_val_f1:.4f} | "
        f"Frozen Threshold={best_threshold:.4f} | "
        f"Time={training_hours:.3f}h"
    )

    # ==========================================================
    # Persist frozen threshold and validation summary
    # (test evaluation happens in test.py ONLY)
    # ==========================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)
    val_summary = {
        "fold_id":                fold_id,
        "seed_used":              seed,
        "best_threshold":         best_threshold,
        "best_validation_f1":     best_val_f1,
        "training_time_hours":    training_hours,
        "status":                 "training_complete_threshold_frozen",
        "note": (
            "Test evaluation has NOT been run. "
            "Execute test.py --fold X to apply the frozen threshold "
            "to the held-out test set."
        ),
    }
    summary_path = os.path.join(RESULTS_DIR, f"fold_{fold_id}_train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(val_summary, f, indent=2)
    logger.info(f"[Fold {fold_id}] Train summary → {summary_path}")

    return val_summary


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MRSCN — one fold at a time (distributed-safe)"
    )
    parser.add_argument(
        "--fold", type=int, required=True,
        choices=list(range(1, N_FOLDS + 1)),
        help="Fold index (1–5). Each fold is independently executable.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint for this fold.",
    )
    args = parser.parse_args()

    logger.info(f"Starting training | Fold={args.fold} | Resume={args.resume}")
    logger.info(f"Device: {DEVICE}")

    result = train_one_fold(fold_id=args.fold, resume=args.resume)

    print("\n" + "="*60)
    print(f"FOLD {args.fold} TRAINING COMPLETE")
    print(f"  Best Val F1   : {result['best_validation_f1']:.4f}")
    print(f"  Frozen Thresh : {result['best_threshold']:.4f}")
    print(f"  Training Time : {result['training_time_hours']:.3f} hours")
    print("="*60)
    print("Next step: python test.py --fold", args.fold)
