# utils/seed_utils.py
# =============================================================
# Deterministic seed management — loaded from seed_registry.json
# NEVER generate seeds dynamically at runtime.
# =============================================================

import json
import os
import random
import numpy as np
import torch


def load_seed_registry(registry_path: str) -> dict:
    """Load the immutable seed registry from disk."""
    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"seed_registry.json not found at {registry_path}. "
            "Generate it once with utils/generate_folds.py and commit it to version control."
        )
    with open(registry_path, "r") as f:
        registry = json.load(f)
    return registry


def get_fold_seed(fold_id: int, registry_path: str) -> int:
    """
    Return the deterministic seed assigned to fold_id (1-indexed).
    Raises KeyError if the fold is not registered.
    """
    registry = load_seed_registry(registry_path)
    key = f"fold_{fold_id}"
    if key not in registry:
        raise KeyError(
            f"Fold '{key}' not found in seed registry. "
            "Valid folds are fold_1 … fold_5."
        )
    return registry[key]


def apply_seed(seed: int, deterministic: bool = True) -> None:
    """
    Apply seed to every random subsystem.
    Must be called BEFORE:
      - model initialisation
      - dataloader creation
      - dataset splitting
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Best-effort determinism; may reduce performance on some ops
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch versions may not support this
            pass
