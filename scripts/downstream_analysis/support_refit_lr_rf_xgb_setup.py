# ============================================================
# LR/RF/XGBoost refit notebook - Cell 1: Setup
# Loads the TB dataset, structured-feature cache, locked splits, and locked
# seeds. Self-contained, matches the loading pattern used throughout every
# earlier Phase/Step this revision.
# ============================================================
import json
import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'

df = pd.read_csv(f'{ROOT}/data/processed/TB/step2_deduplicated_dataset.csv')
binary_label = df['binary_label'].values.astype(np.int64)

struct_full = np.load(f'{ROOT}/cache/TB/structured_features.npy')
with open(f'{ROOT}/cache/TB/structured_features_meta.json') as f:
    struct_meta = json.load(f)

RUN_SPLITS = {}
for run_id in range(1, 6):
    with open(f'{ROOT}/splits/TB/run_{run_id}_indices.json') as f:
        RUN_SPLITS[str(run_id)] = json.load(f)

with open(f'{ROOT}/locked_configs/run_seeds_locked.json') as f:
    RUN_SEEDS = json.load(f)['run_seeds']

print(f"[OK] {len(df):,} rows, struct_features={struct_full.shape}, "
      f"{len(RUN_SPLITS)} run splits, {len(RUN_SEEDS)} seeds loaded.")
