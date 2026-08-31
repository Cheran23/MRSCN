# configs/config.py
# =============================================================
# MRSCN Framework — Central Configuration
# =============================================================

import os

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH       = os.path.join(BASE_DIR, "datasets", "TB_extracted_features.parquet")
FOLDS_DIR       = os.path.join(BASE_DIR, "folds")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
SEED_REGISTRY   = os.path.join(BASE_DIR, "seed_registry.json")

# ----------------------------------------------------------
# Transformer
# ----------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 64

# ----------------------------------------------------------
# Character CNN
# ----------------------------------------------------------
CHAR_VOCAB    = list("abcdefghijklmnopqrstuvwxyz0123456789-._:/?=&%")
MAX_CHAR_LEN  = 200

# ----------------------------------------------------------
# Training
# ----------------------------------------------------------
BATCH_SIZE          = 64
EPOCHS              = 10
LR                  = 1e-5
PATIENCE            = 2
MIN_DELTA           = 0.0005
GRAD_ACCUM_STEPS    = 2          
AMP_ENABLED         = True       

# ----------------------------------------------------------
# Data Split  (80 / 10 / 10 — fixed, no leakage)
# ----------------------------------------------------------
TEST_SIZE_OUTER  = 0.20   # first split: 80% train, 20% temp
TEST_SIZE_INNER  = 0.50   # second split on temp: 50/50 → 10% val, 10% test

# ----------------------------------------------------------
# Cross-Validation
# ----------------------------------------------------------
N_FOLDS = 5

# ----------------------------------------------------------
# Threshold Optimisation  (validation set ONLY)
# ----------------------------------------------------------
THRESH_MIN  = 0.10
THRESH_MAX  = 0.91
THRESH_STEP = 0.01

# ----------------------------------------------------------
# Shortened URL domains
# ----------------------------------------------------------
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "cutt.ly", "rebrand.ly",
    "shorturl.at", "tiny.cc", "rb.gy", "shorte.st",
}

# ----------------------------------------------------------
# Profiling
# ----------------------------------------------------------
LATENCY_WARMUP_RUNS   = 10
LATENCY_MEASURE_RUNS  = 100
