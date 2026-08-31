# utils/data_utils.py

import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from configs.config import BATCH_SIZE, CHAR_VOCAB
from utils.url_utils import encode_chars

char2idx = {c: i + 1 for i, c in enumerate(CHAR_VOCAB)}

_BERT_FEATURES = None

def get_bert_features(path):
    global _BERT_FEATURES
    if _BERT_FEATURES is None:
        print("[INFO] Loading BERT feature cache...")
        _BERT_FEATURES = np.load(path)
        print(f"[INFO] BERT features loaded: {_BERT_FEATURES.shape}")
    return _BERT_FEATURES


class URLDataset(Dataset):
    def __init__(self, parquet_path, indices, feature_cols, scaler=None):
        self.indices = np.asarray(indices, dtype=np.int64)

        bert_path  = os.path.join(os.path.dirname(parquet_path), "bert_features.npy")
        bert_feats = get_bert_features(bert_path)
        self.bert_features = bert_feats[self.indices]

        needed_cols = ["url", "label"] + feature_cols
        df = pd.read_parquet(parquet_path, columns=needed_cols)
        df = df.iloc[self.indices].reset_index(drop=True)

        self.urls   = df["url"].astype(str).tolist()
        self.labels = df["label"].values.astype(np.int64)

        raw_feats = df[feature_cols].values.astype(np.float32)
        if scaler is not None:
            raw_feats = scaler.transform(raw_feats).astype(np.float32)
        self.features = raw_feats

        self.char_ids = np.array([encode_chars(u) for u in self.urls], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "bert_features": torch.tensor(self.bert_features[idx], dtype=torch.float32),
            "char_ids":      torch.tensor(self.char_ids[idx],      dtype=torch.long),
            "features":      torch.tensor(self.features[idx],      dtype=torch.float32),
            "label":         torch.tensor(self.labels[idx],        dtype=torch.long),
        }


def load_fold_indices(folds_dir, fold_id):
    path = os.path.join(folds_dir, f"fold_{fold_id}_indices.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fold index file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_holdout_indices(folds_dir):
    path = os.path.join(folds_dir, "holdout_indices.json")
    with open(path, "r") as f:
        return json.load(f)


def get_feature_cols(parquet_path):
    df_sample = pd.read_parquet(parquet_path).head(1)
    exclude = {"no", "url", "type", "label"}
    return [
        c for c in df_sample.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df_sample[c])
    ]


def build_dataloaders(parquet_path, fold_indices, feature_cols, tokenizer,
                      scaler, batch_size=BATCH_SIZE, num_workers=0, seed=42):
    def _make_loader(indices, shuffle):
        ds = URLDataset(parquet_path, indices, feature_cols, scaler)
        g  = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            generator=g if shuffle else None,
        )

    train_loader = _make_loader(fold_indices["train_indices"], shuffle=True)
    val_loader   = _make_loader(fold_indices["val_indices"],   shuffle=False)
    test_loader  = _make_loader(fold_indices["test_indices"],  shuffle=False)
    return train_loader, val_loader, test_loader


def prepare_scaler(parquet_path, train_indices, feature_cols):
    from sklearn.preprocessing import StandardScaler
    df = pd.read_parquet(parquet_path, columns=feature_cols)
    train_df = df.iloc[train_indices]
    scaler   = StandardScaler()
    scaler.fit(train_df.values)
    return scaler
