# ============================================================
# Phase 6 addendum notebook - Cell 4: Add step14_finetuned_distilbert to
# item 26's GPU computational-complexity measurements
#
# Needed to give the "frozen vs. partial fine-tune" trade-off paragraph
# real numbers on the cost side, not just the accuracy/calibration side
# already covered by items 23/25. Same measurement functions as the
# original Cell 16 (storage via state_dict->BytesIO, TRUE batch=1 latency
# with cuda synchronize + warmup, FLOPs/MACs via thop) - copied verbatim,
# just pointed at the step14 architecture/checkpoint instead.
# ============================================================
import subprocess
try:
    import thop
except ImportError:
    subprocess.run(['pip', 'install', '-q', 'thop'])
    import thop

import os, json, io, math, gc, time, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertModel
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {DEVICE}")

with open(f'{ROOT}/locked_configs/run_seeds_locked.json') as f:
    RUN_SEEDS = json.load(f)['run_seeds']

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BERT_HIDDEN_SIZE = 768
CHAR_VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789-._:/?=&%")
MAX_CHAR_LEN = 200

class StructuralConsistencyModule(nn.Module):
    def __init__(self, bert_dim, char_dim, struct_dim):
        super().__init__()
        self.proj_bert   = nn.Linear(bert_dim,   128)
        self.proj_char   = nn.Linear(char_dim,   128)
        self.proj_struct = nn.Linear(struct_dim, 128)
    def forward(self, bert_feat, char_feat, struct_feat):
        b = self.proj_bert(bert_feat); c = self.proj_char(char_feat); s = self.proj_struct(struct_feat)
        bc = torch.cosine_similarity(b, c, dim=1)
        bs = torch.cosine_similarity(b, s, dim=1)
        cs = torch.cosine_similarity(c, s, dim=1)
        return torch.stack([bc, bs, cs], dim=1)

class PartiallyFineTunedMRSCN(nn.Module):
    """Verbatim from MRSCN_Phase_3.ipynb Step 14 / the DistilBERT gap
    notebook's Cell 2 - reproduced here again so this cell is self-contained."""
    def __init__(self, structured_dim=25, num_classes=2, n_unfrozen_layers=2):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(MODEL_NAME)
        n_layers = len(self.distilbert.transformer.layer)
        freeze_until = n_layers - n_unfrozen_layers
        for p in self.distilbert.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.distilbert.transformer.layer):
            grad_flag = i >= freeze_until
            for p in layer.parameters():
                p.requires_grad = grad_flag

        self.bert_proj = nn.Sequential(nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE),
                                        nn.LayerNorm(BERT_HIDDEN_SIZE), nn.Dropout(0.2))
        self.char_embedding = nn.Embedding(len(CHAR_VOCAB) + 1, 32, padding_idx=0)
        self.char_cnn = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.char_fc = nn.Linear(128 * (MAX_CHAR_LEN // 4), 128)
        self.struct_fc = nn.Sequential(nn.Linear(structured_dim, 64), nn.ReLU(),
                                        nn.Dropout(0.2), nn.Linear(64, 32))
        self.consistency_module = StructuralConsistencyModule(bert_dim=BERT_HIDDEN_SIZE, char_dim=128, struct_dim=32)
        fusion_dim = BERT_HIDDEN_SIZE + 128 + 32 + 3
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes))

    def forward(self, input_ids, attention_mask, char_ids, features):
        bert_out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_out.last_hidden_state[:, 0, :]
        bert_feat = self.bert_proj(cls_token)
        x = self.char_embedding(char_ids).permute(0, 2, 1)
        x = self.char_cnn(x); x = x.flatten(start_dim=1)
        char_feat = self.char_fc(x)
        struct_feat = self.struct_fc(features)
        consistency_feat = self.consistency_module(bert_feat, char_feat, struct_feat)
        combined = torch.cat([bert_feat, char_feat, struct_feat, consistency_feat], dim=1)
        return self.classifier(combined)

def build_step14_partial_ft(run_id, seed):
    torch.manual_seed(seed)
    model = PartiallyFineTunedMRSCN(structured_dim=25)
    ckpt_path = f'{ROOT}/checkpoints/step14/run_{run_id}_best_model.pt'
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    vocab_size = model.distilbert.config.vocab_size
    def make_inputs():
        return (torch.randint(0, vocab_size, (1, MAX_LEN), device=DEVICE, dtype=torch.long),
                torch.ones(1, MAX_LEN, device=DEVICE, dtype=torch.long),
                torch.randint(0, len(CHAR_VOCAB) + 1, (1, MAX_CHAR_LEN), device=DEVICE, dtype=torch.long),
                torch.randn(1, 25, device=DEVICE))
    return model, make_inputs

# ------------------------------------------------------------
# Measurement helpers - identical to the original Cell 16
# ------------------------------------------------------------
def measure_storage_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 ** 2)

def measure_latency_ms(model, make_inputs, n_trials=50, n_warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            inputs = make_inputs()
            _ = model(*inputs)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        times_ms = []
        for _ in range(n_trials):
            inputs = make_inputs()
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(*inputs)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
    return float(np.mean(times_ms)), float(np.std(times_ms))

def measure_flops(model, make_inputs):
    try:
        inputs = make_inputs()
        macs, params = thop.profile(model, inputs=inputs, verbose=False)
        return float(macs), float(macs * 2), float(params), None
    except Exception as e:
        return float('nan'), float('nan'), float('nan'), f"{type(e).__name__}: {e}"

# ------------------------------------------------------------
# Measure (run 1's checkpoint/seed, same convention as the original 6)
# ------------------------------------------------------------
print("=== step14_finetuned_distilbert (MRSCN partial FT DistilBERT) ===")
gc.collect()
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()
seed = RUN_SEEDS['1'] if '1' in RUN_SEEDS else RUN_SEEDS[1]
model, make_inputs = build_step14_partial_ft(1, seed)

storage_mb = measure_storage_mb(model)
latency_mean_ms, latency_std_ms = measure_latency_ms(model, make_inputs)
macs, flops, thop_params, flops_error = measure_flops(model, make_inputs)

print(f"  storage_mb={storage_mb:.3f}  latency_ms={latency_mean_ms:.4f}+-{latency_std_ms:.4f}  "
      + (f"MACs={macs:,.0f}  FLOPs(=2xMACs)={flops:,.0f}  thop_params={thop_params:,.0f}"
         if not math.isnan(macs) else f"FLOPs=[FAILED: {flops_error}]"))

new_row = {
    'config_key': 'step14_finetuned_distilbert', 'label': 'MRSCN (partial FT DistilBERT)',
    'device': 'cuda' if DEVICE.type == 'cuda' else 'cpu',
    'storage_mb': storage_mb, 'latency_ms_batch1_mean': latency_mean_ms,
    'latency_ms_batch1_std': latency_std_ms, 'macs': macs, 'flops_2x_macs': flops,
    'thop_params': thop_params, 'flops_error': flops_error,
}
del model
gc.collect()
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# Append to the existing gpu_storage_latency_flops.csv (backup first)
# ------------------------------------------------------------
gpu_csv_path = f'{ROOT}/results/step21_item26/gpu_storage_latency_flops.csv'
df_old = pd.read_csv(gpu_csv_path)
if 'step14_finetuned_distilbert' in df_old['config_key'].values:
    print("\n[SKIP] step14_finetuned_distilbert already present in gpu_storage_latency_flops.csv - not duplicating.")
    df_final = df_old
else:
    ts = int(time.time())
    backup_path = gpu_csv_path.replace('.csv', f'_before_step14_addendum_{ts}.csv')
    shutil.copyfile(gpu_csv_path, backup_path)
    print(f"\n[BACKUP] {gpu_csv_path} -> {backup_path}")
    df_final = pd.concat([df_old, pd.DataFrame([new_row])[df_old.columns.tolist()]], ignore_index=True)
    df_final.to_csv(gpu_csv_path, index=False)
    print(f"[SAVED] {gpu_csv_path} ({df_final.shape[0]} rows, was {df_old.shape[0]})")

print("\n" + "=" * 100)
print(df_final.to_string(index=False))

# ------------------------------------------------------------
# Sanity: compare storage_mb against the checkpoint file actually on disk
# ------------------------------------------------------------
real_ckpt_path = f'{ROOT}/checkpoints/step14/run_1_best_model.pt'
if os.path.exists(real_ckpt_path):
    real_size_mb = os.path.getsize(real_ckpt_path) / (1024 ** 2)
    print(f"\n[CROSS-CHECK] step14 run_1 real checkpoint file = {real_size_mb:.2f} MB vs. this cell's "
          f"state_dict measurement = {storage_mb:.2f} MB "
          f"({'MATCH' if abs(real_size_mb - storage_mb) < 5 else 'MISMATCH - investigate'})")
