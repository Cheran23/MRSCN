# ============================================================
# Phase 6 addendum notebook - Cell 6: C1_full_mrscn "true end-to-end"
# latency/FLOPs - live frozen DistilBERT forward pass + head, matching
# step14's measurement scope exactly, for a fair single-fresh-URL
# comparison.
#
# The existing gpu_storage_latency_flops.csv / computational_complexity_
# consolidated.csv rows for C1_full_mrscn are NOT touched - those measure
# the classification head alone (bert_features fed in precomputed), which
# is the correct number if embeddings really are precomputed/cached in a
# separate stage. This cell adds a clearly-separate SUPPLEMENTARY
# measurement - same architecture, but chained with a live frozen
# distilbert-base-uncased forward pass, exactly like step14's benchmark -
# so the two can be compared on equal footing for the "cost of scoring a
# URL the model has never seen before, right now" scenario.
#
# The tokenizer settings used here (MAX_LEN=64,
# distilbert-base-uncased, [CLS] pooling) match step14's verified live
# tokenization procedure. The original cached DistilBERT embeddings used
# by item 11 are no longer available, so exact equivalence with the
# historical embedding-generation configuration cannot be independently
# verified.
# ============================================================
import subprocess
try:
    import thop
except ImportError:
    subprocess.run(['pip', 'install', '-q', 'thop'])
    import thop

import os, json, math, gc, time
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

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BERT_HIDDEN_SIZE = 768
CHAR_VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789-._:/?=&%")
MAX_CHAR_LEN = 200

# ------------------------------------------------------------
# ConfigurableMRSCN - verbatim from the original Cell 16 (the head C1
# actually uses; its own state_dict has no DistilBERT weights, since it
# was trained on precomputed bert_features).
# ------------------------------------------------------------
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

class ConfigurableMRSCN(nn.Module):
    def __init__(self, structured_dim=25, num_classes=2,
                 use_bert=True, use_char=True, use_struct=True, use_scm=True):
        super().__init__()
        self.use_bert, self.use_char, self.use_struct = use_bert, use_char, use_struct
        self.use_scm = use_scm and use_bert and use_char and use_struct
        fusion_dim = 0
        if use_bert:
            self.bert_proj = nn.Sequential(nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE),
                                            nn.LayerNorm(BERT_HIDDEN_SIZE), nn.Dropout(0.2))
            fusion_dim += BERT_HIDDEN_SIZE
        if use_char:
            self.char_embedding = nn.Embedding(len(CHAR_VOCAB) + 1, 32, padding_idx=0)
            self.char_cnn = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
            self.char_fc = nn.Linear(128 * (MAX_CHAR_LEN // 4), 128)
            fusion_dim += 128
        if use_struct:
            self.struct_fc = nn.Sequential(nn.Linear(structured_dim, 64), nn.ReLU(),
                                            nn.Dropout(0.2), nn.Linear(64, 32))
            fusion_dim += 32
        if self.use_scm:
            self.consistency_module = StructuralConsistencyModule(bert_dim=BERT_HIDDEN_SIZE, char_dim=128, struct_dim=32)
            fusion_dim += 3
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes))
    def forward(self, bert_features=None, char_ids=None, features=None):
        parts = []
        bert_feat = char_feat = struct_feat = None
        if self.use_bert:
            bert_feat = self.bert_proj(bert_features); parts.append(bert_feat)
        if self.use_char:
            x = self.char_embedding(char_ids).permute(0, 2, 1)
            x = self.char_cnn(x); x = x.flatten(start_dim=1)
            char_feat = self.char_fc(x); parts.append(char_feat)
        if self.use_struct:
            struct_feat = self.struct_fc(features); parts.append(struct_feat)
        if self.use_scm:
            parts.append(self.consistency_module(bert_feat, char_feat, struct_feat))
        combined = torch.cat(parts, dim=1)
        return self.classifier(combined)

class C1TrueEndToEnd(nn.Module):
    """C1_full_mrscn's real trained head, chained behind a LIVE frozen
    distilbert-base-uncased forward pass - the cost of scoring a URL the
    model has never seen before, right now, with no precomputed embedding
    available. distilbert-base-uncased weights stay frozen/untrained
    throughout (this is still the FROZEN variant, not step14's)."""
    def __init__(self, structured_dim=25):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(MODEL_NAME)
        for p in self.distilbert.parameters():
            p.requires_grad = False
        self.mrscn_head = ConfigurableMRSCN(structured_dim=structured_dim,
                                             use_bert=True, use_char=True, use_struct=True, use_scm=True)
    def forward(self, input_ids, attention_mask, char_ids, features):
        bert_out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        bert_features = bert_out.last_hidden_state[:, 0, :]
        return self.mrscn_head(bert_features=bert_features, char_ids=char_ids, features=features)

# ------------------------------------------------------------
# Build: live frozen DistilBERT + C1's real trained head weights
# ------------------------------------------------------------
model = C1TrueEndToEnd(structured_dim=25)
ckpt_path = f'{ROOT}/checkpoints/step13/C1_full_mrscn/run_1_best_model.pt'
head_state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.mrscn_head.load_state_dict(head_state)
model = model.to(DEVICE)
vocab_size = model.distilbert.config.vocab_size

def make_inputs():
    return (torch.randint(0, vocab_size, (1, MAX_LEN), device=DEVICE, dtype=torch.long),
            torch.ones(1, MAX_LEN, device=DEVICE, dtype=torch.long),
            torch.randint(0, len(CHAR_VOCAB) + 1, (1, MAX_CHAR_LEN), device=DEVICE, dtype=torch.long),
            torch.randn(1, 25, device=DEVICE))

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

print("=== C1_full_mrscn - TRUE end-to-end (live frozen DistilBERT + head) ===")
latency_mean_ms, latency_std_ms = measure_latency_ms(model, make_inputs)
macs, flops, thop_params, flops_error = measure_flops(model, make_inputs)
print(f"  latency_ms={latency_mean_ms:.4f}+-{latency_std_ms:.4f}  "
      + (f"MACs={macs:,.0f}  FLOPs(=2xMACs)={flops:,.0f}  thop_params(incl. frozen DistilBERT)={thop_params:,.0f}"
         if not math.isnan(macs) else f"FLOPs=[FAILED: {flops_error}]"))
del model
gc.collect()
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# Save as a clearly-separate supplementary file - does NOT touch the
# existing gpu_storage_latency_flops.csv / consolidated table.
# ------------------------------------------------------------
out_dir = f'{ROOT}/results/step21_item26'
os.makedirs(out_dir, exist_ok=True)
supp_row = pd.DataFrame([{
    'config_key': 'C1_full_mrscn_true_e2e',
    'label': 'MRSCN (full) - TRUE end-to-end, incl. live frozen DistilBERT forward pass',
    'note': 'SUPPLEMENTARY - not part of the main 13-model table. Measures the same scope as '
            'step14_finetuned_distilbert (live DistilBERT forward pass + head) for a fair '
            'single-fresh-URL comparison. The main table\'s C1_full_mrscn row measures head-only '
            'cost (precomputed bert_features), correct if embeddings are genuinely precomputed/'
            'cached in a separate offline stage.',
    'latency_ms_batch1_TRUE_mean': latency_mean_ms,
    'latency_ms_batch1_TRUE_std': latency_std_ms,
    'macs': macs, 'flops_2x_macs': flops, 'thop_params_incl_frozen_distilbert': thop_params,
}])
supp_path = f'{out_dir}/c1_true_end_to_end_supplementary.csv'
supp_row.to_csv(supp_path, index=False)
print(f"\n[SAVED] {supp_path} (supplementary only - main table untouched)")

# ------------------------------------------------------------
# Fair side-by-side: same measurement scope for both
# ------------------------------------------------------------
consolidated_path = f'{ROOT}/results/step21_item26/computational_complexity_consolidated.csv'
df_main = pd.read_csv(consolidated_path)
step14_row = df_main[df_main['config_key'] == 'step14_finetuned_distilbert'].iloc[0]
c1_head_only_row = df_main[df_main['config_key'] == 'C1_full_mrscn'].iloc[0]

print("\n" + "=" * 100)
print("[FAIR COMPARISON] Same measurement scope (live DistilBERT forward pass + head) for both:")
print("=" * 100)
print(f"{'':30s} {'C1 (frozen, TRUE e2e)':>28s} {'step14 (partial FT)':>22s} {'C1 (head-only, main table)':>28s}")
print(f"{'Latency (ms, batch=1)':30s} {latency_mean_ms:>28.4f} {step14_row['latency_ms_batch1_TRUE_mean']:>22.4f} {c1_head_only_row['latency_ms_batch1_TRUE_mean']:>28.4f}")
print(f"{'FLOPs':30s} {flops:>28,.0f} {step14_row['flops_2x_macs']:>22,.0f} {c1_head_only_row['flops_2x_macs']:>28,.0f}")

ratio_e2e = step14_row['flops_2x_macs'] / flops if flops and not math.isnan(flops) else float('nan')
ratio_headonly = step14_row['flops_2x_macs'] / c1_head_only_row['flops_2x_macs']
print(f"\nstep14 vs C1 FLOPs ratio: {ratio_e2e:.2f}x (TRUE end-to-end basis) vs. {ratio_headonly:.1f}x "
      "(head-only basis, the main table's number)")
print("\n[NOTE] The 'TRUE e2e' FLOPs figure includes the SAME frozen distilbert-base-uncased forward "
      "pass cost for both C1 and step14 - the only difference left is the head architecture "
      "(unchanged, same size) plus whichever of the 6 transformer layers actually differ in learned "
      "weights (forward-pass FLOPs cost is identical whether a layer is frozen or fine-tuned - "
      "freezing only changes backprop, not the forward matmuls). So on a truly fair single-fresh-URL "
      "basis, the FLOPs/latency gap should be far smaller than the head-only table suggests - decide "
      "with your co-authors which framing (head-only vs true e2e) matches how the manuscript "
      "describes MRSCN's deployment/inference pipeline before writing the trade-off paragraph.")
