# ============================================================
# PHASE 6 - Cell 16: Item 26 (GPU models) - model storage size, TRUE
# single-sample forward-pass latency, and FLOPs/MACs.
#
# Architectures/build_fn's copied VERBATIM from Step 20 Cell 2 (already
# validated - same source that produced the confirmed peak-memory numbers).
# Storage size is measured the same weight-value-independent way as memory
# was (a float32 tensor takes the same bytes whether trained or not), via
# torch.save(state_dict) to an in-memory buffer - no dependence on the 3
# missing checkpoints (remya_resmlp/he_bertcnn/do_tcnmhsa), confirmed by
# Cell 14. TrackPhish's real checkpoint (172.90 MB on disk, per Cell 14)
# is used as a live cross-check against this method's own measurement.
#
# Latency is TRUE batch=1 timing (torch.cuda.synchronize() before/after,
# GPU warm-up passes, repeated trials) - this is the "forward-pass, not
# end-to-end" latency Reviewer 4 Comment 2 asked for, distinct from
# results.json's latency_ms_per_url (which was averaged over a
# batch_size=64 DataLoader loop and so is a THROUGHPUT-derived number, not
# true single-request latency).
#
# FLOPs/MACs via thop (installed here - Cell 14 confirmed it's not
# preinstalled). thop's hooks are unreliable for nn.LSTM (BiLSTM) and for
# HuggingFace BertModel's internals (He-BERTCNN) - both are wrapped in
# try/except and reported as N/A with a note if thop fails or the count
# looks obviously wrong, rather than silently trusting a bad number.
# ============================================================
import subprocess
try:
    import thop
except ImportError:
    subprocess.run(['pip', 'install', '-q', 'thop'])
    import thop

import os, json, io, math, gc, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {DEVICE}")

with open(f'{ROOT}/locked_configs/run_seeds_locked.json') as f:
    RUN_SEEDS = json.load(f)['run_seeds']

# ============================================================
# Architectures (verbatim from Step 20 Cell 2 / original notebook source)
# ============================================================
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

class ConfigurableMRSCN(nn.Module):
    def __init__(self, structured_dim=25, num_classes=2,
                 use_bert=True, use_char=True, use_struct=True, use_scm=True):
        super().__init__()
        assert use_bert or use_char or use_struct
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

class BiLSTMBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                             bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout * 0.67),
            nn.Linear(32, num_classes))
    def forward(self, char_ids):
        x = self.embedding(char_ids)
        _, (h_n, _) = self.lstm(x)
        feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(feat)

PROJ_DIM_TP = 128
MAX_LEN_TP = 80

class FusionAttention(nn.Module):
    def __init__(self, input_dims, proj_dim=128):
        super().__init__()
        self.sources = list(input_dims.keys())
        self.proj_layers = nn.ModuleDict({s: nn.Linear(input_dims[s], proj_dim) for s in self.sources})
        self.attn_score = nn.Sequential(nn.Linear(proj_dim, proj_dim // 2), nn.ReLU(), nn.Linear(proj_dim // 2, 1))
    def forward(self, emb_dict):
        projected = [self.proj_layers[s](emb_dict[s]).unsqueeze(2) for s in self.sources]
        stacked = torch.cat(projected, dim=2)
        B, L, S, P = stacked.size()
        scores = self.attn_score(stacked.view(B * L * S, P)).view(B, L, S)
        alpha = torch.softmax(scores, dim=-1)
        fused = (stacked * alpha.unsqueeze(-1)).sum(dim=2)
        return fused, alpha

class TrackPhish1DCNN(nn.Module):
    def __init__(self, proj_dim=128, num_classes=2, conv_filters=[128, 128, 128], kernel_sizes=[2, 3, 4], dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(proj_dim, conv_filters[i], kernel_sizes[i]) for i in range(len(kernel_sizes))])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(sum(conv_filters), 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_outs = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        feats = torch.cat(conv_outs, dim=1)
        feats = self.dropout(feats)
        return self.fc(feats)

class TrackPhishModel(nn.Module):
    def __init__(self, emb_matrices, proj_dim=128):
        super().__init__()
        self.sources = list(emb_matrices.keys())
        self.embeddings = nn.ModuleDict()
        input_dims = {}
        for s, mat in emb_matrices.items():
            emb = nn.Embedding.from_pretrained(torch.tensor(mat, dtype=torch.float32), freeze=True, padding_idx=0)
            self.embeddings[s] = emb
            input_dims[s] = mat.shape[1]
        self.fusion = FusionAttention(input_dims, proj_dim=proj_dim)
        self.cnn = TrackPhish1DCNN(proj_dim=proj_dim)
    def forward(self, id_seq):
        emb_dict = {s: self.embeddings[s](id_seq) for s in self.sources}
        fused, _ = self.fusion(emb_dict)
        return self.cnn(fused)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, expansion_factor, out_channels, stride=1):
        super().__init__()
        exp_channels = int(in_channels * expansion_factor)
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, 1, bias=False),
            nn.BatchNorm2d(exp_channels), nn.ReLU(inplace=True))
        self.depthwise = nn.Sequential(
            nn.Conv2d(exp_channels, exp_channels, 3, stride=stride, padding=1, groups=exp_channels, bias=False),
            nn.Conv2d(exp_channels, exp_channels, 1, bias=False),
            nn.BatchNorm2d(exp_channels), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))
    def forward(self, x):
        out = self.expand(x); out = self.depthwise(out); out = self.project(out)
        if self.use_residual:
            out = out + x
        return out

class ResMLPNet(nn.Module):
    def __init__(self, in_channels=1, n_blocks=7):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[InvertedResidualBlock(64, expansion_factor=2, out_channels=64) for _ in range(n_blocks)])
        self.conv_final = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False),
                                         nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d((2, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 3, 128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, 2)
    def forward(self, x):
        x = self.stem(x); x = self.blocks(x); x = self.conv_final(x); x = self.pool(x); x = self.flatten(x)
        x = F.relu(self.fc1(x)); x = self.drop1(x)
        x = F.relu(self.fc2(x)); x = self.drop2(x)
        return self.out(x)

from transformers import BertConfig, BertModel

class BertCNNClassifier(nn.Module):
    def __init__(self, config, cnn_filters=128, kernel_size=3, pool_size=4):
        super().__init__()
        self.bert = BertModel(config)
        hidden = config.hidden_size
        self.conv = nn.Conv1d(hidden, cnn_filters, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)
        pooled_len = math.floor(FINAL_LEN_HE / pool_size)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(cnn_filters * pooled_len, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, ids, mask):
        x = self.bert(ids, attention_mask=mask, return_dict=True).last_hidden_state
        x = x.transpose(1, 2); x = self.conv(x); x = self.relu(x); x = self.pool(x)
        logits = self.fc(x)
        return logits.view(-1)

BASE_LEN_HE = 128
EXT_K_HE = 10
FINAL_LEN_HE = BASE_LEN_HE + EXT_K_HE

MAX_URL_LEN_TCN = 350
EMBED_SIZE_TCN = 32
TCN_FILTERS = 16
TCN_KERNEL = 3
TCN_DILATIONS = (1, 2, 4, 8)
MHSA_HEADS = 2
MHSA_KEY_DIM = 2
DROPOUT_TCN = 0.5
WORD_VOCAB_SIZE_CAP = 20000

def build_char_map():
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    cmap = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for c in chars:
        if c not in cmap:
            cmap[c] = idx
            idx += 1
    return cmap

CHAR_MAP_TCN = build_char_map()
CHAR_VOCAB_SIZE_TCN = max(CHAR_MAP_TCN.values()) + 1

class TCNBlock(nn.Module):
    def __init__(self, in_channels, filters=16, kernel_size=3, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilations = dilations
        convs, bns, projs = [], [], []
        c_in = in_channels
        for d in dilations:
            convs.append(nn.Conv1d(c_in, filters, kernel_size, dilation=d, padding=0, bias=True))
            bns.append(nn.BatchNorm1d(filters))
            projs.append(nn.Conv1d(c_in, filters, 1) if c_in != filters else nn.Identity())
            c_in = filters
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.projs = nn.ModuleList(projs)
    def forward(self, x):
        for d, conv, bn, proj in zip(self.dilations, self.convs, self.bns, self.projs):
            pad = (self.kernel_size - 1) * d
            x_causal = F.pad(x, (pad, 0))
            conv_out = F.relu(bn(conv(x_causal)))
            residual = proj(x)
            x = F.relu(residual + conv_out)
        return x

class SmallMHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, key_dim):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        inner_dim = num_heads * key_dim
        self.q_proj = nn.Linear(embed_dim, inner_dim)
        self.k_proj = nn.Linear(embed_dim, inner_dim)
        self.v_proj = nn.Linear(embed_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, embed_dim)
    def forward(self, x):
        B, T, E = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.key_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.key_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.key_dim)
        return self.out_proj(out)

class TCNMHSANet(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, embed_size=EMBED_SIZE_TCN, maxlen=MAX_URL_LEN_TCN,
                 tcn_filters=TCN_FILTERS, mhsa_heads=MHSA_HEADS, mhsa_key_dim=MHSA_KEY_DIM, dropout=DROPOUT_TCN):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, embed_size, padding_idx=0)
        self.word_emb = nn.Embedding(word_vocab_size, embed_size, padding_idx=0)
        self.tcn = TCNBlock(embed_size * 2, filters=tcn_filters, kernel_size=TCN_KERNEL, dilations=TCN_DILATIONS)
        self.mhsa = SmallMHSA(tcn_filters, mhsa_heads, mhsa_key_dim)
        self.ln = nn.LayerNorm(tcn_filters)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(tcn_filters, 64)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
    def forward(self, char_ids, word_ids):
        c = self.char_emb(char_ids); w = self.word_emb(word_ids)
        seq = torch.cat([c, w], dim=-1).transpose(1, 2)
        tcn_out = self.tcn(seq).transpose(1, 2)
        attn = self.mhsa(tcn_out)
        merged = self.ln(tcn_out + attn)
        pooled = merged.mean(dim=1)
        x = self.drop1(pooled)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x).squeeze(-1)

# ============================================================
# Build_fn's -> (model, forward_fn, inputs_tuple_fn)
# inputs_tuple_fn() returns a fresh positional-args tuple matching
# forward()'s signature, for thop.profile().
# ============================================================
def build_c1_full_mrscn(run_id, seed):
    torch.manual_seed(seed)
    model = ConfigurableMRSCN(structured_dim=25, use_bert=True, use_char=True, use_struct=True, use_scm=True)
    ckpt_path = f'{ROOT}/checkpoints/step13/C1_full_mrscn/run_{run_id}_best_model.pt'
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    def make_inputs():
        return (torch.randn(1, BERT_HIDDEN_SIZE, device=DEVICE),
                torch.randint(0, len(CHAR_VOCAB) + 1, (1, MAX_CHAR_LEN), device=DEVICE, dtype=torch.long),
                torch.randn(1, 25, device=DEVICE))
    return model, make_inputs

def build_bilstm(run_id, seed):
    torch.manual_seed(seed)
    model = BiLSTMBaseline(vocab_size=len(CHAR_VOCAB) + 1)
    ckpt_path = f'{ROOT}/checkpoints/step16/BiLSTM/run_{run_id}_best_model.pt'
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    def make_inputs():
        return (torch.randint(0, len(CHAR_VOCAB) + 1, (1, MAX_CHAR_LEN), device=DEVICE, dtype=torch.long),)
    return model, make_inputs

def build_trackphish(run_id, seed):
    torch.manual_seed(seed)
    ckpt_path = f'{ROOT}/checkpoints/step17/trackphish/run_{run_id}_best_model.pt'
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    vocab_size = state['embeddings.w2v.weight'].shape[0]
    embed_dim = state['embeddings.w2v.weight'].shape[1]
    dummy_mats = {s: np.zeros((vocab_size, embed_dim), dtype=np.float32) for s in ['w2v', 'ft', 'glove']}
    model = TrackPhishModel(dummy_mats, proj_dim=PROJ_DIM_TP)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    def make_inputs():
        return (torch.randint(0, vocab_size, (1, MAX_LEN_TP), device=DEVICE, dtype=torch.long),)
    return model, make_inputs

def build_resmlp(run_id, seed):
    torch.manual_seed(seed)
    model = ResMLPNet().to(DEVICE)
    def make_inputs():
        return (torch.randn(1, 1, 9, 3, device=DEVICE),)
    return model, make_inputs

def build_he_bertcnn(run_id, seed):
    torch.manual_seed(seed)
    config = BertConfig.from_pretrained("prajjwal1/bert-tiny")
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.num_attention_heads = 4
    config.intermediate_size = 128 * 4
    config.max_position_embeddings = max(config.max_position_embeddings, FINAL_LEN_HE)
    model = BertCNNClassifier(config, cnn_filters=128, kernel_size=3, pool_size=4).to(DEVICE)
    vocab_size = config.vocab_size
    def make_inputs():
        return (torch.randint(0, vocab_size, (1, FINAL_LEN_HE), device=DEVICE, dtype=torch.long),
                torch.ones(1, FINAL_LEN_HE, device=DEVICE, dtype=torch.long))
    return model, make_inputs

def build_do_tcnmhsa(run_id, seed):
    torch.manual_seed(seed)
    model = TCNMHSANet(CHAR_VOCAB_SIZE_TCN, WORD_VOCAB_SIZE_CAP).to(DEVICE)
    def make_inputs():
        return (torch.randint(0, CHAR_VOCAB_SIZE_TCN, (1, MAX_URL_LEN_TCN), device=DEVICE, dtype=torch.long),
                torch.randint(0, WORD_VOCAB_SIZE_CAP, (1, MAX_URL_LEN_TCN), device=DEVICE, dtype=torch.long))
    return model, make_inputs

GPU_MODELS = [
    {'config_key': 'C1_full_mrscn', 'label': 'MRSCN (full)',          'build_fn': build_c1_full_mrscn},
    {'config_key': 'BiLSTM',        'label': 'BiLSTM',                'build_fn': build_bilstm},
    {'config_key': 'trackphish',    'label': 'TrackPhish (Kondaiah)', 'build_fn': build_trackphish},
    {'config_key': 'remya_resmlp',  'label': 'ResMLP (Remya)',        'build_fn': build_resmlp},
    {'config_key': 'he_bertcnn',    'label': 'BERT-CNN (He)',         'build_fn': build_he_bertcnn},
    {'config_key': 'do_tcnmhsa',    'label': 'TCN-Attention (Do)',    'build_fn': build_do_tcnmhsa},
]

# ------------------------------------------------------------
# Measurement helpers
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
# Run for all 6 GPU models, run 1's checkpoint/seed (architecture + storage
# + FLOPs don't vary meaningfully by run; latency is timed fresh either way)
# ------------------------------------------------------------
rows = []
for m in GPU_MODELS:
    config_key = m['config_key']
    print(f"\n=== {config_key} ({m['label']}) ===")
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    seed = RUN_SEEDS['1'] if '1' in RUN_SEEDS else RUN_SEEDS[1]
    model, make_inputs = m['build_fn'](1, seed)

    storage_mb = measure_storage_mb(model)
    latency_mean_ms, latency_std_ms = measure_latency_ms(model, make_inputs)
    macs, flops, thop_params, flops_error = measure_flops(model, make_inputs)

    print(f"  storage_mb={storage_mb:.3f}  latency_ms={latency_mean_ms:.4f}+-{latency_std_ms:.4f}  "
          f"MACs={macs:,.0f}  FLOPs(=2xMACs)={flops:,.0f}  thop_params={thop_params:,.0f}"
          if not math.isnan(macs) else
          f"  storage_mb={storage_mb:.3f}  latency_ms={latency_mean_ms:.4f}+-{latency_std_ms:.4f}  "
          f"FLOPs=[FAILED: {flops_error}]")

    rows.append({
        'config_key': config_key, 'label': m['label'], 'device': 'cuda' if DEVICE.type == 'cuda' else 'cpu',
        'storage_mb': storage_mb, 'latency_ms_batch1_mean': latency_mean_ms,
        'latency_ms_batch1_std': latency_std_ms, 'macs': macs, 'flops_2x_macs': flops,
        'thop_params': thop_params, 'flops_error': flops_error,
    })
    del model
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

df_gpu_complexity = pd.DataFrame(rows)
print("\n[SUMMARY]")
print(df_gpu_complexity.to_string(index=False))

out_dir = f'{ROOT}/results/step21_item26'
os.makedirs(out_dir, exist_ok=True)
df_gpu_complexity.to_csv(f'{out_dir}/gpu_storage_latency_flops.csv', index=False)
print(f"\nSaved: {out_dir}/gpu_storage_latency_flops.csv")

# Cross-check: TrackPhish's real checkpoint file size vs. this method's measurement
real_ckpt_path = f'{ROOT}/checkpoints/step17/trackphish/run_1_best_model.pt'
if os.path.exists(real_ckpt_path):
    real_size_mb = os.path.getsize(real_ckpt_path) / (1024 ** 2)
    measured = df_gpu_complexity.loc[df_gpu_complexity['config_key'] == 'trackphish', 'storage_mb'].iloc[0]
    print(f"\n[CROSS-CHECK] TrackPhish real checkpoint file = {real_size_mb:.2f} MB vs. this cell's "
          f"state_dict measurement = {measured:.2f} MB "
          f"({'MATCH (close enough - real file also stores optimizer/metadata overhead if any)' if abs(real_size_mb - measured) < 5 else 'MISMATCH - investigate'})")

df_gpu_complexity
