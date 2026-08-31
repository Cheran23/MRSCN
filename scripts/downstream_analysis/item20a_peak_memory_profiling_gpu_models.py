# ============================================================
# STEP 20 - Cell 2 (CORRECTED v3): GPU/PyTorch peak inference memory profiling
#
# History of fixes in this cell:
#  v1: TrackPhish (Kondaiah et al.) architecture was reconstructed from memory
#      and did NOT match the real saved checkpoint. Fixed by copying the real
#      classes (FusionAttention / TrackPhish1DCNN / TrackPhishModel) verbatim
#      from Step 17 Cell 24/25 source.
#  v2: Added GPU warm-up passes, hypothesizing the large numbers (191/225/360/
#      186/204/189 MB) were one-time cuDNN/cuBLAS kernel-workspace allocation
#      counted against the first measured forward pass.
#  v3 (this version): warm-up produced BYTE-IDENTICAL numbers to v2, which
#      rules out the lazy-kernel-init theory (warm-up should have absorbed
#      that cost and lowered the reading - it didn't move at all). The real
#      suspect: torch.cuda.reset_peak_memory_stats() resets the PEAK counter,
#      not the CURRENTLY ALLOCATED bytes - so if GPU memory is already
#      resident from earlier work in this long Colab session (Cell 0/1,
#      Step 19, Phase 3/4 training, prior Cell 2 attempts) and was never
#      freed, every "peak" reading in this cell is floored by that leftover
#      baseline rather than reflecting each model's own footprint. This
#      version adds an explicit baseline diagnostic BEFORE any model is
#      built, plus more aggressive gc.collect()+empty_cache() between models,
#      so we can see directly whether that's what's going on.
#
# Relies on Cell 1 already having run in this session (ROOT, DEVICE, RUN_SEEDS
# all in scope). Safe to re-run on its own after Cell 1.
#
# IMPORTANT: for a trustworthy reading, do Runtime > Restart session first,
# then re-run Cell 1, then run this cell - see the baseline check below.
# ============================================================
import os, json, math, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = False  # keep memory measurements deterministic

# ------------------------------------------------------------
# Baseline diagnostic - run BEFORE anything else touches the GPU in this
# cell, so we can see whether memory is already resident from earlier work
# in this session (which would contaminate every measurement below).
# ------------------------------------------------------------
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(DEVICE)
_baseline_allocated_mb = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)
_baseline_reserved_mb = torch.cuda.memory_reserved(DEVICE) / (1024 ** 2)
print(f"[BASELINE CHECK] Before this cell builds any model:")
print(f"  torch.cuda.memory_allocated = {_baseline_allocated_mb:.2f} MB")
print(f"  torch.cuda.memory_reserved  = {_baseline_reserved_mb:.2f} MB")
if _baseline_allocated_mb > 20:
    print(f"  [WARNING] {_baseline_allocated_mb:.1f} MB is already resident on the GPU before "
          f"this cell has built anything. This is very likely leftover state from earlier "
          f"cells/steps in this session and WILL inflate every 'peak inference memory' number "
          f"below. STRONGLY RECOMMENDED: Runtime > Restart session, then re-run Cell 1, then "
          f"re-run this cell, for a clean/trustworthy baseline before trusting these numbers "
          f"for the paper.")
else:
    print(f"  [OK] Baseline is small - readings below should be trustworthy without a restart.")

# ------------------------------------------------------------
# Local GPU profiling helper (self-contained - does not depend on Cell 1's
# version, in case that signature differs from what's assumed here)
# ------------------------------------------------------------
def profile_gpu_inference_memory(model, forward_fn, n_trials=5, n_warmup=3):
    model = model.to(DEVICE).eval()

    # Warm-up passes (kept from v2 - harmless, and still correct practice
    # even though it wasn't the actual cause of the inflated v2 numbers).
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = forward_fn(model)
    torch.cuda.synchronize()

    peaks_mb = []
    with torch.no_grad():
        for _ in range(n_trials):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(DEVICE)
            _ = forward_fn(model)
            torch.cuda.synchronize()
            peaks_mb.append(torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2))
    return float(np.mean(peaks_mb)), float(np.std(peaks_mb))


# ============================================================
# Architecture 1/6 - C1_full_mrscn (Step 13, verbatim)
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


# ============================================================
# Architecture 2/6 - BiLSTM (Step 16, verbatim)
# ============================================================
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


# ============================================================
# Architecture 3/6 - TrackPhish / Kondaiah et al. (Step 17 Cell 24/25,
# copied VERBATIM this time from the real notebook source - this is the fix)
# ============================================================
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


# ============================================================
# Architecture 4/6 - ResMLP / Remya et al. (Step 17 Cell 26, verbatim)
# ============================================================
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


# ============================================================
# Architecture 5/6 - He-BERTCNN / He et al. (Step 17 Cell 27, verbatim)
# ============================================================
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
FINAL_LEN_HE = BASE_LEN_HE + EXT_K_HE  # 138


# ============================================================
# Architecture 6/6 - Do-TCNMHSA / Do et al. (Step 17 Cell 28, verbatim)
# ============================================================
MAX_URL_LEN_TCN = 350
EMBED_SIZE_TCN = 32
TCN_FILTERS = 16
TCN_KERNEL = 3
TCN_DILATIONS = (1, 2, 4, 8)
MHSA_HEADS = 2
MHSA_KEY_DIM = 2
DROPOUT_TCN = 0.5
WORD_VOCAB_SIZE_CAP = 20000  # cap used at training time; exact per-run vocab was data-dependent
                              # and not persisted, so we profile at this upper-bound cap
                              # (memory-footprint profiling is weight-value-independent, and
                              # a larger embedding table only makes this a conservative/upper
                              # estimate, never an underestimate)

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
# Model builders (one per config_key) -> (model, forward_fn)
# ============================================================
def build_c1_full_mrscn(run_id, seed):
    torch.manual_seed(seed)
    model = ConfigurableMRSCN(structured_dim=25, use_bert=True, use_char=True, use_struct=True, use_scm=True)
    ckpt_path = f'{ROOT}/checkpoints/step13/C1_full_mrscn/run_{run_id}_best_model.pt'
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    def fwd(m):
        bert_features = torch.randn(1, BERT_HIDDEN_SIZE, device=DEVICE)
        char_ids = torch.randint(0, len(CHAR_VOCAB) + 1, (1, MAX_CHAR_LEN), device=DEVICE, dtype=torch.long)
        features = torch.randn(1, 25, device=DEVICE)
        return m(bert_features=bert_features, char_ids=char_ids, features=features)
    return model, fwd

def build_bilstm(run_id, seed):
    torch.manual_seed(seed)
    model = BiLSTMBaseline(vocab_size=len(CHAR_VOCAB) + 1)
    ckpt_path = f'{ROOT}/checkpoints/step16/BiLSTM/run_{run_id}_best_model.pt'
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    def fwd(m):
        char_ids = torch.randint(0, len(CHAR_VOCAB) + 1, (1, MAX_CHAR_LEN), device=DEVICE, dtype=torch.long)
        return m(char_ids)
    return model, fwd

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
    def fwd(m):
        id_seq = torch.randint(0, vocab_size, (1, MAX_LEN_TP), device=DEVICE, dtype=torch.long)
        return m(id_seq)
    return model, fwd

def build_resmlp(run_id, seed):
    # No saved checkpoint -> architecture-only reconstruction (valid: memory
    # footprint depends on tensor shapes/dtypes, not weight values).
    torch.manual_seed(seed)
    model = ResMLPNet().to(DEVICE)
    def fwd(m):
        x = torch.randn(1, 1, 9, 3, device=DEVICE)
        return m(x)
    return model, fwd

def build_he_bertcnn(run_id, seed):
    # No saved checkpoint -> architecture-only reconstruction.
    torch.manual_seed(seed)
    config = BertConfig.from_pretrained("prajjwal1/bert-tiny")
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.num_attention_heads = 4
    config.intermediate_size = 128 * 4
    config.max_position_embeddings = max(config.max_position_embeddings, FINAL_LEN_HE)
    model = BertCNNClassifier(config, cnn_filters=128, kernel_size=3, pool_size=4).to(DEVICE)
    vocab_size = config.vocab_size
    def fwd(m):
        ids = torch.randint(0, vocab_size, (1, FINAL_LEN_HE), device=DEVICE, dtype=torch.long)
        mask = torch.ones(1, FINAL_LEN_HE, device=DEVICE, dtype=torch.long)
        return m(ids, mask)
    return model, fwd

def build_do_tcnmhsa(run_id, seed):
    # No saved checkpoint -> architecture-only reconstruction, word vocab
    # sized at the training-time cap (see WORD_VOCAB_SIZE_CAP note above).
    torch.manual_seed(seed)
    model = TCNMHSANet(CHAR_VOCAB_SIZE_TCN, WORD_VOCAB_SIZE_CAP).to(DEVICE)
    def fwd(m):
        char_ids = torch.randint(0, CHAR_VOCAB_SIZE_TCN, (1, MAX_URL_LEN_TCN), device=DEVICE, dtype=torch.long)
        word_ids = torch.randint(0, WORD_VOCAB_SIZE_CAP, (1, MAX_URL_LEN_TCN), device=DEVICE, dtype=torch.long)
        return m(char_ids, word_ids)
    return model, fwd


GPU_MODELS = [
    {'config_key': 'C1_full_mrscn', 'label': 'MRSCN (full)',            'has_ckpt': True,  'build_fn': build_c1_full_mrscn},
    {'config_key': 'BiLSTM',        'label': 'BiLSTM',                  'has_ckpt': True,  'build_fn': build_bilstm},
    {'config_key': 'trackphish',    'label': 'TrackPhish (Kondaiah)',   'has_ckpt': True,  'build_fn': build_trackphish},
    {'config_key': 'remya_resmlp',  'label': 'ResMLP (Remya)',          'has_ckpt': False, 'build_fn': build_resmlp},
    {'config_key': 'he_bertcnn',    'label': 'BERT-CNN (He)',           'has_ckpt': False, 'build_fn': build_he_bertcnn},
    {'config_key': 'do_tcnmhsa',    'label': 'TCN-Attention (Do)',      'has_ckpt': False, 'build_fn': build_do_tcnmhsa},
]

GPU_RESULTS_DIR = f'{ROOT}/results/step20'
os.makedirs(GPU_RESULTS_DIR, exist_ok=True)

# No skip-if-file-exists logic: profiling is cheap (seconds per model), so
# this always recomputes and overwrites rather than risk trusting a stale or
# wrong-schema file again.
gpu_results = []
for m in GPU_MODELS:
    config_key = m['config_key']
    out_path = f'{GPU_RESULTS_DIR}/{config_key}_memory.json'

    print(f"\n=== GPU profiling: {config_key} ({m['label']}) ===")
    per_run_means = []
    failed = False
    for run_id in range(1, 6):
        seed = RUN_SEEDS[str(run_id)] if str(run_id) in RUN_SEEDS else RUN_SEEDS[run_id]
        try:
            # Aggressive cleanup BEFORE building this run's model, so no
            # tensor from the previous run/model can still be resident and
            # inflate this measurement's baseline.
            gc.collect()
            torch.cuda.empty_cache()
            model, fwd = m['build_fn'](run_id, seed)
            mean_mb, std_mb = profile_gpu_inference_memory(model, fwd, n_trials=5, n_warmup=3)
            per_run_means.append(mean_mb)
            print(f"  run {run_id}: peak={mean_mb:.2f} MB (std over 5 trials={std_mb:.4f})")
            del model, fwd
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] run {run_id} failed: {type(e).__name__}: {e}")
            failed = True
            break

    if failed or not per_run_means:
        print(f"[FAILED] {config_key} - skipping, will retry on next Cell run.")
        continue

    rec = {
        'config_key': config_key, 'label': m['label'], 'framework': 'pytorch', 'device': 'cuda',
        'has_ckpt': m['has_ckpt'], 'n_runs': len(per_run_means),
        'mean_mb': float(np.mean(per_run_means)), 'std_mb': float(np.std(per_run_means)),
        'per_run_mb': per_run_means,
        'baseline_allocated_mb_before_cell': _baseline_allocated_mb,
    }
    with open(out_path, 'w') as f:
        json.dump(rec, f, indent=2)
    gpu_results.append(rec)
    print(f"[DONE] {config_key}: {rec['mean_mb']:.2f} +/- {rec['std_mb']:.2f} MB")

print("\nGPU profiling pass complete.")
print(f"{len(gpu_results)}/{len(GPU_MODELS)} models profiled successfully.")
if _baseline_allocated_mb > 20:
    print(f"\n[REMINDER] Baseline GPU memory before this cell ran was {_baseline_allocated_mb:.1f} MB "
          f"(see [BASELINE CHECK] above). If you have NOT restarted the runtime since earlier "
          f"steps in this session, treat the numbers above as suspect and re-run after "
          f"Runtime > Restart session + re-running Cell 1.")
