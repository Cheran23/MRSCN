# ============================================================
# Partial-fine-tune DistilBERT gap notebook - Cell 2: Reload checkpoints +
# regenerate predictions (NO retraining - all 5 checkpoints confirmed
# present by Cell 1)
#
# Model/dataset classes below are copied VERBATIM from the original
# training notebook (MRSCN_Phase_3.ipynb, Step 14 cells) so the rebuilt
# architecture's state_dict shapes match the saved checkpoints exactly.
# Validated the same way as every other regenerated-prediction cell this
# phase: an F1 cross-check against the original locked results.json,
# loud on mismatch, not silently trusted.
#
# Run this AFTER Cell 1 (needs: ROOT, RESULTS_DIR, CHECKPOINTS_DIR, DEVICE,
# binary_label, struct_full, char_full, input_ids_full, attn_mask_full,
# RUN_SPLITS, MODEL_NAME).
# ============================================================
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                              roc_auc_score, average_precision_score, matthews_corrcoef,
                              balanced_accuracy_score, confusion_matrix)
from transformers import DistilBertModel

# ------------------------------------------------------------
# Constants - must match the original Step 14 run exactly
# ------------------------------------------------------------
BATCH_SIZE = 64
AMP_ENABLED = True
CHAR_VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789-._:/?=&%")
MAX_CHAR_LEN = 200
BERT_HIDDEN_SIZE = 768

# ------------------------------------------------------------
# Dataset + model classes - verbatim from MRSCN_Phase_3.ipynb Step 14
# ------------------------------------------------------------
class FineTuneMRSCNDataset(Dataset):
    def __init__(self, indices, input_ids_src, attn_mask_src, char_src, struct_src, labels):
        self.indices = np.asarray(indices)
        self.input_ids_src, self.attn_mask_src = input_ids_src, attn_mask_src
        self.char_src, self.struct_src = char_src, struct_src
        self.labels = labels
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        return {
            "input_ids": torch.tensor(self.input_ids_src[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attn_mask_src[idx], dtype=torch.long),
            "char_ids": torch.tensor(self.char_src[idx].astype(np.int64), dtype=torch.long),
            "features": torch.tensor(self.struct_src[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

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
    """Full MRSCN (all branches + SCM), with the last 2 DistilBERT transformer
    layers unfrozen for fine-tuning; embeddings + first 4 layers stay frozen."""
    def __init__(self, structured_dim=25, num_classes=2, n_unfrozen_layers=2):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(MODEL_NAME)
        n_layers = len(self.distilbert.transformer.layer)  # 6 for distilbert-base
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

def compute_metrics(y_true, preds, probs):
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "sensitivity": recall_score(y_true, preds, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1": f1_score(y_true, preds, zero_division=0),
        "macro_f1": f1_score(y_true, preds, average='macro', zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
        "mcc": matthews_corrcoef(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def run_inference_ft(model, loader):
    model.eval()
    y_list, prob_list = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            char_ids = batch['char_ids'].to(DEVICE)
            features = batch['features'].to(DEVICE)
            with autocast('cuda', enabled=AMP_ENABLED):
                logits = model(input_ids=input_ids, attention_mask=attention_mask, char_ids=char_ids, features=features)
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            y_list.append(batch['label'].numpy())
            prob_list.append(probs.cpu().numpy())
    return np.concatenate(y_list), np.concatenate(prob_list)

def save_predictions_npz(step, config_key, run_id, test_idx, y_true, y_prob, y_pred):
    out_dir = f'{ROOT}/results/{step}/{config_key}'
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f'{out_dir}/run_{run_id}_predictions.npz',
                         test_idx=test_idx, y_true=y_true, y_prob=y_prob, y_pred=y_pred)

# ------------------------------------------------------------
# Main loop: reload each run's checkpoint, run inference, cross-check, save
# ------------------------------------------------------------
STEP = 'step14'
CONFIG_KEY = 'step14_finetuned_distilbert'

mismatch_warnings = []
print(f"=== Reloading checkpoints + regenerating predictions: {CONFIG_KEY} (step14) ===")
for run_id in range(1, 6):
    t0 = time.time()
    result_json_path = f'{RESULTS_DIR}/step14/run_{run_id}_results.json'
    with open(result_json_path) as f:
        original = json.load(f)
    locked_threshold = original['best_threshold']
    original_f1 = original['test_metrics']['f1']

    split = RUN_SPLITS[str(run_id)]
    train_idx = np.array(split['train_indices'])
    test_idx = np.array(split['test_indices'])

    scaler = StandardScaler()
    scaler.fit(struct_full[train_idx])
    struct_scaled = scaler.transform(struct_full).astype(np.float32)

    test_ds = FineTuneMRSCNDataset(test_idx, input_ids_full, attn_mask_full, char_full, struct_scaled, binary_label)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = PartiallyFineTunedMRSCN(structured_dim=25).to(DEVICE)
    best_model_path = f'{CHECKPOINTS_DIR}/run_{run_id}_best_model.pt'
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=False))

    y_test, y_prob = run_inference_ft(model, test_loader)
    y_pred = (y_prob >= locked_threshold).astype(int)
    regenerated_metrics = compute_metrics(y_test, y_pred, y_prob)
    regenerated_f1 = regenerated_metrics['f1']

    match = abs(regenerated_f1 - original_f1) < 1e-3
    flag = "OK" if match else "MISMATCH"
    if not match:
        mismatch_warnings.append(f"run {run_id}: original F1={original_f1:.4f}, regenerated F1={regenerated_f1:.4f}")

    save_predictions_npz(STEP, CONFIG_KEY, run_id, test_idx, y_test, y_prob, y_pred)

    elapsed = time.time() - t0
    print(f"  run {run_id}: original F1={original_f1:.4f} | regenerated F1={regenerated_f1:.4f} [{flag}] "
          f"| {elapsed:.1f}s")

    del model
    torch.cuda.empty_cache()

print("\n" + "=" * 80)
if mismatch_warnings:
    print(f"[WARNING] {len(mismatch_warnings)} run(s) did NOT reproduce the original F1 within tolerance:")
    for w in mismatch_warnings:
        print(f"  - {w}")
    print("Investigate before trusting these regenerated predictions.")
else:
    print("[OK] All 5 regenerated F1 scores match the original locked results.json within 1e-3. "
          f"Predictions saved to results/{STEP}/{CONFIG_KEY}/run_{{1-5}}_predictions.npz - "
          "step14_finetuned_distilbert is now unblocked for items 22/23/25 and Phase 7's ROC/PR curves.")
print("=" * 80)
