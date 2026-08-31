# models/mrscn.py

import torch
import torch.nn as nn
from configs.config import CHAR_VOCAB, MAX_CHAR_LEN

char2idx = {c: i + 1 for i, c in enumerate(CHAR_VOCAB)}
BERT_HIDDEN_SIZE = 768


class StructuralConsistencyModule(nn.Module):
    def __init__(self, bert_dim, char_dim, struct_dim):
        super().__init__()
        self.proj_bert   = nn.Linear(bert_dim,   128)
        self.proj_char   = nn.Linear(char_dim,   128)
        self.proj_struct = nn.Linear(struct_dim, 128)

    def forward(self, bert_feat, char_feat, struct_feat):
        b = self.proj_bert(bert_feat)
        c = self.proj_char(char_feat)
        s = self.proj_struct(struct_feat)
        bc = torch.cosine_similarity(b, c, dim=1)
        bs = torch.cosine_similarity(b, s, dim=1)
        cs = torch.cosine_similarity(c, s, dim=1)
        return torch.stack([bc, bs, cs], dim=1)


class MRSCNModel(nn.Module):
    def __init__(self, structured_dim, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # BERT stream — accepts pre-extracted 768-dim features
        self.bert_proj = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE),
            nn.LayerNorm(BERT_HIDDEN_SIZE),
            nn.Dropout(0.2),
        )

        # Character CNN stream
        self.char_embedding = nn.Embedding(len(char2idx) + 1, 32, padding_idx=0)
        self.char_cnn = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.char_fc = nn.Linear(128 * (MAX_CHAR_LEN // 4), 128)

        # Structured feature stream
        self.struct_fc = nn.Sequential(
            nn.Linear(structured_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
        )

        # Structural Consistency Module
        self.consistency_module = StructuralConsistencyModule(
            bert_dim=BERT_HIDDEN_SIZE, char_dim=128, struct_dim=32
        )

        # Fusion classifier
        fusion_dim = BERT_HIDDEN_SIZE + 128 + 32 + 3
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, bert_features, char_ids, features,
                input_ids=None, attention_mask=None):
        bert_feat   = self.bert_proj(bert_features)

        x = self.char_embedding(char_ids).permute(0, 2, 1)
        x = self.char_cnn(x)
        x = x.flatten(start_dim=1)
        char_feat = self.char_fc(x)

        struct_feat = self.struct_fc(features)

        consistency_feat = self.consistency_module(bert_feat, char_feat, struct_feat)

        combined = torch.cat([bert_feat, char_feat, struct_feat, consistency_feat], dim=1)
        return self.classifier(combined)
