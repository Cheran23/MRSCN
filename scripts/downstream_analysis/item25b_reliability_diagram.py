# ============================================================
# PHASE 6 - Cell 13: Item 25 - Reliability diagram for the representative
# run (run 4, the same median-F1 run Cell 2 established for item 21).
# One panel per model (3x3 grid, same 9 models as Cell 12), each showing
# observed accuracy vs. mean predicted probability per bin against the
# y=x perfect-calibration diagonal.
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'
REPRESENTATIVE_RUN = 4

MODELS = {
    'MRSCN (C1_full_mrscn)': ('step13', 'C1_full_mrscn'),
    'BiLSTM': ('step16', 'BiLSTM'),
    'BERT-CNN': ('step17', 'he_bertcnn'),
    'TCN-Attention': ('step17', 'do_tcnmhsa'),
    'TrackPhish': ('step17', 'trackphish'),
    'ResMLP': ('step17', 'remya_resmlp'),
    'Khalife-RF': ('step17', 'khalife_rf'),
    'Mohanty-GBoost': ('step17', 'mohanty_gboost_fst'),
    'Omolara-AdaBoost': ('step17', 'omolara_adaboost'),
}

def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))

def ece_score(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    bin_details = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            bin_details.append({'bin_center': (lo + hi) / 2, 'n': 0,
                                 'confidence': float('nan'), 'accuracy': float('nan')})
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (n_bin / n) * abs(accuracy - confidence)
        bin_details.append({'bin_center': (lo + hi) / 2, 'n': n_bin,
                             'confidence': confidence, 'accuracy': accuracy})
    return float(ece), bin_details

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for ax, (model_label, (step, key)) in zip(axes, MODELS.items()):
    fp = f'{ROOT}/results/{step}/{key}/run_{REPRESENTATIVE_RUN}_predictions.npz'
    if not os.path.exists(fp):
        ax.set_title(f'{model_label}\n[no data]')
        ax.axis('off')
        continue
    pred = np.load(fp)
    if 'y_prob' not in pred.files:
        ax.set_title(f'{model_label}\n[no y_prob]')
        ax.axis('off')
        continue
    y_true = pred['y_true'].astype(np.float64)
    y_prob = pred['y_prob'].astype(np.float64)

    brier = brier_score(y_true, y_prob)
    ece, bin_details = ece_score(y_true, y_prob, n_bins=10)
    bin_df = pd.DataFrame(bin_details).dropna()

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Perfect calibration')
    ax.plot(bin_df['confidence'], bin_df['accuracy'], marker='o', linewidth=2, color='#1f77b4',
            label='Observed')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted probability (bin)')
    ax.set_ylabel('Observed accuracy (fraction positive)')
    ax.set_title(f'{model_label}\nBrier={brier:.4f}  ECE={ece:.4f}  (run {REPRESENTATIVE_RUN})')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(f'Reliability diagrams - representative run (run {REPRESENTATIVE_RUN})', fontsize=14, y=1.01)
fig.tight_layout()

out_dir = f'{ROOT}/results/step21_item25'
os.makedirs(out_dir, exist_ok=True)
fig_path = f'{out_dir}/reliability_diagram_run{REPRESENTATIVE_RUN}.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fig_path}")
plt.show()
