# ============================================================
# PHASE 7 - Cell 3: Item 28 - Normalized confusion-matrix heatmap for
# MRSCN, representative run, annotated with both the row-normalized
# proportion and the raw count.
#
# Row-normalized = each row (true class) divided by its own row total, so
# the diagonal directly shows sensitivity/specificity as proportions -
# the standard reading for "how well does the model do on each true
# class," which is what makes FP/FN behavior easier to interpret visually
# than raw counts (Editor Comment 30).
#
# Sequential colormap (single hue, light->dark blue - dataviz skill's
# default sequential ramp) since this encodes magnitude (proportion of a
# row), not category.
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from google.colab import drive, files
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'
RESULTS_DIR = f'{ROOT}/results'
OUT_DIR = f'{ROOT}/results/step27_item28'
os.makedirs(OUT_DIR, exist_ok=True)

REPRESENTATIVE_RUN = 4  # re-derived and confirmed by Cell 1

# ------------------------------------------------------------
# Load MRSCN's representative-run predictions
# ------------------------------------------------------------
pred = np.load(f'{RESULTS_DIR}/step13/C1_full_mrscn/run_{REPRESENTATIVE_RUN}_predictions.npz')
y_true, y_pred = pred['y_true'], pred['y_pred']

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
cm_normalized = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

# ------------------------------------------------------------
# Sequential blue ramp (dataviz skill reference palette, 100->700 steps)
# ------------------------------------------------------------
SEQ_BLUE_STEPS = ['#cde2fb', '#9ec5f4', '#5598e7', '#256abf', '#104281', '#0d366b']
seq_cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE_STEPS, N=256)

labels = ['Benign', 'Malicious']
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm_normalized, cmap=seq_cmap, vmin=0, vmax=1, aspect='equal')

for i in range(2):
    for j in range(2):
        proportion = cm_normalized[i, j]
        count = cm[i, j]
        # Text color: dark ink on light cells, white ink on dark cells (readability)
        text_color = '#ffffff' if proportion > 0.55 else '#0b0b0b'
        ax.text(j, i, f"{proportion:.4f}\n({count:,})", ha='center', va='center',
                 color=text_color, fontsize=11, fontweight='bold' if i == j else 'normal')

ax.set_xticks([0, 1]); ax.set_xticklabels([f'Predicted {l}' for l in labels])
ax.set_yticks([0, 1]); ax.set_yticklabels([f'True {l}' for l in labels])
ax.set_title(f'Normalized Confusion Matrix - MRSCN (Run {REPRESENTATIVE_RUN})', fontsize=12, pad=12)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Row-normalized proportion')

fig.tight_layout()
png_path = f'{OUT_DIR}/mrscn_confusion_matrix_heatmap.png'
pdf_path = f'{OUT_DIR}/mrscn_confusion_matrix_heatmap.pdf'
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
plt.close(fig)
print(f"[SAVED] {png_path}")
print(f"[SAVED] {pdf_path}")
files.download(png_path)
files.download(pdf_path)

# ------------------------------------------------------------
# Save the underlying numbers too
# ------------------------------------------------------------
df_out = pd.DataFrame({
    'true_label': ['Benign', 'Benign', 'Malicious', 'Malicious'],
    'predicted_label': ['Benign', 'Malicious', 'Benign', 'Malicious'],
    'count': [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]],
    'row_normalized_proportion': [cm_normalized[0, 0], cm_normalized[0, 1], cm_normalized[1, 0], cm_normalized[1, 1]],
})
csv_path = f'{OUT_DIR}/mrscn_confusion_matrix_values.csv'
df_out.to_csv(csv_path, index=False)
print(f"[SAVED] {csv_path}")
print(df_out.to_string(index=False))

print(f"\n[SUMMARY] Specificity (TN rate) = {cm_normalized[0,0]:.4f} | "
      f"FPR = {cm_normalized[0,1]:.4f} | FNR = {cm_normalized[1,0]:.4f} | "
      f"Sensitivity/Recall (TP rate) = {cm_normalized[1,1]:.4f}")
