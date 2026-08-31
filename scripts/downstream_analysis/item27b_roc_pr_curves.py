# ============================================================
# PHASE 7 - Cell 2: Item 27 - ROC and PR curves for MRSCN vs. the Table 7
# baseline group and, separately, vs. the Table 12 baseline group, for the
# representative run (run 4, re-derived in Cell 1).
#
# PR curves report Average Precision via sklearn's average_precision_score
# - NOT a trapezoidal auc(recall, precision) - per the workflow spec's
# explicit correction (Editor Comment 29 also flags the old PR-AUC
# terminology as wrong; this cell produces the metric correctly at
# source, the terminology fix in the manuscript text is separate work).
#
# Colors: validated categorical palette (dataviz skill), 7-slot ordering
# passes CVD/normal-vision separation for the "adjacent" (line-chart)
# pairlist. MRSCN is deliberately NOT one of the categorical hues - it's
# drawn in solid black, thicker, so the proposed model is unambiguous
# against 5-6 baselines even in grayscale print. Every baseline also gets
# a distinct linestyle as a secondary (non-color) encoding, satisfying
# the palette's contrast-WARN relief rule for the three lower-contrast
# hues (aqua/yellow/magenta) via direct legend labels + linestyle, not
# color alone.
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from google.colab import drive, files
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'
RESULTS_DIR = f'{ROOT}/results'
OUT_DIR = f'{ROOT}/results/step27_item27'
os.makedirs(OUT_DIR, exist_ok=True)

REPRESENTATIVE_RUN = 4  # re-derived and confirmed by Cell 1

MRSCN = ('step13', 'C1_full_mrscn', 'MRSCN (proposed)')
TABLE7_GROUP = [
    ('step16', 'LogisticRegression', 'Logistic Regression'),
    ('step16', 'RandomForest', 'Random Forest'),
    ('step16', 'XGBoost', 'XGBoost'),
    ('step17', 'he_bertcnn', 'BERT-CNN'),
    ('step17', 'do_tcnmhsa', 'TCN-Attention'),
    ('step16', 'BiLSTM', 'BiLSTM'),
]
TABLE12_GROUP = [
    ('step17', 'khalife_rf', 'Khalife et al. (RF)'),
    ('step17', 'mohanty_gboost_fst', 'Mohanty & Acharya (GBoost)'),
    ('step17', 'omolara_adaboost', 'Omolara & Alawida (AdaBoost)'),
    ('step17', 'remya_resmlp', 'ResMLP (Remya et al.)'),
    ('step17', 'trackphish', 'TrackPhish (Kondaiah et al.)'),
]

# Validated categorical palette (dataviz skill reference palette, light mode,
# fixed order - not cycled/reassigned across groups)
CATEGORICAL_HUES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
MRSCN_COLOR = '#000000'

def load_predictions(step, config_key, run_id):
    p = f'{RESULTS_DIR}/{step}/{config_key}/run_{run_id}_predictions.npz'
    d = np.load(p)
    return d['y_true'], d['y_prob']

def build_roc_pr_figure(group, group_label, save_stem):
    y_true_mrscn, y_prob_mrscn = load_predictions(*MRSCN[:2], REPRESENTATIVE_RUN)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))

    curve_rows = []

    def plot_one(ax_roc, ax_pr, label, y_true, y_prob, color, linestyle, is_mrscn):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)

        lw = 2.6 if is_mrscn else 1.6
        zorder = 10 if is_mrscn else 5
        ax_roc.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=lw, zorder=zorder,
                    label=f"{label} (AUC={auc_val:.4f})")
        ax_pr.plot(recall, precision, color=color, linestyle=linestyle, linewidth=lw, zorder=zorder,
                   label=f"{label} (AP={ap_val:.4f})")

        for fp, tp in zip(fpr, tpr):
            curve_rows.append({'model': label, 'curve': 'ROC', 'x_fpr_or_recall': fp, 'y_tpr_or_precision': tp})
        for rec, prec in zip(recall, precision):
            curve_rows.append({'model': label, 'curve': 'PR', 'x_fpr_or_recall': rec, 'y_tpr_or_precision': prec})
        return auc_val, ap_val

    summary_rows = []
    auc_val, ap_val = plot_one(ax_roc, ax_pr, MRSCN[2], y_true_mrscn, y_prob_mrscn,
                                MRSCN_COLOR, '-', is_mrscn=True)
    summary_rows.append({'model': MRSCN[2], 'roc_auc': auc_val, 'average_precision': ap_val})

    for i, (step, key, label) in enumerate(group):
        y_true, y_prob = load_predictions(step, key, REPRESENTATIVE_RUN)
        assert np.array_equal(y_true, y_true_mrscn), f"{label}: y_true mismatch vs MRSCN - test set misaligned"
        color = CATEGORICAL_HUES[i % len(CATEGORICAL_HUES)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]
        auc_val, ap_val = plot_one(ax_roc, ax_pr, label, y_true, y_prob, color, linestyle, is_mrscn=False)
        summary_rows.append({'model': label, 'roc_auc': auc_val, 'average_precision': ap_val})

    # Reference lines
    ax_roc.plot([0, 1], [0, 1], color='#8a8a86', linestyle=':', linewidth=1.2, zorder=1, label='Chance (AUC=0.500)')
    prevalence = float(y_true_mrscn.mean())
    ax_pr.axhline(prevalence, color='#8a8a86', linestyle=':', linewidth=1.2, zorder=1,
                  label=f'Chance (AP={prevalence:.4f})')

    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'ROC Curve - {group_label} (Run {REPRESENTATIVE_RUN})')
    ax_roc.set_xlim(-0.02, 1.02); ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.legend(loc='lower right', fontsize=8, frameon=True)
    ax_roc.grid(True, alpha=0.25, linewidth=0.6)

    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f'Precision-Recall Curve - {group_label} (Run {REPRESENTATIVE_RUN})')
    ax_pr.set_xlim(-0.02, 1.02); ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.legend(loc='lower left', fontsize=8, frameon=True)
    ax_pr.grid(True, alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    png_path = f'{OUT_DIR}/{save_stem}_roc_pr.png'
    pdf_path = f'{OUT_DIR}/{save_stem}_roc_pr.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")
    files.download(png_path)
    files.download(pdf_path)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = f'{OUT_DIR}/{save_stem}_auc_ap_summary.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"[SAVED] {summary_path}")
    print(df_summary.to_string(index=False))

    df_curves = pd.DataFrame(curve_rows)
    curves_path = f'{OUT_DIR}/{save_stem}_curve_points.csv'
    df_curves.to_csv(curves_path, index=False)
    print(f"[SAVED] {curves_path} ({len(df_curves):,} points)")

    return df_summary

print("=" * 100)
print("[TABLE 7] MRSCN vs. Logistic Regression, Random Forest, XGBoost, BERT-CNN, TCN-Attention, BiLSTM")
print("=" * 100)
summary_table7 = build_roc_pr_figure(TABLE7_GROUP, 'MRSCN vs. Table 7 Baselines', 'table7')

print("\n" + "=" * 100)
print("[TABLE 12] MRSCN vs. Khalife-RF, Mohanty-GBoost, Omolara-AdaBoost, ResMLP, TrackPhish")
print("=" * 100)
summary_table12 = build_roc_pr_figure(TABLE12_GROUP, 'MRSCN vs. Table 12 Baselines', 'table12')

print("\n[DONE] Both figure pairs saved to results/step27_item27/. Send the PNGs back to view them.")
