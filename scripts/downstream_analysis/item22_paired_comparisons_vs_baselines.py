# ============================================================
# PHASE 6 - Cell 5: Item 22 - Paired comparisons vs. top baselines
# McNemar's test (continuity-corrected) + Holm correction across the 3
# comparisons, per run, with discordant-pair odds ratios + 95% CIs.
# MRSCN vs. BERT-CNN, vs. BiLSTM, vs. TCN-Attention.
# Addresses Editor Comments 5, 6, 8.
#
# Cell 4 confirmed all 4 models share the identical test_idx SET and ORDER
# for every run, so y_pred arrays can be compared elementwise without
# re-sorting. This cell re-asserts that per run anyway (cheap, and catches
# any drift if these predictions get regenerated later).
# ============================================================
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'

MRSCN_LOC = ('step13', 'C1_full_mrscn')
BASELINES = {
    'BERT-CNN': ('step17', 'he_bertcnn'),
    'BiLSTM': ('step16', 'BiLSTM'),
    'TCN-Attention': ('step17', 'do_tcnmhsa'),
}

def load_pred(step, key, run_id):
    return np.load(f'{ROOT}/results/{step}/{key}/run_{run_id}_predictions.npz')

def mcnemar_continuity_corrected(b, c):
    """b = MRSCN correct & baseline wrong; c = MRSCN wrong & baseline correct."""
    n_discordant = b + c
    if n_discordant == 0:
        return float('nan'), float('nan')
    stat = (abs(b - c) - 1) ** 2 / n_discordant
    p = float(chi2.sf(stat, df=1))
    return float(stat), p

def discordant_odds_ratio(b, c):
    """OR of the discordant pairs (b/c), with Haldane-Anscombe 0.5 correction
    when either count is 0, and a log-scale 95% CI."""
    b_c, c_c = (b + 0.5, c + 0.5) if (b == 0 or c == 0) else (b, c)
    or_val = b_c / c_c
    se_log = math.sqrt(1.0 / b_c + 1.0 / c_c)
    log_or = math.log(or_val)
    ci_low = math.exp(log_or - 1.96 * se_log)
    ci_high = math.exp(log_or + 1.96 * se_log)
    return or_val, ci_low, ci_high

def holm_correction(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    same order as the input list."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted

all_rows = []
for run_id in range(1, 6):
    mrscn_pred = load_pred(*MRSCN_LOC, run_id)
    y_true_ref = mrscn_pred['y_true']
    y_pred_mrscn = mrscn_pred['y_pred']
    correct_mrscn = (y_pred_mrscn == y_true_ref)

    run_rows = []
    raw_pvals = []
    for baseline_label, (step, key) in BASELINES.items():
        base_pred = load_pred(step, key, run_id)
        assert np.array_equal(base_pred['test_idx'], mrscn_pred['test_idx']), \
            f"[FATAL] run {run_id}: {baseline_label} test_idx no longer matches MRSCN's."
        assert np.array_equal(base_pred['y_true'], y_true_ref), \
            f"[FATAL] run {run_id}: {baseline_label} y_true no longer matches MRSCN's."

        correct_base = (base_pred['y_pred'] == y_true_ref)
        b = int(np.sum(correct_mrscn & ~correct_base))   # MRSCN right, baseline wrong
        c = int(np.sum(~correct_mrscn & correct_base))   # MRSCN wrong, baseline right
        both_right = int(np.sum(correct_mrscn & correct_base))
        both_wrong = int(np.sum(~correct_mrscn & ~correct_base))

        stat, p_raw = mcnemar_continuity_corrected(b, c)
        or_val, or_low, or_high = discordant_odds_ratio(b, c)

        run_rows.append({
            'run_id': run_id,
            'comparison': f'MRSCN vs. {baseline_label}',
            'n_total': len(y_true_ref),
            'both_correct': both_right,
            'both_wrong': both_wrong,
            'mrscn_right_baseline_wrong_b': b,
            'mrscn_wrong_baseline_right_c': c,
            'mcnemar_chi2_cc': stat,
            'p_raw': p_raw,
            'odds_ratio_b_over_c': or_val,
            'or_ci_low': or_low,
            'or_ci_high': or_high,
        })
        raw_pvals.append(p_raw)

    p_holm = holm_correction(raw_pvals)
    for row, p_adj in zip(run_rows, p_holm):
        row['p_holm'] = float(p_adj)
        row['significant_holm_0.05'] = bool(p_adj < 0.05)
    all_rows.extend(run_rows)

df_mcnemar = pd.DataFrame(all_rows)
cols = ['run_id', 'comparison', 'n_total', 'both_correct', 'both_wrong',
        'mrscn_right_baseline_wrong_b', 'mrscn_wrong_baseline_right_c',
        'mcnemar_chi2_cc', 'p_raw', 'p_holm', 'significant_holm_0.05',
        'odds_ratio_b_over_c', 'or_ci_low', 'or_ci_high']
df_mcnemar = df_mcnemar[cols]

print("[ITEM 22] McNemar paired comparisons (continuity-corrected), Holm-adjusted within each "
      "run's 3-comparison family. odds_ratio_b_over_c > 1 favors MRSCN (b = pairs where MRSCN "
      "was right and the baseline was wrong; c = the reverse).")
print()
print(df_mcnemar.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

n_sig = int(df_mcnemar['significant_holm_0.05'].sum())
n_total_tests = len(df_mcnemar)
print(f"\n[SUMMARY] {n_sig}/{n_total_tests} run x comparison cells reach Holm-adjusted "
      f"significance at alpha=0.05.")

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
import os
out_dir = f'{ROOT}/results/step21_item22'
os.makedirs(out_dir, exist_ok=True)
df_mcnemar.to_csv(f'{out_dir}/mcnemar_vs_top_baselines.csv', index=False)
print(f"\nSaved: {out_dir}/mcnemar_vs_top_baselines.csv")

df_mcnemar
