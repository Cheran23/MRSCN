# ============================================================
# PHASE 6 - Cell 6: Item 23 - Paired comparisons for ablations/variants
# Same McNemar (continuity-corrected) + odds-ratio-CI procedure as item 22,
# per run, for full MRSCN vs. its ablations. Addresses Editor Comments
# 6, 7, 8, 13, 14.
#
# This script computes the four Step 13 ablation comparisons for which
# trained checkpoints/predictions were available at this stage:
# w/o SCM, w/o threshold tuning, w/o class balancing, and static-only.
# The fifth comparison, frozen vs. partially fine-tuned DistilBERT, is
# added subsequently by item23b_extend_with_distilbert_partial_ft.py.
# That addendum recomputes the Holm correction across the full five-
# comparison family for each run.
# 'static-only' is mapped to C9_static_only_21feat (its own config label
# is literally "Static-only (21-feat, no externally-dependent features)"),
# not C5_structured_only ("Structured-only (25-feat)"), since C9 is the
# one that matches the spec's "static-only" terminology.
# ============================================================
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'

MRSCN_LOC = ('step13', 'C1_full_mrscn')
ABLATIONS = {
    'w/o SCM': ('step13', 'C2_no_scm'),
    'w/o threshold tuning': ('step13', 'C3_no_threshold_tuning'),
    'w/o class balancing': ('step13', 'C4_no_class_balancing'),
    'static-only (C9, 21-feat)': ('step13', 'C9_static_only_21feat'),
}

def load_pred(step, key, run_id):
    return np.load(f'{ROOT}/results/{step}/{key}/run_{run_id}_predictions.npz')

def mcnemar_continuity_corrected(b, c):
    n_discordant = b + c
    if n_discordant == 0:
        return float('nan'), float('nan')
    stat = (abs(b - c) - 1) ** 2 / n_discordant
    return float(stat), float(chi2.sf(stat, df=1))

def discordant_odds_ratio(b, c):
    b_c, c_c = (b + 0.5, c + 0.5) if (b == 0 or c == 0) else (b, c)
    or_val = b_c / c_c
    se_log = math.sqrt(1.0 / b_c + 1.0 / c_c)
    log_or = math.log(or_val)
    return or_val, math.exp(log_or - 1.96 * se_log), math.exp(log_or + 1.96 * se_log)

def holm_correction(pvals):
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
    correct_mrscn = (mrscn_pred['y_pred'] == y_true_ref)

    run_rows, raw_pvals = [], []
    for variant_label, (step, key) in ABLATIONS.items():
        var_pred = load_pred(step, key, run_id)
        # Same notebook, same RUN_SPLITS[str(run_id)]['test_indices'] source for every
        # Step 13 config -> alignment is guaranteed by construction, but verified anyway.
        assert np.array_equal(var_pred['test_idx'], mrscn_pred['test_idx']), \
            f"[FATAL] run {run_id}: {variant_label} test_idx does not match MRSCN's."
        assert np.array_equal(var_pred['y_true'], y_true_ref), \
            f"[FATAL] run {run_id}: {variant_label} y_true does not match MRSCN's."

        correct_var = (var_pred['y_pred'] == y_true_ref)
        b = int(np.sum(correct_mrscn & ~correct_var))   # full MRSCN right, variant wrong
        c = int(np.sum(~correct_mrscn & correct_var))   # full MRSCN wrong, variant right
        both_right = int(np.sum(correct_mrscn & correct_var))
        both_wrong = int(np.sum(~correct_mrscn & ~correct_var))

        stat, p_raw = mcnemar_continuity_corrected(b, c)
        or_val, or_low, or_high = discordant_odds_ratio(b, c)

        run_rows.append({
            'run_id': run_id,
            'comparison': f'Full MRSCN vs. {variant_label}',
            'n_total': len(y_true_ref),
            'both_correct': both_right,
            'both_wrong': both_wrong,
            'full_right_variant_wrong_b': b,
            'full_wrong_variant_right_c': c,
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

df_ablation = pd.DataFrame(all_rows)
cols = ['run_id', 'comparison', 'n_total', 'both_correct', 'both_wrong',
        'full_right_variant_wrong_b', 'full_wrong_variant_right_c',
        'mcnemar_chi2_cc', 'p_raw', 'p_holm', 'significant_holm_0.05',
        'odds_ratio_b_over_c', 'or_ci_low', 'or_ci_high']
df_ablation = df_ablation[cols]

print("[ITEM 23] McNemar paired comparisons for the four Step 13 ablations "
      "(continuity-corrected), with Holm adjustment across these four comparisons. "
      "odds_ratio_b_over_c > 1 favors the full MRSCN over the ablated/variant model.")
print()
print(df_ablation.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

n_sig = int(df_ablation['significant_holm_0.05'].sum())
print(f"\n[SUMMARY] {n_sig}/{len(df_ablation)} run x comparison cells reach Holm-adjusted "
      f"significance at alpha=0.05.")

print("\n" + "=" * 100)
print(
    "\n[NOTE] The fifth comparison, frozen vs. partially fine-tuned DistilBERT, "
    "is added by item23b_extend_with_distilbert_partial_ft.py, which then "
    "recomputes Holm adjustment across the complete five-comparison family."
)

print("=" * 100)

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
import os
out_dir = f'{ROOT}/results/step21_item23'
os.makedirs(out_dir, exist_ok=True)
df_ablation.to_csv(f'{out_dir}/mcnemar_vs_ablations.csv', index=False)
print(f"\nSaved: {out_dir}/mcnemar_vs_ablations.csv")

df_ablation
