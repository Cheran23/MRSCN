# ============================================================
# PHASE 6 - Cell 10: Item 24 - Quantitative FP/FN error characterization
# Compares FP vs TN and FN vs TP on MRSCN's own predictions, per run,
# summarized across all 5 runs. Addresses Editor Comment 26.
#
# Uses the item-6-gap variables built and cached by Cell 9
# (shortening_flag, obfuscation_score/any_obfuscation, lexical_complexity,
# external_missing_count - all defined in Phase 6, documented in
# step21_item24_derived_features_definitions.json) plus raw columns
# verified by Cell 8 (url_length, has_https, domain_age, ip_blacklisted).
#
# Continuous variables -> Mann-Whitney U (reports medians per group).
# Binary variables -> Fisher's exact test (reports proportions per group).
# domain_age and ip_blacklisted use -1 as their verified missing sentinel
# (Cell 8): the raw value is compared only among non-missing rows, and
# missingness itself is compared separately as its own binary variable.
# No multiple-comparison correction is applied within this item (the spec
# doesn't ask for one here, unlike items 22/23's Holm correction) - raw
# per-run p-values are reported as-is.
# ============================================================
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'

df = pd.read_csv(f'{ROOT}/data/processed/TB/step2_deduplicated_dataset.csv')
derived = np.load(f'{ROOT}/cache/TB/step21_item24_derived_features.npz')

# Full-dataset arrays, positional (row order matches step2_deduplicated_dataset.csv)
url_length_all = df['url_length'].values.astype(np.float64)
has_https_all = df['has_https'].values.astype(int)
domain_age_all = df['domain_age'].values.astype(np.float64)
ip_blacklisted_all = df['ip_blacklisted'].values.astype(int)
shortening_flag_all = derived['shortening_flag']
obfuscation_score_all = derived['obfuscation_score']
any_obfuscation_all = derived['any_obfuscation']
lexical_complexity_all = derived['lexical_complexity']
external_missing_count_all = derived['external_missing_count']


def mwu_compare(vals_a, vals_b, label):
    vals_a = vals_a[~np.isnan(vals_a)]
    vals_b = vals_b[~np.isnan(vals_b)]
    if len(vals_a) < 1 or len(vals_b) < 1:
        return {'variable': label, 'test': 'mannwhitneyu', 'group_a_median': float('nan'),
                'group_b_median': float('nan'), 'statistic': float('nan'), 'p_value': float('nan'),
                'n_a': len(vals_a), 'n_b': len(vals_b)}
    stat, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
    return {'variable': label, 'test': 'mannwhitneyu',
            'group_a_median': float(np.median(vals_a)), 'group_b_median': float(np.median(vals_b)),
            'statistic': float(stat), 'p_value': float(p), 'n_a': len(vals_a), 'n_b': len(vals_b)}


def fisher_compare(bin_a, bin_b, label):
    if len(bin_a) < 1 or len(bin_b) < 1:
        return {'variable': label, 'test': 'fisher_exact', 'group_a_rate': float('nan'),
                'group_b_rate': float('nan'), 'statistic': float('nan'), 'p_value': float('nan'),
                'n_a': len(bin_a), 'n_b': len(bin_b)}
    a1 = int(bin_a.sum()); a0 = len(bin_a) - a1
    b1 = int(bin_b.sum()); b0 = len(bin_b) - b1
    odds, p = fisher_exact([[a1, a0], [b1, b0]])
    return {'variable': label, 'test': 'fisher_exact',
            'group_a_rate': a1 / len(bin_a), 'group_b_rate': b1 / len(bin_b),
            'statistic': float(odds), 'p_value': float(p), 'n_a': len(bin_a), 'n_b': len(bin_b)}


all_rows = []
for run_id in range(1, 6):
    pred = np.load(f'{ROOT}/results/step13/C1_full_mrscn/run_{run_id}_predictions.npz')
    test_idx = pred['test_idx']
    y_true = pred['y_true']
    y_pred = pred['y_pred']

    fp_mask = (y_true == 0) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)

    def gather(arr_all, mask):
        return arr_all[test_idx[mask]]

    comparisons = {
        'FP_vs_TN': (fp_mask, tn_mask),
        'FN_vs_TP': (fn_mask, tp_mask),
    }

    for comp_label, (mask_a, mask_b) in comparisons.items():
        rows_this_comp = []

        rows_this_comp.append(mwu_compare(gather(url_length_all, mask_a),
                                           gather(url_length_all, mask_b), 'url_length'))
        rows_this_comp.append(mwu_compare(gather(lexical_complexity_all, mask_a),
                                           gather(lexical_complexity_all, mask_b), 'lexical_complexity'))
        rows_this_comp.append(mwu_compare(gather(obfuscation_score_all, mask_a).astype(float),
                                           gather(obfuscation_score_all, mask_b).astype(float), 'obfuscation_score'))
        rows_this_comp.append(mwu_compare(gather(external_missing_count_all, mask_a).astype(float),
                                           gather(external_missing_count_all, mask_b).astype(float),
                                           'external_missing_count'))

        # domain_age: compare raw value only among non-missing (!= -1) rows
        da_a = gather(domain_age_all, mask_a); da_b = gather(domain_age_all, mask_b)
        rows_this_comp.append(mwu_compare(da_a[da_a != -1], da_b[da_b != -1], 'domain_age_value_if_known'))

        rows_this_comp.append(fisher_compare(gather(shortening_flag_all, mask_a),
                                              gather(shortening_flag_all, mask_b), 'shortening_flag'))
        rows_this_comp.append(fisher_compare(gather(any_obfuscation_all, mask_a),
                                              gather(any_obfuscation_all, mask_b), 'any_obfuscation_indicator'))
        rows_this_comp.append(fisher_compare(gather(has_https_all, mask_a),
                                              gather(has_https_all, mask_b), 'has_https'))
        rows_this_comp.append(fisher_compare((gather(domain_age_all, mask_a) == -1).astype(int),
                                              (gather(domain_age_all, mask_b) == -1).astype(int),
                                              'domain_age_missing'))
        rows_this_comp.append(fisher_compare((gather(ip_blacklisted_all, mask_a) == -1).astype(int),
                                              (gather(ip_blacklisted_all, mask_b) == -1).astype(int),
                                              'ip_blacklisted_missing'))

        ipb_a = gather(ip_blacklisted_all, mask_a); ipb_b = gather(ip_blacklisted_all, mask_b)
        ipb_a_known = ipb_a[ipb_a != -1]; ipb_b_known = ipb_b[ipb_b != -1]
        rows_this_comp.append(fisher_compare((ipb_a_known == 1).astype(int), (ipb_b_known == 1).astype(int),
                                              'ip_blacklisted_positive_among_known'))

        for row in rows_this_comp:
            row['run_id'] = run_id
            row['comparison'] = comp_label
            row['n_group_a_total'] = int(mask_a.sum())
            row['n_group_b_total'] = int(mask_b.sum())
        all_rows.extend(rows_this_comp)

df_fpfn = pd.DataFrame(all_rows)
front_cols = ['run_id', 'comparison', 'variable', 'test', 'n_group_a_total', 'n_group_b_total']
other_cols = [c for c in df_fpfn.columns if c not in front_cols]
df_fpfn = df_fpfn[front_cols + other_cols]

print("[ITEM 24] FP-vs-TN and FN-vs-TP error characterization, per run. "
      "Group A = FP (or FN); Group B = TN (or TP).")
print(df_fpfn.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ------------------------------------------------------------
# Summary across the 5 runs: mean of point estimates + how many runs
# reach raw p < 0.05 for each (comparison, variable)
# ------------------------------------------------------------
def summarize(g):
    out = {'n_runs_p_lt_0.05': int((g['p_value'] < 0.05).sum())}
    if 'group_a_median' in g and g['group_a_median'].notna().any():
        out['mean_group_a_median'] = g['group_a_median'].mean()
        out['mean_group_b_median'] = g['group_b_median'].mean()
    if 'group_a_rate' in g and g['group_a_rate'].notna().any():
        out['mean_group_a_rate'] = g['group_a_rate'].mean()
        out['mean_group_b_rate'] = g['group_b_rate'].mean()
    return pd.Series(out)

df_summary = df_fpfn.groupby(['comparison', 'variable']).apply(summarize).reset_index()
print("\n[SUMMARY ACROSS 5 RUNS]")
print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
import os
out_dir = f'{ROOT}/results/step21_item24'
os.makedirs(out_dir, exist_ok=True)
df_fpfn.to_csv(f'{out_dir}/fp_fn_characterization_per_run.csv', index=False)
df_summary.to_csv(f'{out_dir}/fp_fn_characterization_summary.csv', index=False)
print(f"\nSaved: {out_dir}/fp_fn_characterization_per_run.csv")
print(f"Saved: {out_dir}/fp_fn_characterization_summary.csv")

df_summary
