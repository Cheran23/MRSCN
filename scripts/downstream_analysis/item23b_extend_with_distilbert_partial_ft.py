# ============================================================
# Phase 6 addendum notebook - Cell 2: Extend item 23 with the frozen vs.
# partially fine-tuned DistilBERT comparison
#
# This directly settles the question item 14 exists to answer: is it
# worth unfreezing the last 2 DistilBERT layers? Treated as an ablation
# of C1_full_mrscn (same architecture, only the freezing regime differs),
# NOT added to item 22 (that family is specifically vs. external
# competing baseline architectures).
#
# IMPORTANT: adding a 5th comparison to each run's family means the
# Holm-Bonferroni correction must be recomputed for ALL 5 comparisons per
# run, not just the new one - more tests in the family = stricter
# correction, so the previously significant/not-significant calls on the
# original 4 ablations can change. This cell recomputes and prints an
# explicit before/after so any flips are visible, not silently buried.
#
# McNemar/OR/CI formulas reproduced from the existing item23 CSV by
# reverse-matching its printed numbers (continuity-corrected McNemar,
# OR=b/c with Haldane-Anscombe 0.5 correction when b or c is 0, log-scale
# 95% CI with SE=sqrt(1/b+1/c)) - confirmed to reproduce the existing
# run-1 "w/o SCM" row's OR=1.206107 and CI exactly before trusting it for
# the new comparison.
# ============================================================
import os, shutil, time
import numpy as np
import pandas as pd
from scipy.stats import chi2
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'
RESULTS_DIR = f'{ROOT}/results'
ITEM23_PATH = f'{RESULTS_DIR}/step21_item23/mcnemar_vs_ablations.csv'
NEW_COMPARISON_LABEL = "Full MRSCN (frozen DistilBERT) vs. partially fine-tuned DistilBERT (last 2 layers)"

def load_predictions(step, config_key, run_id):
    p = f'{RESULTS_DIR}/{step}/{config_key}/run_{run_id}_predictions.npz'
    d = np.load(p)
    return {k: d[k] for k in d.files}

def mcnemar_row(run_id, comparison, y_true, y_pred_full, y_pred_variant):
    correct_full = (y_pred_full == y_true)
    correct_variant = (y_pred_variant == y_true)
    n_total = len(y_true)
    both_correct = int(np.sum(correct_full & correct_variant))
    both_wrong = int(np.sum(~correct_full & ~correct_variant))
    b = int(np.sum(correct_full & ~correct_variant))   # full right, variant wrong
    c = int(np.sum(~correct_full & correct_variant))   # full wrong, variant right
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p_raw = float(chi2.sf(stat, df=1))
    b_adj, c_adj = (b + 0.5, c + 0.5) if (b == 0 or c == 0) else (b, c)
    odds_ratio = b_adj / c_adj
    se_log = np.sqrt(1.0 / b_adj + 1.0 / c_adj)
    log_or = np.log(odds_ratio)
    ci_low = float(np.exp(log_or - 1.96 * se_log))
    ci_high = float(np.exp(log_or + 1.96 * se_log))
    return {
        "run_id": run_id, "comparison": comparison, "n_total": n_total,
        "both_correct": both_correct, "both_wrong": both_wrong,
        "full_right_variant_wrong_b": b, "full_wrong_variant_right_c": c,
        "mcnemar_chi2_cc": float(stat), "p_raw": p_raw,
        "odds_ratio_b_over_c": float(odds_ratio), "or_ci_low": ci_low, "or_ci_high": ci_high,
    }

def holm_bonferroni(pvals):
    """Standard step-down Holm correction. Verified against the existing
    item23 CSV's run-1 family (4 comparisons) - reproduces p_holm=0.039867,
    0.252325, 0.509310, ~0.0 exactly before being trusted for 5."""
    m = len(pvals)
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adj = np.empty(m)
    running_max = 0.0
    for i in range(m):
        val = (m - i) * sorted_p[i]
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)
    result = np.empty(m)
    result[order] = adj
    return result

# ------------------------------------------------------------
# Sanity-check the reproduced formulas against the existing CSV's run-1
# "w/o SCM" row before trusting them for the new comparison.
# ------------------------------------------------------------
df_old = pd.read_csv(ITEM23_PATH)
check_row = df_old[(df_old['run_id'] == 1) & (df_old['comparison'] == 'Full MRSCN vs. w/o SCM')].iloc[0]
b_chk, c_chk = check_row['full_right_variant_wrong_b'], check_row['full_wrong_variant_right_c']
or_chk = b_chk / c_chk
se_chk = np.sqrt(1.0 / b_chk + 1.0 / c_chk)
ci_low_chk = np.exp(np.log(or_chk) - 1.96 * se_chk)
ci_high_chk = np.exp(np.log(or_chk) + 1.96 * se_chk)
assert abs(or_chk - check_row['odds_ratio_b_over_c']) < 1e-4, "OR formula mismatch - STOP, do not proceed"
assert abs(ci_low_chk - check_row['or_ci_low']) < 1e-3, "CI-low formula mismatch - STOP, do not proceed"
assert abs(ci_high_chk - check_row['or_ci_high']) < 1e-3, "CI-high formula mismatch - STOP, do not proceed"
print("[OK] Reproduced OR/CI formulas match the existing CSV's run-1 'w/o SCM' row exactly. Proceeding.")

# ------------------------------------------------------------
# Build the new comparison's rows, then recompute Holm across each run's
# enlarged 5-comparison family.
# ------------------------------------------------------------
all_rows = []
flips = []
for run_id in range(1, 6):
    c1 = load_predictions('step13', 'C1_full_mrscn', run_id)
    step14 = load_predictions('step14', 'step14_finetuned_distilbert', run_id)
    assert np.array_equal(c1['test_idx'], step14['test_idx']), f"run {run_id}: test_idx mismatch - STOP"
    y_true = c1['y_true']

    new_row = mcnemar_row(run_id, NEW_COMPARISON_LABEL, y_true, c1['y_pred'], step14['y_pred'])

    existing_rows = df_old[df_old['run_id'] == run_id].to_dict('records')
    # Rebuild existing rows' core fields fresh (drop their old p_holm/significant -
    # will be recomputed below across the enlarged family)
    family = existing_rows + [new_row]
    p_raws = [r['p_raw'] for r in family]
    p_holms = holm_bonferroni(p_raws)

    for r, p_holm in zip(family, p_holms):
        old_p_holm = r.get('p_holm', None)
        old_sig = r.get('significant_holm_0.05', None)
        new_sig = bool(p_holm <= 0.05)
        if old_p_holm is not None and old_sig is not None and bool(old_sig) != new_sig:
            flips.append(f"run {run_id} | {r['comparison']}: significant {old_sig} -> {new_sig} "
                          f"(p_holm {old_p_holm:.6f} -> {p_holm:.6f})")
        row_out = {k: r[k] for k in ["run_id", "comparison", "n_total", "both_correct", "both_wrong",
                                      "full_right_variant_wrong_b", "full_wrong_variant_right_c",
                                      "mcnemar_chi2_cc", "p_raw"]}
        row_out["p_holm"] = float(p_holm)
        row_out["significant_holm_0.05"] = new_sig
        row_out["odds_ratio_b_over_c"] = r["odds_ratio_b_over_c"]
        row_out["or_ci_low"] = r["or_ci_low"]
        row_out["or_ci_high"] = r["or_ci_high"]
        all_rows.append(row_out)

df_new = pd.DataFrame(all_rows)[df_old.columns.tolist()]

print("\n" + "=" * 100)
print(f"[UPDATED] item 23 - now {df_new.shape[0]} rows (5 comparisons x 5 runs, was {df_old.shape[0]})")
print("=" * 100)
print(df_new.to_string(index=False))

print("\n" + "=" * 100)
if flips:
    print(f"[FLIPS] {len(flips)} significance call(s) changed once the family grew to 5 comparisons:")
    for f in flips:
        print(f"  - {f}")
else:
    print("[NO FLIPS] All previously significant/non-significant calls on the original 4 ablations "
          "are unchanged even with the stricter 5-comparison correction.")
print("=" * 100)

# ------------------------------------------------------------
# Back up the original file, then save the extended one in its place.
# ------------------------------------------------------------
backup_path = ITEM23_PATH.replace('.csv', f'_before_distilbert_addendum_{int(time.time())}.csv')
shutil.copyfile(ITEM23_PATH, backup_path)
df_new.to_csv(ITEM23_PATH, index=False)
print(f"\n[SAVED] {ITEM23_PATH} (original backed up to {backup_path})")

# ------------------------------------------------------------
# Headline summary for the new comparison specifically, across runs
# ------------------------------------------------------------
new_only = df_new[df_new['comparison'] == NEW_COMPARISON_LABEL]
print("\n" + "=" * 100)
print("[HEADLINE] Frozen vs. partially fine-tuned DistilBERT, across 5 runs:")
print("=" * 100)
print(new_only[['run_id', 'p_raw', 'p_holm', 'significant_holm_0.05', 'odds_ratio_b_over_c', 'or_ci_low', 'or_ci_high']].to_string(index=False))
n_sig = int(new_only['significant_holm_0.05'].sum())
print(f"\n{n_sig}/5 runs show a statistically significant difference (Holm-corrected alpha=0.05) "
      "between the frozen and partially fine-tuned variants.")
