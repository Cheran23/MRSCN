# ============================================================
# PHASE 6 - Cell 9: Build item 24's derived features (shortening flag,
# obfuscation indicators, lexical-complexity score, external-feature
# missingness count) and cache them to Drive.
#
# [DEFINED IN PHASE 6 - NOT A PHASE 1 RETRIEVAL] Per your decision: since
# item 6's Phase 1 deliverable was never actually produced/persisted
# anywhere on Drive (confirmed by Cell 7/7b), these are defined here, now,
# using only columns verified to exist and encodings verified by Cell 8:
#
# - shortening_flag: URL's domain (parsed from the 'url' column, prepending
#   "http://" when no scheme is present, since Cell 8 showed both forms
#   occur) matched against a documented list of known URL-shortener domains.
# - obfuscation indicators: the 7 EXISTING binary lexical flags whose Cell 8
#   value_counts confirmed are 0/1 and whose per-class means point toward
#   obfuscation/evasion tricks: has_ip, prefix_suffix,
#   double_forward_slash_redirect, exe, url_anchor, request_url,
#   sensitive_words. obfuscation_score = count of these 7 that are 1 (0-7);
#   any_obfuscation_indicator = obfuscation_score > 0. (sub_domain and
#   free_hosting were deliberately excluded - sub_domain is a structural
#   feature already reported separately as url structure, not obfuscation
#   per se, and free_hosting's Cell 8 direction ran counter to expectation.)
# - lexical_complexity: documented as the mean of 3 z-scored components
#   (z-scored against the full 640,845-row dataset): Shannon entropy of the
#   URL string, digit_ratio (number_of_digits / url_length), and
#   percentage_special_chars (already a ratio, used as-is before z-scoring).
# - external_missing_count: computed DIRECTLY from the 4 features
#   feature_taxonomy_locked.json names as externally dependent
#   (domain_age, ip_blacklisted, web_traffic, SFH). A value of -1 denotes
#   an external lookup/extraction failure for these features. For
#   web_traffic specifically, 0 denotes a domain not found/ranked in
#   Tranco, whereas -1 is reserved for an exception during the lookup.
#   Deliberately NOT
#   reusing the existing 'count_NAN' column - Cell 8 showed count_NAN
#   ranges 1-372 (mean ~9.2), which cannot be a 0-4 count of missingness
#   among just these 4 features, so it must be counting NaNs from a wider,
#   earlier raw-extraction column set that predates the locked 25-feature
#   cache. Flagging that count_NAN's real meaning is unclear/likely
#   unrelated to the "external-feature missingness" the spec means.
# ============================================================
import json
import math
import os
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

ROOT = '/content/drive/MyDrive/MRSCN_Revision'
df = pd.read_csv(f'{ROOT}/data/processed/TB/step2_deduplicated_dataset.csv')
n_total = len(df)
print(f"[INFO] Loaded {n_total:,} rows.")

# ------------------------------------------------------------
# 1. Shortening flag
# ------------------------------------------------------------
SHORTENER_DOMAINS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly',
    'adf.ly', 'tiny.cc', 'rebrand.ly', 'shorte.st', 'cutt.ly', 'soo.gd',
    'bl.ink', 's.id', 'rb.gy', 'v.gd', 'x.co', 'po.st', 'tr.im', 'qr.ae',
    '1url.com', 'clck.ru', 'shorturl.at', 'lnkd.in', 'db.tt', 'qr.net',
    'ity.im', 'q.gs', 'viralurl.com', 'zzb.bz', 'adcrun.ch', 'u.to',
    'cli.gs', 'kutt.it', 'shorturl.com', 'go2l.ink', 'tiny.pl', 'vzturl.com',
}

def extract_domain(u):
    try:
        u2 = u if '://' in str(u) else f'http://{u}'
        netloc = urlparse(u2).netloc.lower()
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return netloc.split(':')[0]
    except Exception:
        return ''

print("[BUILD] Extracting domains + shortening flag...")
domains = df['url'].apply(extract_domain)
shortening_flag = domains.isin(SHORTENER_DOMAINS).astype(int).values
print(f"  {shortening_flag.sum():,} / {n_total:,} rows flagged as using a known shortener domain.")

# ------------------------------------------------------------
# 2. Obfuscation indicators (from existing, Cell-8-verified 0/1 columns)
# ------------------------------------------------------------
OBFUSCATION_COLS = ['has_ip', 'prefix_suffix', 'double_forward_slash_redirect',
                     'exe', 'url_anchor', 'request_url', 'sensitive_words']
obfuscation_score = df[OBFUSCATION_COLS].sum(axis=1).values.astype(int)
any_obfuscation = (obfuscation_score > 0).astype(int)
print(f"[BUILD] obfuscation_score computed from {OBFUSCATION_COLS}; "
      f"{any_obfuscation.sum():,} / {n_total:,} rows have >=1 indicator.")

# ------------------------------------------------------------
# 3. Lexical complexity = mean of 3 z-scored components
# ------------------------------------------------------------
def shannon_entropy(s):
    if not isinstance(s, str) or len(s) == 0:
        return 0.0
    length = len(s)
    counts = Counter(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())

print("[BUILD] Computing Shannon entropy per URL (one-time cost, ~640K strings)...")
entropy = df['url'].apply(shannon_entropy).values.astype(np.float64)

digit_ratio = np.where(df['url_length'].values > 0,
                        df['number_of_digits'].values / np.maximum(df['url_length'].values, 1),
                        0.0)
special_char_pct = df['percentage_special_chars'].values.astype(np.float64)

def zscore(x):
    return (x - x.mean()) / x.std(ddof=0)

lexical_complexity = (zscore(entropy) + zscore(digit_ratio) + zscore(special_char_pct)) / 3.0
print(f"[BUILD] lexical_complexity computed: mean={lexical_complexity.mean():.4f}, "
      f"std={lexical_complexity.std():.4f} (z-scored components -> ~mean 0 by construction).")

# ------------------------------------------------------------
# 4. External-feature missingness count (computed directly, NOT via count_NAN)
# ------------------------------------------------------------
EXTERNAL_COLS = ['domain_age', 'ip_blacklisted', 'web_traffic', 'SFH']
external_missing_count = (df[EXTERNAL_COLS] == -1).sum(axis=1).values.astype(int)
print(f"[BUILD] external_missing_count (0-4) from {EXTERNAL_COLS}: "
      f"value_counts={pd.Series(external_missing_count).value_counts().sort_index().to_dict()}")

# ------------------------------------------------------------
# Save cache (aligned to step2_deduplicated_dataset.csv row order/positional index)
# ------------------------------------------------------------
cache_dir = f'{ROOT}/cache/TB'
os.makedirs(cache_dir, exist_ok=True)
cache_path = f'{cache_dir}/step21_item24_derived_features.npz'
np.savez_compressed(
    cache_path,
    shortening_flag=shortening_flag,
    obfuscation_score=obfuscation_score,
    any_obfuscation=any_obfuscation,
    lexical_complexity=lexical_complexity.astype(np.float32),
    entropy=entropy.astype(np.float32),
    digit_ratio=digit_ratio.astype(np.float32),
    external_missing_count=external_missing_count,
)
print(f"\nSaved: {cache_path}")

definition_doc = {
    "shortening_flag": "1 if the URL's parsed domain matches a known shortener-domain list "
                        f"({len(SHORTENER_DOMAINS)} domains); defined in Phase 6, not Phase 1.",
    "obfuscation_score": f"count (0-7) of these binary columns == 1: {OBFUSCATION_COLS}",
    "any_obfuscation_indicator": "obfuscation_score > 0",
    "lexical_complexity": "mean of z-scored(entropy), z-scored(digit_ratio), "
                           "z-scored(percentage_special_chars); z-scored against the full "
                           "640,845-row deduplicated dataset.",
    "external_missing_count": "count (0-4) of {domain_age, ip_blacklisted, web_traffic, SFH} "
                           "== -1, representing external lookup/extraction failure. For "
                           "web_traffic, 0 instead denotes not found/ranked in Tranco. NOT the same "
                               "as the existing 'count_NAN' column, whose range (1-372) does not "
                               "match a 0-4 external-feature-missingness count.",
}
with open(f'{cache_dir}/step21_item24_derived_features_definitions.json', 'w') as f:
    json.dump(definition_doc, f, indent=2)
print(f"Saved: {cache_dir}/step21_item24_derived_features_definitions.json")

# ------------------------------------------------------------
# Sanity check: print means by class (should show malicious > benign for
# shortening/obfuscation/lexical_complexity if these are reasonable proxies)
# ------------------------------------------------------------
binary_label = df['binary_label'].values
print("\n[SANITY CHECK] mean by class:")
for name, arr in [('shortening_flag', shortening_flag), ('obfuscation_score', obfuscation_score),
                   ('any_obfuscation', any_obfuscation), ('lexical_complexity', lexical_complexity),
                   ('external_missing_count', external_missing_count)]:
    print(f"  {name}: benign={arr[binary_label == 0].mean():.4f}  "
          f"malicious={arr[binary_label == 1].mean():.4f}")
