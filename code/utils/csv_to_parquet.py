#!/usr/bin/env python3
# utils/csv_to_parquet.py
# =============================================================
# ONE-TIME SCRIPT: Convert raw CSV to Parquet for memory-safe
# Usage:
#   python utils/csv_to_parquet.py --csv 
# =============================================================

import argparse
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from configs.config import DATA_PATH, SHORTENER_DOMAINS

try:
    import requests
    from urllib.parse import urlparse, unquote
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


def is_shortened(url: str) -> bool:
    try:
        domain = urlparse(url).netloc.lower()
        return domain in SHORTENER_DOMAINS
    except Exception:
        return False


def expand_short_url(url: str, timeout: int = 5) -> str:
    try:
        url = unquote(url)
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        return r.url
    except Exception:
        return url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to raw CSV file (TB_extracted_features.csv)."
    )
    parser.add_argument(
        "--expand_urls", action="store_true",
        help="Expand shortened URLs (slow — only enable if needed)."
    )
    parser.add_argument(
        "--chunksize", type=int, default=50_000,
        help="Rows per chunk to avoid OOM during conversion."
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    print(f"[INFO] Converting {args.csv} → {DATA_PATH}")
    print(f"[INFO] Chunk size: {args.chunksize:,}")

    chunks = []
    reader = pd.read_csv(args.csv, chunksize=args.chunksize)

    for chunk in tqdm(reader, desc="Processing chunks"):
        # Normalise label
        if "label" not in chunk.columns and "type" in chunk.columns:
            chunk["label"] = chunk["type"].apply(
                lambda x: 0 if str(x).lower() == "legitimate" else 1
            )

        # Expand shortened URLs (optional, slow)
        if args.expand_urls and _REQUESTS_AVAILABLE:
            chunk["url"] = [
                expand_short_url(u) if is_shortened(u) else u
                for u in chunk["url"].astype(str)
            ]

        # Clean
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0, inplace=True)

        if "web_traffic" in chunk.columns:
            chunk["web_traffic"] = np.log1p(
                pd.to_numeric(chunk["web_traffic"], errors="coerce").clip(lower=0)
            )

        chunks.append(chunk)

    print("[INFO] Concatenating chunks …")
    df = pd.concat(chunks, ignore_index=True)
    print(f"[INFO] Total rows: {len(df):,}")

    df.to_parquet(DATA_PATH, index=False, engine="pyarrow", compression="snappy")
    print(f"[INFO] Parquet saved → {DATA_PATH}")


if __name__ == "__main__":
    main()
