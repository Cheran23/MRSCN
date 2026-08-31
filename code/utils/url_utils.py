# utils/url_utils.py
# =============================================================
# URL preprocessing utilities
# =============================================================

import requests
from urllib.parse import urlparse, unquote
from configs.config import SHORTENER_DOMAINS, CHAR_VOCAB, MAX_CHAR_LEN

char2idx = {c: i + 1 for i, c in enumerate(CHAR_VOCAB)}


def is_shortened(url: str) -> bool:
    try:
        domain = urlparse(url).netloc.lower()
        return domain in SHORTENER_DOMAINS
    except Exception:
        return False


def expand_short_url(url: str, timeout: int = 5) -> str:
    try:
        url = unquote(url)
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        return response.url
    except Exception:
        return url


def encode_chars(url: str) -> list:
    """Encode URL as fixed-length character index sequence."""
    url = str(url).lower()
    ids = [char2idx.get(c, 0) for c in url[:MAX_CHAR_LEN]]
    if len(ids) < MAX_CHAR_LEN:
        ids += [0] * (MAX_CHAR_LEN - len(ids))
    return ids
