import numpy as np
import re
import unicodedata

def normalize_score_bm25(score):
    s = np.array(score, dtype=float)
    mn, mx = s.min(), s.max()
    if mx - mn < 10^ (-9):
        return np.ones_like(s)
    return (s - mn) / (mx - mn)

def tokenize_bm25(text):
    # Unicode normalize
    text = unicodedata.normalize("NFC", text)
    # Lowercase
    text = text.lower()
    # Loại bỏ ký tự lạ, chỉ giữ chữ + số
    text = re.sub(r"[^0-9a-zA-Z\u00C0-\u1EF9\s]", " ", text)  # giữ tiếng Việt có dấu
    # Tách chữ và số rời
    text = re.sub(r"(\d)", r" \1 ", text)
    tokens = text.split()
    return tokens