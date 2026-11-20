"""Modeller arasında paylaşılacak temel metin normalizasyonu."""

import re
import unicodedata


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"https?://\S+", " urltoken ", text)
    return re.sub(r"\s+", " ", text).strip()
