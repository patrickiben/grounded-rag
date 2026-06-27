from __future__ import annotations

import re
from typing import List

_WORD = re.compile(r"[A-Za-z0-9.]+")
_SENT = re.compile(r"(?<=[.!?])\s+")
# standalone numbers; lookbehind avoids matching digits inside identifiers (CP101)
_NUM = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)")

_STOP = {
    "the", "a", "an", "of", "in", "to", "and", "or", "was", "were", "is", "are",
    "be", "at", "on", "for", "with", "by", "as", "that", "this", "from", "had",
    "has", "have", "it", "its", "not", "no", "least", "one", "what", "which",
    "how", "many", "much", "did", "do", "does", "any", "were", "been",
}


def tokenize(text: str) -> List[str]:
    """Lowercased content tokens (numbers kept, stopwords dropped)."""
    out: List[str] = []
    for w in _WORD.findall(text.lower()):
        w = w.strip(".")
        if w and w not in _STOP:
            out.append(w)
    return out


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def numbers(text: str) -> List[float]:
    out: List[float] = []
    for m in _NUM.finditer(text):
        try:
            out.append(float(m.group(1).replace(",", "")))
        except ValueError:  # pragma: no cover
            continue
    return out
