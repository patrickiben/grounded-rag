from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Tuple

from .corpus import Document
from .text import tokenize


class TfidfRetriever:
    """A small, dependency-free lexical retriever (TF-IDF + cosine similarity).

    The interface (``query(text, k) -> [(score, Document), ...]``) is kept
    deliberately minimal so a dense / embedding-backed retriever or a vector
    database can be dropped in later without touching the rest of the pipeline.
    """

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self._tokens = [tokenize(f"{d.title} {d.text}") for d in documents]

        df: Counter = Counter()
        for toks in self._tokens:
            df.update(set(toks))
        n = max(1, len(documents))
        self.idf: Dict[str, float] = {
            t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()
        }
        self._vecs = [self._vectorize(toks) for toks in self._tokens]

    def _vectorize(self, toks: List[str]) -> Dict[str, float]:
        tf = Counter(toks)
        total = max(1, len(toks))
        return {t: (c / total) * self.idf.get(t, 0.0) for t, c in tf.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return num / (na * nb) if na and nb else 0.0

    def query(self, text: str, k: int = 3) -> List[Tuple[float, Document]]:
        qv = self._vectorize(tokenize(text))
        scored = [
            (self._cosine(qv, dv), doc)
            for dv, doc in zip(self._vecs, self.documents)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]
