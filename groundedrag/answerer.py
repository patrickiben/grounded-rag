from __future__ import annotations

from typing import List, Tuple

from .corpus import Document
from .text import split_sentences, tokenize


def extractive_answer(
    query: str,
    retrieved: List[Tuple[float, Document]],
    max_sentences: int = 2,
) -> Tuple[str, List[str]]:
    """A zero-dependency stand-in for an LLM generator.

    Returns the retrieved sentences most relevant to the query. It is grounded
    *by construction* -- which is exactly the point: it lets the grounding
    *evaluator* be demonstrated with no API key. Swap this for a real model and
    the grounding check (the valuable part) applies unchanged.
    """
    q = set(tokenize(query))
    candidates: List[Tuple[float, str, str]] = []
    for _, doc in retrieved:
        for sent in split_sentences(doc.text):
            st = set(tokenize(sent))
            if not st:
                continue
            overlap = len(q & st) / len(q) if q else 0.0
            candidates.append((overlap, sent, doc.id))

    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = [c for c in candidates[:max_sentences] if c[0] > 0]
    answer = " ".join(c[1] for c in chosen)
    citations = sorted({c[2] for c in chosen})
    return answer, citations
