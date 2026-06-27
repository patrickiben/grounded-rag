from __future__ import annotations

from typing import Any, Dict

from .answerer import extractive_answer
from .grounding import evaluate_grounding
from .retriever import TfidfRetriever


def answer_question(query: str, retriever: TfidfRetriever, k: int = 3) -> Dict[str, Any]:
    """Retrieve -> answer -> evaluate grounding, in one call."""
    retrieved = retriever.query(query, k=k)
    answer, citations = extractive_answer(query, retrieved)
    grounding = evaluate_grounding(answer, retrieved)
    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "retrieved": [(round(score, 3), doc.id) for score, doc in retrieved],
        "grounding": grounding,
    }
