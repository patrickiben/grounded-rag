"""Dependency-free RAG demo with lexical and numeric screening.

Retrieve, extract source sentences, and screen numeric membership, word overlap
and simple negation mismatch. A passing verdict does not establish semantic
entailment; explicit known-gap tests preserve that distinction.
"""
from .answerer import extractive_answer
from .corpus import Document, load_corpus
from .grounding import GroundingFinding, GroundingReport, evaluate_grounding
from .pipeline import answer_question
from .retriever import TfidfRetriever

__all__ = [
    "Document",
    "load_corpus",
    "TfidfRetriever",
    "extractive_answer",
    "GroundingReport",
    "GroundingFinding",
    "evaluate_grounding",
    "answer_question",
]
__version__ = "0.1.0"
