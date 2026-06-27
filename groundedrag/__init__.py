"""grounded-rag -- a tiny, dependency-free RAG demo with a grounding evaluator.

Retrieve over a corpus, answer extractively (a stand-in for an LLM), and -- the
part most RAG demos skip -- *evaluate whether the answer is actually supported
by the retrieved sources*: it flags answer sentences with no support and numbers
that appear in no source. The grounding check is the same discipline as the
companion ``numguard`` project, applied to retrieval-augmented answers.

No third-party dependencies. Run ``python selftest.py``.
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
