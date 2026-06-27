"""Minimal usage example: retrieve, answer, and check grounding."""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from groundedrag import TfidfRetriever, answer_question, load_corpus  # noqa: E402

docs = load_corpus(os.path.join(BASE, "data", "corpus.json"))
retriever = TfidfRetriever(docs)

for q in ["What dose levels were studied?", "What was the mean terminal half-life?"]:
    res = answer_question(q, retriever, k=3)
    print(f"\nQ: {q}")
    print(f"A: {res['answer']}")
    print(f"   cited: {res['citations']}   retrieved: {res['retrieved']}")
    print(res["grounding"].scorecard())
