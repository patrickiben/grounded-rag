#!/usr/bin/env python3
"""Zero-dependency self-test for grounded-rag.

Verifies three things:
  1. Retrieval finds the gold document for every question (hit@k, MRR).
  2. A well-formed (extractive) answer passes the grounding evaluator.
  3. An answer with an injected unsupported claim and a fabricated number is
     flagged by the grounding evaluator.

Exits 0 if all expectations hold, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from groundedrag.corpus import load_corpus  # noqa: E402
from groundedrag.grounding import evaluate_grounding  # noqa: E402
from groundedrag.pipeline import answer_question  # noqa: E402
from groundedrag.retriever import TfidfRetriever  # noqa: E402

K = 3


def main() -> int:
    docs = load_corpus(os.path.join(BASE, "data", "corpus.json"))
    with open(os.path.join(BASE, "data", "questions.json"), encoding="utf-8") as fh:
        questions = json.load(fh)["questions"]
    retriever = TfidfRetriever(docs)
    ok = True

    print("=" * 72)
    print("RETRIEVAL")
    hits = 0
    rr_sum = 0.0
    for item in questions:
        ids = [d.id for _, d in retriever.query(item["q"], k=K)]
        hit = item["gold"] in ids
        rank = ids.index(item["gold"]) + 1 if hit else 0
        rr_sum += (1.0 / rank) if rank else 0.0
        hits += 1 if hit else 0
        ok = ok and hit
        print(f"  [{'OK ' if hit else 'MISS'}] gold={item['gold']:7s} top{K}={ids}")
        print(f"        q: {item['q']}")
    n = len(questions)
    print(f"  hit@{K} = {hits}/{n} ({hits / n:.0%})    MRR = {rr_sum / n:.3f}")

    print("=" * 72)
    print("GROUNDING -- well-formed (extractive) answer")
    res = answer_question(
        "What proportion of participants reported an adverse event?", retriever, k=K
    )
    print(f"  answer: {res['answer']}")
    print(f"  citations: {res['citations']}   retrieved: {res['retrieved']}")
    print(res["grounding"].scorecard())
    if not res["grounding"].grounded:
        ok = False
        print("  UNEXPECTED: a well-formed answer was flagged as ungrounded.")

    print("=" * 72)
    print("GROUNDING -- answer with injected hallucinations")
    context = retriever.query("adverse events", k=K)
    tampered = (
        res["answer"]
        + " The treatment was considered curative in most cases."
        + " Separately, 85% of participants discontinued."
    )
    report = evaluate_grounding(tampered, context)
    print(f"  answer: {tampered}")
    print(report.scorecard())
    kinds = {f.kind for f in report.findings}
    if report.grounded or "ungrounded_number" not in kinds or "unsupported_claim" not in kinds:
        ok = False
        print("  UNEXPECTED: tampered answer was not fully flagged.")

    print("=" * 72)
    if ok:
        print("SELFTEST PASSED -- retrieval hits the gold docs; source extracts pass lexical checks; "
              "hallucinated content is flagged.")
        return 0
    print("SELFTEST FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
