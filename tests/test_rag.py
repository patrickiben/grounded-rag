import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from groundedrag import (  # noqa: E402
    TfidfRetriever,
    answer_question,
    evaluate_grounding,
    load_corpus,
)


def _retriever():
    docs = load_corpus(os.path.join(BASE, "data", "corpus.json"))
    return TfidfRetriever(docs)


def test_retrieval_hits_gold_docs():
    r = _retriever()
    with open(os.path.join(BASE, "data", "questions.json"), encoding="utf-8") as fh:
        questions = json.load(fh)["questions"]
    for item in questions:
        ids = [d.id for _, d in r.query(item["q"], k=3)]
        assert item["gold"] in ids, f"{item['gold']} not in {ids}"


def test_top1_is_gold_for_a_clear_question():
    r = _retriever()
    ids = [d.id for _, d in r.query("What was the mean terminal half-life?", k=1)]
    assert ids == ["pk"]


def test_grounded_answer_passes():
    r = _retriever()
    res = answer_question(
        "What proportion of participants reported an adverse event?", r, k=3
    )
    assert res["grounding"].grounded, res["grounding"].scorecard()
    assert "ae" in res["citations"]


def test_ungrounded_number_is_flagged():
    r = _retriever()
    ctx = r.query("adverse events", k=3)
    rep = evaluate_grounding("In total, 85% of participants discontinued.", ctx)
    assert not rep.grounded
    assert any(f.kind == "ungrounded_number" for f in rep.findings)


def test_unsupported_claim_is_flagged():
    r = _retriever()
    ctx = r.query("adverse events", k=3)
    rep = evaluate_grounding("The treatment was considered curative in most cases.", ctx)
    assert not rep.grounded
    assert any(f.kind == "unsupported_claim" for f in rep.findings)


def test_supported_sentence_passes():
    r = _retriever()
    ctx = r.query("adverse events", k=3)
    rep = evaluate_grounding("No serious adverse events were reported.", ctx)
    assert rep.grounded, rep.scorecard()
