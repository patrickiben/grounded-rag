# grounded-rag

**A tiny, dependency-free RAG demo — with the part most RAG demos skip: an evaluator that checks whether the answer is actually supported by the retrieved sources.**

![ci](https://github.com/patrickiben/grounded-rag/actions/workflows/ci.yml/badge.svg)
&nbsp;·&nbsp; Python ≥ 3.9 &nbsp;·&nbsp; zero runtime dependencies &nbsp;·&nbsp; MIT

Retrieval-augmented generation is easy to demo and hard to trust: the model can still answer with something the sources don't say. `grounded-rag` shows the whole loop — retrieve → answer → **grounding evaluation** — and treats the evaluation as the main event. Every answer sentence is checked against the retrieved context: a sentence with **no lexical support** is flagged as an unsupported claim, and a **number that appears in no source** is flagged as a hallucination. It's the same grounding discipline as the companion [`numguard`](https://github.com/patrickiben/numguard) project, applied to RAG answers.

---

## Quickstart (no install, no dependencies)

```bash
python selftest.py
```

Real output:

```
========================================================================
RETRIEVAL
  [OK ] gold=ae      top3=['ae', 'elig', 'dosing']
        q: What proportion of participants reported an adverse event?
  [OK ] gold=pk      top3=['pk', 'elig', 'dosing']
        q: What was the mean terminal half-life?
  [OK ] gold=dosing  top3=['dosing', 'pk', 'elig']
        q: What dose levels were studied?
  [OK ] gold=elig    top3=['elig', 'dosing', 'ae']
        q: What were the age eligibility criteria?
  hit@3 = 4/4 (100%)    MRR = 1.000
========================================================================
GROUNDING -- well-formed (extractive) answer
  answer: In the safety population of 24 participants, 19 participants (79.2%) ...
  grounding: GROUNDED  (supported 2/2, coverage 100%)
   - every answer sentence is supported by the sources.
========================================================================
GROUNDING -- answer with injected hallucinations
  answer: ... The treatment was considered curative in most cases. Separately, 85% of participants discontinued.
  grounding: NOT GROUNDED  (supported 2/4, coverage 50%)
   - [ERROR] unsupported_claim: not supported by sources (support 20%): "The treatment was considered curative in most cases."
   - [ERROR] ungrounded_number: number(s) 85 appear in no source: "Separately, 85% of participants discontinued."
========================================================================
SELFTEST PASSED -- retrieval hits the gold docs; grounded answers pass; hallucinated content is flagged.
```

---

## Programmatic use

```python
from groundedrag import load_corpus, TfidfRetriever, answer_question

retriever = TfidfRetriever(load_corpus("data/corpus.json"))
result = answer_question("What was the mean terminal half-life?", retriever, k=3)

print(result["answer"])            # "The mean terminal half-life was approximately 7.8 hours."
print(result["citations"])         # ['pk']
print(result["grounding"].grounded)  # True
print(result["grounding"].scorecard())
```

---

## How it works

```
   question ──► retrieve (TF-IDF) ──► answer ──► grounding eval ──► (answer, citations, grounding report)
                  top-k docs           extractive    every sentence checked
                                       stand-in       against the retrieved sources
```

Three small, swappable pieces:

| Stage | Module | What it does |
|-------|--------|--------------|
| **Retrieve** | `retriever.py` | Dependency-free TF-IDF + cosine over the corpus. Reports `hit@k` / `MRR` against gold docs. The `query()` interface is minimal so a dense / embedding / vector-DB retriever drops in unchanged. |
| **Answer** | `answerer.py` | An **extractive stand-in for an LLM** — returns the retrieved sentences most relevant to the query. Grounded by construction, so the demo runs with no API key. Swap in a real model and the grounding check still applies. |
| **Evaluate grounding** | `grounding.py` | The point of the project. Each answer sentence must (a) share enough content words with some context sentence and (b) contain only numbers that appear in the context. Otherwise it's flagged `unsupported_claim` or `ungrounded_number`. |

Why "extractive answerer"? So the **valuable, differentiated part — the grounding evaluator and the retrieval metrics — is fully real and runs anywhere**, instead of hiding behind an API key. The honest framing matters: this demonstrates the *evaluation harness*, not a production generator.

---

## Project layout

```
grounded-rag/
├── groundedrag/
│   ├── corpus.py       # Document + corpus loader
│   ├── text.py         # tokenization, sentence split, number extraction
│   ├── retriever.py    # TF-IDF retriever (pluggable interface)
│   ├── answerer.py     # extractive LLM stand-in
│   ├── grounding.py    # the grounding evaluator  <-- the core
│   └── pipeline.py     # retrieve -> answer -> evaluate
├── data/corpus.json    # synthetic CP-101 clinical corpus
├── data/questions.json # eval questions with gold doc ids
├── tests/test_rag.py
├── examples/demo.py
├── selftest.py         # zero-dependency runnable proof
└── .github/workflows/ci.yml
```

Run the tests with `pytest -q` (or just `python selftest.py`).

---

## Roadmap

- [ ] Pluggable **embedding / vector-DB retriever** behind the same interface
- [ ] **LLM-backed answerer** (local model or API) with the grounding check unchanged
- [ ] **Span-level citations** (point to the exact supporting sentence)
- [ ] Richer grounding (negation/contradiction, units, date grounding)
- [ ] Shared grounding core with [`numguard`](https://github.com/patrickiben/numguard)

---

## Status & honesty note

A **working proof-of-concept** on a synthetic corpus, not a production RAG system. The retriever is lexical (not semantic) and the answerer is extractive (not generative) — both deliberately, so the grounding evaluator can be demonstrated with zero dependencies. The grounding evaluation itself is the contribution: a concrete, testable way to catch answers the sources don't support.

## License

MIT — see [LICENSE](LICENSE). Built by Patrick Iben (`p.iben@saeny.net`).
