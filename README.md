# grounded-rag

**A tiny, dependency-free RAG demo with numeric-membership and lexical screening, including explicit tests of its semantic blind spots.**

![ci](https://github.com/patrickiben/grounded-rag/actions/workflows/ci.yml/badge.svg)
&nbsp;·&nbsp; Python ≥ 3.9 &nbsp;·&nbsp; zero runtime dependencies &nbsp;·&nbsp; MIT

The loop is retrieve → extractive answer → numeric and lexical screening. The evaluator flags low word overlap, recognized numbers missing from the retrieved context, and simple negation mismatches. It is a deterministic review aid, not an entailment model.

> **Read this first:** passing lexical checks does not prove source support. Replacing “increased” with “decreased,” swapping units, or attaching a real number to a different claim can still pass. These limits are executable in `tests/test_screening.py`. The output says **PASSES LEXICAL CHECKS**; `.grounded` remains a compatibility alias for `.passes_lexical_checks`, with the same limited meaning. The negation heuristic can miss scope or flag benign differences; it is not a contradiction model. An empty answer fails with zero coverage. See [the numeric grammar and synchronization contract](NUMERIC_CONTRACT.md).

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
  grounding: PASSES LEXICAL CHECKS  (passing sentences 2/2, coverage 100%)
   - every answer sentence passes numeric/lexical screening; semantic support is unverified.
========================================================================
GROUNDING -- answer with injected hallucinations
  answer: ... The treatment was considered curative in most cases. Separately, 85% of participants discontinued.
  grounding: FAILS LEXICAL CHECKS  (passing sentences 2/4, coverage 50%)
   - [ERROR] unsupported_claim: low lexical overlap (support 20%): "The treatment was considered curative in most cases."
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
print(result["grounding"].passes_lexical_checks)  # True; not proof of entailment
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
| **Answer** | `answerer.py` | An **extractive stand-in for an LLM** — returns the retrieved sentences most relevant to the query. Copied from retrieved text, so the demo runs with no API key. Swap in a real model and the grounding check still applies. |
| **Evaluate grounding** | `grounding.py` | The point of the project. Each nonempty answer sentence must share enough content words with a source, contain recognized numbers present in the context, and avoid a simple negation mismatch with its closest source. Findings include `unsupported_claim`, `ungrounded_number`, `numeric_format`, `possible_contradiction`, and `empty_answer`. None establishes entailment. |

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

Run `python -m pytest -q` (or just `python selftest.py`). CI checks Python 3.9, 3.12 and 3.14 and imports an installed wheel in an isolated environment. Invalid settings or malformed source numeric formats raise `ValueError`; malformed answer formats yield a failing report.

---

## Roadmap

- [ ] Pluggable **embedding / vector-DB retriever** behind the same interface
- [ ] **LLM-backed answerer** (local model or API) with the grounding check unchanged
- [ ] **Span-level citations** (point to the exact supporting sentence)
- [ ] Semantic entailment beyond the implemented simple negation heuristic; units and date grounding
- [x] Shared vendored numeric contract with [`numguard`](https://github.com/patrickiben/numguard), including synchronized regression fixtures

---

## Status & honesty note

A **working proof-of-concept** on a synthetic corpus, not a production RAG system. The retriever is lexical (not semantic) and the answerer is extractive (not generative) — both deliberately, so the grounding evaluator can be demonstrated with zero dependencies. The contribution is a small, testable lexical/numeric screen with a declared boundary: some unsupported answers will pass.

## License

MIT — see [LICENSE](LICENSE). Built by Patrick Iben (`p.iben@saeny.net`).
