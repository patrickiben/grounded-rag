from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from ._numeric import finite_number, validate_tolerances
from .corpus import Document
from .text import numbers, split_sentences, tokenize


@dataclass
class GroundingFinding:
    severity: str
    kind: str
    message: str


@dataclass
class GroundingReport:
    findings: List[GroundingFinding] = field(default_factory=list)
    supported: int = 0
    total: int = 0

    @property
    def passes_lexical_checks(self) -> bool:
        """A lexical screening verdict; not a claim of semantic entailment."""
        return self.total > 0 and all(f.severity != "ERROR" for f in self.findings)

    @property
    def grounded(self) -> bool:
        """Compatibility alias for passes_lexical_checks; does not prove support."""
        return self.passes_lexical_checks

    @property
    def coverage(self) -> float:
        return self.supported / self.total if self.total else 0.0

    def scorecard(self) -> str:
        head = "PASSES LEXICAL CHECKS" if self.passes_lexical_checks else "FAILS LEXICAL CHECKS"
        lines = [
            f"  grounding: {head}  "
            f"(passing sentences {self.supported}/{self.total}, coverage {self.coverage:.0%})"
        ]
        if not self.findings:
            if self.total:
                lines.append("   - every answer sentence passes numeric/lexical screening; semantic support is unverified.")
            else:
                lines.append("   - no evaluable answer content.")
        for f in self.findings:
            lines.append(f"   - [{f.severity}] {f.kind}: {f.message}")
        return "\n".join(lines)


def evaluate_grounding(
    answer: str,
    context: List[Tuple[float, Document]],
    overlap_threshold: float = 0.5,
    num_tol: float = 0.05,
) -> GroundingReport:
    """Screen numeric membership, lexical overlap and simple negation mismatch.

    This heuristic does not establish entailment. Word swaps, units, attribution
    and nuanced negation remain known gaps even when every sentence passes.
    Empty answers fail, with zero coverage. Invalid settings/context raise
    ValueError; malformed numeric tokens in an answer produce a failing report.
    """
    finite_number(overlap_threshold, "overlap_threshold")
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be in [0, 1]")
    validate_tolerances(num_tol, 0.001)
    context_text = " ".join(d.text for _, d in context)
    context_sentences = split_sentences(context_text)
    ctx_sentence_tokens = [_content(s) for s in context_sentences]
    ctx_numbers = numbers(context_text)

    report = GroundingReport()
    for sent in split_sentences(answer):
        st = _content(sent)
        if not st:
            continue
        report.total += 1
        try:
            answer_numbers = numbers(sent)
        except ValueError as error:
            report.findings.append(GroundingFinding("ERROR", "numeric_format", str(error)))
            continue
        bad = [
            n for n in answer_numbers
            if not any(abs(n - c) <= max(num_tol, 0.001 * abs(c)) for c in ctx_numbers)
        ]
        if bad:
            shown = ", ".join(str(int(b) if float(b).is_integer() else b) for b in bad)
            report.findings.append(GroundingFinding(
                "ERROR", "ungrounded_number",
                f'number(s) {shown} appear in no source: "{sent}"',
            ))
            continue

        overlaps = [len(st & cs) / len(st) for cs in ctx_sentence_tokens]
        best = max(overlaps, default=0.0)
        if not overlaps or best < overlap_threshold:
            report.findings.append(GroundingFinding(
                "ERROR", "unsupported_claim",
                f'low lexical overlap (support {best:.0%}): "{sent}"',
            ))
            continue
        strongest = [source for source, overlap in zip(context_sentences, overlaps) if overlap == best]
        if strongest and all(_negated(source) != _negated(sent) for source in strongest):
            report.findings.append(GroundingFinding(
                "ERROR", "possible_contradiction",
                f'negation differs from the closest source sentence: "{sent}"',
            ))
            continue
        report.supported += 1

    if report.total == 0:
        report.findings.append(GroundingFinding(
            "ERROR", "empty_answer", "No evaluable answer content; coverage is zero.",
        ))
    return report


_NEGATION = re.compile(r"\b(?:no|not|never|without|neither|nor)\b|\b\w+n['’]t\b", re.I)


def _negated(sentence: str) -> bool:
    return bool(_NEGATION.search(sentence))


def _content(sentence: str) -> set:
    return set(tokenize(_NEGATION.sub(" ", sentence)))
