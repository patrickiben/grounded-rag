from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

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
    def grounded(self) -> bool:
        return all(f.severity != "ERROR" for f in self.findings)

    @property
    def coverage(self) -> float:
        return self.supported / self.total if self.total else 1.0

    def scorecard(self) -> str:
        head = "GROUNDED" if self.grounded else "NOT GROUNDED"
        lines = [
            f"  grounding: {head}  "
            f"(supported {self.supported}/{self.total}, coverage {self.coverage:.0%})"
        ]
        if not self.findings:
            lines.append("   - every answer sentence is supported by the sources.")
        for f in self.findings:
            lines.append(f"   - [{f.severity}] {f.kind}: {f.message}")
        return "\n".join(lines)


def evaluate_grounding(
    answer: str,
    context: List[Tuple[float, Document]],
    overlap_threshold: float = 0.5,
    num_tol: float = 0.05,
) -> GroundingReport:
    """Flag answer content not supported by the retrieved context.

    An answer sentence is *supported* when it shares enough content words with
    some context sentence AND every number it contains appears in the context.
    Ungrounded numbers are the hard failure (a fabricated figure); low lexical
    support flags an unsupported qualitative claim. Same grounding discipline as
    the ``numguard`` project, applied to RAG answers.
    """
    context_text = " ".join(d.text for _, d in context)
    ctx_sentence_tokens = [set(tokenize(s)) for s in split_sentences(context_text)]
    ctx_numbers = numbers(context_text)

    report = GroundingReport()
    for sent in split_sentences(answer):
        st = set(tokenize(sent))
        if not st:
            continue
        report.total += 1

        bad = [
            n for n in numbers(sent)
            if not any(abs(n - c) <= max(num_tol, 0.001 * abs(c)) for c in ctx_numbers)
        ]
        if bad:
            shown = ", ".join(str(int(b) if float(b).is_integer() else b) for b in bad)
            report.findings.append(GroundingFinding(
                "ERROR", "ungrounded_number",
                f'number(s) {shown} appear in no source: "{sent}"',
            ))
            continue

        best = max((len(st & cs) / len(st) for cs in ctx_sentence_tokens), default=0.0)
        if best < overlap_threshold:
            report.findings.append(GroundingFinding(
                "ERROR", "unsupported_claim",
                f'not supported by sources (support {best:.0%}): "{sent}"',
            ))
            continue

        report.supported += 1

    return report
