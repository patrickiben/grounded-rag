"""Boundary checks: lexical screening is not semantic entailment."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from groundedrag import evaluate_grounding
from groundedrag.corpus import Document


def context(text):
    return [(1.0, Document("test", "Test", text))]


@pytest.mark.parametrize("answer", ["", "   ", "the and of"])
def test_empty_or_unevaluable_answers_fail_with_zero_coverage(answer):
    report = evaluate_grounding(answer, context("No serious events occurred."))
    assert not report.passes_lexical_checks
    assert not report.grounded  # compatibility property has identical meaning
    assert report.coverage == 0
    assert report.findings[0].kind == "empty_answer"


@pytest.mark.parametrize("source,answer", [
    ("No serious adverse events were reported.", "Serious adverse events were reported."),
    ("Serious adverse events were reported.", "No serious adverse events were reported."),
])
def test_simple_negation_reversal_is_flagged(source, answer):
    report = evaluate_grounding(answer, context(source))
    assert not report.passes_lexical_checks
    assert report.findings[0].kind == "possible_contradiction"


def test_same_negation_passes():
    text = "No serious adverse events were reported."
    report = evaluate_grounding(text, context(text))
    assert report.passes_lexical_checks
    assert "PASSES LEXICAL CHECKS" in report.scorecard()
    assert "semantic support is unverified" in report.scorecard()


@pytest.mark.parametrize("answer", ["The change was -2.3 mg.", "The change was .5 mg.", "The change was 2.3e9 mg."])
def test_numeric_meaning_cannot_alias_a_different_source_value(answer):
    assert not evaluate_grounding(answer, context("The change was 2.3 mg.")).passes_lexical_checks


def test_malformed_draft_number_fails_screening():
    report = evaluate_grounding("The rate was 79,2%.", context("The rate was 79.2%."))
    assert report.findings[0].kind == "numeric_format"


@pytest.mark.parametrize("kwargs", [{"num_tol": float("inf")}, {"num_tol": -1}, {"overlap_threshold": float("nan")}, {"overlap_threshold": 1.1}])
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        evaluate_grounding("The rate was 1%.", context("The rate was 1%."), **kwargs)


def test_known_gap_verb_reversal_can_pass_lexical_screening():
    # Keep the limit executable: overlap does not prove semantic support.
    source = "Treatment A increased participant survival."
    report = evaluate_grounding("Treatment A decreased participant survival.", context(source))
    assert report.passes_lexical_checks


def test_known_gap_unit_swap_can_pass_lexical_screening():
    report = evaluate_grounding("The dose was 10 g.", context("The dose was 10 mg."))
    assert report.passes_lexical_checks


def test_missing_sources_fail_even_if_overlap_check_is_disabled():
    report = evaluate_grounding("A statement without sources.", [], overlap_threshold=0)
    assert not report.passes_lexical_checks
