"""ContextRecall must skip (passed=True, score=1.0) when expected_output is
missing — not return 0.0 (which made it look like a quality failure when
the data shape just didn't support the metric).

Surfaced by a 0.8.1 dogfood pass: EvalSuite.for_rag() auto-adds ContextRecall
and users running RAG cases without expected_output saw a confusing 0.0
result tagged "Requires both case.context and case.expected_output".
"""
from __future__ import annotations

from multivon_eval import EvalCase
from multivon_eval.evaluators.llm_judge import ContextRecall


def test_context_recall_skips_when_expected_output_missing():
    case = EvalCase(input="Q", context="some retrieved context")
    result = ContextRecall().evaluate(case, output="A")
    assert result.passed is True
    assert result.score == 1.0
    assert "[skipped]" in result.reason
    assert result.metadata.get("skipped") is True


def test_context_recall_skips_when_context_missing():
    case = EvalCase(input="Q", expected_output="A")
    result = ContextRecall().evaluate(case, output="A")
    assert result.passed is True
    assert result.score == 1.0
    assert "[skipped]" in result.reason


def test_context_recall_skips_when_both_missing():
    case = EvalCase(input="Q")
    result = ContextRecall().evaluate(case, output="A")
    assert result.passed is True
    assert "[skipped]" in result.reason
