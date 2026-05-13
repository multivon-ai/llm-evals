"""Tests for the multivon-eval pytest plugin.

These tests do NOT spawn a pytester sub-pytest (would require pytest's
pytester fixture and adds setup complexity). They directly exercise the
plugin's public surface so the contract is locked in.
"""
from __future__ import annotations

import pytest

from multivon_eval import EvalCase
from multivon_eval.evaluators.deterministic import NotEmpty, ExactMatch
from multivon_eval.pytest_plugin import (
    EvaluatorFailure,
    assert_evaluators,
)


class TestAssertEvaluators:
    def test_passes_when_all_evaluators_pass(self):
        case = EvalCase(input="x", expected_output="hello")
        results = assert_evaluators(case, "hello", [NotEmpty(), ExactMatch()])
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_raises_evaluator_failure_when_any_fails(self):
        case = EvalCase(input="x", expected_output="hello")
        with pytest.raises(EvaluatorFailure) as info:
            assert_evaluators(case, "wrong", [ExactMatch()])
        # EvaluatorFailure is an AssertionError so pytest treats it normally.
        assert isinstance(info.value, AssertionError)

    def test_failure_message_includes_evaluator_details(self):
        case = EvalCase(input="x", expected_output="hello", context="some context")
        with pytest.raises(EvaluatorFailure) as info:
            assert_evaluators(case, "wrong", [ExactMatch()])
        msg = str(info.value)
        assert "exact_match" in msg
        assert "score=0.000" in msg or "score=0.0" in msg
        assert "case input" in msg
        assert "actual out" in msg

    def test_failure_carries_structured_results(self):
        case = EvalCase(input="x", expected_output="hello")
        with pytest.raises(EvaluatorFailure) as info:
            assert_evaluators(case, "wrong", [NotEmpty(), ExactMatch()])
        failures = [r for r in info.value.results if not r.passed]
        assert len(failures) == 1
        assert failures[0].evaluator == "exact_match"

    def test_passes_with_runs_gt_1(self):
        # Sync deterministic evaluator: 3 runs always produce identical results.
        case = EvalCase(input="x")
        results = assert_evaluators(case, "non-empty", [NotEmpty()], runs=3)
        assert len(results) == 1
        assert results[0].passed
        assert results[0].metadata.get("runs") == 3
        assert results[0].metadata.get("all_scores") == [1.0, 1.0, 1.0]

    def test_majority_vote_with_split_results(self, monkeypatch):
        """If 2 of 3 runs pass, majority verdict passes."""
        from multivon_eval.evaluators.base import Evaluator
        from multivon_eval.result import EvalResult

        class _Flipper(Evaluator):
            name = "flipper"

            def __init__(self):
                super().__init__(threshold=0.5)
                self.n = 0

            def evaluate(self, case, output):
                self.n += 1
                passed = self.n != 2  # 1st pass, 2nd fail, 3rd pass
                return self._result(
                    score=1.0 if passed else 0.0,
                    reason=f"call {self.n}",
                )

        case = EvalCase(input="x")
        results = assert_evaluators(case, "y", [_Flipper()], runs=3)
        assert results[0].passed  # 2/3 majority
        assert "2/3" in results[0].reason


class TestEvaluatorFailureClass:
    def test_format_includes_failed_evaluators_only(self):
        from multivon_eval.result import EvalResult

        case = EvalCase(input="x")
        passing = EvalResult(evaluator="ok", score=1.0, passed=True)
        failing = EvalResult(evaluator="bad", score=0.1, passed=False, reason="boom")
        exc = EvaluatorFailure([passing, failing], case=case, output="o")
        msg = str(exc)
        assert "bad" in msg
        assert "boom" in msg
        # passing evaluators don't show up
        assert "evaluator=ok" not in msg
