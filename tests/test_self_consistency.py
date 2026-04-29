"""Tests for SelfConsistency evaluator."""
from __future__ import annotations
from unittest.mock import patch, MagicMock
import pytest

from multivon_eval import SelfConsistency, EvalCase, JudgeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _case(text: str = "What is the capital of France?") -> EvalCase:
    return EvalCase(input=text)


def _consistent_samples() -> list[str]:
    return [
        "Paris is the capital of France.",
        "France's capital is Paris.",
        "The city of Paris serves as France's capital.",
    ]


def _contradictory_samples() -> list[str]:
    return [
        "The capital of France is Lyon.",
        "France's capital is Marseille.",
        "Berlin is the capital of France.",
    ]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_construction():
    e = SelfConsistency()
    assert e.name == "self_consistency"
    assert e.threshold == 0.7
    assert e._n == 5
    assert e._max_n == 20
    assert e._adaptive is True
    assert e._backend == "auto"


def test_custom_params():
    e = SelfConsistency(n=3, max_n=10, adaptive=False, backend="llm", threshold=0.8)
    assert e._n == 3
    assert e._max_n == 10
    assert e._adaptive is False
    assert e._backend == "llm"
    assert e.threshold == 0.8


def test_judge_config_stored():
    cfg = JudgeConfig(provider="openai", model="gpt-4o-mini")
    e = SelfConsistency(judge=cfg)
    assert e._judge_cfg is cfg


# ---------------------------------------------------------------------------
# No samples / no model_fn
# ---------------------------------------------------------------------------

def test_no_samples_no_model_fn_returns_neutral():
    e = SelfConsistency()
    result = e.evaluate(_case(), "Paris is the capital.", samples=[])
    assert result.score == 0.5
    assert not result.passed


def test_no_model_fn_and_empty_samples_returns_neutral():
    """No model_fn and no samples → neutral 0.5 score, no crash."""
    e = SelfConsistency(n=3)
    result = e.evaluate(_case(), "Paris.", samples=[])
    assert result.score == 0.5


# ---------------------------------------------------------------------------
# LLM backend (mocked)
# ---------------------------------------------------------------------------

def _mock_judge_consistent(*args, **kwargs) -> str:
    return "Consistent"


def _mock_judge_contradicts(*args, **kwargs) -> str:
    return "Contradicts"


@patch("multivon_eval.evaluators.consistency.make_judge_call", side_effect=_mock_judge_consistent)
def test_llm_backend_all_consistent(mock_call):
    e = SelfConsistency(backend="llm")
    result = e.evaluate(_case(), "Paris is the capital of France.", samples=_consistent_samples())
    assert result.score == pytest.approx(1.0)
    assert result.passed
    assert "llm backend" in result.reason


@patch("multivon_eval.evaluators.consistency.make_judge_call", side_effect=_mock_judge_contradicts)
def test_llm_backend_all_contradictory(mock_call):
    e = SelfConsistency(backend="llm", threshold=0.7)
    result = e.evaluate(_case(), "Paris is the capital of France.", samples=_contradictory_samples())
    assert result.score == pytest.approx(0.0)
    assert not result.passed


@patch("multivon_eval.evaluators.consistency.make_judge_call", side_effect=_mock_judge_consistent)
def test_llm_backend_uses_pre_generated_samples_no_model_call(mock_call):
    """When enough samples are pre-generated (>= n), model_fn must not be called."""
    model_fn = MagicMock(return_value="some output")
    # Set n=3 so 3 pre-generated samples satisfy the requirement exactly
    e = SelfConsistency(model_fn=model_fn, n=3, backend="llm")
    e.evaluate(_case(), "Paris.", samples=_consistent_samples())
    model_fn.assert_not_called()


@patch("multivon_eval.evaluators.consistency.make_judge_call", side_effect=_mock_judge_consistent)
def test_llm_backend_calls_model_fn_when_samples_short(mock_call):
    """If fewer samples than n are provided, model_fn is called to top up."""
    model_fn = MagicMock(return_value="Paris is the capital.")
    e = SelfConsistency(model_fn=model_fn, n=5, backend="llm")
    # Provide 2 samples, need 5 — model_fn should be called 3 times
    e.evaluate(_case(), "Paris.", samples=_consistent_samples()[:2])
    assert model_fn.call_count == 3


# ---------------------------------------------------------------------------
# Adaptive escalation
# ---------------------------------------------------------------------------

@patch("multivon_eval.evaluators.consistency.make_judge_call")
def test_adaptive_escalates_on_borderline_score(mock_call):
    """Borderline score (between 0.3 and 0.7) should trigger escalation."""
    call_count = {"n": 0}

    def alternating(*args, **kwargs):
        call_count["n"] += 1
        return "Consistent" if call_count["n"] % 2 == 0 else "Contradicts"

    mock_call.side_effect = alternating

    model_fn = MagicMock(return_value="Some output.")
    e = SelfConsistency(model_fn=model_fn, n=3, max_n=10, adaptive=True, backend="llm")
    result = e.evaluate(_case(), "Some output.", samples=_consistent_samples()[:3])

    # Score alternates 50/50 → borderline → escalation → model_fn called
    assert model_fn.called


@patch("multivon_eval.evaluators.consistency.make_judge_call", side_effect=_mock_judge_consistent)
def test_no_escalation_when_adaptive_false(mock_call):
    model_fn = MagicMock(return_value="output")
    e = SelfConsistency(model_fn=model_fn, n=3, adaptive=False, backend="llm")
    e.evaluate(_case(), "output", samples=_consistent_samples()[:3])
    model_fn.assert_not_called()


@patch("multivon_eval.evaluators.consistency.make_judge_call", side_effect=_mock_judge_consistent)
def test_no_escalation_when_score_clear(mock_call):
    """High-confidence score should not trigger escalation."""
    model_fn = MagicMock(return_value="output")
    e = SelfConsistency(model_fn=model_fn, n=3, max_n=10, adaptive=True, backend="llm")
    e.evaluate(_case(), "output", samples=_consistent_samples())
    model_fn.assert_not_called()


# ---------------------------------------------------------------------------
# NLI backend (mocked)
# ---------------------------------------------------------------------------

def _make_nli_pipe(contradiction_score: float):
    """Return a mock NLI pipeline that always returns given contradiction prob."""
    def pipe(inputs, **kwargs):
        return [[
            {"label": "CONTRADICTION", "score": contradiction_score},
            {"label": "ENTAILMENT", "score": 1.0 - contradiction_score},
        ]]
    return pipe


@patch("multivon_eval.evaluators.consistency._load_nli_pipeline")
def test_nli_backend_high_contradiction(mock_load):
    mock_load.return_value = _make_nli_pipe(contradiction_score=0.9)
    e = SelfConsistency(backend="nli")
    result = e.evaluate(
        _case(), "Paris is the capital of France.",
        samples=_contradictory_samples(),
    )
    # High contradiction → low consistency score
    assert result.score < 0.5
    assert "nli backend" in result.reason


@patch("multivon_eval.evaluators.consistency._load_nli_pipeline")
def test_nli_backend_low_contradiction(mock_load):
    mock_load.return_value = _make_nli_pipe(contradiction_score=0.05)
    e = SelfConsistency(backend="nli")
    result = e.evaluate(
        _case(), "Paris is the capital of France.",
        samples=_consistent_samples(),
    )
    # Low contradiction → high consistency score
    assert result.score > 0.8
    assert result.passed


@patch("multivon_eval.evaluators.consistency._load_nli_pipeline")
def test_nli_backend_falls_back_to_llm_when_unavailable(mock_load):
    """If NLI pipeline returns None (transformers not installed), fall back to LLM."""
    mock_load.return_value = None
    with patch(
        "multivon_eval.evaluators.consistency.make_judge_call",
        side_effect=_mock_judge_consistent,
    ):
        e = SelfConsistency(backend="nli")
        result = e.evaluate(_case(), "Paris.", samples=_consistent_samples())
        assert result.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Auto backend selection
# ---------------------------------------------------------------------------

@patch("multivon_eval.evaluators.consistency._load_nli_pipeline")
def test_auto_selects_nli_when_available(mock_load):
    mock_load.return_value = _make_nli_pipe(0.1)
    e = SelfConsistency(backend="auto")
    result = e.evaluate(_case(), "Paris.", samples=_consistent_samples())
    assert "nli" in result.reason


@patch("multivon_eval.evaluators.consistency._load_nli_pipeline")
@patch("multivon_eval.evaluators.consistency.make_judge_call", side_effect=_mock_judge_consistent)
def test_auto_selects_llm_when_nli_unavailable(mock_call, mock_load):
    mock_load.return_value = None
    e = SelfConsistency(backend="auto")
    result = e.evaluate(_case(), "Paris.", samples=_consistent_samples())
    assert "llm" in result.reason


# ---------------------------------------------------------------------------
# Reason string
# ---------------------------------------------------------------------------

@patch("multivon_eval.evaluators.consistency.make_judge_call", side_effect=_mock_judge_consistent)
def test_reason_contains_sample_count(mock_call):
    e = SelfConsistency(backend="llm")
    result = e.evaluate(_case(), "Paris.", samples=_consistent_samples())
    assert "3 samples" in result.reason


@patch("multivon_eval.evaluators.consistency.make_judge_call", side_effect=_mock_judge_consistent)
def test_reason_contains_backend_name(mock_call):
    e = SelfConsistency(backend="llm")
    result = e.evaluate(_case(), "Paris.", samples=_consistent_samples())
    assert "llm backend" in result.reason
