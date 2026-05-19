"""Regression tests for the 0.8.1 bug fix: built-in adapters now auto-inject
``EvalCase.context`` into the LLM prompt when called via the suite.

Pre-0.8.1, ``run_with_anthropic`` / ``run_with_openai`` dropped context
entirely — every RAG case effectively ran with no grounding. These tests
mock the underlying client + verify the system prompt actually receives
the context block.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from multivon_eval import EvalCase, EvalSuite, NotEmpty
from multivon_eval.adapters import (
    AnthropicAdapter, OpenAIAdapter, LiteLLMAdapter,
    _format_context_block, _RAG_SYSTEM_PREFIX,
)


# ─── _format_context_block (helper) ──────────────────────────────────────


def test_format_context_block_handles_none():
    assert _format_context_block(None) == ""


def test_format_context_block_handles_empty_string():
    assert _format_context_block("") == ""


def test_format_context_block_handles_string():
    out = _format_context_block("Refunds within 30 days.")
    assert "Refunds within 30 days." in out
    assert "Context:" in out


def test_format_context_block_handles_list():
    out = _format_context_block(["First chunk.", "Second chunk."])
    assert "First chunk." in out
    assert "Second chunk." in out
    assert "[chunk 1]" in out
    assert "[chunk 2]" in out


def test_format_context_block_skips_falsy_chunks_in_list():
    out = _format_context_block(["A", "", None, "B"])
    assert "[chunk 1]" in out
    assert "A" in out
    assert "B" in out


# ─── AnthropicAdapter context injection ──────────────────────────────────


def _stub_anthropic_response(text: str) -> Any:
    resp = MagicMock()
    block = MagicMock()
    block.text = text
    resp.content = [block]
    return resp


def test_anthropic_adapter_call_with_case_injects_context():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _stub_anthropic_response("Paris")
    adapter = AnthropicAdapter("claude-haiku-4-5-20251001", client=fake_client)

    case = EvalCase(
        input="What is the capital of France?",
        context="France is a country in Western Europe. Its capital is Paris.",
    )
    out = adapter._call_with_case(case)

    assert out == "Paris"
    # The call must have happened with a system prompt containing the context
    _, kw = fake_client.messages.create.call_args
    assert "system" in kw
    assert "France is a country in Western Europe" in kw["system"]
    assert "Paris" in kw["system"]
    assert _RAG_SYSTEM_PREFIX[:30] in kw["system"]


def test_anthropic_adapter_call_with_case_with_no_context():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _stub_anthropic_response("4")
    adapter = AnthropicAdapter("claude-haiku-4-5-20251001", client=fake_client)

    case = EvalCase(input="2+2?")
    out = adapter._call_with_case(case)

    assert out == "4"
    _, kw = fake_client.messages.create.call_args
    # No context → no system prompt added unless one was configured.
    assert "system" not in kw or _RAG_SYSTEM_PREFIX not in kw.get("system", "")


def test_anthropic_adapter_preserves_user_system_prompt_when_no_context():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _stub_anthropic_response("ok")
    adapter = AnthropicAdapter(
        "claude-haiku-4-5-20251001",
        client=fake_client,
        system_prompt="You are a helpful assistant.",
    )

    out = adapter._call_with_case(EvalCase(input="hello"))
    _, kw = fake_client.messages.create.call_args
    assert kw["system"] == "You are a helpful assistant."


def test_anthropic_adapter_combines_user_system_prompt_with_rag_context():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _stub_anthropic_response("ok")
    adapter = AnthropicAdapter(
        "claude-haiku-4-5-20251001",
        client=fake_client,
        system_prompt="You are a helpful assistant.",
    )

    case = EvalCase(input="What is X?", context="X is Y.")
    adapter._call_with_case(case)
    _, kw = fake_client.messages.create.call_args
    assert "You are a helpful assistant." in kw["system"]
    assert "X is Y." in kw["system"]


def test_anthropic_adapter_handles_list_context():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _stub_anthropic_response("ok")
    adapter = AnthropicAdapter("claude-haiku-4-5-20251001", client=fake_client)

    case = EvalCase(input="x", context=["chunk A.", "chunk B."])
    adapter._call_with_case(case)
    _, kw = fake_client.messages.create.call_args
    assert "chunk A." in kw["system"]
    assert "chunk B." in kw["system"]


# ─── OpenAIAdapter context injection ─────────────────────────────────────


def _stub_openai_response(text: str) -> Any:
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    resp.choices = [choice]
    return resp


def test_openai_adapter_call_with_case_injects_context():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _stub_openai_response("Paris")
    adapter = OpenAIAdapter("gpt-4o-mini", client=fake_client)

    case = EvalCase(input="capital?", context="France's capital is Paris.")
    out = adapter._call_with_case(case)

    assert out == "Paris"
    _, kw = fake_client.chat.completions.create.call_args
    system_msg = next(m for m in kw["messages"] if m["role"] == "system")
    assert "Paris" in system_msg["content"]
    assert _RAG_SYSTEM_PREFIX[:30] in system_msg["content"]


def test_openai_adapter_call_with_case_no_context_no_rag_prefix():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _stub_openai_response("ok")
    adapter = OpenAIAdapter("gpt-4o-mini", client=fake_client)

    adapter._call_with_case(EvalCase(input="hello"))
    _, kw = fake_client.chat.completions.create.call_args
    # System prompt absent → only user message
    roles = [m["role"] for m in kw["messages"]]
    assert "user" in roles
    if "system" in roles:
        sysm = next(m for m in kw["messages"] if m["role"] == "system")
        assert _RAG_SYSTEM_PREFIX[:30] not in sysm["content"]


# ─── End-to-end via suite.run() ──────────────────────────────────────────


def test_suite_routes_to_call_with_case_when_available():
    """Suite must prefer _call_with_case over __call__ when adapter exposes it."""
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _stub_anthropic_response("Paris")
    adapter = AnthropicAdapter("claude-haiku-4-5-20251001", client=fake_client)

    suite = EvalSuite("Test")
    suite.add_cases([
        EvalCase(input="capital?", context="France's capital is Paris."),
    ])
    suite.add_evaluators(NotEmpty())
    report = suite.run(adapter, verbose=False)

    assert report.total == 1
    assert report.pass_rate == 1.0  # NotEmpty passes
    # The adapter received context in its system prompt
    _, kw = fake_client.messages.create.call_args
    assert "Paris" in kw["system"]


def test_suite_falls_back_to_call_for_plain_callables():
    """A plain function with no _call_with_case attribute uses the old path."""
    calls: list[str] = []

    def plain_model(input: str) -> str:
        calls.append(input)
        return f"echo: {input}"

    suite = EvalSuite("Plain")
    suite.add_cases([EvalCase(input="hi", context="ignored")])
    suite.add_evaluators(NotEmpty())
    report = suite.run(plain_model, verbose=False)

    assert report.pass_rate == 1.0
    assert calls == ["hi"]
