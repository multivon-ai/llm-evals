"""
Synthetic dataset generation for multivon-eval.

Eliminates the cold-start problem: point at your docs or text and get
eval cases ready to run. No manually labeled data required to get started.

Usage:
    from multivon_eval import generate_from_text, generate_from_file

    # From raw text
    cases = generate_from_text(my_docs, n=20, task="qa")

    # From a file
    cases = generate_from_file("docs/faq.txt", n=15)

    # Hallucination pairs (faithful + hallucinated)
    pairs = generate_hallucination_pairs(my_docs, n=10)

    # Use immediately
    suite.add_cases(cases)
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Literal

from .case import EvalCase
from .evaluators.llm_judge import _judge_call


TaskType = Literal["qa", "summarization", "hallucination"]


def generate_from_text(
    text: str,
    n: int = 10,
    task: TaskType = "qa",
    context_window: int = 3000,
) -> list[EvalCase]:
    """
    Generate eval cases from raw text.

    Args:
        text:           Source text (docs, knowledge base, FAQ, etc.)
        n:              Number of cases to generate.
        task:           "qa" — question/answer pairs with context.
                        "summarization" — doc chunk + expected summary.
                        "hallucination" — faithful answer + expected_output="faithful".
        context_window: Max characters of source to include in each prompt.

    Returns:
        List of EvalCase objects ready to add to a suite.
    """
    text = text.strip()
    chunks = _chunk_text(text, context_window)

    if task == "qa":
        return _generate_qa(chunks, n)
    elif task == "summarization":
        return _generate_summarization(chunks, n)
    elif task == "hallucination":
        return _generate_hallucination(chunks, n)
    else:
        raise ValueError(f"Unknown task: {task!r}. Use 'qa', 'summarization', or 'hallucination'.")


def generate_from_file(
    path: str,
    n: int = 10,
    task: TaskType = "qa",
) -> list[EvalCase]:
    """
    Generate eval cases from a text file (.txt, .md, .rst, .py, etc.).

    Args:
        path:   Path to the source file.
        n:      Number of cases to generate.
        task:   See generate_from_text.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return generate_from_text(text, n=n, task=task)


def generate_hallucination_pairs(
    text: str,
    n: int = 10,
) -> list[dict]:
    """
    Generate faithful + hallucinated answer pairs for hallucination benchmarking.

    Returns a list of dicts: {question, context, faithful_answer, hallucinated_answer}.
    These can be used to build hallucination detection benchmarks like HaluEval.
    """
    chunks = _chunk_text(text, 3000)
    prompt = f"""You are building a hallucination detection benchmark dataset.

Source text:
\"\"\"
{chunks[0]}
\"\"\"

Generate {n} question-answer pairs. For each, provide:
1. A specific factual question answerable from the text
2. A faithful answer (directly grounded in the text)
3. A hallucinated answer (plausible-sounding but containing at least one false claim)

Return a JSON array. Each element:
{{
  "question": "...",
  "context": "the relevant excerpt from the source text",
  "faithful_answer": "...",
  "hallucinated_answer": "..."
}}

Return ONLY the JSON array, no commentary."""

    try:
        raw = _judge_call(prompt, max_tokens=3000)
        data = _extract_json_array(raw)
        return data[:n]
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}\nRaw response: {raw[:500] if 'raw' in dir() else 'none'}")


# ── Private helpers ────────────────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    step = max_chars - 200  # 200-char overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + max_chars]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _generate_qa(chunks: list[str], n: int) -> list[EvalCase]:
    cases = []
    per_chunk = max(1, n // len(chunks))
    remaining = n

    for chunk in chunks:
        if remaining <= 0:
            break
        batch = min(per_chunk, remaining)
        prompt = f"""Generate {batch} question-answer pairs from this text. Questions should require understanding the text, not just keyword lookup.

Text:
\"\"\"
{chunk}
\"\"\"

Return a JSON array. Each element:
{{
  "question": "...",
  "answer": "...",
  "context_excerpt": "the 1-3 sentences from the text that contain the answer"
}}

Return ONLY the JSON array."""

        try:
            raw = _judge_call(prompt, max_tokens=2000)
            data = _extract_json_array(raw)
            for item in data[:batch]:
                cases.append(EvalCase(
                    input=item.get("question", ""),
                    expected_output=item.get("answer", ""),
                    context=item.get("context_excerpt", chunk[:500]),
                    metadata={"generated": True, "task": "qa"},
                ))
            remaining -= len(data[:batch])
        except Exception:
            continue

    return cases[:n]


def _generate_summarization(chunks: list[str], n: int) -> list[EvalCase]:
    cases = []
    for chunk in chunks[:n]:
        prompt = f"""Write a 2-3 sentence faithful summary of this text. Include only information present in the text.

Text:
\"\"\"
{chunk[:2000]}
\"\"\"

Return a JSON object:
{{
  "summary": "..."
}}

Return ONLY the JSON object."""

        try:
            raw = _judge_call(prompt, max_tokens=300)
            data = _extract_json_object(raw)
            cases.append(EvalCase(
                input="Summarize the following text.",
                expected_output=data.get("summary", ""),
                context=chunk[:2000],
                metadata={"generated": True, "task": "summarization"},
            ))
        except Exception:
            continue

        if len(cases) >= n:
            break

    return cases[:n]


def _generate_hallucination(chunks: list[str], n: int) -> list[EvalCase]:
    cases = []
    per_chunk = max(1, n // len(chunks))
    remaining = n

    for chunk in chunks:
        if remaining <= 0:
            break
        batch = min(per_chunk, remaining)
        prompt = f"""Generate {batch} QA pairs from this text. Each should have a faithful answer grounded in the text.

Text:
\"\"\"
{chunk}
\"\"\"

Return a JSON array. Each element:
{{
  "question": "...",
  "faithful_answer": "answer grounded in the text",
  "context_excerpt": "the 1-3 relevant sentences from the text"
}}

Return ONLY the JSON array."""

        try:
            raw = _judge_call(prompt, max_tokens=2000)
            data = _extract_json_array(raw)
            for item in data[:batch]:
                cases.append(EvalCase(
                    input=item.get("question", ""),
                    expected_output="faithful",
                    context=item.get("context_excerpt", chunk[:500]),
                    metadata={
                        "generated": True,
                        "task": "hallucination",
                        "faithful_answer": item.get("faithful_answer", ""),
                    },
                ))
            remaining -= len(data[:batch])
        except Exception:
            continue

    return cases[:n]


def _extract_json_array(text: str) -> list:
    text = text.strip()
    # Try to find a JSON array in the response
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)
