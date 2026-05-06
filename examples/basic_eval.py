"""
Basic eval — deterministic evaluators, no LLM judge needed.
Zero API cost. Run this as a sanity check on any model.
"""
from dotenv import load_dotenv
load_dotenv()

import anthropic
from multivon_eval import EvalSuite, EvalCase, ExactMatch, NotEmpty, WordCount, RegexMatch

client = anthropic.Anthropic()


def my_model(prompt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


cases = [
    EvalCase(
        input="What is 2 + 2? Reply with just the number.",
        expected_output="4",
        tags=["math"],
    ),
    EvalCase(
        input="Name three primary colors. Reply with a comma-separated list.",
        tags=["knowledge"],
    ),
    EvalCase(
        input="Write a one-sentence description of photosynthesis.",
        tags=["science"],
    ),
    EvalCase(
        input="What is the capital of France? One word only.",
        expected_output="Paris",
        tags=["geography"],
    ),
]

suite = EvalSuite("Basic Deterministic Eval", model_id="claude-haiku-4-5")
suite.add_cases(cases)
suite.add_evaluators(
    NotEmpty(),
    WordCount(min_words=1, max_words=100),
)

# Per-case evaluators: add ExactMatch only to cases with expected_output
for case in cases:
    if case.expected_output:
        suite.add_evaluators(ExactMatch())
        break

report = suite.run(my_model)
report.save_json("eval_results.json")
print("\nSaved to eval_results.json")
