"""
multivon-eval pytest plugin — runs evaluators inside pytest test functions.

Pytest is the obvious adoption surface for any team that already has a
test suite. This plugin lets a developer write::

    from multivon_eval import EvalCase
    from multivon_eval.evaluators.llm_judge import Faithfulness
    from multivon_eval.pytest_plugin import assert_evaluators

    def test_refund_policy(my_model):
        case = EvalCase(
            input="What is the refund window?",
            context="Customers may request refunds within 14 days.",
        )
        output = my_model(case.input)
        assert_evaluators(case, output, [Faithfulness()])

That's it. The plugin is enabled automatically when ``multivon-eval`` is
installed — pytest picks it up through the entry point registered in
``pyproject.toml``. No conftest setup required.

Why a plugin, not just helpers? Three things only a real plugin can do:

  1. **Aggregated cost summary** at the end of the pytest run, instead of
     printing per-test. Adds one line to the terminal summary:
     ``multivon-eval: 47 judge calls, $0.0234 estimated``.
  2. **--multivon-runs=N** command-line flag to bump every multivon
     assertion's runs count, so a flaky-test investigation is a single
     re-run with N=5.
  3. **Pretty failure output** — when an evaluator fails, the message
     shows the per-evaluator score, threshold, reason, and the relevant
     prompt content, not just ``AssertionError``.
"""
from __future__ import annotations

import statistics
from typing import Iterable

import pytest

from .case import EvalCase
from .costs import CostTracker, reset_token, set_active_tracker
from .evaluators.base import Evaluator
from .result import EvalResult


# Module-level state used by the plugin hooks. pytest is single-process by
# default, so this is safe; tests running under pytest-xdist will each have
# their own process and their own tracker.
_PROCESS_TRACKER: CostTracker | None = None
_PROCESS_TOKEN = None
_FAILURE_DETAIL: list[str] = []


# ── public helpers ──────────────────────────────────────────────────────────


class EvaluatorFailure(AssertionError):
    """Rich AssertionError surfaced when one or more evaluators fail.

    Inherits from ``AssertionError`` so pytest renders it as a normal
    assertion failure, but carries the structured details so tooling can
    inspect them.
    """

    def __init__(self, results: list[EvalResult], case: EvalCase, output: str):
        self.results = results
        self.case = case
        self.output = output
        super().__init__(self._format())

    def _format(self) -> str:
        lines = ["multivon-eval: one or more evaluators failed:"]
        for r in self.results:
            if r.passed:
                continue
            lines.append(
                f"  • {r.evaluator}  score={r.score:.3f}  passed=False"
            )
            if r.reason:
                lines.append(f"    reason: {r.reason}")
        lines.append("")
        lines.append(f"  case input : {self.case.input[:120]}")
        lines.append(f"  case ctx   : {(self.case.context_str() or '<none>')[:120]}")
        lines.append(f"  actual out : {self.output[:120]}")
        return "\n".join(lines)


def assert_evaluators(
    case: EvalCase,
    output: str,
    evaluators: Iterable[Evaluator],
    *,
    runs: int = 1,
) -> list[EvalResult]:
    """Run every evaluator on ``(case, output)``. Raise on any failure.

    Args:
        case:        The :class:`EvalCase` being checked.
        output:      Model output to evaluate.
        evaluators:  One or more :class:`Evaluator` instances.
        runs:        Number of times to run each evaluator. Use > 1 to
                     detect flaky tests via majority vote. The
                     ``--multivon-runs=N`` pytest flag overrides this.

    Returns the list of :class:`EvalResult` (one per evaluator). Raises
    :class:`EvaluatorFailure` if any did not pass.
    """
    override = _runs_override()
    if override is not None:
        runs = override

    results: list[EvalResult] = []
    for ev in evaluators:
        if runs <= 1:
            r = ev.evaluate(case, output)
            results.append(r)
            continue
        # Multi-run: take majority verdict + mean score.
        sub: list[EvalResult] = [ev.evaluate(case, output) for _ in range(runs)]
        passes = sum(1 for s in sub if s.passed)
        avg = statistics.fmean(s.score for s in sub)
        verdict_passed = passes > runs / 2
        reason = f"{passes}/{runs} runs passed; mean score {avg:.3f}"
        results.append(EvalResult(
            evaluator=sub[0].evaluator,
            score=avg,
            passed=verdict_passed,
            reason=reason,
            metadata={"all_scores": [s.score for s in sub], "runs": runs},
        ))

    if any(not r.passed for r in results):
        raise EvaluatorFailure(results=results, case=case, output=output)
    return results


def _runs_override() -> int | None:
    """Pytest --multivon-runs override, if active."""
    return _RUNS_OVERRIDE


_RUNS_OVERRIDE: int | None = None


# ── pytest hooks ────────────────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("multivon-eval", description="multivon-eval pytest options")
    group.addoption(
        "--multivon-runs",
        action="store",
        type=int,
        default=None,
        help=(
            "Override the `runs=` argument to assert_evaluators(). Useful for "
            "flaky-test investigation: re-run the suite with `--multivon-runs=5`."
        ),
    )
    group.addoption(
        "--multivon-no-costs",
        action="store_true",
        default=False,
        help="Disable multivon-eval cost-tracking and the terminal summary line.",
    )


def pytest_configure(config: pytest.Config) -> None:
    global _PROCESS_TRACKER, _PROCESS_TOKEN, _RUNS_OVERRIDE
    _RUNS_OVERRIDE = config.getoption("--multivon-runs")

    if not config.getoption("--multivon-no-costs"):
        _PROCESS_TRACKER = CostTracker()
        _PROCESS_TOKEN = set_active_tracker(_PROCESS_TRACKER)


def pytest_unconfigure(config: pytest.Config) -> None:
    global _PROCESS_TRACKER, _PROCESS_TOKEN
    if _PROCESS_TOKEN is not None:
        try:
            reset_token(_PROCESS_TOKEN)
        except Exception:
            pass
    _PROCESS_TRACKER = None
    _PROCESS_TOKEN = None


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    if _PROCESS_TRACKER is None:
        return
    snap = _PROCESS_TRACKER.snapshot()
    if snap.total_calls == 0:
        return
    terminalreporter.write_sep("=", "multivon-eval costs")
    terminalreporter.write_line(str(snap))


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def multivon_costs() -> CostTracker | None:
    """Fixture returning the per-process CostTracker, or None if disabled.

    Use in tests that want to assert cost ceilings::

        def test_cheap_eval(multivon_costs):
            ...  # run some asserts
            assert multivon_costs.snapshot().total_cost_usd or 0 < 0.05
    """
    return _PROCESS_TRACKER
