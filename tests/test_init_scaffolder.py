"""Tests for the `multivon-eval init` scaffolder + templates module."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from multivon_eval.templates import TEMPLATES, list_templates, render


# ─────────────────────────────────────────────────────────────────────────────
# templates module — pure-Python invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_list_templates_matches_registry():
    """`list_templates` must enumerate every key in TEMPLATES."""
    listed = list_templates()
    assert set(listed) == set(TEMPLATES.keys()), (
        f"list_templates() out of sync with TEMPLATES: "
        f"listed={listed} vs registry={list(TEMPLATES.keys())}"
    )


@pytest.mark.parametrize("template", list(TEMPLATES.keys()))
def test_every_template_has_required_files(template):
    """Every template must ship at least: eval.py + README.md + requirements.txt."""
    files = TEMPLATES[template]
    assert "eval.py" in files, f"{template} missing eval.py"
    assert "README.md" in files, f"{template} missing README.md"
    assert "requirements.txt" in files, f"{template} missing requirements.txt"


@pytest.mark.parametrize("template", list(TEMPLATES.keys()))
def test_every_template_eval_py_is_valid_python(template):
    """Every shipped eval.py must parse — regression for typos/syntax."""
    src = TEMPLATES[template]["eval.py"]
    try:
        ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"{template}/eval.py has SyntaxError: {e}")


@pytest.mark.parametrize("template", list(TEMPLATES.keys()))
def test_every_template_requirements_pins_correct_version(template):
    """requirements.txt must pin multivon-eval >= the version we're shipping
    init from. Without this, a Colab/CI run could install an older lib that
    doesn't have the features the template uses (assert_budget, agent fix,
    match_any, etc.)."""
    from multivon_eval import __version__
    req = TEMPLATES[template]["requirements.txt"]
    assert "multivon-eval" in req
    assert ">=" in req, f"{template} requirements should pin a minimum version, got: {req!r}"
    # Pinned minimum must be <= current shipped __version__ (so an installed
    # current dev version satisfies it).
    import re
    m = re.search(r">=([\d.]+)", req)
    assert m, f"can't parse minimum version from: {req!r}"
    pinned = tuple(int(x) for x in m.group(1).split("."))
    current = tuple(int(x) for x in __version__.split(".")[:3])
    assert pinned <= current, (
        f"{template} pins multivon-eval>={pinned} but lib is at {current}; "
        f"would fail to install on fresh systems"
    )


# ─────────────────────────────────────────────────────────────────────────────
# render() — file map building
# ─────────────────────────────────────────────────────────────────────────────

def test_render_unknown_template_raises():
    with pytest.raises(ValueError) as exc:
        render("nonexistent")
    assert "nonexistent" in str(exc.value)
    # Error should also list the available templates so user can fix.
    for t in list_templates():
        assert t in str(exc.value)


def test_render_returns_a_copy_not_the_registry():
    """Mutating render() output must NOT mutate the registry — otherwise one
    bad caller could corrupt every subsequent init."""
    files = render("quickstart")
    files["evil.py"] = "raise"
    assert "evil.py" not in TEMPLATES["quickstart"], (
        "render() returned a reference, not a copy — registry was mutated"
    )


def test_render_with_github_ci_adds_workflow():
    files = render("rag", with_ci="github")
    assert ".github/workflows/eval.yml" in files
    ci_yaml = files[".github/workflows/eval.yml"]
    # Sanity-check the workflow content references the template name.
    assert "rag" in ci_yaml
    # Should reference the right env-var secrets.
    assert "ANTHROPIC_API_KEY" in ci_yaml
    assert "OPENAI_API_KEY" in ci_yaml


def test_render_without_ci_omits_workflow():
    files = render("rag")
    assert ".github/workflows/eval.yml" not in files


def test_render_unsupported_ci_flavor_raises():
    with pytest.raises(ValueError):
        render("rag", with_ci="gitlab")


# ─────────────────────────────────────────────────────────────────────────────
# CLI subcommand — end-to-end via subprocess
# ─────────────────────────────────────────────────────────────────────────────

def _run_cli(*args, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "multivon_eval", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def test_cli_init_scaffolds_quickstart(tmp_path: Path):
    """init --template quickstart writes a runnable project to disk."""
    target = tmp_path / "myeval"
    res = _run_cli("init", "--template", "quickstart", "--dir", str(target), cwd=tmp_path)
    assert res.returncode == 0, f"stderr: {res.stderr}"

    # All expected files present.
    assert (target / "eval.py").exists()
    assert (target / "README.md").exists()
    assert (target / "requirements.txt").exists()
    assert (target / ".gitignore").exists()
    # No .env.example for quickstart (no API key needed).
    assert not (target / ".env.example").exists()


def test_cli_init_rag_writes_env_example(tmp_path: Path):
    target = tmp_path / "rag-proj"
    res = _run_cli("init", "--template", "rag", "--dir", str(target), cwd=tmp_path)
    assert res.returncode == 0, f"stderr: {res.stderr}"
    assert (target / ".env.example").exists()
    assert "ANTHROPIC_API_KEY" in (target / ".env.example").read_text()


def test_cli_init_with_ci_writes_workflow(tmp_path: Path):
    target = tmp_path / "ragci"
    res = _run_cli("init", "-t", "rag", "-d", str(target), "--ci", "github", cwd=tmp_path)
    assert res.returncode == 0, f"stderr: {res.stderr}"
    workflow = target / ".github" / "workflows" / "eval.yml"
    assert workflow.exists(), "github CI flag did not write workflow"
    text = workflow.read_text()
    assert "name: eval" in text


def test_cli_init_refuses_to_clobber_nonempty_dir(tmp_path: Path):
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "existing.txt").write_text("don't touch me")
    res = _run_cli("init", "-t", "quickstart", "-d", str(target), cwd=tmp_path)
    assert res.returncode == 1
    # Original file untouched.
    assert (target / "existing.txt").read_text() == "don't touch me"
    # eval.py NOT written.
    assert not (target / "eval.py").exists()


def test_cli_init_force_overrides_clobber_check(tmp_path: Path):
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "existing.txt").write_text("don't touch me")
    res = _run_cli("init", "-t", "quickstart", "-d", str(target), "--force", cwd=tmp_path)
    assert res.returncode == 0
    assert (target / "eval.py").exists()
    # The existing file should still be there (we don't delete, only add).
    assert (target / "existing.txt").read_text() == "don't touch me"


def test_cli_init_help_lists_all_templates():
    """--help should mention every template name so users can discover them."""
    res = subprocess.run(
        [sys.executable, "-m", "multivon_eval", "init", "--help"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0
    for t in list_templates():
        assert t in res.stdout, f"--help did not list template {t!r}"


def test_cli_init_quickstart_runs_offline(tmp_path: Path):
    """End-to-end smoke: scaffold quickstart, run it, get a passing report.

    Skipped if pytest isn't installed in the harness venv (the test runs
    `python eval.py` as a subprocess and the quickstart template only uses
    deterministic evaluators, so no API keys needed — but we still depend
    on the multivon-eval install being usable).
    """
    target = tmp_path / "smoke"
    res = _run_cli("init", "-t", "quickstart", "-d", str(target), cwd=tmp_path)
    assert res.returncode == 0

    run_res = subprocess.run(
        [sys.executable, "eval.py"],
        capture_output=True, text=True, cwd=target,
    )
    assert run_res.returncode == 0, f"eval.py failed:\n{run_res.stderr}"

    # Report saved.
    report_path = target / "eval-reports" / "quickstart.json"
    assert report_path.exists(), "save_json output missing"
    data = json.loads(report_path.read_text())
    # Quickstart hand-tunes the model_fn to satisfy the checks → 100% pass.
    assert data["summary"]["pass_rate"] == 1.0, (
        f"expected 100% pass on hand-tuned quickstart, got: {data['summary']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Regressions for codex review on the init scaffolder
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("template", ["rag", "agent", "regulated"])
def test_template_loads_dotenv_before_judge_setup(template):
    """README tells users to `cp .env.example .env`; eval.py must actually
    load it. Codex review caught this — without load_dotenv() the .env
    values are never read."""
    src = TEMPLATES[template]["eval.py"]
    assert "load_dotenv" in src, (
        f"{template}/eval.py does not call load_dotenv() — .env contents "
        f"won't be loaded by the 3-command flow."
    )


@pytest.mark.parametrize("template", ["rag", "agent", "regulated"])
def test_template_gates_with_fail_threshold(template):
    """suite.run() must pass fail_threshold so CI exits 1 on eval failures.
    Codex review caught that the GitHub workflow only ran `python eval.py`
    and a low pass_rate would still exit 0."""
    src = TEMPLATES[template]["eval.py"]
    assert "fail_threshold=" in src, (
        f"{template}/eval.py runs suite without fail_threshold — CI can go "
        f"green even on eval failure."
    )


def test_template_local_ollama_sets_dummy_api_key():
    """OpenAI SDK refuses to instantiate without an api_key, even for local
    base_urls. The RAG template must set OPENAI_API_KEY before constructing
    the local JudgeConfig."""
    src = TEMPLATES["rag"]["eval.py"]
    assert "OPENAI_API_KEY" in src and "ollama" in src.lower(), (
        "RAG local Ollama fallback must set a sentinel OPENAI_API_KEY so the "
        "OpenAI SDK can instantiate even without a real key."
    )


def test_regulated_template_saves_report_to_eval_reports():
    """The CI workflow uploads eval-reports/ — the regulated template must
    save there too, not only audit-logs/, or the artifact upload misses
    the report. Codex review caught this."""
    src = TEMPLATES["regulated"]["eval.py"]
    assert "eval-reports" in src, (
        "regulated template only writes audit-logs/ — CI artifact upload "
        "will miss the report."
    )


def test_ci_workflow_uploads_both_eval_reports_and_audit_logs():
    """The generated GitHub workflow must upload both directories, since
    the regulated template writes audit-logs/ and the upload-artifact
    action ignores missing dirs."""
    workflow = render("regulated", with_ci="github")[".github/workflows/eval.yml"]
    assert "eval-reports/" in workflow
    assert "audit-logs/" in workflow


def test_cli_init_dir_points_at_a_file_is_clean_error(tmp_path: Path):
    """--dir pointing at an existing FILE should fail with a clear error,
    not a NotADirectoryError traceback. Codex review caught this edge case."""
    target = tmp_path / "iam-a-file"
    target.write_text("hello")
    res = _run_cli("init", "-t", "quickstart", "-d", str(target), cwd=tmp_path)
    assert res.returncode != 0
    assert "must be a directory" in (res.stderr + res.stdout).lower(), (
        f"Expected clean error, got stderr={res.stderr!r} stdout={res.stdout!r}"
    )
