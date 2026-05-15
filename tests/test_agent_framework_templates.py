"""D16: end-to-end smoke tests for the agent-langgraph and
agent-openai-sdk templates.

These tests prove the persona walkthrough findings are fixed:

  - Templates SCAFFOLD without errors via the CLI.
  - eval.py IMPORTS cleanly without an API key set — so a user can
    read the file before they've signed up for an API.
  - requirements.txt + .env.example are produced.

We do NOT run the eval (that needs a real LLM). The optional live
test file (test_live_agent_frameworks.py) handles that under
``MULTIVON_EVAL_LIVE=1``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from multivon_eval.templates import TEMPLATES, list_templates


# ─────────────────────────────────────────────────────────────────────────────
# Both templates are registered
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_langgraph_is_registered():
    assert "agent-langgraph" in TEMPLATES
    files = TEMPLATES["agent-langgraph"]
    assert "eval.py" in files and "README.md" in files
    assert "requirements.txt" in files and ".env.example" in files


def test_agent_openai_sdk_is_registered():
    assert "agent-openai-sdk" in TEMPLATES
    files = TEMPLATES["agent-openai-sdk"]
    assert "eval.py" in files and "README.md" in files
    assert "requirements.txt" in files and ".env.example" in files


def test_list_templates_includes_new_templates():
    names = list_templates()
    assert "agent-langgraph" in names
    assert "agent-openai-sdk" in names
    # Display order: the framework-specific templates come right after
    # the generic ``agent`` so the user sees them as alternatives.
    assert names.index("agent") < names.index("agent-langgraph") < names.index("agent-openai-sdk")


# ─────────────────────────────────────────────────────────────────────────────
# CLI scaffold produces a usable project
# ─────────────────────────────────────────────────────────────────────────────

def test_cli_scaffolds_agent_langgraph(tmp_path):
    target = tmp_path / "lg_proj"
    rc = subprocess.run(
        [sys.executable, "-m", "multivon_eval", "init",
         "-t", "agent-langgraph", "-d", str(target)],
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 0, f"scaffold failed: {rc.stderr}"
    assert (target / "eval.py").exists()
    assert (target / "README.md").exists()
    # requirements.txt names the right extra
    reqs = (target / "requirements.txt").read_text()
    assert "multivon-eval[langgraph]" in reqs


def test_cli_scaffolds_agent_openai_sdk(tmp_path):
    target = tmp_path / "oai_proj"
    rc = subprocess.run(
        [sys.executable, "-m", "multivon_eval", "init",
         "-t", "agent-openai-sdk", "-d", str(target)],
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 0, f"scaffold failed: {rc.stderr}"
    assert (target / "eval.py").exists()
    reqs = (target / "requirements.txt").read_text()
    assert "multivon-eval[openai-agents]" in reqs


# ─────────────────────────────────────────────────────────────────────────────
# Persona finding: eval.py must IMPORT without an API key
# (was a hard-fail in OpenAI Agents SDK template; codex D16 cycle 3 fix)
# ─────────────────────────────────────────────────────────────────────────────

def _can_import(file_path: Path, env: dict) -> tuple[bool, str]:
    """Try to `import` the eval.py file's module body without
    executing the __main__ block. Returns (success, stderr)."""
    rc = subprocess.run(
        [sys.executable, "-c",
         f"import importlib.util, sys; "
         f"spec = importlib.util.spec_from_file_location('eval_under_test', {str(file_path)!r}); "
         f"mod = importlib.util.module_from_spec(spec); "
         f"spec.loader.exec_module(mod); print('OK')"],
        capture_output=True, text=True, env=env, timeout=20,
    )
    return rc.returncode == 0, rc.stderr


@pytest.mark.skipif(
    "openai-agents" not in (subprocess.run(
        [sys.executable, "-m", "pip", "list"], capture_output=True, text=True
    ).stdout.lower()),
    reason="openai-agents SDK not installed in this environment",
)
def test_openai_sdk_template_imports_without_api_key(tmp_path):
    """Codex D16 cycle 3 persona B finding: the OpenAI Agents SDK
    template used to raise RuntimeError at IMPORT time when
    OPENAI_API_KEY was missing — preventing the user from even
    reading eval.py to learn how it works. The check must now be
    deferred to __main__."""
    target = tmp_path / "oai_import_test"
    subprocess.run(
        [sys.executable, "-m", "multivon_eval", "init",
         "-t", "agent-openai-sdk", "-d", str(target)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    ok, err = _can_import(target / "eval.py", env)
    assert ok, (
        "eval.py raised on import without OPENAI_API_KEY — the check "
        f"must be deferred to runtime. stderr:\n{err}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# README updates from persona C: framework templates are discoverable
# ─────────────────────────────────────────────────────────────────────────────

def test_readme_mentions_agent_langgraph_template():
    """Persona C: existing user must SEE the new template in the
    'Pick your path' table. Discoverability is the whole point."""
    readme = Path(__file__).parent.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "agent-langgraph" in text, "agent-langgraph missing from README"
    assert "agent-openai-sdk" in text, "agent-openai-sdk missing from README"


def test_template_readme_documents_tracer_wiring():
    """Persona A finding: the callback-forwarding pattern is the
    crux. Must be documented in the template README, not just buried
    in eval.py."""
    lg_readme = TEMPLATES["agent-langgraph"]["README.md"]
    assert "callbacks" in lg_readme.lower()
    assert "**kwargs" in lg_readme or "kwargs.get" in lg_readme

    oai_readme = TEMPLATES["agent-openai-sdk"]["README.md"]
    assert "capture" in oai_readme.lower()
    assert "tracer.capture" in oai_readme or "TRACER.capture" in oai_readme


def test_template_readme_has_migration_note():
    """Persona C finding: existing `agent` template users need a
    'how to migrate' tip."""
    lg_readme = TEMPLATES["agent-langgraph"]["README.md"]
    assert "migrat" in lg_readme.lower(), "agent-langgraph README needs a migration section"

    oai_readme = TEMPLATES["agent-openai-sdk"]["README.md"]
    assert "migrat" in oai_readme.lower(), "agent-openai-sdk README needs a migration section"
