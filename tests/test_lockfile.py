"""Tests for the suite.lock content-addressed fingerprint."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from multivon_eval import EvalCase, EvalSuite, LockMismatch, SuiteLock
from multivon_eval.evaluators.deterministic import NotEmpty, ExactMatch
from multivon_eval.evaluators.llm_judge import Faithfulness
from multivon_eval.lockfile import build_suite_lock, fingerprint_evaluator


class TestLockBuild:
    def test_lock_carries_library_version(self):
        suite = EvalSuite("v").add_evaluators(NotEmpty())
        lock = suite.lock()
        from multivon_eval import __version__
        assert lock.library_version == __version__

    def test_lock_carries_suite_name_and_evaluators(self):
        suite = EvalSuite("My Suite").add_evaluators(NotEmpty(), ExactMatch())
        lock = suite.lock()
        assert lock.suite_name == "My Suite"
        names = {e.name for e in lock.evaluators}
        assert names == {"not_empty", "exact_match"}

    def test_lock_is_deterministic(self):
        suite = EvalSuite("det")
        suite.add_evaluators(NotEmpty(), ExactMatch())
        suite.add_cases([EvalCase(input="a"), EvalCase(input="b")])
        a = suite.lock()
        b = suite.lock()
        assert a.suite_hash == b.suite_hash
        assert a.to_dict() == b.to_dict()

    def test_case_count_and_hash_present(self):
        suite = EvalSuite("c").add_evaluators(NotEmpty())
        suite.add_cases([EvalCase(input="a"), EvalCase(input="b")])
        lock = suite.lock()
        assert lock.case_count == 2
        assert lock.cases_hash and len(lock.cases_hash) == 64

    def test_no_cases_gives_null_hash(self):
        suite = EvalSuite("c").add_evaluators(NotEmpty())
        lock = suite.lock()
        assert lock.case_count == 0
        assert lock.cases_hash is None


class TestRoundtrip:
    def test_to_json_round_trip(self):
        suite = EvalSuite("r").add_evaluators(NotEmpty(), ExactMatch())
        original = suite.lock()
        restored = SuiteLock.from_json(original.to_json())
        assert restored.suite_hash == original.suite_hash
        assert restored.to_dict() == original.to_dict()

    def test_write_lock_creates_file(self, tmp_path):
        suite = EvalSuite("write").add_evaluators(NotEmpty())
        path = suite.write_lock(tmp_path / "suite.lock")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["suite_name"] == "write"
        assert data["library_version"]


class TestDiffAndVerify:
    def test_identical_suite_verifies(self):
        suite = EvalSuite("v").add_evaluators(NotEmpty(), ExactMatch())
        suite.add_cases([EvalCase(input="a")])
        lock = suite.lock()
        # No mutation — verify must pass.
        suite.verify_lock(lock)

    def test_added_evaluator_detected(self):
        suite = EvalSuite("v").add_evaluators(NotEmpty())
        before = suite.lock()
        suite.add_evaluator(ExactMatch())
        with pytest.raises(LockMismatch) as info:
            suite.verify_lock(before)
        assert any("evaluator added" in d for d in info.value.differences)

    def test_removed_evaluator_detected(self):
        full = EvalSuite("v").add_evaluators(NotEmpty(), ExactMatch())
        lock = full.lock()
        reduced = EvalSuite("v").add_evaluators(NotEmpty())
        with pytest.raises(LockMismatch) as info:
            reduced.verify_lock(lock)
        assert any("evaluator removed" in d for d in info.value.differences)

    def test_threshold_change_detected(self):
        suite = EvalSuite("v").add_evaluators(NotEmpty(threshold=0.5))
        lock = suite.lock()
        # Mutate threshold via a fresh evaluator with different threshold.
        suite._evaluators[0].threshold = 0.8
        with pytest.raises(LockMismatch) as info:
            suite.verify_lock(lock)
        assert any("threshold" in d for d in info.value.differences)

    def test_case_count_change_detected(self):
        suite = EvalSuite("v").add_evaluators(NotEmpty())
        suite.add_cases([EvalCase(input="a")])
        lock = suite.lock()
        suite.add_cases([EvalCase(input="b")])
        with pytest.raises(LockMismatch) as info:
            suite.verify_lock(lock)
        assert any("case_count" in d for d in info.value.differences)

    def test_verify_lock_accepts_path(self, tmp_path):
        suite = EvalSuite("v").add_evaluators(NotEmpty())
        path = suite.write_lock(tmp_path / "s.lock")
        suite.verify_lock(path)

    def test_verify_lock_accepts_json_string(self):
        suite = EvalSuite("v").add_evaluators(NotEmpty())
        raw = suite.lock().to_json()
        suite.verify_lock(raw)

    def test_verify_lock_rejects_unknown_type(self):
        suite = EvalSuite("v").add_evaluators(NotEmpty())
        with pytest.raises(TypeError):
            suite.verify_lock(12345)  # type: ignore[arg-type]


class TestLockMismatchPayload:
    def test_mismatch_carries_structured_differences(self):
        suite = EvalSuite("m").add_evaluators(NotEmpty())
        lock = suite.lock()
        suite.add_evaluator(ExactMatch())
        try:
            suite.verify_lock(lock)
        except LockMismatch as exc:
            assert isinstance(exc.differences, list)
            assert len(exc.differences) >= 1
        else:
            pytest.fail("expected LockMismatch")


class TestFingerprintFields:
    def test_judge_fingerprint_present_for_faithfulness(self):
        f = Faithfulness()
        fp = fingerprint_evaluator(f)
        # The Faithfulness evaluator resolves a JudgeConfig via JudgeConfig defaults
        # — it may or may not expose `judge` directly; the resolver yields one.
        # At minimum the fingerprint should not error and should record name/class.
        assert fp.name == "faithfulness"
        assert "llm_judge" in fp.class_path

    def test_judge_change_affects_hash(self):
        from multivon_eval.judge import JudgeConfig
        suite_a = EvalSuite("j").add_evaluators(
            Faithfulness(judge=JudgeConfig(provider="openai", model="gpt-4o-mini")),
        )
        suite_b = EvalSuite("j").add_evaluators(
            Faithfulness(judge=JudgeConfig(provider="openai", model="gpt-4o")),
        )
        # Different judge models => different evaluator fingerprint => different suite hash
        assert suite_a.lock().suite_hash != suite_b.lock().suite_hash

    def test_calibration_provenance_attached_when_available(self):
        # Faithfulness w/ gpt-4o-mini has a calibrated threshold (0.9) in v1.json.
        from multivon_eval.judge import JudgeConfig
        f = Faithfulness(judge=JudgeConfig(provider="openai", model="gpt-4o-mini"))
        fp = fingerprint_evaluator(f)
        # The calibration field must be populated for a known judge+evaluator pair.
        assert fp.calibration is not None
        assert fp.calibration["dataset_hash"]
        # And the judge fingerprint must include the model.
        assert fp.judge and fp.judge["model"] == "gpt-4o-mini"


class TestLibraryVersionDrift:
    def test_library_version_mismatch_surfaces(self):
        suite = EvalSuite("lib").add_evaluators(NotEmpty())
        lock = suite.lock()
        lock.library_version = "0.4.2"  # pretend an old lock
        diffs = suite.lock().diff(lock)
        assert any("library_version" in d for d in diffs)
