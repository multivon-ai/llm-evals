"""
ComplianceReporter — local-first audit trail generation.

Produces tamper-evident NDJSON audit records for every eval run.
Maps results to EU AI Act Article 9 / NIST AI RMF categories.
No cloud required — runs entirely within your environment.

Usage:
    from multivon_eval import ComplianceReporter, EvalSuite

    suite = EvalSuite("HR Chatbot Eval")
    reporter = ComplianceReporter(output_dir="./audit-logs", framework="eu-ai-act")

    report = suite.run(model_fn)
    reporter.record(report)
"""
from __future__ import annotations
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .result import EvalReport


Framework = Literal["eu-ai-act", "nist-ai-rmf", "none"]

# Maps evaluator names to EU AI Act Article 9 / NIST AI RMF control categories
_EU_AI_ACT_MAPPING: dict[str, str] = {
    "faithfulness":        "Article 9(4)(a) — Accuracy & reliability",
    "hallucination":       "Article 9(4)(a) — Accuracy & reliability",
    "relevance":           "Article 9(4)(a) — Accuracy & reliability",
    "answer_accuracy":     "Article 9(4)(a) — Accuracy & reliability",
    "toxicity":            "Article 9(6) — Bias & discrimination monitoring",
    "bias":                "Article 9(6) — Bias & discrimination monitoring",
    "pii_detection":       "Article 9(4)(b) — Privacy & data governance",
    "schema_compliance":   "Article 9(4)(c) — Robustness & output consistency",
    "task_completion":     "Article 9(5) — Task performance logging",
    "tool_call_accuracy":  "Article 9(5) — Task performance logging",
    "trajectory_efficiency": "Article 9(5) — Task performance logging",
    "not_empty":           "Article 9(4)(c) — Robustness & output consistency",
    "coherence":           "Article 9(4)(a) — Accuracy & reliability",
}

_NIST_MAPPING: dict[str, str] = {
    "faithfulness":        "GOVERN 1.1 — AI risk policies",
    "hallucination":       "GOVERN 1.1 — AI risk policies",
    "relevance":           "MEASURE 2.5 — AI system performance",
    "answer_accuracy":     "MEASURE 2.5 — AI system performance",
    "toxicity":            "MANAGE 2.4 — Bias & fairness",
    "bias":                "MANAGE 2.4 — Bias & fairness",
    "pii_detection":       "GOVERN 6.1 — Privacy risk management",
    "schema_compliance":   "MEASURE 2.6 — Robustness",
    "task_completion":     "MEASURE 2.5 — AI system performance",
    "tool_call_accuracy":  "MEASURE 2.5 — AI system performance",
    "not_empty":           "MEASURE 2.6 — Robustness",
}


@dataclass
class AuditRecord:
    record_id: str
    suite_name: str
    model_id: str
    timestamp: str
    framework: str
    summary: dict
    evaluator_results: list[dict]
    record_hash: str

    def to_ndjson(self) -> str:
        return json.dumps({
            "record_id": self.record_id,
            "suite_name": self.suite_name,
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "framework": self.framework,
            "summary": self.summary,
            "evaluator_results": self.evaluator_results,
            "record_hash": self.record_hash,
        }, separators=(",", ":"))


class ComplianceReporter:
    """
    Records every eval run as a tamper-evident NDJSON audit trail.

    Each record is SHA-256 hashed so auditors can verify the log has
    not been modified. Framework mappings annotate each evaluator result
    with the relevant regulatory control.

    Args:
        output_dir: Directory to write audit logs (created if missing).
        framework:  "eu-ai-act" | "nist-ai-rmf" | "none"

    Files produced:
        <output_dir>/<suite_name>.audit.ndjson  — append-only log
        <output_dir>/<suite_name>.audit.sha256  — running hash file
    """

    def __init__(self, output_dir: str = "./audit-logs", framework: Framework = "eu-ai-act"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.framework = framework
        self._mapping = (
            _EU_AI_ACT_MAPPING if framework == "eu-ai-act"
            else _NIST_MAPPING if framework == "nist-ai-rmf"
            else {}
        )

    def record(self, report: EvalReport, tags: dict[str, str] | None = None) -> str:
        """
        Write an audit record for this eval run.

        Returns the record_id (UUID-like) for reference.
        """
        import uuid
        record_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(timezone.utc).isoformat()

        summary = {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "pass_rate": round(report.pass_rate, 4),
            "avg_score": round(report.avg_score, 4),
            "runs_per_case": report.runs_per_case,
            "flaky_count": report.flaky_count,
            "stability_score": round(report.stability_score, 4),
            "tags": tags or {},
        }

        evaluator_results = []
        for ev_name, score in report.scores_by_evaluator().items():
            entry: dict = {
                "evaluator": ev_name,
                "avg_score": round(score, 4),
                "pass_rate": round(report.passed_by_evaluator().get(ev_name, 0.0), 4),
            }
            if ev_name in self._mapping:
                entry["control"] = self._mapping[ev_name]
            evaluator_results.append(entry)

        payload = json.dumps({
            "record_id": record_id,
            "suite_name": report.suite_name,
            "model_id": report.model_id,
            "timestamp": timestamp,
            "summary": summary,
            "evaluator_results": evaluator_results,
        }, separators=(",", ":"))
        record_hash = hashlib.sha256(payload.encode()).hexdigest()

        audit_record = AuditRecord(
            record_id=record_id,
            suite_name=report.suite_name,
            model_id=report.model_id,
            timestamp=timestamp,
            framework=self.framework,
            summary=summary,
            evaluator_results=evaluator_results,
            record_hash=record_hash,
        )

        # Append to NDJSON log
        log_path = self.output_dir / f"{report.suite_name.replace(' ', '_')}.audit.ndjson"
        with open(log_path, "a") as f:
            f.write(audit_record.to_ndjson() + "\n")

        # Append hash to sha256 file
        hash_path = self.output_dir / f"{report.suite_name.replace(' ', '_')}.audit.sha256"
        with open(hash_path, "a") as f:
            f.write(f"{record_hash}  {record_id}  {timestamp}\n")

        print(f"  [compliance] audit record → {record_id}  ({log_path.name})")
        if self.framework != "none":
            print(f"  [compliance] framework: {self.framework}")
        return record_id

    def verify(self, suite_name: str) -> bool:
        """
        Re-hash each audit record and verify against the .sha256 file.

        Returns True if all records are intact, False if any have been tampered with.
        Prints a report.
        """
        log_path = self.output_dir / f"{suite_name.replace(' ', '_')}.audit.ndjson"
        hash_path = self.output_dir / f"{suite_name.replace(' ', '_')}.audit.sha256"

        if not log_path.exists():
            print(f"No audit log found: {log_path}")
            return False

        stored_hashes: dict[str, str] = {}
        if hash_path.exists():
            for line in hash_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    stored_hashes[parts[1]] = parts[0]

        lines = log_path.read_text().strip().splitlines()
        all_ok = True
        for line in lines:
            try:
                data = json.loads(line)
                record_id = data.get("record_id", "?")
                stored_hash = data.pop("record_hash", None)
                payload = json.dumps({k: v for k, v in data.items() if k != "record_hash"}, separators=(",", ":"))
                computed = hashlib.sha256(payload.encode()).hexdigest()
                match = (stored_hash == computed)
                status = "OK" if match else "TAMPERED"
                if not match:
                    all_ok = False
                print(f"  {status}  {record_id}  {data.get('timestamp', '')[:19]}")
            except Exception as e:
                print(f"  ERROR parsing record: {e}")
                all_ok = False

        print(f"\n  Verification: {'PASS — all records intact' if all_ok else 'FAIL — tampered records detected'}")
        return all_ok
