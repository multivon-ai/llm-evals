"""
Calibration sweep — emits `_calibration_data/v2.json` directly.

Reuses the threshold-sweep machinery from `run_threshold_calibration.py`
but writes its output in the schema the runtime loader expects, with
provenance fields filled in (`dataset`, `dataset_hash`, `n`, `precision`,
`recall`, `f1`, `measured_at`, `judge_aliases`).

Designed to be merge-friendly with the existing `v1.json`. Default
behavior:

  • Run the sweep for every (evaluator × judge) pair listed.
  • Merge with `v1.json` entries that weren't re-measured (so a partial
    sweep doesn't drop coverage).
  • Bump schema_version to 2 in the output.

Cost budget: by default, ~$10 across the four most-impactful missing
judges (gpt-4o, claude-opus-4-7, gpt-5.5, claude-sonnet-4-7). Override
with --judges if you want to be cheaper.

Usage:
  python benchmarks/run_calibration_v2.py
  python benchmarks/run_calibration_v2.py --judges anthropic/claude-haiku-4-5-20251001
  python benchmarks/run_calibration_v2.py --output multivon_eval/_calibration_data/v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse the existing sweep mechanics.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_threshold_calibration import (  # noqa: E402
    _load_halueval_qa,
    _load_halueval_summ,
    _load_relevance_golden,
    _collect_scores,
    _best_threshold,
    _score_hallucination,
    _score_faithfulness,
    _score_relevance,
)


# Default judges to sweep when --judges isn't passed.
# Bias toward filling v1.json's null-F1 entries and adding flagship models.
DEFAULT_JUDGES_TO_ADD = [
    # Already in v1.json but with null F1 — re-measure to fill them in.
    "anthropic/claude-opus-4-7",
    "openai/gpt-4o",
    # New flagship — first calibration row for gpt-5.5.
    "openai/gpt-5.5",
]


# Dataset metadata for provenance.
DATASET_META = {
    "hallucination": {
        "dataset": "HaluEval QA",
        "dataset_hash": "halueval-qa-2024-50c",  # 50-case sweep, balanced
    },
    "faithfulness": {
        "dataset": "HaluEval Summarization",
        "dataset_hash": "halueval-sum-2024-30c",  # 30-case sweep, balanced
    },
    "relevance": {
        "dataset": "Curated relevance golden set",
        "dataset_hash": "relevance-gold-2026-40c",
    },
}


# Alias maps for stable lookup against short model ids.
JUDGE_ALIASES = {
    "claude-haiku-4-5-20251001": ["claude-haiku-4-5"],
    "claude-sonnet-4-6-20251030": ["claude-sonnet-4-6"],
    "claude-sonnet-4-7-20251030": ["claude-sonnet-4-7"],
    "claude-opus-4-7-20251101": ["claude-opus-4-7"],
    "gpt-4o-2024-11-20": ["gpt-4o"],
    "gpt-5.5-2026-04-23": ["gpt-5.5", "gpt-5"],
}


def _provenance_entry(
    *,
    evaluator: str,
    judge_model: str,
    sweep_result: dict,
    n: int,
    measured_at: str,
) -> dict:
    """Convert one (evaluator × judge) sweep result into a v2 entry."""
    meta = DATASET_META[evaluator]
    opt = sweep_result["optimal"]
    aliases = JUDGE_ALIASES.get(judge_model, [])
    return {
        "evaluator": evaluator,
        "judge_model": judge_model,
        "judge_aliases": aliases,
        "threshold": opt["threshold"],
        "dataset": meta["dataset"],
        "dataset_hash": meta["dataset_hash"],
        "n": n,
        "precision": opt["precision"],
        "recall": opt["recall"],
        "f1": opt["f1"],
        "measured_at": measured_at,
        "notes": "Threshold maximises F1 over a 0.30..0.90 sweep against human labels.",
    }


def _merge_with_v1(new_entries: list[dict], v1_path: Path) -> list[dict]:
    """Merge new entries with v1 by (evaluator, judge_model), new wins."""
    if not v1_path.exists():
        return new_entries
    v1 = json.loads(v1_path.read_text())
    v1_entries = v1.get("entries", [])
    new_keys = {(e["evaluator"], e["judge_model"]) for e in new_entries}
    merged: list[dict] = list(new_entries)
    for old in v1_entries:
        if (old["evaluator"], old["judge_model"]) not in new_keys:
            merged.append(old)
    return sorted(merged, key=lambda e: (e["evaluator"], e["judge_model"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES_TO_ADD)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-hal", type=int, default=50,
                    help="HaluEval QA sample size for hallucination evaluator")
    ap.add_argument("--n-faith", type=int, default=30,
                    help="HaluEval Summarization sample size for faithfulness")
    ap.add_argument("--n-rel", type=int, default=40,
                    help="Relevance golden set sample size")
    ap.add_argument("--evaluators", nargs="+", default=["hallucination", "faithfulness", "relevance"])
    ap.add_argument("--output", default="multivon_eval/_calibration_data/v2.json")
    ap.add_argument("--no-merge-v1", action="store_true",
                    help="Don't merge with v1 entries (output only what you re-measured)")
    args = ap.parse_args()

    from multivon_eval import configure, JudgeConfig, Hallucination, Faithfulness, Relevance

    print(f"\n  Loading datasets...", flush=True)
    hal_items = _load_halueval_qa(args.n_hal)
    faith_items = _load_halueval_summ(args.n_faith)
    try:
        rel_items = _load_relevance_golden()
    except Exception:
        from run_relevance_benchmark import GOLDEN_SET
        rel_items = [
            {"question": g["question"], "output": g["answer"], "label": g["label"]}
            for g in GOLDEN_SET
        ][: args.n_rel]
    print(f"  Loaded: hal={len(hal_items)}, faith={len(faith_items)}, rel={len(rel_items)}")
    print(f"  Sweeping {len(args.judges)} judge(s) × {len(args.evaluators)} evaluator(s)\n")

    new_entries: list[dict] = []
    measured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    overall_t0 = time.time()

    for judge_str in args.judges:
        if "/" not in judge_str:
            print(f"  [skip] invalid judge: {judge_str}", file=sys.stderr)
            continue
        provider, model = judge_str.split("/", 1)
        configure(JudgeConfig(provider=provider, model=model))
        print(f"  Judge: {judge_str}")

        for ev_name in args.evaluators:
            if ev_name == "hallucination":
                ev = Hallucination(threshold=0.5)
                scores = _collect_scores(hal_items, _score_hallucination, ev, args.workers)
                n = len(scores)
            elif ev_name == "faithfulness":
                ev = Faithfulness(threshold=0.5)
                scores = _collect_scores(faith_items, _score_faithfulness, ev, args.workers)
                n = len(scores)
            elif ev_name == "relevance":
                ev = Relevance(threshold=0.5)
                scores = _collect_scores(rel_items, _score_relevance, ev, args.workers)
                n = len(scores)
            else:
                print(f"    [skip] unknown evaluator: {ev_name}", file=sys.stderr)
                continue

            sweep_t0 = time.time()
            sweep = _best_threshold(scores)
            elapsed = time.time() - sweep_t0
            opt = sweep["optimal"]
            print(f"    {ev_name:<15} → threshold={opt['threshold']:.2f}  F1={opt['f1']:.3f}  ({elapsed:.0f}s)")

            new_entries.append(_provenance_entry(
                evaluator=ev_name,
                judge_model=model,
                sweep_result=sweep,
                n=n,
                measured_at=measured_at,
            ))

        print()

    overall_elapsed = time.time() - overall_t0
    print(f"  Sweep complete in {overall_elapsed:.0f}s. New entries: {len(new_entries)}\n")

    # Merge with v1.
    v1_path = Path(__file__).resolve().parent.parent / "multivon_eval" / "_calibration_data" / "v1.json"
    if args.no_merge_v1:
        merged = sorted(new_entries, key=lambda e: (e["evaluator"], e["judge_model"]))
    else:
        merged = _merge_with_v1(new_entries, v1_path)

    output = {
        "schema_version": 2,
        "generated_at": measured_at,
        "methodology": (
            "Threshold sweep over [0.30..0.90] in steps of 0.05, selecting the "
            "value that maximises F1 against human labels. Reproduced by "
            "benchmarks/run_calibration_v2.py."
        ),
        "based_on_v1": not args.no_merge_v1,
        "entries": merged,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Wrote {out_path}")
    print(f"  Total entries: {len(merged)}  (new: {len(new_entries)}, merged from v1: {len(merged) - len(new_entries)})")


if __name__ == "__main__":
    main()
