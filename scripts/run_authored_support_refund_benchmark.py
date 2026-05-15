"""Run authored support/refunds benchmark for MELM Guard + Memory OS."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import load_authored_support_refund_dataset
from melm.evaluation import bootstrap_mean_ci, bootstrap_paired_difference_ci
from melm.guard import evaluate_guard_benchmark
from melm.memory import SupportMemoryOS, evaluate_memory_os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="benchmarks/support_refunds_authored.jsonl")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--out-json", default="reports/melm_authored_support_refund_benchmark.json")
    parser.add_argument("--out-md", default="reports/melm_authored_support_refund_benchmark.md")
    args = parser.parse_args()

    dataset = load_authored_support_refund_dataset(args.dataset)
    fixture = dataset.fixture
    guard_report = evaluate_guard_benchmark(
        fixture.facts,
        fixture.rules,
        fixture.guard_cases,
        current_time=fixture.current_time,
    )
    memory_report = evaluate_memory_os(
        SupportMemoryOS(fixture.events),
        fixture.memory_cases,
        k=args.k,
    )
    guard_melm_correct = [
        prediction.melm_status == prediction.expected_status
        for prediction in guard_report.predictions
    ]
    guard_schema_correct = [
        prediction.schema_only_status == prediction.expected_status
        for prediction in guard_report.predictions
    ]
    guard_prompt_correct = [
        prediction.prompt_only_status == prediction.expected_status
        for prediction in guard_report.predictions
    ]
    memory_os_correct = [
        prediction.memory_os_correct
        for prediction in memory_report.predictions
    ]
    memory_vector_correct = [
        prediction.vector_correct
        for prediction in memory_report.predictions
    ]
    memory_temporal_correct = [
        prediction.temporal_entity_correct
        for prediction in memory_report.predictions
    ]

    externally_blind = not bool(dataset.metadata.get("requires_external_blind_batch"))
    payload = {
        "benchmark": "melm_authored_support_refunds_v0_1",
        "dataset_path": dataset.path,
        "dataset_metadata": dataset.metadata,
        "schema_validation_passed": not dataset.validation_errors,
        "validation_errors": list(dataset.validation_errors),
        "events": len(fixture.events),
        "facts": len(fixture.facts),
        "turns": len(dataset.turns),
        "guard_cases": len(fixture.guard_cases),
        "memory_cases": len(fixture.memory_cases),
        "k": args.k,
        "bootstrap_samples": args.bootstrap_samples,
        "guard_report": asdict(guard_report),
        "memory_report": asdict(memory_report),
        "statistics": {
            "guard_melm_accuracy_ci": asdict(
                bootstrap_mean_ci(
                    [float(value) for value in guard_melm_correct],
                    samples=args.bootstrap_samples,
                    seed=311,
                )
            ),
            "guard_melm_vs_schema_accuracy_diff_ci": asdict(
                bootstrap_paired_difference_ci(
                    guard_melm_correct,
                    guard_schema_correct,
                    samples=args.bootstrap_samples,
                    seed=312,
                )
            ),
            "guard_melm_vs_prompt_accuracy_diff_ci": asdict(
                bootstrap_paired_difference_ci(
                    guard_melm_correct,
                    guard_prompt_correct,
                    samples=args.bootstrap_samples,
                    seed=313,
                )
            ),
            "memory_os_accuracy_ci": asdict(
                bootstrap_mean_ci(
                    [float(value) for value in memory_os_correct],
                    samples=args.bootstrap_samples,
                    seed=314,
                )
            ),
            "memory_os_vs_vector_accuracy_diff_ci": asdict(
                bootstrap_paired_difference_ci(
                    memory_os_correct,
                    memory_vector_correct,
                    samples=args.bootstrap_samples,
                    seed=315,
                )
            ),
            "memory_os_vs_temporal_entity_accuracy_diff_ci": asdict(
                bootstrap_paired_difference_ci(
                    memory_os_correct,
                    memory_temporal_correct,
                    samples=args.bootstrap_samples,
                    seed=316,
                )
            ),
        },
        "confusion": {
            "guard_schema_only": _guard_confusion(guard_report.predictions, "schema_only_status"),
            "guard_prompt_only": _guard_confusion(guard_report.predictions, "prompt_only_status"),
            "guard_melm": _guard_confusion(guard_report.predictions, "melm_status"),
            "memory_vector": _boolean_confusion(memory_report.predictions, "vector_correct"),
            "memory_temporal_entity": _boolean_confusion(memory_report.predictions, "temporal_entity_correct"),
            "memory_os": _boolean_confusion(memory_report.predictions, "memory_os_correct"),
        },
        "authored_batch_gate_passed": (
            not dataset.validation_errors
            and guard_report.gate_passed
            and memory_report.gate_passed
        ),
        "publication_grade_ready": (
            externally_blind
            and not dataset.validation_errors
            and guard_report.gate_passed
            and memory_report.gate_passed
        ),
        "recommendation": (
            "freeze_external_blind_batch_and_rerun"
            if externally_blind and guard_report.gate_passed and memory_report.gate_passed
            else "use_this_as_seed_and_author_external_blind_batch"
        ),
    }
    _write_outputs(payload, Path(args.out_json), Path(args.out_md))

    print("MELM authored support/refunds benchmark")
    print(f"- schema_validation_passed={payload['schema_validation_passed']}")
    print(f"- authored_batch_gate_passed={payload['authored_batch_gate_passed']}")
    print(f"- publication_grade_ready={payload['publication_grade_ready']}")
    print(f"- events/facts/turns={payload['events']}/{payload['facts']}/{payload['turns']}")
    print(f"- guard_cases={payload['guard_cases']} gate={guard_report.gate_passed}")
    print(f"- memory_cases={payload['memory_cases']} gate={memory_report.gate_passed}")
    print(f"- recommendation={payload['recommendation']}")


def _guard_confusion(predictions, field_name: str) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for prediction in predictions:
        expected = prediction.expected_status
        observed = getattr(prediction, field_name)
        matrix.setdefault(expected, {})
        matrix[expected][observed] = matrix[expected].get(observed, 0) + 1
    return matrix


def _boolean_confusion(predictions, field_name: str) -> dict[str, int]:
    return {
        "correct": sum(1 for prediction in predictions if getattr(prediction, field_name)),
        "incorrect": sum(1 for prediction in predictions if not getattr(prediction, field_name)),
    }


def _write_outputs(payload: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")


def _markdown(payload: dict) -> str:
    guard = payload["guard_report"]
    memory = payload["memory_report"]
    stats = payload["statistics"]
    lines = [
        "# MELM Authored Support/Refunds Benchmark",
        "",
        f"Dataset: `{payload['dataset_path']}`",
        f"Schema validation passed: `{payload['schema_validation_passed']}`",
        f"Authored batch gate passed: `{payload['authored_batch_gate_passed']}`",
        f"Publication-grade ready: `{payload['publication_grade_ready']}`",
        f"Recommendation: `{payload['recommendation']}`",
        "",
        f"- Turns/events/facts: `{payload['turns']}` / `{payload['events']}` / `{payload['facts']}`",
        f"- Guard cases: `{payload['guard_cases']}`",
        f"- Memory cases: `{payload['memory_cases']}`",
        f"- Bootstrap samples: `{payload['bootstrap_samples']}`",
        "",
        "## Guard",
        "",
        f"- Gate passed: `{guard['gate_passed']}`",
        f"- MELM accuracy: `{guard['melm_accuracy']:.2%}`",
        f"- Schema-only accuracy: `{guard['schema_only_accuracy']:.2%}`",
        f"- Prompt-only accuracy: `{guard['prompt_only_accuracy']:.2%}`",
        f"- MELM false-allow rate: `{guard['melm_false_allow_rate']:.2%}`",
        f"- Schema-only false-allow rate: `{guard['schema_only_false_allow_rate']:.2%}`",
        f"- False-allow reduction vs schema: `{guard['false_allow_reduction_vs_schema']:.2%}`",
        f"- Valid-action allow rate: `{guard['valid_action_allow_rate']:.2%}`",
        f"- Traceability: `{guard['traceability']:.2%}`",
        "",
        "| Estimate | Mean | 95% CI |",
        "|---|---:|---:|",
        _ci_row("Guard MELM accuracy", stats["guard_melm_accuracy_ci"]),
        _ci_row("Guard MELM - schema accuracy", stats["guard_melm_vs_schema_accuracy_diff_ci"]),
        _ci_row("Guard MELM - prompt accuracy", stats["guard_melm_vs_prompt_accuracy_diff_ci"]),
        "",
        "## Memory OS",
        "",
        f"- Gate passed: `{memory['gate_passed']}`",
        f"- Vector RAG accuracy: `{memory['vector_accuracy']:.2%}`",
        f"- Temporal/entity RAG accuracy: `{memory['temporal_entity_accuracy']:.2%}`",
        f"- Memory OS accuracy: `{memory['memory_os_accuracy']:.2%}`",
        f"- Memory OS gain vs vector: `{memory['memory_os_gain_vs_vector']:.2%}`",
        f"- Positive recall: `{memory['positive_recall']:.2%}`",
        f"- Negative abstention: `{memory['negative_abstention']:.2%}`",
        "",
        "| Estimate | Mean | 95% CI |",
        "|---|---:|---:|",
        _ci_row("Memory OS accuracy", stats["memory_os_accuracy_ci"]),
        _ci_row("Memory OS - vector accuracy", stats["memory_os_vs_vector_accuracy_diff_ci"]),
        _ci_row("Memory OS - temporal/entity accuracy", stats["memory_os_vs_temporal_entity_accuracy_diff_ci"]),
        "",
        "## Dataset Checks",
        "",
    ]
    if payload["validation_errors"]:
        lines.extend(f"- {error}" for error in payload["validation_errors"])
    else:
        lines.append("- No schema or coverage validation errors.")
    lines.extend(
        [
            "",
            "## Candid Interpretation",
            "",
            "This is a stronger authored seed batch, not yet a publishable external benchmark. "
            "It validates that the MELM pipeline can ingest non-generator JSONL, preserve "
            "evidence provenance, score Guard and Memory OS together, and report uncertainty. "
            "Publication-grade evidence still requires an independently authored blind batch "
            "or real support logs with human labels.",
            "",
        ]
    )
    return "\n".join(lines)


def _ci_row(label: str, ci: dict) -> str:
    return f"| {label} | {ci['estimate']:.2%} | [{ci['low']:.2%}, {ci['high']:.2%}] |"


if __name__ == "__main__":
    main()
