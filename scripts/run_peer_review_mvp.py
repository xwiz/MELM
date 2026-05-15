"""Run peer-review-style MELM Guard + Memory OS validation."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import generate_support_refund_benchmark
from melm.guard import evaluate_guard_benchmark
from melm.memory import SupportMemoryOS, evaluate_memory_os
from melm.evaluation import bootstrap_mean_ci, bootstrap_paired_difference_ci


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--out-json", default="reports/melm_peer_review_mvp.json")
    parser.add_argument("--out-md", default="reports/melm_peer_review_mvp.md")
    args = parser.parse_args()

    development = _run_split(
        "development",
        scenario_repeats=args.scenario_repeats,
        seed=17,
        start_order_index=2000,
        bootstrap_samples=args.bootstrap_samples,
    )
    heldout = _run_split(
        "heldout_seed",
        scenario_repeats=args.scenario_repeats,
        seed=91,
        start_order_index=9000,
        bootstrap_samples=args.bootstrap_samples,
    )
    passed = (
        development["guard_report"]["gate_passed"]
        and development["memory_report"]["gate_passed"]
        and heldout["guard_report"]["gate_passed"]
        and heldout["memory_report"]["gate_passed"]
        and heldout["statistics"]["guard_melm_vs_schema_accuracy_diff_ci"]["low"] > 0.0
        and heldout["statistics"]["memory_os_vs_vector_accuracy_diff_ci"]["low"] > 0.0
    )
    payload = {
        "benchmark": "melm_guard_memory_peer_review_mvp",
        "no_api_required": True,
        "scenario_repeats": args.scenario_repeats,
        "bootstrap_samples": args.bootstrap_samples,
        "peer_review_gate_passed": passed,
        "development": development,
        "heldout_seed": heldout,
        "recommendation": (
            "advance_to_external_human_labeled_support_logs"
            if passed
            else "hold_for_benchmark_or_rule_rework"
        ),
        "limitations": [
            "Synthetic support/refund cases are deterministic and policy-derived.",
            "No live LLM baseline is required; prompt-only is simulated by fixed proposals.",
            "Temporal/entity RAG and Memory OS share the same raw event annotations.",
            "The next peer-review step must add external human-labeled logs or independently authored fixtures.",
        ],
    }
    _write_outputs(payload, Path(args.out_json), Path(args.out_md))

    print("MELM peer-review MVP validation")
    print(f"- peer_review_gate_passed={passed}")
    print(f"- development_guard={development['guard_report']['gate_passed']}")
    print(f"- development_memory={development['memory_report']['gate_passed']}")
    print(f"- heldout_guard={heldout['guard_report']['gate_passed']}")
    print(f"- heldout_memory={heldout['memory_report']['gate_passed']}")
    print(f"- recommendation={payload['recommendation']}")


def _run_split(
    split_name: str,
    *,
    scenario_repeats: int,
    seed: int,
    start_order_index: int,
    bootstrap_samples: int,
) -> dict:
    fixture = generate_support_refund_benchmark(
        scenario_repeats=scenario_repeats,
        seed=seed,
        start_order_index=start_order_index,
    )
    guard_report = evaluate_guard_benchmark(
        fixture.facts,
        fixture.rules,
        fixture.guard_cases,
        current_time=fixture.current_time,
    )
    memory_report = evaluate_memory_os(
        SupportMemoryOS(fixture.events),
        fixture.memory_cases,
        k=2,
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
    return {
        "split": split_name,
        "seed": seed,
        "events": len(fixture.events),
        "facts": len(fixture.facts),
        "guard_cases": len(fixture.guard_cases),
        "memory_cases": len(fixture.memory_cases),
        "guard_report": _drop_predictions(asdict(guard_report)),
        "memory_report": _drop_predictions(asdict(memory_report)),
        "robustness": _robustness_curve(
            fixture,
            seed=seed + 100,
            bootstrap_samples=bootstrap_samples,
        ),
        "statistics": {
            "guard_melm_accuracy_ci": asdict(
                bootstrap_mean_ci(
                    [float(value) for value in guard_melm_correct],
                    samples=bootstrap_samples,
                    seed=seed,
                )
            ),
            "guard_melm_vs_schema_accuracy_diff_ci": asdict(
                bootstrap_paired_difference_ci(
                    guard_melm_correct,
                    guard_schema_correct,
                    samples=bootstrap_samples,
                    seed=seed + 1,
                )
            ),
            "guard_melm_vs_prompt_accuracy_diff_ci": asdict(
                bootstrap_paired_difference_ci(
                    guard_melm_correct,
                    guard_prompt_correct,
                    samples=bootstrap_samples,
                    seed=seed + 2,
                )
            ),
            "memory_os_accuracy_ci": asdict(
                bootstrap_mean_ci(
                    [float(value) for value in memory_os_correct],
                    samples=bootstrap_samples,
                    seed=seed + 3,
                )
            ),
            "memory_os_vs_vector_accuracy_diff_ci": asdict(
                bootstrap_paired_difference_ci(
                    memory_os_correct,
                    memory_vector_correct,
                    samples=bootstrap_samples,
                    seed=seed + 4,
                )
            ),
            "memory_os_vs_temporal_entity_accuracy_diff_ci": asdict(
                bootstrap_paired_difference_ci(
                    memory_os_correct,
                    memory_temporal_correct,
                    samples=bootstrap_samples,
                    seed=seed + 5,
                )
            ),
        },
    }


def _robustness_curve(fixture, *, seed: int, bootstrap_samples: int) -> list[dict]:
    rows: list[dict] = []
    for rate in (0.05, 0.10, 0.20):
        degraded = _drop_non_policy_evidence(fixture, rate=rate, seed=seed + int(rate * 1000))
        guard_report = evaluate_guard_benchmark(
            degraded["facts"],
            fixture.rules,
            fixture.guard_cases,
            current_time=fixture.current_time,
        )
        memory_report = evaluate_memory_os(
            SupportMemoryOS(degraded["events"]),
            fixture.memory_cases,
            k=2,
        )
        memory_os_correct = [prediction.memory_os_correct for prediction in memory_report.predictions]
        memory_vector_correct = [prediction.vector_correct for prediction in memory_report.predictions]
        rows.append(
            {
                "drop_rate": rate,
                "dropped_events": degraded["dropped_events"],
                "guard_melm_accuracy": guard_report.melm_accuracy,
                "guard_false_allow_rate": guard_report.melm_false_allow_rate,
                "guard_valid_action_allow_rate": guard_report.valid_action_allow_rate,
                "memory_os_accuracy": memory_report.memory_os_accuracy,
                "memory_os_gain_vs_vector": memory_report.memory_os_gain_vs_vector,
                "memory_os_vs_vector_ci": asdict(
                    bootstrap_paired_difference_ci(
                        memory_os_correct,
                        memory_vector_correct,
                        samples=bootstrap_samples,
                        seed=seed + int(rate * 10000),
                    )
                ),
            }
        )
    return rows


def _drop_non_policy_evidence(fixture, *, rate: float, seed: int) -> dict:
    rng = random.Random(seed)
    candidates = [
        event.event_id
        for event in fixture.events
        if not event.event_id.startswith("policy_")
    ]
    drop_count = max(1, round(len(candidates) * rate))
    dropped = set(rng.sample(candidates, min(drop_count, len(candidates))))
    return {
        "events": [event for event in fixture.events if event.event_id not in dropped],
        "facts": [fact for fact in fixture.facts if fact.source_event_id not in dropped],
        "dropped_events": len(dropped),
    }


def _drop_predictions(report: dict) -> dict:
    report = dict(report)
    report["predictions"] = f"{len(report.get('predictions', []))} records omitted; see benchmark-specific report for case table"
    if report.get("by_category"):
        report["by_category"] = {
            category: _drop_predictions(category_report)
            for category, category_report in report["by_category"].items()
        }
    return report


def _write_outputs(payload: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")


def _markdown(payload: dict) -> str:
    lines = [
        "# MELM Guard + Memory OS Peer-Review MVP",
        "",
        f"No API required: `{payload['no_api_required']}`",
        f"Scenario repeats per split: `{payload['scenario_repeats']}`",
        f"Bootstrap samples: `{payload['bootstrap_samples']}`",
        f"Peer-review gate passed: `{payload['peer_review_gate_passed']}`",
        f"Recommendation: `{payload['recommendation']}`",
        "",
    ]
    for split_key in ("development", "heldout_seed"):
        split = payload[split_key]
        guard = split["guard_report"]
        memory = split["memory_report"]
        stats = split["statistics"]
        lines.extend(
            [
                f"## {split['split'].replace('_', ' ').title()} Split",
                "",
                f"- Seed: `{split['seed']}`",
                f"- Events/facts: `{split['events']}` / `{split['facts']}`",
                f"- Guard cases: `{split['guard_cases']}`",
                f"- Memory cases: `{split['memory_cases']}`",
                f"- Guard gate: `{guard['gate_passed']}`",
                f"- Guard MELM accuracy: `{guard['melm_accuracy']:.2%}`",
                f"- Guard schema-only accuracy: `{guard['schema_only_accuracy']:.2%}`",
                f"- Guard false-allow reduction vs schema: `{guard['false_allow_reduction_vs_schema']:.2%}`",
                f"- Guard traceability: `{guard['traceability']:.2%}`",
                f"- Memory gate: `{memory['gate_passed']}`",
                f"- Vector RAG accuracy: `{memory['vector_accuracy']:.2%}`",
                f"- Temporal/entity RAG accuracy: `{memory['temporal_entity_accuracy']:.2%}`",
                f"- Memory OS accuracy: `{memory['memory_os_accuracy']:.2%}`",
                f"- Memory OS gain vs vector: `{memory['memory_os_gain_vs_vector']:.2%}`",
                f"- Negative abstention: `{memory['negative_abstention']:.2%}`",
                "",
                "| Estimate | Mean | 95% CI |",
                "|---|---:|---:|",
                _ci_row("Guard MELM accuracy", stats["guard_melm_accuracy_ci"]),
                _ci_row("Guard MELM - schema accuracy", stats["guard_melm_vs_schema_accuracy_diff_ci"]),
                _ci_row("Guard MELM - prompt accuracy", stats["guard_melm_vs_prompt_accuracy_diff_ci"]),
                _ci_row("Memory OS accuracy", stats["memory_os_accuracy_ci"]),
                _ci_row("Memory OS - vector accuracy", stats["memory_os_vs_vector_accuracy_diff_ci"]),
                _ci_row("Memory OS - temporal/entity accuracy", stats["memory_os_vs_temporal_entity_accuracy_diff_ci"]),
                "",
                "### Evidence Dropout Robustness",
                "",
                "| Dropped | Guard accuracy | Guard false-allow | Valid allow | Memory OS accuracy | OS-vector gain | 95% CI |",
                "|---:|---:|---:|---:|---:|---:|---:|",
                *[_robustness_row(row) for row in split["robustness"]],
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            *[f"- {item}" for item in payload["limitations"]],
            "",
            "Interpretation: this is now a stronger internal benchmark, not a "
            "publication by itself. It adds balanced categories, held-out seed "
            "replication, paired bootstrap intervals, and explicit limitations. "
            "The next publishable step is independent human-labeled support logs "
            "or fixtures authored by someone who did not implement the system.",
            "",
        ]
    )
    return "\n".join(lines)


def _ci_row(label: str, ci: dict) -> str:
    return f"| {label} | {ci['estimate']:.2%} | [{ci['low']:.2%}, {ci['high']:.2%}] |"


def _robustness_row(row: dict) -> str:
    ci = row["memory_os_vs_vector_ci"]
    return (
        f"| {row['drop_rate']:.0%} ({row['dropped_events']}) | "
        f"{row['guard_melm_accuracy']:.2%} | "
        f"{row['guard_false_allow_rate']:.2%} | "
        f"{row['guard_valid_action_allow_rate']:.2%} | "
        f"{row['memory_os_accuracy']:.2%} | "
        f"{row['memory_os_gain_vs_vector']:.2%} | "
        f"[{ci['low']:.2%}, {ci['high']:.2%}] |"
    )


if __name__ == "__main__":
    main()
