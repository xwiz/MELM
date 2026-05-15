"""Run combined MELM Guard + Memory OS MVP benchmark."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import support_refund_fixture
from melm.guard import evaluate_guard_benchmark
from melm.memory import Event, SupportMemoryOS, evaluate_memory_os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--sizes", default="100,1000,5000")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--out-json", default="reports/melm_guard_memory_runtime.json")
    parser.add_argument("--out-md", default="reports/melm_guard_memory_runtime.md")
    args = parser.parse_args()

    fixture = support_refund_fixture()
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
    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    resource_rows = [run_resource_probe(size, queries=args.queries, k=args.k) for size in sizes]
    runtime_gate = guard_report.gate_passed and memory_report.gate_passed
    payload = {
        "source": "support_refund_fixture",
        "runtime_gate_passed": runtime_gate,
        "guard_report": asdict(guard_report),
        "memory_os_report": asdict(memory_report),
        "resource_probe": {
            "implementation": "pure-Python indexed SupportMemoryOS over EventMemory",
            "queries_per_size": args.queries,
            "rows": resource_rows,
            "conclusion": (
                "This MVP reports latency honestly. The support indexes reduce "
                "state lookups, but vector and structured retrieval still keep "
                "a Python EventMemory path; a production CPU win needs a tighter "
                "indexed graph or native sidecar."
            ),
        },
        "recommendation": (
            "advance_to_external_review_dataset"
            if runtime_gate
            else "hold_for_harder_fixture_or_rule_rework"
        ),
    }
    _write_outputs(payload, Path(args.out_json), Path(args.out_md))

    print("MELM Guard + Memory OS runtime benchmark")
    print(f"- runtime_gate_passed={runtime_gate}")
    print(f"- guard_gate_passed={guard_report.gate_passed}")
    print(f"- memory_os_gate_passed={memory_report.gate_passed}")
    print(f"- recommendation={payload['recommendation']}")
    for row in resource_rows:
        print(
            f"- events={row['events']} rag={row['vector_rag_ms']:.3f}ms "
            f"temporal_entity={row['temporal_entity_ms']:.3f}ms "
            f"state_lookup={row['state_lookup_ms']:.3f}ms"
        )


def run_resource_probe(events: int, *, queries: int, k: int) -> dict:
    generated = _make_support_like_events(events)
    t0 = time.perf_counter()
    memory = SupportMemoryOS(generated)
    build_seconds = time.perf_counter() - t0
    query_texts = [f"What is the latest status for order o{1000 + (index % events):04d}?" for index in range(queries)]

    vector_ms = _avg_query_ms(lambda query: memory.event_memory.retrieve_rag(query, k=k), query_texts)
    temporal_ms = _avg_query_ms(lambda query: memory.retrieve_temporal_entity_rag(query, k=k), query_texts)
    state_ms = _avg_query_ms(
        lambda query: memory.resolve_order_state(_order_id_from_query(query), "status"),
        query_texts,
    )
    return {
        "events": events,
        "build_seconds": round(build_seconds, 6),
        "vector_rag_ms": round(vector_ms, 6),
        "temporal_entity_ms": round(temporal_ms, 6),
        "state_lookup_ms": round(state_ms, 6),
    }


def _make_support_like_events(count: int) -> list[Event]:
    events: list[Event] = []
    statuses = ("delivered", "shipped", "refunded", "returned")
    for index in range(count):
        order_id = f"o{1000 + index:04d}"
        status = statuses[index % len(statuses)]
        events.append(
            Event(
                event_id=f"resource_{index:06d}",
                source_span=f"Order {order_id} is currently {status}.",
                time_index=index,
                actors=("support",),
                action_or_state="status",
                objects=(order_id, "status"),
                location="support_queue",
                metadata={
                    "order_id": order_id,
                    "fact_subject": f"order:{order_id}",
                    "fact_predicate": "status",
                    "fact_value": status,
                },
            )
        )
    return events


def _avg_query_ms(fn, queries: list[str]) -> float:
    t0 = time.perf_counter()
    for query in queries:
        fn(query)
    return ((time.perf_counter() - t0) / len(queries)) * 1000 if queries else 0.0


def _order_id_from_query(query: str) -> str:
    return next((part.lower().strip("?.!,") for part in query.split() if part.lower().startswith("o")), "")


def _write_outputs(payload: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")


def _markdown(payload: dict) -> str:
    guard = payload["guard_report"]
    memory = payload["memory_os_report"]
    resource = payload["resource_probe"]
    lines = [
        "# MELM Guard + Memory OS Runtime Benchmark",
        "",
        f"Source: `{payload['source']}`",
        f"Runtime gate passed: `{payload['runtime_gate_passed']}`",
        f"Recommendation: `{payload['recommendation']}`",
        "",
        "## Guard Signal",
        "",
        f"- Gate passed: `{guard['gate_passed']}`",
        f"- False-allow reduction vs schema: `{guard['false_allow_reduction_vs_schema']:.2%}`",
        f"- Valid-action allow rate: `{guard['valid_action_allow_rate']:.2%}`",
        f"- Traceability: `{guard['traceability']:.2%}`",
        "",
        "## Memory OS Signal",
        "",
        f"- Gate passed: `{memory['gate_passed']}`",
        f"- Vector RAG accuracy: `{memory['vector_accuracy']:.2%}`",
        f"- Temporal/entity RAG accuracy: `{memory['temporal_entity_accuracy']:.2%}`",
        f"- Memory OS accuracy: `{memory['memory_os_accuracy']:.2%}`",
        f"- Memory OS gain vs vector: `{memory['memory_os_gain_vs_vector']:.2%}`",
        f"- Positive recall: `{memory['positive_recall']:.2%}`",
        f"- Negative abstention: `{memory['negative_abstention']:.2%}`",
        "",
        "## Resource Probe",
        "",
        f"Implementation: `{resource['implementation']}`",
        f"Queries per size: `{resource['queries_per_size']}`",
        "",
        "| Events | Build s | Vector RAG ms | Temporal/entity ms | State lookup ms |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in resource["rows"]:
        lines.append(
            f"| {row['events']} | {row['build_seconds']:.4f} | "
            f"{row['vector_rag_ms']:.3f} | {row['temporal_entity_ms']:.3f} | "
            f"{row['state_lookup_ms']:.3f} |"
        )
    lines.extend(["", resource["conclusion"], ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
