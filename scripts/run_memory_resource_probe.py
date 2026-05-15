"""Measure current MELM event-memory resource behavior against plain RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.memory import Event, EventMemory


NAMES = ("maya", "leo", "sam", "nora", "owen", "iris", "lila", "ben", "rafi", "mina")
OBJECTS = (
    "rainbow bead",
    "yellow box",
    "star blocks",
    "green bead",
    "purple lunchbox",
    "tiny shell",
    "red pail",
    "blue bucket",
    "silver tag",
    "moon sticker",
)
PLACES = (
    "puzzle shelf",
    "reading rug",
    "blue mat",
    "window",
    "sandbox",
    "craft tray",
    "red bin",
    "music room",
)
ACTIONS = ("put", "moved", "slid", "carried", "stacked", "rebuilt", "saw", "reminded")


@dataclass(frozen=True)
class ResourceProbeRow:
    events: int
    build_seconds: float
    rag_query_ms: float
    event_memory_query_ms: float
    event_memory_overhead_ratio: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="12,100,1000,5000")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--out-json", default="reports/melm_vs_rag_resource_efficiency.json")
    parser.add_argument("--out-md", default="reports/melm_vs_rag_resource_efficiency.md")
    args = parser.parse_args()

    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    rows = [run_probe(size, queries=args.queries) for size in sizes]
    payload = {
        "implementation": "current pure-Python brute-force EventMemory",
        "queries_per_size": args.queries,
        "rows": [asdict(row) for row in rows],
        "conclusion": (
            "Current MELM event memory is more evidence/context efficient than RAG on "
            "the validation tasks, but this Python implementation is not more CPU "
            "efficient than RAG because both retrievers scan the full event list."
        ),
    }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print("MELM vs RAG resource probe")
    for row in rows:
        print(
            f"- events={row.events} rag={row.rag_query_ms:.3f}ms "
            f"event_memory={row.event_memory_query_ms:.3f}ms "
            f"overhead={row.event_memory_overhead_ratio:.2f}x"
        )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def run_probe(events: int, *, queries: int) -> ResourceProbeRow:
    t0 = time.perf_counter()
    memory = EventMemory(_make_events(events))
    build_seconds = time.perf_counter() - t0
    query_texts = [
        f"Where did {NAMES[(index * 5) % len(NAMES)].title()} put the "
        f"{OBJECTS[(index * 7) % len(OBJECTS)]}?"
        for index in range(queries)
    ]

    rag_ms = _avg_query_ms(lambda query: memory.retrieve_rag(query, k=2), query_texts)
    event_ms = _avg_query_ms(lambda query: memory.retrieve_event_memory(query, k=2), query_texts)
    return ResourceProbeRow(
        events=events,
        build_seconds=round(build_seconds, 6),
        rag_query_ms=round(rag_ms, 6),
        event_memory_query_ms=round(event_ms, 6),
        event_memory_overhead_ratio=round(event_ms / rag_ms, 3) if rag_ms else 0.0,
    )


def _make_events(count: int) -> list[Event]:
    events: list[Event] = []
    for index in range(count):
        actor = NAMES[index % len(NAMES)]
        obj = OBJECTS[index % len(OBJECTS)]
        place = PLACES[(index * 3) % len(PLACES)]
        action = ACTIONS[(index * 5) % len(ACTIONS)]
        events.append(
            Event(
                event_id=f"e{index:06d}",
                source_span=f"{actor.title()} {action} the {obj} near the {place}.",
                time_index=index,
                actors=(actor,),
                action_or_state=action,
                objects=(obj, place),
                location=place,
                previous_event_id=f"e{index - 1:06d}" if index else None,
                next_event_id=f"e{index + 1:06d}" if index < count - 1 else None,
            )
        )
    return events


def _avg_query_ms(fn, queries: list[str]) -> float:
    t0 = time.perf_counter()
    for query in queries:
        fn(query)
    return ((time.perf_counter() - t0) / len(queries)) * 1000


def _markdown(payload: dict) -> str:
    lines = [
        "# MELM vs RAG Resource Efficiency",
        "",
        f"Implementation: `{payload['implementation']}`",
        f"Queries per size: `{payload['queries_per_size']}`",
        "",
        payload["conclusion"],
        "",
        "| Events | Build s | RAG ms/query | Event-memory ms/query | Overhead |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['events']} | {row['build_seconds']:.4f} | "
            f"{row['rag_query_ms']:.3f} | {row['event_memory_query_ms']:.3f} | "
            f"{row['event_memory_overhead_ratio']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the current win is in answerability, context budget, "
            "and false-answer control. A true embedded resource win requires an "
            "indexed event/state graph or Rust/C sidecar rather than brute-force "
            "Python scans.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
