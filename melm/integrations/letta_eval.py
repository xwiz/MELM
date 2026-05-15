"""Letta Evals export helpers for MELM benchmarks.

The export is intentionally simple and transparent:
- JSONL records follow Letta's documented `input`, `ground_truth`, `tags`,
  `metadata`, and `agent_args` shape.
- The suite file is a runnable template once a real Letta agent file or agent id
  is supplied.
- No official Letta score is produced by this module; it only prepares the same
  dataset and memory store for side-by-side runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from melm.appliance import MemoryRecord
from melm.benchmarks.public_memory import PublicMemoryBenchmark


@dataclass(frozen=True)
class LettaEvalPack:
    dataset_path: str
    suite_path: str
    memory_path: str
    readme_path: str
    samples: int
    memories: int


def export_locomo_letta_eval_pack(
    benchmark: PublicMemoryBenchmark,
    out_dir: str | Path,
    *,
    max_questions: int | None = None,
    agent_file: str = "agent.af",
    base_url: str = "http://localhost:8283",
    gate_value: float = 0.60,
) -> LettaEvalPack:
    """Export LoCoMo as a Letta Evals-style pack."""

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = target_dir / "locomo_letta_dataset.jsonl"
    memory_path = target_dir / "locomo_memory.jsonl"
    suite_path = target_dir / "locomo_letta_suite.yaml"
    readme_path = target_dir / "README.md"

    memories = locomo_memory_records(benchmark)
    _write_jsonl(memory_path, [_memory_to_json(record) for record in memories])

    samples = _letta_dataset_records(
        benchmark,
        memory_path=memory_path,
        max_questions=max_questions,
    )
    validate_letta_dataset_records(samples)
    _write_jsonl(dataset_path, samples)
    suite_path.write_text(
        _suite_yaml(
            dataset_path=dataset_path.name,
            agent_file=agent_file,
            base_url=base_url,
            gate_value=gate_value,
        ),
        encoding="utf-8",
    )
    readme_path.write_text(
        _readme(
            dataset_path=dataset_path.name,
            suite_path=suite_path.name,
            memory_path=memory_path.name,
        ),
        encoding="utf-8",
    )
    return LettaEvalPack(
        dataset_path=str(dataset_path),
        suite_path=str(suite_path),
        memory_path=str(memory_path),
        readme_path=str(readme_path),
        samples=len(samples),
        memories=len(memories),
    )


def locomo_memory_records(benchmark: PublicMemoryBenchmark) -> list[MemoryRecord]:
    """Return LoCoMo sessions as MELM appliance memory records."""

    records: list[MemoryRecord] = []
    for document in benchmark.documents:
        records.append(
            MemoryRecord(
                memory_id=document.doc_id,
                text=document.raw_text,
                kind="locomo_session",
                created_at=document.date_time,
                metadata={
                    "sample_id": document.sample_id,
                    "session_id": document.session_id,
                    "observation": document.observation,
                    "summary": document.session_summary,
                    "event_summary": document.event_summary,
                },
            )
        )
    return records


def validate_letta_dataset_records(records: list[dict[str, Any]]) -> list[str]:
    """Return validation errors for the exported Letta dataset shape."""

    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        if "input" not in record or not isinstance(record["input"], str) or not record["input"]:
            errors.append(f"record {index}: input must be a non-empty string")
        if "ground_truth" not in record or not isinstance(record["ground_truth"], str):
            errors.append(f"record {index}: ground_truth must be a string")
        if "tags" in record and not isinstance(record["tags"], list):
            errors.append(f"record {index}: tags must be a list")
        if "metadata" not in record or not isinstance(record["metadata"], dict):
            errors.append(f"record {index}: metadata must be an object")
        if "agent_args" not in record or not isinstance(record["agent_args"], dict):
            errors.append(f"record {index}: agent_args must be an object")
    return errors


def _letta_dataset_records(
    benchmark: PublicMemoryBenchmark,
    *,
    memory_path: Path,
    max_questions: int | None,
) -> list[dict[str, Any]]:
    questions = [
        question
        for question in benchmark.questions
        if question.has_gold_answer and question.evidence_session_ids
    ]
    if max_questions is not None:
        questions = questions[:max_questions]

    records: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        records.append(
            {
                "id": index,
                "input": (
                    "Answer from long-term memory. Keep the answer concise and cite "
                    "the memory/session ids that support it.\n\n"
                    f"Question: {question.question}"
                ),
                "ground_truth": question.answer,
                "tags": ["locomo", "memory", question.category],
                "metadata": {
                    "benchmark": benchmark.name,
                    "question_id": question.question_id,
                    "sample_id": question.sample_id,
                    "question": question.question,
                    "category": question.category,
                    "category_id": question.category_id,
                    "evidence_dialog_ids": list(question.evidence_dialog_ids),
                    "evidence_session_ids": list(question.evidence_session_ids),
                },
                "agent_args": {
                    "memory_store_path": str(memory_path),
                    "sample_id": question.sample_id,
                    "evidence_session_ids": list(question.evidence_session_ids),
                },
            }
        )
    return records


def _suite_yaml(
    *,
    dataset_path: str,
    agent_file: str,
    base_url: str,
    gate_value: float,
) -> str:
    return "\n".join(
        [
            "name: melm-locomo-letta-memory-eval",
            "description: LoCoMo long-term memory QA pack exported by MELM for official Letta Evals runs.",
            f"dataset: {dataset_path}",
            "",
            "target:",
            "  kind: agent",
            f"  agent_file: {agent_file}",
            f"  base_url: {base_url}",
            "",
            "graders:",
            "  answer_contains:",
            "    kind: tool",
            "    function: contains",
            "    extractor: last_assistant",
            "",
            "gate:",
            "  kind: simple",
            "  metric_key: answer_contains",
            "  aggregation: avg_score",
            "  op: gte",
            f"  value: {gate_value:.2f}",
            "",
        ]
    )


def _readme(*, dataset_path: str, suite_path: str, memory_path: str) -> str:
    return "\n".join(
        [
            "# MELM Letta Evaluation Pack",
            "",
            "This directory contains a LoCoMo memory evaluation pack exported from MELM.",
            "",
            f"- Dataset: `{dataset_path}`",
            f"- Suite template: `{suite_path}`",
            f"- Memory store for MELM/local targets: `{memory_path}`",
            "",
            "Run MELM locally on the same dataset:",
            "",
            "```powershell",
            f"python scripts\\run_melm_letta_style_eval.py --dataset artifacts\\letta_eval\\{dataset_path} --memory artifacts\\letta_eval\\{memory_path}",
            "```",
            "",
            "Run Letta Evals after supplying a real Letta agent file/server:",
            "",
            "```powershell",
            "pip install letta-evals",
            f"letta-evals validate artifacts\\letta_eval\\{suite_path}",
            f"letta-evals run artifacts\\letta_eval\\{suite_path}",
            "```",
            "",
            "Candid limitation: the suite is a template until `agent_file` points to a real Letta agent configured with equivalent memory.",
            "",
        ]
    )


def _memory_to_json(record: MemoryRecord) -> dict[str, Any]:
    return {
        "schema": "melm.appliance.memory.v1",
        "memory_id": record.memory_id,
        "text": record.text,
        "kind": record.kind,
        "created_at": record.created_at,
        "metadata": record.metadata,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
