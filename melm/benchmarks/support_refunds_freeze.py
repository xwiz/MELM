"""Freeze and preregistration helpers for support/refunds blind batches."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .support_refunds_authored import (
    AuthoredSupportRefundDataset,
    load_authored_support_refund_dataset,
)


DEFAULT_SUPPORT_REFUNDS_PREREGISTRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "support_refunds_external_blind_preregistration.json"
)


@dataclass(frozen=True)
class SupportRefundDatasetSummary:
    turns: int
    fact_events: int
    facts: int
    guard_cases: int
    memory_cases: int
    guard_category_counts: dict[str, int]
    memory_category_counts: dict[str, int]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of the exact bytes on disk."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_support_refunds_preregistration(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the preregistration JSON for an external support/refunds batch."""

    preregistration_path = Path(path) if path else DEFAULT_SUPPORT_REFUNDS_PREREGISTRATION_PATH
    return json.loads(preregistration_path.read_text(encoding="utf-8"))


def summarize_support_refund_dataset(
    dataset: AuthoredSupportRefundDataset,
) -> SupportRefundDatasetSummary:
    """Count the key coverage dimensions used by the blind-batch protocol."""

    return SupportRefundDatasetSummary(
        turns=len(dataset.turns),
        fact_events=len(dataset.fixture.events),
        facts=len(dataset.fixture.facts),
        guard_cases=len(dataset.fixture.guard_cases),
        memory_cases=len(dataset.fixture.memory_cases),
        guard_category_counts=dict(Counter(case.category for case in dataset.fixture.guard_cases)),
        memory_category_counts=dict(Counter(case.category for case in dataset.fixture.memory_cases)),
    )


def validate_support_refunds_preregistration(
    dataset: AuthoredSupportRefundDataset,
    preregistration: dict[str, Any],
) -> list[str]:
    """Return coverage/protocol errors against the preregistered blind-batch plan."""

    errors: list[str] = []
    metadata = dataset.metadata
    summary = summarize_support_refund_dataset(dataset)

    expected_dataset_id = preregistration.get("dataset_id")
    if expected_dataset_id and metadata.get("dataset_id") != expected_dataset_id:
        errors.append(
            f"metadata.dataset_id {metadata.get('dataset_id')!r} does not match "
            f"preregistration dataset_id {expected_dataset_id!r}"
        )

    if preregistration.get("authoring_mode") == "external_blind_batch":
        if metadata.get("external_blind_batch") is not True:
            errors.append("metadata.external_blind_batch must be true")
        if metadata.get("requires_external_blind_batch") is not False:
            errors.append("metadata.requires_external_blind_batch must be false")

    minimums = preregistration.get("minimums", {})
    _check_minimum(errors, "turns", summary.turns, minimums.get("turns"))
    _check_minimum(errors, "fact_events", summary.fact_events, minimums.get("fact_events"))
    _check_minimum(errors, "guard_cases", summary.guard_cases, minimums.get("guard_cases"))
    _check_minimum(errors, "memory_cases", summary.memory_cases, minimums.get("memory_cases"))

    required_annotators = minimums.get("annotator_count")
    if required_annotators is not None:
        observed = int(metadata.get("annotator_count", 0) or 0)
        _check_minimum(errors, "metadata.annotator_count", observed, required_annotators)

    required_overlap = minimums.get("overlap_labeled_percent")
    if required_overlap is not None:
        observed = float(metadata.get("overlap_labeled_percent", 0.0) or 0.0)
        _check_minimum(errors, "metadata.overlap_labeled_percent", observed, required_overlap)

    adjudication_required = bool(preregistration.get("adjudication_required", True))
    if adjudication_required and not metadata.get("adjudication_record_path"):
        errors.append("metadata.adjudication_record_path is required")

    _check_category_counts(
        errors,
        "guard",
        summary.guard_category_counts,
        preregistration.get("required_guard_category_counts", {}),
    )
    _check_category_counts(
        errors,
        "memory",
        summary.memory_category_counts,
        preregistration.get("required_memory_category_counts", {}),
    )
    return errors


def build_support_refunds_freeze_manifest(
    dataset_path: str | Path,
    preregistration_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a freeze manifest for a support/refunds JSONL dataset."""

    path = Path(dataset_path)
    prereg_path = Path(preregistration_path) if preregistration_path else DEFAULT_SUPPORT_REFUNDS_PREREGISTRATION_PATH
    dataset = load_authored_support_refund_dataset(path)
    preregistration = load_support_refunds_preregistration(prereg_path)
    summary = summarize_support_refund_dataset(dataset)
    preregistration_errors = validate_support_refunds_preregistration(dataset, preregistration)

    return {
        "schema": "melm.support_refunds.freeze_manifest.v1",
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "dataset_path": str(path),
        "dataset_sha256": sha256_file(path),
        "dataset_size_bytes": path.stat().st_size,
        "preregistration_path": str(prereg_path),
        "preregistration_id": preregistration.get("preregistration_id", ""),
        "preregistration_sha256": sha256_file(prereg_path),
        "dataset_metadata": dataset.metadata,
        "summary": {
            "turns": summary.turns,
            "fact_events": summary.fact_events,
            "facts": summary.facts,
            "guard_cases": summary.guard_cases,
            "memory_cases": summary.memory_cases,
            "guard_category_counts": summary.guard_category_counts,
            "memory_category_counts": summary.memory_category_counts,
        },
        "schema_validation_passed": not dataset.validation_errors,
        "validation_errors": list(dataset.validation_errors),
        "preregistration_passed": not preregistration_errors,
        "preregistration_errors": preregistration_errors,
        "frozen_before_scoring": True,
    }


def verify_support_refunds_freeze(
    dataset_path: str | Path,
    manifest_path: str | Path,
) -> list[str]:
    """Return errors if the dataset no longer matches a freeze manifest."""

    path = Path(dataset_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("schema") != "melm.support_refunds.freeze_manifest.v1":
        errors.append("freeze manifest has an unsupported schema")
    current_hash = sha256_file(path)
    if manifest.get("dataset_sha256") != current_hash:
        errors.append(
            "dataset SHA-256 does not match freeze manifest: "
            f"{current_hash} != {manifest.get('dataset_sha256')}"
        )
    if manifest.get("frozen_before_scoring") is not True:
        errors.append("freeze manifest must set frozen_before_scoring=true")
    if manifest.get("schema_validation_passed") is not True:
        errors.append("freeze manifest was created from a dataset that failed schema validation")
    if manifest.get("preregistration_passed") is not True:
        errors.append("freeze manifest was created from a dataset that failed preregistration")
    return errors


def support_refunds_freeze_markdown(manifest: dict[str, Any]) -> str:
    """Render a compact Markdown freeze report."""

    summary = manifest["summary"]
    lines = [
        "# Support/Refunds External Blind Freeze Manifest",
        "",
        f"Dataset: `{manifest['dataset_path']}`",
        f"Dataset SHA-256: `{manifest['dataset_sha256']}`",
        f"Preregistration: `{manifest['preregistration_id']}`",
        f"Preregistration SHA-256: `{manifest['preregistration_sha256']}`",
        f"Generated UTC: `{manifest['generated_utc']}`",
        "",
        f"- Schema validation passed: `{manifest['schema_validation_passed']}`",
        f"- Preregistration passed: `{manifest['preregistration_passed']}`",
        f"- Frozen before scoring: `{manifest['frozen_before_scoring']}`",
        f"- Turns/fact events/facts: `{summary['turns']}` / `{summary['fact_events']}` / `{summary['facts']}`",
        f"- Guard cases: `{summary['guard_cases']}`",
        f"- Memory cases: `{summary['memory_cases']}`",
        "",
        "## Guard Categories",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{category}` | {count} |"
        for category, count in sorted(summary["guard_category_counts"].items())
    )
    lines.extend(["", "## Memory Categories", "", "| Category | Count |", "|---|---:|"])
    lines.extend(
        f"| `{category}` | {count} |"
        for category, count in sorted(summary["memory_category_counts"].items())
    )
    lines.extend(["", "## Validation", ""])
    if manifest["validation_errors"] or manifest["preregistration_errors"]:
        lines.extend(f"- {error}" for error in manifest["validation_errors"])
        lines.extend(f"- {error}" for error in manifest["preregistration_errors"])
    else:
        lines.append("- No schema, coverage, or preregistration errors.")
    lines.append("")
    return "\n".join(lines)


def _check_minimum(
    errors: list[str],
    label: str,
    observed: float,
    minimum: Any,
) -> None:
    if minimum is None:
        return
    if observed < float(minimum):
        errors.append(f"{label}={observed:g} is below preregistered minimum {float(minimum):g}")


def _check_category_counts(
    errors: list[str],
    label: str,
    observed: dict[str, int],
    required: dict[str, Any],
) -> None:
    for category, minimum in sorted(required.items()):
        count = observed.get(category, 0)
        if count < int(minimum):
            errors.append(
                f"{label} category {category!r} count {count} is below preregistered minimum {int(minimum)}"
            )
