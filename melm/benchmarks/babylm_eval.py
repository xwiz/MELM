"""Adapters for local BabyLM 2026 evaluation assets."""

from __future__ import annotations

import json
from pathlib import Path

from .minimal_pairs import MinimalPairCase
from .multiple_choice import MultipleChoiceCase


def profile_blimp_fast(data_dir: str | Path) -> dict[str, int]:
    """Return file/case counts for a BabyLM fast BLiMP directory."""

    root = Path(data_dir)
    files = sorted(root.glob("*.jsonl"))
    cases = 0
    for path in files:
        cases += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "files": len(files),
        "cases": cases,
    }


def profile_jsonl_directory(data_dir: str | Path) -> dict[str, int]:
    """Return file/case counts for a JSONL benchmark directory."""

    root = Path(data_dir)
    files = sorted(root.glob("*.jsonl"))
    cases = 0
    for path in files:
        cases += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "files": len(files),
        "cases": cases,
    }


def load_blimp_fast_cases(
    data_dir: str | Path,
    *,
    max_files: int | None = None,
    max_cases_per_file: int | None = None,
) -> list[MinimalPairCase]:
    """Load BabyLM 2026 fast BLiMP JSONL files as minimal pairs."""

    root = Path(data_dir)
    files = sorted(root.glob("*.jsonl"))
    if max_files is not None:
        files = files[:max_files]

    cases: list[MinimalPairCase] = []
    for path in files:
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            uid = str(item.get("UID") or path.stem)
            pair_id = str(item.get("pair_id", count))
            cases.append(
                MinimalPairCase(
                    case_id=f"{uid}:{pair_id}",
                    category=uid,
                    good=str(item["sentence_good"]),
                    bad=str(item["sentence_bad"]),
                )
            )
            count += 1
            if max_cases_per_file is not None and count >= max_cases_per_file:
                break
    return cases


def load_entity_tracking_fast_cases(
    data_dir: str | Path,
    *,
    max_files: int | None = None,
    max_cases_per_file: int | None = None,
) -> list[MultipleChoiceCase]:
    """Load BabyLM 2026 fast entity-tracking JSONL files."""

    root = Path(data_dir)
    files = sorted(root.glob("*.jsonl"))
    if max_files is not None:
        files = files[:max_files]

    cases: list[MultipleChoiceCase] = []
    for path in files:
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = str(item.get("sample_id", count))
            numops = item.get("numops", "unknown")
            cases.append(
                MultipleChoiceCase(
                    case_id=f"{path.stem}:{sample_id}",
                    category=f"{path.stem}_{numops}_ops",
                    prompt=str(item["input_prefix"]),
                    options=tuple(str(option) for option in item["options"]),
                    label_index=0,
                )
            )
            count += 1
            if max_cases_per_file is not None and count >= max_cases_per_file:
                break
    return cases
