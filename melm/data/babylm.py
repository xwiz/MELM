"""Local BabyLM-style corpus manifest adapter.

This adapter intentionally does not assume a particular download URL or hosted
dataset name. It scans a local corpus directory, preserves explicit split names
when present, and emits the same `CorpusManifest` used by tokenizer and training
scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .manifest import CorpusDocument, CorpusManifest


TRAIN_ALIASES = frozenset({"train", "training", "train_10m", "train_100m"})
VALIDATION_ALIASES = frozenset({"dev", "valid", "validation", "val"})
TEST_ALIASES = frozenset({"test", "eval", "evaluation"})
DEFAULT_SUFFIXES = (".txt", ".text", ".train", ".dev", ".valid", ".test")


@dataclass(frozen=True)
class BabyLMScanSummary:
    manifest: CorpusManifest
    explicit_splits: int
    fallback_splits: int


def scan_babylm_corpus(
    root: str | Path,
    *,
    name: str = "babylm_local",
    version: str = "local",
    track: str = "unknown",
    license_name: str = "unknown",
    suffixes: tuple[str, ...] = DEFAULT_SUFFIXES,
) -> BabyLMScanSummary:
    """Scan a local BabyLM-style corpus into a reproducible manifest."""

    root_path = Path(root)
    files = _iter_files(root_path, suffixes)
    pending: list[tuple[str, Path, bytes, str, str, str | None]] = []

    source = f"babylm:{track}"
    for file_path in files:
        data = file_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        relative = file_path.relative_to(root_path).as_posix() if root_path.is_dir() else file_path.name
        doc_id = hashlib.sha1(f"{source}:{relative}:{digest}".encode("utf-8")).hexdigest()[:16]
        pending.append((digest, file_path, data, relative, doc_id, _infer_split(Path(relative))))

    fallback_items = [
        (doc_id, digest)
        for digest, _file_path, _data, _relative, doc_id, split in pending
        if split is None
    ]
    fallback_splits = _stable_splits(fallback_items)

    documents: list[CorpusDocument] = []
    explicit_count = 0
    fallback_count = 0
    for digest, file_path, data, _relative, doc_id, explicit_split in sorted(pending, key=lambda item: item[0]):
        if explicit_split is None:
            split = fallback_splits.get(doc_id, "train")
            fallback_count += 1
        else:
            split = explicit_split
            explicit_count += 1
        documents.append(
            CorpusDocument(
                doc_id=doc_id,
                path=str(file_path),
                source=source,
                split=split,
                bytes=len(data),
                sha256=digest,
                license=license_name,
            )
        )

    return BabyLMScanSummary(
        manifest=CorpusManifest(
            name=name,
            version=version,
            documents=documents,
        ),
        explicit_splits=explicit_count,
        fallback_splits=fallback_count,
    )


def _infer_split(relative: Path) -> str | None:
    parts = [_normalize(part) for part in relative.parts]
    filename_tokens: list[str] = []
    for raw_part in relative.parts:
        normalized = _normalize(raw_part)
        filename_tokens.extend(token for token in normalized.replace(".", "_").split("_") if token)
    candidates = (
        parts
        + [_normalize(relative.stem), _normalize(relative.suffix.lstrip("."))]
        + filename_tokens
    )
    for candidate in candidates:
        if candidate in TRAIN_ALIASES or candidate.startswith("train_"):
            return "train"
        if candidate in VALIDATION_ALIASES or candidate.startswith(("dev_", "valid_", "validation_")):
            return "validation"
        if candidate in TEST_ALIASES or candidate.startswith(("test_", "eval_")):
            return "test"
    return None


def _normalize(value: str) -> str:
    return value.lower().strip().replace("-", "_")


def _iter_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in suffixes else []
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(
        file_path
        for file_path in root.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in suffixes
        and "__pycache__" not in file_path.parts
        and ".git" not in file_path.parts
    )


def _stable_splits(items: list[tuple[str, str]]) -> dict[str, str]:
    if not items:
        return {}
    ordered = sorted(items, key=lambda item: item[1])
    if len(ordered) == 1:
        return {ordered[0][0]: "train"}
    if len(ordered) == 2:
        return {ordered[0][0]: "train", ordered[1][0]: "validation"}

    validation_count = max(1, round(len(ordered) * 0.10))
    test_count = max(1, round(len(ordered) * 0.10))
    train_count = max(1, len(ordered) - validation_count - test_count)
    splits: dict[str, str] = {}
    for index, (doc_id, _digest) in enumerate(ordered):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        splits[doc_id] = split
    return splits
