"""Corpus manifest helpers for reproducible data experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    path: str
    source: str
    split: str
    bytes: int
    sha256: str
    license: str = "unknown"


@dataclass(frozen=True)
class CorpusManifest:
    name: str
    version: str
    documents: list[CorpusDocument]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "CorpusManifest":
        raw = json.loads(text)
        return cls(
            name=raw["name"],
            version=raw["version"],
            documents=[CorpusDocument(**item) for item in raw["documents"]],
        )


def scan_corpus(
    root: str | Path,
    *,
    name: str,
    version: str = "0.1.0",
    source: str = "local",
    license_name: str = "unknown",
    suffixes: tuple[str, ...] = (".txt", ".md"),
    exclude_dirs: tuple[str, ...] = ("reports", ".git", "__pycache__", ".pytest_cache"),
) -> CorpusManifest:
    """Scan local files into a deterministic manifest.

    Splits are assigned by stable document hash:
    train < 80%, validation < 90%, test otherwise.
    """

    root_path = Path(root)
    files = _iter_files(root_path, suffixes, exclude_dirs)
    pending: list[tuple[str, Path, bytes, str, str]] = []

    for file_path in files:
        data = file_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        relative = file_path.relative_to(root_path).as_posix() if root_path.is_dir() else file_path.name
        doc_id = hashlib.sha1(f"{source}:{relative}:{digest}".encode("utf-8")).hexdigest()[:16]
        pending.append((digest, file_path, data, relative, doc_id))

    pending.sort(key=lambda item: item[0])
    split_by_doc_id = _stable_splits([(item[4], item[0]) for item in pending])

    documents: list[CorpusDocument] = []
    for digest, file_path, data, _relative, doc_id in pending:
        documents.append(
            CorpusDocument(
                doc_id=doc_id,
                path=str(file_path),
                source=source,
                split=split_by_doc_id[doc_id],
                bytes=len(data),
                sha256=digest,
                license=license_name,
            )
        )

    return CorpusManifest(name=name, version=version, documents=documents)


def load_manifest(path: str | Path) -> CorpusManifest:
    return CorpusManifest.from_json(Path(path).read_text(encoding="utf-8"))


def save_manifest(manifest: CorpusManifest, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.to_json(), encoding="utf-8")


def texts_for_split(manifest: CorpusManifest, split: str) -> list[str]:
    texts: list[str] = []
    for document in manifest.documents:
        if document.split == split:
            texts.append(Path(document.path).read_text(encoding="utf-8"))
    return texts


def _iter_files(root: Path, suffixes: tuple[str, ...], exclude_dirs: tuple[str, ...]) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in suffixes else []
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(
        file_path
        for file_path in root.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in suffixes
        and not any(part in exclude_dirs for part in file_path.relative_to(root).parts[:-1])
    )


def _stable_splits(items: list[tuple[str, str]]) -> dict[str, str]:
    """Assign stable splits while preserving validation/test for small corpora."""

    total = len(items)
    if total == 0:
        return {}
    if total == 1:
        return {items[0][0]: "train"}
    if total == 2:
        return {items[0][0]: "train", items[1][0]: "validation"}

    validation_count = max(1, round(total * 0.10))
    test_count = max(1, round(total * 0.10))
    train_count = max(1, total - validation_count - test_count)

    splits: dict[str, str] = {}
    for index, (doc_id, _digest) in enumerate(items):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        splits[doc_id] = split
    return splits
