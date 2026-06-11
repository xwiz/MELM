"""Deterministic seed builder for validated SenseCandidate JSONL inputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from melm.contracts import ContractValidationError

from .assistant_lexicon import (
    configure_lexicon_router_families,
    lexicon_ingest,
    validate_lexicon_router_families,
)
from .assistant_os_store import AssistantOSStore


DEFAULT_BUILD_TIMESTAMP = "2000-01-01T00:00:00+00:00"


@dataclass(frozen=True)
class LexiconCandidateSource:
    path: Path
    provenance: str


@dataclass(frozen=True)
class _LoadedCandidate:
    candidate: dict[str, Any]
    provenance: str
    path: Path
    line_number: int


@dataclass(frozen=True)
class LexiconSeedBuildReport:
    passed: bool
    output_db: str
    output_db_sha256: str
    manifest_path: str
    source_files: tuple[dict[str, Any], ...]
    candidates_read: int
    candidates_applied: int
    candidates_rejected: int
    duplicate_candidates: int
    source_counts: dict[str, int]
    table_counts: dict[str, int]
    collisions: tuple[dict[str, Any], ...]
    rejections: tuple[dict[str, Any], ...]
    build_timestamp: str
    router_families: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "melm.lexicon_seed_build_report.v1",
            "passed": self.passed,
            "output_db": self.output_db,
            "output_db_sha256": self.output_db_sha256,
            "manifest_path": self.manifest_path,
            "source_files": list(self.source_files),
            "candidates_read": self.candidates_read,
            "candidates_applied": self.candidates_applied,
            "candidates_rejected": self.candidates_rejected,
            "duplicate_candidates": self.duplicate_candidates,
            "source_counts": self.source_counts,
            "table_counts": self.table_counts,
            "collisions": list(self.collisions),
            "rejections": list(self.rejections),
            "build_timestamp": self.build_timestamp,
            "router_families": list(self.router_families),
        }


def build_lexicon_seed(
    candidate_sources: Iterable[LexiconCandidateSource],
    *,
    output_db: Path,
    manifest_path: Path,
    reset: bool = False,
    build_timestamp: str = DEFAULT_BUILD_TIMESTAMP,
    router_families: tuple[str, ...] = (),
) -> LexiconSeedBuildReport:
    sources = tuple(candidate_sources)
    if not sources:
        raise ValueError("at least one candidate JSONL source is required")
    normalized_router_families = validate_lexicon_router_families(router_families)
    source_files = tuple(_source_file_record(source) for source in sources)
    candidates = _load_candidates(sources)
    candidates.sort(
        key=lambda item: (
            _canonical_candidate(item.candidate),
            str(item.path),
            item.line_number,
        )
    )

    if not reset and output_db.exists():
        raise FileExistsError(f"lexicon seed database already exists: {output_db}")
    output_db.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    build_db = output_db.with_name(f".{output_db.name}.building")
    _remove_sqlite_family(build_db)

    store = AssistantOSStore(build_db)
    source_counts: Counter[str] = Counter()
    applied = 0
    duplicates = 0
    rejections: list[dict[str, Any]] = []
    try:
        for loaded in candidates:
            provenance = loaded.provenance
            source_counts[provenance or "unknown"] += 1
        pending = list(candidates)
        while pending:
            deferred: list[_LoadedCandidate] = []
            progress = 0
            for loaded in pending:
                candidate = loaded.candidate
                try:
                    result = lexicon_ingest(
                        store,
                        candidate,
                        recorded_at=build_timestamp,
                        expected_provenance=loaded.provenance,
                    )
                except ContractValidationError as exc:
                    if "unresolved genus" in str(exc):
                        deferred.append(loaded)
                        continue
                    rejections.append(_rejection_record(loaded, exc))
                    continue
                applied += 1
                duplicates += int(result.duplicate_candidate)
                progress += 1
            if not deferred:
                break
            if progress == 0:
                for loaded in deferred:
                    try:
                        lexicon_ingest(
                            store,
                            loaded.candidate,
                            recorded_at=build_timestamp,
                            expected_provenance=loaded.provenance,
                        )
                    except ContractValidationError as exc:
                        rejections.append(_rejection_record(loaded, exc))
                break
            pending = deferred
        try:
            configure_lexicon_router_families(store, normalized_router_families)
        except ContractValidationError as exc:
            rejections.append(
                {
                    "candidate_hash": "",
                    "lemma": "",
                    "provenance": "release_configuration",
                    "path": "",
                    "line_number": 0,
                    "error": str(exc),
                }
            )
        table_counts = {
            table: store.count(table)
            for table in (
                "lexemes",
                "word_forms",
                "lexical_senses",
                "lexical_provenance",
                "lexical_relation_candidates",
                "lexicon_ingestions",
            )
        }
        collisions = _collision_report(store)
    finally:
        store.close()

    # Normalize free-list pages and force WAL contents into the database file.
    connection = sqlite3.connect(build_db)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("VACUUM")
    finally:
        connection.close()

    report = LexiconSeedBuildReport(
        passed=not rejections,
        output_db=str(output_db),
        output_db_sha256=_sha256_file(build_db) if not rejections else "",
        manifest_path=str(manifest_path),
        source_files=source_files,
        candidates_read=len(candidates),
        candidates_applied=applied,
        candidates_rejected=len(rejections),
        duplicate_candidates=duplicates,
        source_counts=dict(sorted(source_counts.items())),
        table_counts=table_counts,
        collisions=tuple(collisions),
        rejections=tuple(rejections),
        build_timestamp=build_timestamp,
        router_families=normalized_router_families,
    )
    if report.passed:
        _remove_sqlite_family(output_db)
        build_db.replace(output_db)
        manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.building")
        manifest_tmp.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest_tmp.replace(manifest_path)
    else:
        _remove_sqlite_family(build_db)
    return report


def _load_candidates(
    sources: tuple[LexiconCandidateSource, ...],
) -> list[_LoadedCandidate]:
    candidates: list[_LoadedCandidate] = []
    for source in sources:
        path = Path(source.path)
        if not path.is_file():
            raise FileNotFoundError(f"candidate JSONL not found: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: candidate must be a JSON object")
            candidates.append(
                _LoadedCandidate(
                    candidate=payload,
                    provenance=source.provenance,
                    path=path,
                    line_number=line_number,
                )
            )
    return candidates


def _collision_report(store: AssistantOSStore) -> list[dict[str, Any]]:
    rows = store.connection.execute(
        """
        SELECT l.lemma, l.pos, COUNT(*) AS sense_count,
               GROUP_CONCAT(s.semantic_class_id, ',') AS classes
        FROM lexemes AS l
        JOIN lexical_senses AS s ON s.lexeme_id=l.lexeme_id
        GROUP BY l.lexeme_id
        HAVING COUNT(*) > 1
        ORDER BY l.normalized_lemma, l.pos
        """
    ).fetchall()
    return [
        {
            "lemma": str(row["lemma"]),
            "pos": str(row["pos"]),
            "sense_count": int(row["sense_count"]),
            "semantic_class_ids": sorted(str(row["classes"]).split(",")),
        }
        for row in rows
    ]


def _source_file_record(source: LexiconCandidateSource) -> dict[str, Any]:
    path = Path(source.path)
    if not path.is_file():
        raise FileNotFoundError(f"candidate JSONL not found: {path}")
    return {
        "path": str(path),
        "bound_provenance": source.provenance,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _rejection_record(
    loaded: _LoadedCandidate,
    error: ContractValidationError,
) -> dict[str, Any]:
    return {
        "candidate_hash": _sha256_text(_canonical_candidate(loaded.candidate)),
        "lemma": str(loaded.candidate.get("lemma", "")),
        "provenance": loaded.provenance,
        "path": str(loaded.path),
        "line_number": loaded.line_number,
        "error": str(error),
    }


def _canonical_candidate(candidate: dict[str, Any]) -> str:
    return json.dumps(candidate, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_sqlite_family(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            candidate.unlink()
