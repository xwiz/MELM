"""Factored lexicon storage and the single SenseCandidate ingestion gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from time import perf_counter
from typing import Any

from melm.contracts import (
    ContractValidationError,
    load_contract_json,
    validate_reserved_lexemes,
    validate_router_lexicon_families,
    validate_sense_candidate,
)

from .assistant_os_store import AssistantOSStore


RUNTIME_PROVENANCE = frozenset({"user_taught", "cloud_lookup"})
STATUS_RANK = {"quarantined": 0, "dormant": 1, "active": 2}
ROUTER_LEXICON_FAMILIES_KEY = "lexicon_router_families"


@dataclass(frozen=True)
class LexiconIngestResult:
    ingestion_id: str
    candidate_hash: str
    lexeme_id: str
    sense_id: str
    semantic_class_id: str
    status: str
    created_lexeme: bool
    created_sense: bool
    duplicate_candidate: bool
    provenance_merged: bool
    relations_staged: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingestion_id": self.ingestion_id,
            "candidate_hash": self.candidate_hash,
            "lexeme_id": self.lexeme_id,
            "sense_id": self.sense_id,
            "semantic_class_id": self.semantic_class_id,
            "status": self.status,
            "created_lexeme": self.created_lexeme,
            "created_sense": self.created_sense,
            "duplicate_candidate": self.duplicate_candidate,
            "provenance_merged": self.provenance_merged,
            "relations_staged": self.relations_staged,
        }


@dataclass(frozen=True)
class TieredLexicalLookup:
    term: str
    tier: str
    senses: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "tier": self.tier,
            "senses": [dict(item) for item in self.senses],
        }


@dataclass(frozen=True)
class LexiconLookupBenchmark:
    queries: int
    warmup_queries: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    active_hits: int
    dormant_hits: int
    misses: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "melm.lexicon_lookup_benchmark.v1",
            "queries": self.queries,
            "warmup_queries": self.warmup_queries,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
            "active_hits": self.active_hits,
            "dormant_hits": self.dormant_hits,
            "misses": self.misses,
        }


def lexicon_ingest(
    store: AssistantOSStore,
    candidate: dict[str, Any],
    *,
    expected_provenance: str,
    recorded_at: str | None = None,
) -> LexiconIngestResult:
    timestamp = recorded_at or _timestamp()
    candidate_hash = _candidate_hash(candidate)

    try:
        validate_sense_candidate(candidate)
        normalized_lemma = _normalize_term(str(candidate["lemma"]))
        source = dict(candidate["source"])
        provenance = str(source["provenance"])
        if provenance != expected_provenance:
            raise ContractValidationError(
                "$.source.provenance: "
                f"candidate declares {provenance!r}, adapter is bound to {expected_provenance!r}"
            )
        reserved, policy_lexemes = _controlled_lexemes()
        actual_reserved = normalized_lemma in reserved
        actual_policy_overlap = normalized_lemma in policy_lexemes
        if actual_reserved and provenance != "seed_authored":
            raise ContractValidationError(
                f"$.lemma: {normalized_lemma!r} is reserved and cannot be acquired from {provenance}"
            )
        if actual_reserved != bool(candidate["safety"]["reserved_conflict"]):
            raise ContractValidationError(
                "$.safety.reserved_conflict: does not match the release-controlled reserved namespace"
            )
        if actual_policy_overlap != bool(candidate["safety"]["policy_term_overlap"]):
            raise ContractValidationError(
                "$.safety.policy_term_overlap: "
                "does not match the release-controlled policy namespace"
            )
        if actual_policy_overlap and provenance != "seed_authored":
            raise ContractValidationError(
                f"$.lemma: {normalized_lemma!r} overlaps policy vocabulary "
                f"and cannot be acquired from {provenance}"
            )
        for index, form in enumerate(candidate.get("forms", [])):
            normalized_form = _normalize_term(str(form["surface"]))
            if not normalized_form:
                raise ContractValidationError(
                    f"$.forms[{index}].surface: normalizes to an empty form"
                )
            if provenance != "seed_authored" and normalized_form in reserved:
                raise ContractValidationError(
                    f"$.forms[{index}].surface: {normalized_form!r} is reserved "
                    f"and cannot be acquired from {provenance}"
                )
            if provenance != "seed_authored" and normalized_form in policy_lexemes:
                raise ContractValidationError(
                    f"$.forms[{index}].surface: {normalized_form!r} overlaps "
                    f"policy vocabulary and cannot be acquired from {provenance}"
                )

        existing_ingestion = store.connection.execute(
            """
            SELECT ingestion_id, lexeme_id, sense_id, status
            FROM lexicon_ingestions
            WHERE candidate_hash=?
            """,
            (candidate_hash,),
        ).fetchone()
        if existing_ingestion is not None and str(existing_ingestion["status"]) == "applied":
            sense = _sense_row(store, str(existing_ingestion["sense_id"]))
            return LexiconIngestResult(
                ingestion_id=str(existing_ingestion["ingestion_id"]),
                candidate_hash=candidate_hash,
                lexeme_id=str(existing_ingestion["lexeme_id"]),
                sense_id=str(existing_ingestion["sense_id"]),
                semantic_class_id=str(sense["semantic_class_id"]),
                status=str(sense["status"]),
                created_lexeme=False,
                created_sense=False,
                duplicate_candidate=True,
                provenance_merged=False,
                relations_staged=0,
            )

        class_choice = _select_class(candidate["semantic_class_candidates"])
        semantic_class_id = str(class_choice["class_id"])
        genus_lemma = _normalize_term(str(candidate.get("genus_lemma", "")))
        genus_classes = _genus_classes(store, genus_lemma) if genus_lemma else set()
        if genus_lemma and not genus_classes:
            class_confidence = float(class_choice["confidence"])
            if provenance in RUNTIME_PROVENANCE or class_confidence < 0.8:
                raise ContractValidationError(
                    f"$.genus_lemma: unresolved genus {genus_lemma!r} is not eligible for ingestion"
                )
        if genus_classes and not any(
            _class_is_same_or_descendant(semantic_class_id, genus_class)
            for genus_class in genus_classes
        ):
            raise ContractValidationError(
                f"$.genus_lemma: class {semantic_class_id!r} is incompatible with "
                f"{genus_lemma!r} classes {sorted(genus_classes)!r}"
            )

        return _apply_candidate(
            store,
            candidate,
            candidate_hash=candidate_hash,
            normalized_lemma=normalized_lemma,
            semantic_class_id=semantic_class_id,
            genus_lemma=genus_lemma,
            actual_reserved=actual_reserved,
            recorded_at=timestamp,
        )
    except ContractValidationError as exc:
        _record_rejected_ingestion(
            store,
            candidate,
            candidate_hash,
            str(exc),
            recorded_at=timestamp,
        )
        raise


def lookup_lexical_senses(
    store: AssistantOSStore,
    term: str,
    *,
    statuses: tuple[str, ...] = ("active", "dormant"),
) -> list[dict[str, Any]]:
    normalized = _normalize_term(term)
    allowed = tuple(status for status in statuses if status in STATUS_RANK)
    if not normalized or not allowed:
        return []
    placeholders = ",".join("?" for _ in allowed)
    rows = store.connection.execute(
        f"""
        SELECT
            l.lexeme_id, l.lemma, l.language, l.pos, l.reserved,
            s.sense_id, s.semantic_class_id, s.concept_id,
            s.argument_template_id, s.definition, s.genus_lemma,
            s.confidence, s.status, s.use_count
        FROM lexemes AS l
        JOIN lexical_senses AS s ON s.lexeme_id=l.lexeme_id
        LEFT JOIN word_forms AS f ON f.lexeme_id=l.lexeme_id
        WHERE (l.normalized_lemma=? OR f.normalized_surface=?)
          AND s.status IN ({placeholders})
        GROUP BY s.sense_id
        ORDER BY s.confidence DESC, s.semantic_class_id, s.sense_id
        """,
        (normalized, normalized, *allowed),
    ).fetchall()
    return [
        {
            "lexeme_id": str(row["lexeme_id"]),
            "lemma": str(row["lemma"]),
            "language": str(row["language"]),
            "pos": str(row["pos"]),
            "reserved": bool(row["reserved"]),
            "sense_id": str(row["sense_id"]),
            "semantic_class_id": str(row["semantic_class_id"]),
            "concept_id": str(row["concept_id"]),
            "argument_template_id": str(row["argument_template_id"]),
            "definition": str(row["definition"]),
            "genus_lemma": str(row["genus_lemma"]),
            "confidence": float(row["confidence"]),
            "status": str(row["status"]),
            "use_count": int(row["use_count"]),
        }
        for row in rows
    ]


def lexical_classes_for_term(
    store: AssistantOSStore,
    term: str,
    *,
    statuses: tuple[str, ...] = ("active",),
) -> frozenset[str]:
    return frozenset(
        str(item["semantic_class_id"])
        for item in lookup_lexical_senses(store, term, statuses=statuses)
    )


def lookup_lexical_senses_tiered(
    store: AssistantOSStore,
    term: str,
) -> TieredLexicalLookup:
    active = lookup_lexical_senses(store, term, statuses=("active",))
    if active:
        return TieredLexicalLookup(term=term, tier="active", senses=tuple(active))
    dormant = lookup_lexical_senses(store, term, statuses=("dormant",))
    if dormant:
        return TieredLexicalLookup(term=term, tier="dormant", senses=tuple(dormant))
    return TieredLexicalLookup(term=term, tier="miss", senses=())


def benchmark_lexicon_lookup(
    store: AssistantOSStore,
    terms: tuple[str, ...],
    *,
    iterations: int = 1000,
    warmup_queries: int = 50,
) -> LexiconLookupBenchmark:
    normalized_terms = tuple(term for term in terms if _normalize_term(term))
    if not normalized_terms:
        raise ValueError("at least one non-empty lookup term is required")
    if iterations < 1 or warmup_queries < 0:
        raise ValueError("iterations must be positive and warmup_queries non-negative")
    for index in range(warmup_queries):
        lookup_lexical_senses_tiered(
            store,
            normalized_terms[index % len(normalized_terms)],
        )
    durations: list[float] = []
    tier_counts = {"active": 0, "dormant": 0, "miss": 0}
    for index in range(iterations):
        started = perf_counter()
        result = lookup_lexical_senses_tiered(
            store,
            normalized_terms[index % len(normalized_terms)],
        )
        durations.append((perf_counter() - started) * 1000.0)
        tier_counts[result.tier] += 1
    ordered = sorted(durations)
    return LexiconLookupBenchmark(
        queries=iterations,
        warmup_queries=warmup_queries,
        p50_ms=round(_percentile(ordered, 0.50), 4),
        p95_ms=round(_percentile(ordered, 0.95), 4),
        max_ms=round(max(ordered), 4),
        active_hits=tier_counts["active"],
        dormant_hits=tier_counts["dormant"],
        misses=tier_counts["miss"],
    )


def configure_lexicon_router_families(
    store: AssistantOSStore,
    families: tuple[str, ...],
) -> None:
    normalized = validate_lexicon_router_families(families)
    _validate_router_family_coverage(store, normalized)
    with store.connection:
        store.connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (ROUTER_LEXICON_FAMILIES_KEY, json.dumps(normalized, separators=(",", ":"))),
        )


def validate_lexicon_router_families(
    families: tuple[str, ...],
) -> tuple[str, ...]:
    definitions = _router_family_definitions()
    normalized = tuple(
        sorted({str(item).strip().lower() for item in families if str(item).strip()})
    )
    unknown = sorted(set(normalized) - set(definitions))
    if unknown:
        raise ContractValidationError(
            f"unknown lexicon router families: {', '.join(unknown)}"
        )
    return normalized


def load_lexicon_router_families(store: AssistantOSStore) -> frozenset[str]:
    row = store.connection.execute(
        "SELECT value FROM metadata WHERE key=?",
        (ROUTER_LEXICON_FAMILIES_KEY,),
    ).fetchone()
    if row is None:
        return frozenset()
    try:
        payload = json.loads(str(row["value"]))
    except json.JSONDecodeError as exc:
        raise ContractValidationError(
            f"metadata.{ROUTER_LEXICON_FAMILIES_KEY}: invalid JSON"
        ) from exc
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise ContractValidationError(
            f"metadata.{ROUTER_LEXICON_FAMILIES_KEY}: must be an array of strings"
        )
    try:
        return frozenset(validate_lexicon_router_families(tuple(payload)))
    except ContractValidationError as exc:
        raise ContractValidationError(
            f"metadata.{ROUTER_LEXICON_FAMILIES_KEY}: {exc}"
        ) from exc


def _router_family_definitions() -> dict[str, dict[str, tuple[str, ...]]]:
    payload = load_contract_json("router_lexicon_families.v1.json")
    validate_router_lexicon_families(payload)
    return {
        str(family): {
            "required_terms": tuple(str(item) for item in definition["required_terms"]),
            "allowed_classes": tuple(str(item) for item in definition["allowed_classes"]),
        }
        for family, definition in payload["families"].items()
    }


def _validate_router_family_coverage(
    store: AssistantOSStore,
    families: tuple[str, ...],
) -> None:
    definitions = _router_family_definitions()
    for family in families:
        definition = definitions[family]
        allowed_classes = set(definition["allowed_classes"])
        missing: list[str] = []
        mismatched: list[str] = []
        for term in definition["required_terms"]:
            classes = lexical_classes_for_term(store, term, statuses=("active",))
            if not classes:
                missing.append(term)
            elif not classes & allowed_classes:
                mismatched.append(term)
        if missing or mismatched:
            details = []
            if missing:
                details.append(f"missing active terms: {', '.join(missing)}")
            if mismatched:
                details.append(f"class-mismatched terms: {', '.join(mismatched)}")
            raise ContractValidationError(
                f"lexicon router family {family!r} is incomplete: {'; '.join(details)}"
            )


def _apply_candidate(
    store: AssistantOSStore,
    candidate: dict[str, Any],
    *,
    candidate_hash: str,
    normalized_lemma: str,
    semantic_class_id: str,
    genus_lemma: str,
    actual_reserved: bool,
    recorded_at: str,
) -> LexiconIngestResult:
    now = recorded_at
    language = str(candidate["language"]).strip().lower()
    pos = str(candidate["pos"])
    lexeme_id = _stable_id("lex", language, normalized_lemma, pos)
    sense_id = _stable_id("sense", lexeme_id, semantic_class_id)
    source = dict(candidate["source"])
    provenance = str(source["provenance"])
    source_ref = str(source["source_ref"])
    confidence = max(
        float(candidate["confidence_prior"]),
        float(_select_class(candidate["semantic_class_candidates"])["confidence"]),
    )
    requested_status = str(candidate["suggested_status"])
    definition = str(candidate["definition"]).strip()
    frequency_rank = candidate.get("frequency_rank")
    concept_id = f"concept:{semantic_class_id}:{normalized_lemma}"
    ingestion_id = _stable_id("ing", candidate_hash)

    with store.connection:
        existing_lexeme = store.connection.execute(
            "SELECT lexeme_id, frequency_rank FROM lexemes WHERE lexeme_id=?",
            (lexeme_id,),
        ).fetchone()
        created_lexeme = existing_lexeme is None
        if created_lexeme:
            store.connection.execute(
                """
                INSERT INTO lexemes(
                    lexeme_id, lemma, normalized_lemma, language, pos,
                    reserved, frequency_rank, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lexeme_id,
                    str(candidate["lemma"]).strip(),
                    normalized_lemma,
                    language,
                    pos,
                    int(actual_reserved),
                    frequency_rank,
                    now,
                    now,
                ),
            )
        else:
            current_rank = existing_lexeme["frequency_rank"]
            merged_rank = _minimum_optional_int(current_rank, frequency_rank)
            store.connection.execute(
                """
                UPDATE lexemes
                SET reserved=MAX(reserved, ?), frequency_rank=?, updated_at=?
                WHERE lexeme_id=?
                """,
                (int(actual_reserved), merged_rank, now, lexeme_id),
            )

        existing_sense = store.connection.execute(
            """
            SELECT status, confidence, definition, genus_lemma, argument_template_id
            FROM lexical_senses WHERE sense_id=?
            """,
            (sense_id,),
        ).fetchone()
        created_sense = existing_sense is None
        if created_sense:
            status = requested_status
            store.connection.execute(
                """
                INSERT INTO lexical_senses(
                    sense_id, lexeme_id, semantic_class_id, concept_id,
                    argument_template_id, definition, genus_lemma,
                    confidence, status, created_at, last_used_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sense_id,
                    lexeme_id,
                    semantic_class_id,
                    concept_id,
                    str(candidate.get("argument_template_id", "")),
                    definition,
                    genus_lemma,
                    confidence,
                    status,
                    now,
                    now,
                    now,
                ),
            )
        else:
            status = _more_active_status(str(existing_sense["status"]), requested_status)
            merged_confidence = max(float(existing_sense["confidence"]), confidence)
            merged_definition = str(existing_sense["definition"]) or definition
            merged_genus = str(existing_sense["genus_lemma"]) or genus_lemma
            merged_template = (
                str(existing_sense["argument_template_id"])
                or str(candidate.get("argument_template_id", ""))
            )
            store.connection.execute(
                """
                UPDATE lexical_senses
                SET confidence=?, status=?, definition=?, genus_lemma=?,
                    argument_template_id=?, updated_at=?
                WHERE sense_id=?
                """,
                (
                    merged_confidence,
                    status,
                    merged_definition,
                    merged_genus,
                    merged_template,
                    now,
                    sense_id,
                ),
            )

        forms = [
            {
                "surface": str(candidate["lemma"]),
                "morph_features": "Lemma=Yes",
                "provenance": provenance,
            },
            *list(candidate.get("forms", [])),
        ]
        for form in forms:
            normalized_surface = _normalize_term(str(form["surface"]))
            form_id = _stable_id(
                "form",
                lexeme_id,
                normalized_surface,
                str(form.get("morph_features", "")),
            )
            store.connection.execute(
                """
                INSERT INTO word_forms(
                    form_id, lexeme_id, surface, normalized_surface,
                    morph_features, provenance, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(lexeme_id, normalized_surface, morph_features)
                DO NOTHING
                """,
                (
                    form_id,
                    lexeme_id,
                    str(form["surface"]).strip(),
                    normalized_surface,
                    str(form.get("morph_features", "")),
                    str(form.get("provenance", provenance)),
                    now,
                ),
            )

        provenance_id = _stable_id("prov", sense_id, provenance, source_ref)
        existing_provenance = store.connection.execute(
            """
            SELECT 1 FROM lexical_provenance
            WHERE sense_id=? AND provenance=? AND source_ref=?
            """,
            (sense_id, provenance, source_ref),
        ).fetchone()
        store.connection.execute(
            """
            INSERT INTO lexical_provenance(
                provenance_id, sense_id, provenance, source_ref, license,
                retrieved_at, definition, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sense_id, provenance, source_ref) DO UPDATE SET
                license=excluded.license,
                retrieved_at=excluded.retrieved_at,
                definition=excluded.definition,
                confidence=MAX(lexical_provenance.confidence, excluded.confidence)
            """,
            (
                provenance_id,
                sense_id,
                provenance,
                source_ref,
                str(source["license"]),
                str(source.get("retrieved_at", "")),
                definition,
                confidence,
                now,
            ),
        )
        provenance_merged = not created_sense and existing_provenance is None

        relations_staged = 0
        for relation in candidate.get("relations", []):
            target_lemma = _normalize_term(str(relation["target_lemma"]))
            target_ref = str(relation.get("target_sense_ref", ""))
            relation_id = _stable_id(
                "rel",
                sense_id,
                str(relation["relation"]),
                target_lemma,
                target_ref,
            )
            before = store.connection.total_changes
            store.connection.execute(
                """
                INSERT INTO lexical_relation_candidates(
                    relation_id, sense_id, relation, target_lemma,
                    target_sense_ref, status, provenance, source_ref, created_at
                ) VALUES (?, ?, ?, ?, ?, 'quarantined', ?, ?, ?)
                ON CONFLICT(sense_id, relation, target_lemma, target_sense_ref)
                DO NOTHING
                """,
                (
                    relation_id,
                    sense_id,
                    str(relation["relation"]),
                    target_lemma,
                    target_ref,
                    provenance,
                    source_ref,
                    now,
                ),
            )
            relations_staged += int(store.connection.total_changes > before)

        store.connection.execute(
            """
            INSERT INTO lexicon_ingestions(
                ingestion_id, candidate_hash, batch_id, schema_id,
                provenance, source_ref, status, lexeme_id, sense_id,
                error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'applied', ?, ?, '', ?)
            ON CONFLICT(candidate_hash) DO UPDATE SET
                status='applied', lexeme_id=excluded.lexeme_id,
                sense_id=excluded.sense_id, error=''
            """,
            (
                ingestion_id,
                candidate_hash,
                str(candidate.get("batch_id", "")),
                str(candidate["schema_id"]),
                provenance,
                source_ref,
                lexeme_id,
                sense_id,
                now,
            ),
        )

    return LexiconIngestResult(
        ingestion_id=ingestion_id,
        candidate_hash=candidate_hash,
        lexeme_id=lexeme_id,
        sense_id=sense_id,
        semantic_class_id=semantic_class_id,
        status=status,
        created_lexeme=created_lexeme,
        created_sense=created_sense,
        duplicate_candidate=False,
        provenance_merged=provenance_merged,
        relations_staged=relations_staged,
    )


def _record_rejected_ingestion(
    store: AssistantOSStore,
    candidate: dict[str, Any],
    candidate_hash: str,
    error: str,
    *,
    recorded_at: str,
) -> None:
    source = candidate.get("source", {})
    source = source if isinstance(source, dict) else {}
    with store.connection:
        store.connection.execute(
            """
            INSERT INTO lexicon_ingestions(
                ingestion_id, candidate_hash, batch_id, schema_id,
                provenance, source_ref, status, lexeme_id, sense_id,
                error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'rejected', '', '', ?, ?)
            ON CONFLICT(candidate_hash) DO UPDATE SET
                status='rejected', error=excluded.error
            """,
            (
                _stable_id("ing", candidate_hash),
                candidate_hash,
                str(candidate.get("batch_id", "")),
                str(candidate.get("schema_id", "")),
                str(source.get("provenance", "")),
                str(source.get("source_ref", "")),
                error,
                recorded_at,
            ),
        )


def _controlled_lexemes() -> tuple[set[str], set[str]]:
    payload = load_contract_json("reserved_lexemes.v1.json")
    validate_reserved_lexemes(payload)
    return (
        {_normalize_term(str(item)) for item in payload["lexemes"]},
        {_normalize_term(str(item)) for item in payload["policy_lexemes"]},
    )


def _genus_classes(store: AssistantOSStore, genus_lemma: str) -> set[str]:
    rows = store.connection.execute(
        """
        SELECT DISTINCT s.semantic_class_id
        FROM lexemes AS l
        JOIN lexical_senses AS s ON s.lexeme_id=l.lexeme_id
        WHERE l.normalized_lemma=? AND s.status IN ('active', 'dormant')
        """,
        (genus_lemma,),
    ).fetchall()
    return {str(row["semantic_class_id"]) for row in rows}


def _class_is_same_or_descendant(class_id: str, ancestor_id: str) -> bool:
    payload = load_contract_json("semantic_classes.v1.json")
    parents = {
        str(item["class_id"]): (
            str(item["parent_id"]) if item["parent_id"] is not None else None
        )
        for item in payload["classes"]
    }
    current: str | None = class_id
    while current is not None:
        if current == ancestor_id:
            return True
        current = parents.get(current)
    return False


def _select_class(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        candidates,
        key=lambda item: (-float(item["confidence"]), str(item["class_id"])),
    )


def _sense_row(store: AssistantOSStore, sense_id: str):
    row = store.connection.execute(
        "SELECT semantic_class_id, status FROM lexical_senses WHERE sense_id=?",
        (sense_id,),
    ).fetchone()
    if row is None:
        raise ContractValidationError(f"applied ingestion references missing sense {sense_id!r}")
    return row


def _more_active_status(current: str, proposed: str) -> str:
    return current if STATUS_RANK[current] >= STATUS_RANK[proposed] else proposed


def _candidate_hash(candidate: dict[str, Any]) -> str:
    payload = json.dumps(candidate, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _normalize_term(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", value.strip().lower()))


def _minimum_optional_int(current: Any, proposed: Any) -> int | None:
    values = [int(item) for item in (current, proposed) if item is not None]
    return min(values) if values else None


def _percentile(ordered: list[float], fraction: float) -> float:
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _timestamp() -> str:
    # Stable to seconds and consistent with the existing assistant store.
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
