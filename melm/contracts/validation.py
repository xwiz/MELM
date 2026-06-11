"""Dependency-free validation for versioned MELM contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTRACT_ROOT = Path(__file__).resolve().parent


class ContractValidationError(ValueError):
    """Raised when a contract artifact or payload violates its contract."""


def load_contract_json(name: str) -> dict[str, Any]:
    path = (CONTRACT_ROOT / name).resolve()
    if path.parent != CONTRACT_ROOT or not path.is_file():
        raise ContractValidationError(f"unknown contract artifact: {name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"invalid contract artifact {name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractValidationError(f"contract artifact {name} must be an object")
    return payload


def _fail(path: str, message: str) -> None:
    raise ContractValidationError(f"{path}: {message}")


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    if "const" in schema and value != schema["const"]:
        _fail(path, f"must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        _fail(path, f"must be one of {schema['enum']!r}")

    expected = schema.get("type")
    if expected is not None:
        expected_types = expected if isinstance(expected, list) else [expected]
        type_checks = {
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
            "string": lambda item: isinstance(item, str),
            "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "null": lambda item: item is None,
        }
        if not any(type_checks[item](value) for item in expected_types):
            _fail(path, f"must have type {'|'.join(expected_types)}")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                _fail(path, f"missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                _fail(path, f"unknown properties: {', '.join(unknown)}")
        for key, item in value.items():
            if key in properties:
                _validate_schema(item, properties[key], f"{path}.{key}")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < int(min_items):
            _fail(path, f"must contain at least {min_items} item(s)")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, f"{path}[{index}]")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if min_length is not None and len(value) < int(min_length):
            _fail(path, f"must contain at least {min_length} character(s)")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            _fail(path, f"must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            _fail(path, f"must be <= {schema['maximum']}")


@dataclass(frozen=True)
class ContractRegistry:
    schema_id: str
    contracts: dict[str, dict[str, Any]]

    @classmethod
    def load(cls) -> "ContractRegistry":
        payload = load_contract_json("registry.v1.json")
        validate_contract_registry(payload)
        return cls(
            schema_id=str(payload["schema_id"]),
            contracts={str(item["schema_id"]): dict(item) for item in payload["contracts"]},
        )

    def require(self, schema_id: str) -> dict[str, Any]:
        try:
            return dict(self.contracts[schema_id])
        except KeyError as exc:
            raise ContractValidationError(f"unregistered schema_id: {schema_id}") from exc

    def load_schema(self, schema_id: str) -> dict[str, Any]:
        entry = self.require(schema_id)
        path = str(entry.get("path", ""))
        if not path:
            raise ContractValidationError(f"contract {schema_id} has no schema path")
        return load_contract_json(path)


def validate_contract_registry(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.contract_registry.v1":
        _fail("$.schema_id", "must equal 'melm.contract_registry.v1'")
    contracts = payload.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        _fail("$.contracts", "must be a non-empty array")
    seen: set[str] = set()
    for index, entry in enumerate(contracts):
        path = f"$.contracts[{index}]"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        for key in ("schema_id", "owner", "failure_behavior", "safety_critical"):
            if key not in entry:
                _fail(path, f"missing required property {key!r}")
        schema_id = str(entry["schema_id"])
        if schema_id in seen:
            _fail(path, f"duplicate schema_id {schema_id!r}")
        seen.add(schema_id)
        artifact_path = str(entry.get("path", ""))
        if artifact_path:
            load_contract_json(artifact_path)


def _semantic_class_ids() -> set[str]:
    payload = load_contract_json("semantic_classes.v1.json")
    validate_semantic_class_registry(payload)
    return {str(item["class_id"]) for item in payload["classes"]}


def validate_semantic_class_registry(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.semantic_classes.v1":
        _fail("$.schema_id", "must equal 'melm.semantic_classes.v1'")
    classes = payload.get("classes")
    if not isinstance(classes, list) or not classes:
        _fail("$.classes", "must be a non-empty array")
    by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(classes):
        path = f"$.classes[{index}]"
        if not isinstance(item, dict):
            _fail(path, "must be an object")
        class_id = str(item.get("class_id", ""))
        if not class_id:
            _fail(path, "class_id is required")
        if class_id in by_id:
            _fail(path, f"duplicate class_id {class_id!r}")
        flags = item.get("policy_flags", [])
        if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
            _fail(path, "policy_flags must be an array of strings")
        by_id[class_id] = item
    for class_id, item in by_id.items():
        parent_id = item.get("parent_id")
        if parent_id is not None and parent_id not in by_id:
            _fail(f"$.classes[{class_id}].parent_id", f"unknown parent {parent_id!r}")
    for class_id in by_id:
        visited: set[str] = set()
        current: str | None = class_id
        while current is not None:
            if current in visited:
                _fail("$.classes", f"inheritance cycle at {current!r}")
            visited.add(current)
            parent = by_id[current].get("parent_id")
            current = str(parent) if parent is not None else None


def validate_class_maps() -> None:
    class_ids = _semantic_class_ids()
    for name in ("wn_supersense_map.v1.json", "verbnet_map.v1.json"):
        payload = load_contract_json(name)
        mappings = payload.get("mappings")
        if not isinstance(mappings, dict) or not mappings:
            _fail(f"{name}.mappings", "must be a non-empty object")
        unknown = sorted({str(item) for item in mappings.values()} - class_ids)
        if unknown:
            _fail(f"{name}.mappings", f"unknown semantic classes: {', '.join(unknown)}")


def validate_reserved_lexemes(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.reserved_lexemes.v1":
        _fail("$.schema_id", "must equal 'melm.reserved_lexemes.v1'")
    for key in ("lexemes", "policy_lexemes"):
        lexemes = payload.get(key)
        if not isinstance(lexemes, list) or not lexemes:
            _fail(f"$.{key}", "must be a non-empty array")
        normalized = [str(item).strip().lower() for item in lexemes]
        if any(not item for item in normalized):
            _fail(f"$.{key}", "entries must be non-empty strings")
        if normalized != sorted(set(normalized)):
            _fail(f"$.{key}", "entries must be sorted and unique")


def validate_router_lexicon_families(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.router_lexicon_families.v1":
        _fail("$.schema_id", "must equal 'melm.router_lexicon_families.v1'")
    families = payload.get("families")
    if not isinstance(families, dict) or not families:
        _fail("$.families", "must be a non-empty object")
    class_ids = _semantic_class_ids()
    reserved = load_contract_json("reserved_lexemes.v1.json")
    validate_reserved_lexemes(reserved)
    reserved_terms = set(reserved["lexemes"])
    for family, item in families.items():
        path = f"$.families.{family}"
        if not isinstance(item, dict):
            _fail(path, "must be an object")
        if set(item) != {"required_terms", "allowed_classes"}:
            _fail(path, "must contain only required_terms and allowed_classes")
        terms = item["required_terms"]
        classes = item["allowed_classes"]
        if not isinstance(terms, list) or not terms:
            _fail(f"{path}.required_terms", "must be a non-empty array")
        if not isinstance(classes, list) or not classes:
            _fail(f"{path}.allowed_classes", "must be a non-empty array")
        if terms != sorted(set(terms)):
            _fail(f"{path}.required_terms", "must be sorted and unique")
        if classes != sorted(set(classes)):
            _fail(f"{path}.allowed_classes", "must be sorted and unique")
        unreserved = sorted(set(terms) - reserved_terms)
        if unreserved:
            _fail(
                f"{path}.required_terms",
                f"router anchors must be reserved: {', '.join(unreserved)}",
            )
        unknown = sorted(set(classes) - class_ids)
        if unknown:
            _fail(
                f"{path}.allowed_classes",
                f"unknown semantic classes: {', '.join(unknown)}",
            )


def validate_sense_candidate(candidate: dict[str, Any]) -> None:
    registry = ContractRegistry.load()
    schema = registry.load_schema("melm.sense_candidate.v1")
    _validate_schema(candidate, schema)

    class_ids = _semantic_class_ids()
    unknown = sorted(
        {
            str(item["class_id"])
            for item in candidate["semantic_class_candidates"]
            if str(item["class_id"]) not in class_ids
        }
    )
    if unknown:
        _fail("$.semantic_class_candidates", f"unknown semantic classes: {', '.join(unknown)}")

    provenance = candidate["source"]["provenance"]
    status = candidate["suggested_status"]
    safety = candidate["safety"]
    if (
        safety["reserved_conflict"]
        and provenance != "seed_authored"
        and status != "quarantined"
    ):
        _fail("$.suggested_status", "reserved conflicts must remain quarantined")
    if (
        safety["policy_term_overlap"]
        and provenance != "seed_authored"
        and status != "quarantined"
    ):
        _fail("$.suggested_status", "policy-term overlaps must remain quarantined")
    source_policy = {
        "seed_authored": ({"active", "dormant", "quarantined"}, 0.95, {"seed_authored"}),
        "wordnet": ({"active", "dormant"}, 0.85, {"supersense_map", "genus_walk"}),
        "wiktextract": ({"dormant", "quarantined"}, 0.70, {"genus_walk"}),
        "verbnet": ({"active", "dormant"}, 0.85, {"verbnet_map"}),
        "user_taught": ({"quarantined"}, 0.60, {"genus_walk"}),
        "offline_dictionary": ({"quarantined"}, 0.70, {"genus_walk"}),
        "cloud_lookup": ({"quarantined"}, 0.50, {"llm_assigned"}),
        "inferred": ({"quarantined"}, 0.50, {"genus_walk"}),
    }
    allowed_statuses, maximum_prior, allowed_methods = source_policy[provenance]
    if status not in allowed_statuses:
        _fail(
            "$.suggested_status",
            f"{provenance} candidates must use one of {sorted(allowed_statuses)!r}",
        )
    if float(candidate["confidence_prior"]) > maximum_prior:
        _fail(
            "$.confidence_prior",
            f"{provenance} prior must be <= {maximum_prior:.2f}",
        )
    invalid_methods = sorted(
        {
            str(item["method"])
            for item in candidate["semantic_class_candidates"]
            if str(item["method"]) not in allowed_methods
        }
    )
    if invalid_methods:
        _fail(
            "$.semantic_class_candidates",
            f"{provenance} cannot use mapping methods: {', '.join(invalid_methods)}",
        )
