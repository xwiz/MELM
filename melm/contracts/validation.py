"""Dependency-free validation for versioned MELM contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTRACT_ROOT = Path(__file__).resolve().parent

_KNOWN_INTENTS = {
    "assistant_identity", "assistant_status", "story", "weather",
    "common_sense_safety", "media_playback", "health_advice",
    "personal_memory", "autobiographical_memory", "meal_suggestion",
    "social_contact", "social_greeting", "assistant_behavior",
    "personal_goal_advice", "open_domain", "mail",
}


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


def _is_compatible_version(current: str, predecessor: str) -> bool:
    """Return True if *predecessor* version is compatible with *current*.

    Compatibility rule: predecessor major version must be <= current major
    version, and if majors are equal, predecessor minor must be <= current minor.
    This allows v2.x to consume v1.x contracts, but not vice versa.
    """
    try:
        c_parts = [int(p) for p in current.split(".")]
        p_parts = [int(p) for p in predecessor.split(".")]
    except ValueError:
        return False
    # Pad to at least 2 parts
    while len(c_parts) < 2:
        c_parts.append(0)
    while len(p_parts) < 2:
        p_parts.append(0)
    if p_parts[0] > c_parts[0]:
        return False
    if p_parts[0] == c_parts[0] and p_parts[1] > c_parts[1]:
        return False
    return True


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

    def check_compatibility(self) -> list[str]:
        import hashlib
        errors: list[str] = []
        for schema_id, entry in self.contracts.items():
            # Verify loaded artifact hash matches stored schema_hash.
            artifact_path = str(entry.get("path", ""))
            stored_hash = str(entry.get("schema_hash", ""))
            if artifact_path and stored_hash:
                try:
                    raw = (CONTRACT_ROOT / artifact_path).read_bytes()
                    full_content = raw.replace(b"\r\n", b"\n")
                    actual_hash = hashlib.sha256(full_content).hexdigest()[:16]
                    if actual_hash != stored_hash:
                        errors.append(
                            f"{schema_id}: schema_hash mismatch "
                            f"(expected {stored_hash!r}, got {actual_hash!r})"
                        )
                except Exception as exc:
                    errors.append(f"{schema_id}: cannot verify schema_hash: {exc}")
            # Verify predecessor existence and version compatibility.
            current_version = str(entry.get("version", "0.0.0"))
            for pred in entry.get("compatible_predecessors", []):
                if pred not in self.contracts:
                    errors.append(
                        f"{schema_id}: compatible_predecessor {pred!r} not found in registry"
                    )
                    continue
                pred_version = str(self.contracts[pred].get("version", "0.0.0"))
                if not _is_compatible_version(current_version, pred_version):
                    errors.append(
                        f"{schema_id}: version {current_version!r} is not compatible "
                        f"with predecessor {pred!r} at {pred_version!r}"
                    )
        return errors


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
        for key in ("schema_id", "owner", "failure_behavior", "safety_critical", "compatible_predecessors", "version", "schema_hash"):
            if key not in entry:
                _fail(path, f"missing required property {key!r}")
        version = str(entry.get("version", ""))
        if not version:
            _fail(f"{path}.version", "must be a non-empty string")
        schema_hash = str(entry.get("schema_hash", ""))
        if not schema_hash or len(schema_hash) != 16 or not all(c in "0123456789abcdef" for c in schema_hash):
            _fail(f"{path}.schema_hash", "must be a 16-character lowercase hex string")
        predecessors = entry["compatible_predecessors"]
        if not isinstance(predecessors, list) or any(not isinstance(p, str) for p in predecessors):
            _fail(f"{path}.compatible_predecessors", "must be an array of strings")
        schema_id = str(entry["schema_id"])
        if schema_id in seen:
            _fail(path, f"duplicate schema_id {schema_id!r}")
        seen.add(schema_id)
        artifact_path = str(entry.get("path", ""))
        if artifact_path:
            load_contract_json(artifact_path)


def load_semantic_class_ids() -> set[str]:
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
    class_ids = load_semantic_class_ids()
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
    class_ids = load_semantic_class_ids()
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

    class_ids = load_semantic_class_ids()
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
        "auto_research": ({"active", "dormant"}, 0.75, {"genus_walk", "llm_assigned"}),
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


def validate_frame_templates(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.frame_templates.v1":
        _fail("$.schema_id", "must equal 'melm.frame_templates.v1'")
    templates = payload.get("templates")
    if not isinstance(templates, dict) or not templates:
        _fail("$.templates", "must be a non-empty object")
    class_ids = load_semantic_class_ids()
    for fid, item in templates.items():
        path = f"$.templates.{fid}"
        if not isinstance(item, dict):
            _fail(path, "must be an object")
        for key in ("frame_id", "intent", "family", "activation", "threshold", "priority"):
            if key not in item:
                _fail(path, f"missing required property {key!r}")
        if str(item.get("frame_id")) != fid:
            _fail(f"{path}.frame_id", f"must equal template key {fid!r}")
        if str(item.get("intent")) not in _KNOWN_INTENTS:
            _fail(f"{path}.intent", f"unknown intent {item.get('intent')!r}")
        activation = item.get("activation", {})
        if not isinstance(activation, dict):
            _fail(f"{path}.activation", "must be an object")
        for cls_list_key in ("required_classes", "required_all_classes", "optional_classes", "exclude_classes"):
            raw = activation.get(cls_list_key, [])
            if not isinstance(raw, list) or any(not isinstance(c, str) for c in raw):
                _fail(f"{path}.activation.{cls_list_key}", "must be an array of strings")
            if cls_list_key == "required_classes" and not raw:
                _fail(f"{path}.activation.required_classes", "must be a non-empty array")
            unknown = sorted(set(raw) - class_ids)
            if unknown:
                _fail(
                    f"{path}.activation.{cls_list_key}",
                    f"unknown semantic classes: {', '.join(unknown)}",
                )
        sb = item.get("slot_bindings", [])
        if not isinstance(sb, list) or any(not isinstance(s, str) for s in sb):
            _fail(f"{path}.slot_bindings", "must be an array of strings")
        threshold = item.get("threshold")
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            _fail(f"{path}.threshold", "must be in [0, 1]")

        context_gates = item.get("context_gates")
        if context_gates is not None:
            if not isinstance(context_gates, dict):
                _fail(f"{path}.context_gates", "must be an object")
            for gk, gv in context_gates.items():
                if not isinstance(gk, str) or not isinstance(gv, bool):
                    _fail(f"{path}.context_gates.{gk}", "must have string keys and bool values")
        context_score = item.get("context_score")
        if context_score is not None:
            if not isinstance(context_score, dict):
                _fail(f"{path}.context_score", "must be an object")
            for sk, sv in context_score.items():
                if not isinstance(sk, str) or not isinstance(sv, (int, float)) or not 0 <= sv <= 1:
                    _fail(f"{path}.context_score.{sk}", "must be a float in [0, 1]")


def validate_food_tags(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.food_tags.v1":
        _fail("$.schema_id", "must equal 'melm.food_tags.v1'")
    tags = payload.get("tags")
    if not isinstance(tags, dict) or not tags:
        _fail("$.tags", "must be a non-empty object")
    for marker, marker_tags in tags.items():
        path = f"$.tags.{marker}"
        if not isinstance(marker, str) or not marker:
            _fail(path, "marker must be a non-empty string")
        if not isinstance(marker_tags, list) or not marker_tags:
            _fail(f"{path}", "must be a non-empty array of strings")
        for tag in marker_tags:
            if not isinstance(tag, str) or not tag:
                _fail(f"{path}", "each tag must be a non-empty string")


def load_food_tags() -> dict[str, set[str]]:
    payload = load_contract_json("food_tags.v1.json")
    validate_food_tags(payload)
    return {marker: set(tags) for marker, tags in payload["tags"].items()}


def validate_health_disclaimers(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.health_disclaimers.v1":
        _fail("$.schema_id", "must equal 'melm.health_disclaimers.v1'")
    urgent_terms = payload.get("urgent_terms")
    if not isinstance(urgent_terms, list) or not urgent_terms:
        _fail("$.urgent_terms", "must be a non-empty array of strings")
    for t in urgent_terms:
        if not isinstance(t, str) or not t:
            _fail("$.urgent_terms", "each term must be a non-empty string")
    urgent_pairs = payload.get("urgent_pairs")
    if not isinstance(urgent_pairs, list):
        _fail("$.urgent_pairs", "must be an array of [token, token] pairs")
    for pair in urgent_pairs:
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(t, str) and t for t in pair):
            _fail(f"$.urgent_pairs", "each pair must be [str, str]")
    responses = payload.get("responses")
    if not isinstance(responses, dict) or not responses:
        _fail("$.responses", "must be a non-empty object")
    for key, entry in responses.items():
        if not isinstance(key, str) or not key:
            _fail(f"$.responses.{key}", "key must be a non-empty string")
        if not isinstance(entry, dict):
            _fail(f"$.responses.{key}", "must be an object with 'text' and 'triggers'")
        if "text" not in entry or not isinstance(entry["text"], str) or not entry["text"]:
            _fail(f"$.responses.{key}.text", "must be a non-empty string")
        if "triggers" not in entry or not isinstance(entry["triggers"], list):
            _fail(f"$.responses.{key}.triggers", "must be an array of strings")
    if "fallback" not in responses:
        _fail("$.responses", "must include a 'fallback' entry")


def load_health_disclaimers() -> dict[str, Any]:
    payload = load_contract_json("health_disclaimers.v1.json")
    validate_health_disclaimers(payload)
    return dict(payload["responses"])


def validate_safety_policies(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.safety_policies.v1":
        _fail("$.schema_id", "must equal 'melm.safety_policies.v1'")
    policies = payload.get("policies")
    if not isinstance(policies, dict) or not policies:
        _fail("$.policies", "must be a non-empty object")
    for pid, policy in policies.items():
        path = f"$.policies.{pid}"
        if not isinstance(policy, dict):
            _fail(path, "must be an object")
        for key in ("template", "destinations", "triggers"):
            if key not in policy:
                _fail(path, f"missing required property {key!r}")
        template = policy["template"]
        if not isinstance(template, str) or not template:
            _fail(f"{path}.template", "must be a non-empty string")
        triggers = policy["triggers"]
        if not isinstance(triggers, list):
            _fail(f"{path}.triggers", "must be an array of strings")
        for t in triggers:
            if not isinstance(t, str) or not t:
                _fail(f"{path}.triggers", "each trigger must be a non-empty string")
        destinations = policy["destinations"]
        if not isinstance(destinations, dict) or not destinations:
            _fail(f"{path}.destinations", "must be a non-empty object")
        if "default" not in destinations:
            _fail(f"{path}.destinations", "must include a 'default' entry")
        for dkey, dentry in destinations.items():
            dpath = f"{path}.destinations.{dkey}"
            if not isinstance(dkey, str) or not dkey:
                _fail(dpath, "key must be a non-empty string")
            if not isinstance(dentry, dict):
                _fail(dpath, "must be an object with 'triggers' and 'phrase'")
            if "phrase" not in dentry or not isinstance(dentry["phrase"], str) or not dentry["phrase"]:
                _fail(f"{dpath}.phrase", "must be a non-empty string")
            if "triggers" not in dentry or not isinstance(dentry["triggers"], list):
                _fail(f"{dpath}.triggers", "must be an array of strings")


def load_safety_policies() -> dict[str, Any]:
    payload = load_contract_json("safety_policies.v1.json")
    validate_safety_policies(payload)
    return dict(payload["policies"])


def validate_story_components(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.story_components.v1":
        _fail("$.schema_id", "must equal 'melm.story_components.v1'")
    images = payload.get("images")
    if not isinstance(images, dict) or not images:
        _fail("$.images", "must be a non-empty object")
    for sub_key in ("title_keywords", "full_text_keywords", "default"):
        if sub_key not in images:
            _fail(f"$.images", f"must include a '{sub_key}' entry")
        val = images[sub_key]
        if sub_key == "default":
            if not isinstance(val, str) or not val:
                _fail("$.images.default", "must be a non-empty string")
        else:
            if not isinstance(val, dict) or not val:
                _fail(f"$.images.{sub_key}", "must be a non-empty object")
            for k, v in val.items():
                if not isinstance(v, str) or not v:
                    _fail(f"$.images.{sub_key}.{k}", "must be a non-empty string")
    for section in ("challenges", "lessons"):
        data = payload.get(section)
        if not isinstance(data, dict) or not data:
            _fail(f"$.{section}", "must be a non-empty object")
        if "default" not in data:
            _fail(f"$.{section}", "must include a 'default' entry")
        for key, val in data.items():
            if not isinstance(val, str) or not val:
                _fail(f"$.{section}.{key}", "must be a non-empty string")
    templates = payload.get("answer_templates")
    if templates is not None:
        if not isinstance(templates, list) or not templates:
            _fail("$.answer_templates", "if present must be a non-empty list")
        for i, t in enumerate(templates):
            if not isinstance(t, str) or not t:
                _fail(f"$.answer_templates[{i}]", "must be a non-empty string")


def load_story_components() -> dict[str, dict[str, str] | list[str]]:
    payload = load_contract_json("story_components.v1.json")
    validate_story_components(payload)
    result: dict[str, Any] = {
        "images": dict(payload["images"]),
        "challenges": dict(payload["challenges"]),
        "lessons": dict(payload["lessons"]),
    }
    if "answer_templates" in payload:
        result["answer_templates"] = list(payload["answer_templates"])
    return result


def validate_weather_concepts(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.weather_concepts.v1":
        _fail("$.schema_id", "must equal 'melm.weather_concepts.v1'")
    weather_terms = payload.get("weather_terms")
    if not isinstance(weather_terms, list) or not weather_terms:
        _fail("$.weather_terms", "must be a non-empty array of strings")
    for term in weather_terms:
        if not isinstance(term, str) or not term:
            _fail("$.weather_terms", "each term must be a non-empty string")


def load_weather_concepts() -> set[str]:
    payload = load_contract_json("weather_concepts.v1.json")
    validate_weather_concepts(payload)
    return set(payload["weather_terms"])


def validate_meal_scopes(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.meal_scopes.v1":
        _fail("$.schema_id", "must equal 'melm.meal_scopes.v1'")
    scopes = payload.get("scopes")
    if not isinstance(scopes, dict) or not scopes:
        _fail("$.scopes", "must be a non-empty object")
    for token, scope in scopes.items():
        if not isinstance(token, str) or not token:
            _fail(f"$.scopes.{token}", "token must be a non-empty string")
        if not isinstance(scope, str) or not scope:
            _fail(f"$.scopes.{token}", "scope must be a non-empty string")
    if "default" not in payload or not isinstance(payload["default"], str):
        _fail("$.default", "must be a non-empty string")


def load_meal_scopes() -> tuple[tuple[str, str], ...]:
    payload = load_contract_json("meal_scopes.v1.json")
    validate_meal_scopes(payload)
    return tuple(sorted(payload["scopes"].items()))


def validate_assistant_identity(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.assistant_identity.v1":
        _fail("$.schema_id", "must equal 'melm.assistant_identity.v1'")
    templates = payload.get("identity_templates")
    if not isinstance(templates, dict) or not templates:
        _fail("$.identity_templates", "must be a non-empty object")
    for key, template in templates.items():
        if not isinstance(template, str) or not template:
            _fail(f"$.identity_templates.{key}", "must be a non-empty string")
    for required in ("introduction", "status_unavailable", "status_next_steps", "status_default"):
        if required not in templates:
            _fail("$.identity_templates", f"must include a '{required}' entry")


def load_assistant_identity() -> dict[str, str]:
    payload = load_contract_json("assistant_identity.v1.json")
    validate_assistant_identity(payload)
    result = dict(payload["identity_templates"])
    for key in ("first_available_date", "technical_name"):
        if key in payload:
            result[key] = str(payload[key])
    return result


def validate_answer_templates(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.answer_templates.v1":
        _fail("$.schema_id", "must equal 'melm.answer_templates.v1'")
    intents = payload.get("intents")
    if not isinstance(intents, dict) or not intents:
        _fail("$.intents", "must be a non-empty object")
    for intent, entry in intents.items():
        path = f"$.intents.{intent}"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        if "template" in entry:
            if not isinstance(entry["template"], str) or not entry["template"]:
                _fail(f"{path}.template", "must be a non-empty string")
        if "templates" in entry:
            if not isinstance(entry["templates"], dict) or not entry["templates"]:
                _fail(f"{path}.templates", "must be a non-empty object")
            for key, val in entry["templates"].items():
                if not isinstance(val, str) or not val:
                    _fail(f"{path}.templates.{key}", "must be a non-empty string")
        if "requires_evidence" in entry:
            req = entry["requires_evidence"]
            if not isinstance(req, list) or not req:
                _fail(f"{path}.requires_evidence", "must be a non-empty array of strings")
            for item in req:
                if not isinstance(item, str) or not item:
                    _fail(f"{path}.requires_evidence", "each entry must be a non-empty string")
    evidence_targets = payload.get("evidence_count_targets")
    if evidence_targets is not None:
        if not isinstance(evidence_targets, dict):
            _fail("$.evidence_count_targets", "must be an object")
        for intent, target in evidence_targets.items():
            if not isinstance(target, int) or target < 1:
                _fail(f"$.evidence_count_targets.{intent}", "must be a positive integer")
    specificity = payload.get("answer_specificity_phrases")
    if specificity is not None:
        if not isinstance(specificity, dict):
            _fail("$.answer_specificity_phrases", "must be an object")
        for intent, phrase in specificity.items():
            path = f"$.answer_specificity_phrases.{intent}"
            if isinstance(phrase, str):
                if not phrase:
                    _fail(path, "must be a non-empty string")
            elif isinstance(phrase, list):
                if not phrase or not all(isinstance(p, str) and p for p in phrase):
                    _fail(path, "must be a non-empty array of non-empty strings")
            else:
                _fail(path, "must be a string or array of strings")
    bonuses = payload.get("answer_specificity_bonuses")
    if bonuses is not None:
        if not isinstance(bonuses, list):
            _fail("$.answer_specificity_bonuses", "must be an array")
        for i, entry in enumerate(bonuses):
            path = f"$.answer_specificity_bonuses[{i}]"
            if not isinstance(entry, dict):
                _fail(path, "must be an object")
                continue
            if "intent" not in entry or not isinstance(entry["intent"], str) or not entry["intent"]:
                _fail(f"{path}.intent", "must be a non-empty string")
            if "triggers" not in entry or not isinstance(entry["triggers"], list) or not entry["triggers"]:
                _fail(f"{path}.triggers", "must be a non-empty array")
            for j, group in enumerate(entry["triggers"]):
                if not isinstance(group, list) or not group or not all(isinstance(p, str) and p for p in group):
                    _fail(f"{path}.triggers[{j}]", "must be a non-empty array of non-empty strings")
            if "bonus" not in entry or not isinstance(entry["bonus"], (int, float)):
                _fail(f"{path}.bonus", "must be a number")
    reasoning_templates = payload.get("reasoning_templates")
    if reasoning_templates is not None:
        if not isinstance(reasoning_templates, dict) or not reasoning_templates:
            _fail("$.reasoning_templates", "must be a non-empty object")
        for task, templates in reasoning_templates.items():
            tpath = f"$.reasoning_templates.{task}"
            if not isinstance(templates, dict) or not templates:
                _fail(tpath, "must be a non-empty object")
            for key, val in templates.items():
                if not isinstance(val, str) or not val:
                    _fail(f"{tpath}.{key}", "must be a non-empty string")


def load_answer_templates() -> dict[str, Any]:
    payload = load_contract_json("answer_templates.v1.json")
    validate_answer_templates(payload)
    return dict(payload["intents"])


def load_reasoning_templates() -> dict[str, Any]:
    payload = load_contract_json("answer_templates.v1.json")
    validate_answer_templates(payload)
    return dict(payload.get("reasoning_templates", {}))


def validate_open_domain_templates(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.open_domain_templates.v1":
        _fail("$.schema_id", "must equal 'melm.open_domain_templates.v1'")
    intents = payload.get("intents")
    if not isinstance(intents, dict) or not intents:
        _fail("$.intents", "must be a non-empty object")
    for intent, entry in intents.items():
        path = f"$.intents.{intent}"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        templates = entry.get("templates")
        if not isinstance(templates, dict) or not templates:
            _fail(f"{path}.templates", "must be a non-empty object")
        for key, val in templates.items():
            if not isinstance(val, str) or not val:
                _fail(f"{path}.templates.{key}", "must be a non-empty string")
        if "requires_evidence" in entry:
            req = entry["requires_evidence"]
            if not isinstance(req, list) or not req:
                _fail(f"{path}.requires_evidence", "must be a non-empty array of strings")
            for item in req:
                if not isinstance(item, str) or not item:
                    _fail(f"{path}.requires_evidence", "each entry must be a non-empty string")
    speech_act_templates = payload.get("speech_act_templates")
    if not isinstance(speech_act_templates, dict) or not speech_act_templates:
        _fail("$.speech_act_templates", "must be a non-empty object")
    for group_name, entry in speech_act_templates.items():
        path = f"$.speech_act_templates.{group_name}"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        for variant in ("handoff", "learned", "unknown"):
            val = entry.get(variant)
            if not isinstance(val, str) or not val:
                _fail(f"{path}.{variant}", "must be a non-empty string")


def load_open_domain_templates() -> dict[str, Any]:
    payload = load_contract_json("open_domain_templates.v1.json")
    validate_open_domain_templates(payload)
    return dict(payload)


def validate_memory_insights(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.memory_insights.v1":
        _fail("$.schema_id", "must equal 'melm.memory_insights.v1'")
    rules = payload.get("rules")
    if not isinstance(rules, list) or not rules:
        _fail("$.rules", "must be a non-empty array")
    for index, rule in enumerate(rules):
        path = f"$.rules[{index}]"
        if not isinstance(rule, dict):
            _fail(path, "must be an object")
        for key in ("category", "text"):
            if key not in rule:
                _fail(path, f"missing required property {key!r}")
        if rule["category"] not in {"transitions", "open_loops", "action_state", "boundary_controls"}:
            _fail(f"{path}.category", f"unknown category {rule['category']!r}")
        if not isinstance(rule["text"], str) or not rule["text"]:
            _fail(f"{path}.text", "must be a non-empty string")
        for optional in ("intent", "reason", "route"):
            if optional in rule and not isinstance(rule[optional], str):
                _fail(f"{path}.{optional}", "must be a string")
    if "consented_stored_text" not in payload or not isinstance(payload["consented_stored_text"], str) or not payload["consented_stored_text"]:
        _fail("$.consented_stored_text", "must be a non-empty string")


def load_memory_insights() -> dict[str, Any]:
    payload = load_contract_json("memory_insights.v1.json")
    validate_memory_insights(payload)
    return dict(payload)


def validate_router_semantic_aliases(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.router_semantic_aliases.v1":
        _fail("$.schema_id", "must equal 'melm.router_semantic_aliases.v1'")
    for section in ("object_role_tokens", "secondary_hint_groups"):
        aliases = payload.get(section)
        if not isinstance(aliases, dict) or not aliases:
            _fail(f"$.{section}", "must be a non-empty object")
        for intent, tokens in aliases.items():
            path = f"$.{section}.{intent}"
            if not isinstance(intent, str) or not intent:
                _fail(f"$.{section}", "intent keys must be non-empty strings")
            if not isinstance(tokens, list) or not tokens:
                _fail(path, "must be a non-empty array of strings")
            for token in tokens:
                if not isinstance(token, str) or not token:
                    _fail(path, "each token must be a non-empty string")


def load_router_semantic_aliases() -> dict[str, Any]:
    payload = load_contract_json("router_semantic_aliases.v1.json")
    validate_router_semantic_aliases(payload)
    return dict(payload)


def validate_frame_minimal_pairs(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.frame_minimal_pairs.v1":
        _fail("$.schema_id", "must equal 'melm.frame_minimal_pairs.v1'")
    if not isinstance(payload.get("description"), str):
        _fail("$.description", "must be a string")
    lexicon = payload.get("lexicon")
    if not isinstance(lexicon, dict) or not lexicon:
        _fail("$.lexicon", "must be a non-empty object")
    for token, classes in lexicon.items():
        if not isinstance(token, str) or not token:
            _fail(f"$.lexicon.{token!r}", "token must be a non-empty string")
        if not isinstance(classes, list) or not classes:
            _fail(f"$.lexicon.{token!r}", "must be a non-empty array of strings")
        for cls in classes:
            if not isinstance(cls, str) or not cls:
                _fail(f"$.lexicon.{token!r}", "each class must be a non-empty string")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        _fail("$.pairs", "must be a non-empty array")
    for i, pair in enumerate(pairs):
        path = f"$.pairs[{i}]"
        if not isinstance(pair, dict):
            _fail(path, "must be an object")
        for key in ("id", "utterance", "tokens", "is_question_like", "is_request_like", "expected_top_frame", "expected_top3", "near_miss", "discriminator"):
            if key not in pair:
                _fail(path, f"missing required property {key!r}")
        if not isinstance(pair["id"], str) or not pair["id"]:
            _fail(f"{path}.id", "must be a non-empty string")
        if not isinstance(pair["utterance"], str):
            _fail(f"{path}.utterance", "must be a string")
        tokens = pair["tokens"]
        if not isinstance(tokens, list) or not tokens:
            _fail(f"{path}.tokens", "must be a non-empty array of strings")
        for t in tokens:
            if not isinstance(t, str):
                _fail(f"{path}.tokens", "each token must be a string")
        if not isinstance(pair["is_question_like"], bool):
            _fail(f"{path}.is_question_like", "must be a boolean")
        if not isinstance(pair["is_request_like"], bool):
            _fail(f"{path}.is_request_like", "must be a boolean")
        etf = pair["expected_top_frame"]
        if etf is not None and not isinstance(etf, str):
            _fail(f"{path}.expected_top_frame", "must be a string or null")
        et3 = pair["expected_top3"]
        if not isinstance(et3, list):
            _fail(f"{path}.expected_top3", "must be an array")
        for fid in et3:
            if not isinstance(fid, str):
                _fail(f"{path}.expected_top3", "each frame ID must be a string")
        if not isinstance(pair["near_miss"], str):
            _fail(f"{path}.near_miss", "must be a string")
        if not isinstance(pair["discriminator"], str):
            _fail(f"{path}.discriminator", "must be a string")
        if "token_roles" in pair:
            tr = pair["token_roles"]
            if not isinstance(tr, list):
                _fail(f"{path}.token_roles", "must be an array")
            for j, role in enumerate(tr):
                rp = f"{path}.token_roles[{j}]"
                if not isinstance(role, dict):
                    _fail(rp, "must be an object")
                for rk in ("index", "token", "lemma", "role", "meaning"):
                    if rk not in role:
                        _fail(rp, f"missing required key {rk!r}")
        if "precision_target" in pair:
            if not isinstance(pair["precision_target"], bool):
                _fail(f"{path}.precision_target", "must be a boolean")


def load_frame_minimal_pairs() -> dict[str, Any]:
    payload = load_contract_json("frame_minimal_pairs.v1.json")
    validate_frame_minimal_pairs(payload)
    return payload


def validate_capability_manifest(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.capability_manifest.v1":
        _fail("$.schema_id", "must equal 'melm.capability_manifest.v1'")
    families = payload.get("families")
    if not isinstance(families, dict) or not families:
        _fail("$.families", "must be a non-empty object")
    for family, item in families.items():
        path = f"$.families.{family}"
        if not isinstance(item, dict):
            _fail(path, "must be an object")
        for key in ("installed", "handler"):
            if key not in item:
                _fail(path, f"missing required property {key!r}")
        if family not in _KNOWN_INTENTS:
            _fail(path, f"unknown family {family!r}")
        if not isinstance(item["installed"], bool):
            _fail(f"{path}.installed", "must be a boolean")
        if not isinstance(item["handler"], str) or not item["handler"]:
            _fail(f"{path}.handler", "must be a non-empty string")


def validate_uol_parse(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "melm.weighted_functional_grammar.v1":
        _fail("$.schema", "must equal 'melm.weighted_functional_grammar.v1'")
    for key in ("speech_act", "subject", "action", "object", "target", "complement_action", "indirect_object", "pattern"):
        if not isinstance(payload.get(key), str):
            _fail(f"$.{key}", "must be a string")
    if not isinstance(payload.get("parse_score"), (int, float)):
        _fail("$.parse_score", "must be a number")
    if not isinstance(payload.get("syntactic_coverage"), (int, float)):
        _fail("$.syntactic_coverage", "must be a number")
    for key in ("modifiers", "relations", "token_roles", "candidates"):
        if not isinstance(payload.get(key), (dict, list)):
            _fail(f"$.{key}", "must be an object or array")
    if not isinstance(payload.get("semantic_unknown_tokens"), list):
        _fail("$.semantic_unknown_tokens", "must be an array")


def validate_route_decision(payload: dict[str, Any]) -> None:
    if not isinstance(payload.get("utterance"), str):
        _fail("$.utterance", "must be a string")
    if payload.get("intent") not in {
        "assistant_identity", "assistant_status", "story", "weather",
        "common_sense_safety", "media_playback", "health_advice",
        "personal_memory", "autobiographical_memory", "meal_suggestion",
        "social_contact", "social_greeting", "assistant_behavior",
        "personal_goal_advice", "open_domain", "unknown"
    }:
        _fail("$.intent", "must be a valid AssistantIntent")
    if payload.get("route") not in {
        "local_answer", "cached_tool", "device_action",
        "external_fetch", "cloud_handoff", "clarify", "reject"
    }:
        _fail("$.route", "must be a valid AssistantRoute")
    if not isinstance(payload.get("answer"), str):
        _fail("$.answer", "must be a string")
    if not isinstance(payload.get("cloud_needed"), bool):
        _fail("$.cloud_needed", "must be a boolean")
    if not isinstance(payload.get("confidence"), (int, float)):
        _fail("$.confidence", "must be a number")


def validate_evidence_packet(payload: dict[str, Any]) -> None:
    if not isinstance(payload.get("packet_id"), str) or not payload["packet_id"]:
        _fail("$.packet_id", "must be a non-empty string")
    items = payload.get("items")
    if not isinstance(items, list):
        _fail("$.items", "must be an array")
    for idx, item in enumerate(items):
        ipath = f"$.items[{idx}]"
        if not isinstance(item, dict):
            _fail(ipath, "must be an object")
        for key in ("key", "kind", "value", "source", "license"):
            if not isinstance(item.get(key), str):
                _fail(f"{ipath}.{key}", "must be a string")
        if not isinstance(item.get("local_only"), bool):
            _fail(f"{ipath}.local_only", "must be a boolean")
    if not isinstance(payload.get("admitted_count"), int):
        _fail("$.admitted_count", "must be an integer")
    if not isinstance(payload.get("blocked_keys"), list):
        _fail("$.blocked_keys", "must be an array")
    if not isinstance(payload.get("boundary"), str):
        _fail("$.boundary", "must be a string")


def validate_answer_plan(payload: dict[str, Any]) -> None:
    if not isinstance(payload.get("plan_id"), str) or not payload["plan_id"]:
        _fail("$.plan_id", "must be a non-empty string")
    if not isinstance(payload.get("route"), str):
        _fail("$.route", "must be a string")
    if payload.get("mode") not in {"narrative", "factual", "procedural", "reflective"}:
        _fail("$.mode", "must be one of narrative/factual/procedural/reflective")
    for key in ("requires", "forbids"):
        if not isinstance(payload.get(key), list):
            _fail(f"$.{key}", "must be an array")
    if not isinstance(payload.get("evidence_packet_id"), str):
        _fail("$.evidence_packet_id", "must be a string")


def validate_verification_result(payload: dict[str, Any]) -> None:
    for key in ("passed", "schema_valid", "packet_bound", "answer_nonempty"):
        if not isinstance(payload.get(key), bool):
            _fail(f"$.{key}", "must be a boolean")
    if not isinstance(payload.get("failure_codes"), list):
        _fail("$.failure_codes", "must be an array")
    if not isinstance(payload.get("constraint_retention"), (int, float)):
        _fail("$.constraint_retention", "must be a number")


def validate_model_manifest(payload: dict[str, Any]) -> None:
    if not isinstance(payload.get("model_id"), str) or not payload["model_id"]:
        _fail("$.model_id", "must be a non-empty string")
    if payload.get("backend") not in {"template", "llguidance", "llamacpp", "bitnet", "remote"}:
        _fail("$.backend", "must be a valid backend")
    if not isinstance(payload.get("parameters_b"), (int, float)):
        _fail("$.parameters_b", "must be a number")
    if not isinstance(payload.get("max_tokens"), int):
        _fail("$.max_tokens", "must be an integer")
    if not isinstance(payload.get("context_window"), int):
        _fail("$.context_window", "must be an integer")


def validate_pi_benchmark(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.pi_benchmark.v1":
        _fail("$.schema_id", "must equal 'melm.pi_benchmark.v1'")
    for key in ("measurements", "recorded_at", "go_no_go"):
        if key not in payload:
            _fail(f"$.{key}", "is required")
    go_no_go = payload["go_no_go"]
    if not isinstance(go_no_go, dict):
        _fail("$.go_no_go", "must be an object")
    for key in ("template_fallback_ready", "model_loaded", "pi_target_met"):
        if key not in go_no_go:
            _fail(f"$.go_no_go.{key}", "is required")
        if not isinstance(go_no_go[key], bool) and go_no_go[key] is not None:
            _fail(f"$.go_no_go.{key}", "must be a boolean or null")


def load_pi_benchmark() -> dict[str, Any]:
    payload = load_contract_json("pi_benchmark.v1.json")
    validate_pi_benchmark(payload)
    return dict(payload)


def validate_uol_normative_cases(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.uol_normative_cases.v1":
        _fail("$.schema_id", "must equal 'melm.uol_normative_cases.v1'")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        _fail("$.cases", "must be a non-empty array")
    for index, case in enumerate(cases):
        path = f"$.cases[{index}]"
        if not isinstance(case, dict):
            _fail(path, "must be an object")
        for key in ("utterance", "speech_act", "subject", "action", "object", "target"):
            if key not in case:
                _fail(path, f"missing required property {key!r}")
            if not isinstance(case[key], str):
                _fail(f"{path}.{key}", "must be a string")
        utterance = case["utterance"]
        if not utterance:
            _fail(f"{path}.utterance", "must be non-empty")
        if case["speech_act"] not in {"request", "wh_question", "yes_no_question", "statement", "greeting"}:
            _fail(f"{path}.speech_act", "must be a valid speech act")
    if len(cases) < 60:
        _fail("$.cases", f"must contain at least 60 cases (got {len(cases)})")


def load_uol_normative_cases() -> list[dict[str, str]]:
    payload = load_contract_json("uol_normative_cases.v1.json")
    validate_uol_normative_cases(payload)
    return list(payload["cases"])


def validate_igbo_lexicon_seed(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.igbo_lexicon_seed.v1":
        _fail("$.schema_id", "must equal 'melm.igbo_lexicon_seed.v1'")
    language = payload.get("language")
    if not isinstance(language, str) or not language:
        _fail("$.language", "must be a non-empty string")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        _fail("$.entries", "must be a non-empty array")
    class_ids = load_semantic_class_ids()
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        path = f"$.entries[{index}]"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        lemma = entry.get("lemma")
        if not isinstance(lemma, str) or not lemma:
            _fail(f"{path}.lemma", "must be a non-empty string")
        if lemma in seen:
            _fail(f"{path}.lemma", f"duplicate lemma {lemma!r}")
        seen.add(lemma)
        semantic_class = entry.get("semantic_class")
        if not isinstance(semantic_class, str) or not semantic_class:
            _fail(f"{path}.semantic_class", "must be a non-empty string")
        if semantic_class not in class_ids:
            _fail(f"{path}.semantic_class", f"unknown class {semantic_class!r}")
        pos = entry.get("pos")
        if not isinstance(pos, str) or not pos:
            _fail(f"{path}.pos", "must be a non-empty string")
        english_gloss = entry.get("english_gloss")
        if not isinstance(english_gloss, str) or not english_gloss:
            _fail(f"{path}.english_gloss", "must be a non-empty string")


def load_igbo_lexicon_seed() -> dict[str, Any]:
    payload = load_contract_json("igbo_lexicon_seed.v1.json")
    validate_igbo_lexicon_seed(payload)
    return dict(payload)


def validate_yoruba_greetings(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.yoruba_greetings.v1":
        _fail("$.schema_id", "must equal 'melm.yoruba_greetings.v1'")
    language = payload.get("language")
    if not isinstance(language, str) or not language:
        _fail("$.language", "must be a non-empty string")
    greetings = payload.get("greetings")
    if not isinstance(greetings, dict) or not greetings:
        _fail("$.greetings", "must be a non-empty object")
    for key, value in greetings.items():
        if not isinstance(key, str) or not key:
            _fail(f"$.greetings.{key}", "key must be a non-empty string")
        if not isinstance(value, str) or not value:
            _fail(f"$.greetings.{key}", "value must be a non-empty string")


def load_yoruba_greetings() -> dict[str, Any]:
    payload = load_contract_json("yoruba_greetings.v1.json")
    validate_yoruba_greetings(payload)
    return dict(payload)


def validate_swahili_greetings(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.swahili_greetings.v1":
        _fail("$.schema_id", "must equal 'melm.swahili_greetings.v1'")
    language = payload.get("language")
    if not isinstance(language, str) or not language:
        _fail("$.language", "must be a non-empty string")
    greetings = payload.get("greetings")
    if not isinstance(greetings, dict) or not greetings:
        _fail("$.greetings", "must be a non-empty object")
    for key, value in greetings.items():
        if not isinstance(key, str) or not key:
            _fail(f"$.greetings.{key}", "key must be a non-empty string")
        if not isinstance(value, str) or not value:
            _fail(f"$.greetings.{key}", "value must be a non-empty string")


def load_swahili_greetings() -> dict[str, Any]:
    payload = load_contract_json("swahili_greetings.v1.json")
    validate_swahili_greetings(payload)
    return dict(payload)


def validate_igbo_greetings(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.igbo_greetings.v1":
        _fail("$.schema_id", "must equal 'melm.igbo_greetings.v1'")
    language = payload.get("language")
    if not isinstance(language, str) or not language:
        _fail("$.language", "must be a non-empty string")
    greetings = payload.get("greetings")
    if not isinstance(greetings, dict) or not greetings:
        _fail("$.greetings", "must be a non-empty object")
    for key, value in greetings.items():
        if not isinstance(key, str) or not key:
            _fail(f"$.greetings.{key}", "key must be a non-empty string")
        if not isinstance(value, str) or not value:
            _fail(f"$.greetings.{key}", "value must be a non-empty string")


def load_igbo_greetings() -> dict[str, Any]:
    payload = load_contract_json("igbo_greetings.v1.json")
    validate_igbo_greetings(payload)
    return dict(payload)


def validate_prompt_seeds(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.prompt_seeds.v1":
        _fail("$.schema_id", "must equal 'melm.prompt_seeds.v1'")
    seeds = payload.get("seeds")
    if not isinstance(seeds, dict) or not seeds:
        _fail("$.seeds", "must be a non-empty object")
    for intent, seed in seeds.items():
        if not isinstance(intent, str) or not intent:
            _fail(f"$.seeds.{intent}", "key must be a non-empty string")
        if not isinstance(seed, dict):
            _fail(f"$.seeds.{intent}", "value must be an object")
        system = seed.get("system")
        if not isinstance(system, str) or not system:
            _fail(f"$.seeds.{intent}.system", "must be a non-empty string")
        user_prefix = seed.get("user_prefix")
        if user_prefix is not None and (not isinstance(user_prefix, str) or not user_prefix):
            _fail(f"$.seeds.{intent}.user_prefix", "must be a non-empty string when present")
    preferred = payload.get("model_preferred_intents")
    if preferred is not None and not isinstance(preferred, list):
        _fail("$.model_preferred_intents", "must be a list of intent strings when present")


def load_prompt_seeds() -> dict[str, Any]:
    payload = load_contract_json("prompt_seeds.v1.json")
    validate_prompt_seeds(payload)
    return dict(payload)


_VALID_ROLES = {
    "greeting", "wh_word", "modal", "auxiliary", "negation",
    "determiner", "preposition", "conjunction", "frequency",
    "equivalence", "politeness", "discourse_particle", "pronoun", "temporal",
}

_VALID_SUBROLES = {
    "manner", "theme", "time", "location", "selection", "agent", "reason",
    "possibility", "necessity", "obligation", "future", "past", "present",
    "copula", "do_support", "perfect",
    "indefinite", "definite", "demonstrative",
    "topic", "purpose", "source", "destination", "qualifier", "path", "accompaniment",
    "coordination", "contrast", "alternative",
    "agent", "patient", "possessor", "reflexive", "human_collective", "human_indefinite",
    "agent_or_patient",
}

_VALID_ANSWER_TYPES = {
    "entity",
    "person",
    "location",
    "time",
    "reason",
    "manner",
    "selection",
}

_VALID_ATOM_KINDS = {
    "state", "relation", "event", "change", "perception", "mental", "implication",
}


def validate_function_words(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.function_words.v1":
        _fail("$.schema_id", "must equal 'melm.function_words.v1'")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        _fail("$.entries", "must be a non-empty array")
    seen = set()
    for index, entry in enumerate(entries):
        path = f"$.entries[{index}]"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        for key in ("lemma", "language", "role"):
            if key not in entry:
                _fail(path, f"missing required property {key!r}")
        lemma = str(entry.get("lemma", "")).strip().lower()
        if not lemma:
            _fail(f"{path}.lemma", "must be a non-empty string")
        language = str(entry.get("language", "")).strip().lower()
        if not language:
            _fail(f"{path}.language", "must be a non-empty string")
        role = str(entry.get("role", "")).strip().lower()
        if role not in _VALID_ROLES:
            _fail(f"{path}.role", f"must be one of {sorted(_VALID_ROLES)!r}")
        subrole = entry.get("subrole")
        if subrole is not None and str(subrole).strip().lower() not in _VALID_SUBROLES:
            _fail(f"{path}.subrole", f"must be one of {sorted(_VALID_SUBROLES)!r}")
        answer_type = entry.get("answer_type")
        if answer_type is not None and str(answer_type).strip().lower() not in _VALID_ANSWER_TYPES:
            _fail(
                f"{path}.answer_type",
                f"must be one of {sorted(_VALID_ANSWER_TYPES)!r}",
            )
        key = (language, lemma)
        if key in seen:
            _fail(path, f"duplicate entry for language={language!r} lemma={lemma!r}")
        seen.add(key)


def load_function_words() -> dict[str, Any]:
    payload = load_contract_json("function_words.v1.json")
    validate_function_words(payload)
    return dict(payload)


def validate_predicate_inventory(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.predicate_inventory.v1":
        _fail("$.schema_id", "must equal 'melm.predicate_inventory.v1'")
    class_ids = load_semantic_class_ids()
    predicates = payload.get("predicates")
    if not isinstance(predicates, list) or not predicates:
        _fail("$.predicates", "must be a non-empty array")
    seen = set()
    for index, entry in enumerate(predicates):
        path = f"$.predicates[{index}]"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        for key in ("lemma", "predicate_id", "kind", "semantic_class"):
            if key not in entry:
                _fail(path, f"missing required property {key!r}")
        lemma = str(entry.get("lemma", "")).strip().lower()
        if not lemma:
            _fail(f"{path}.lemma", "must be a non-empty string")
        predicate_id = str(entry.get("predicate_id", "")).strip()
        if not predicate_id:
            _fail(f"{path}.predicate_id", "must be a non-empty string")
        kind = str(entry.get("kind", "")).strip().lower()
        if kind not in _VALID_ATOM_KINDS:
            _fail(f"{path}.kind", f"must be one of {sorted(_VALID_ATOM_KINDS)!r}")
        sem_cls = str(entry.get("semantic_class", "")).strip()
        if sem_cls not in class_ids:
            _fail(f"{path}.semantic_class", f"unknown class {sem_cls!r}")
        language = str(entry.get("language", "en")).strip().lower()
        key = (language, lemma)
        if key in seen:
            _fail(path, f"duplicate predicate for language={language!r} lemma={lemma!r}")
        seen.add(key)
    domains = payload.get("content_domains")
    if not isinstance(domains, list):
        _fail("$.content_domains", "must be an array")
    for index, entry in enumerate(domains):
        path = f"$.content_domains[{index}]"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        for key in ("lemma", "domain", "type"):
            if key not in entry:
                _fail(path, f"missing required property {key!r}")
        if not isinstance(entry["lemma"], str) or not entry["lemma"]:
            _fail(f"{path}.lemma", "must be a non-empty string")
        if not isinstance(entry["domain"], str) or not entry["domain"]:
            _fail(f"{path}.domain", "must be a non-empty string")
        if not isinstance(entry["type"], str) or not entry["type"]:
            _fail(f"{path}.type", "must be a non-empty string")


def load_predicate_inventory() -> dict[str, Any]:
    payload = load_contract_json("predicate_inventory.v1.json")
    validate_predicate_inventory(payload)
    return dict(payload)


# ---------------------------------------------------------------------------
# Mood / Affect contracts (M10+)
# ---------------------------------------------------------------------------

def validate_mood_states(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.mood_states.v1":
        _fail("$.schema_id", "must equal 'melm.mood_states.v1'")
    moods = payload.get("moods")
    if not isinstance(moods, list) or not moods:
        _fail("$.moods", "must be a non-empty array")
    seen = set()
    for idx, entry in enumerate(moods):
        path = f"$.moods[{idx}]"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        for key in ("mood_id", "valence", "arousal", "response_mode", "engagement_floor"):
            if key not in entry:
                _fail(path, f"missing required property {key!r}")
        mid = str(entry.get("mood_id", ""))
        if mid in seen:
            _fail(path, f"duplicate mood_id {mid!r}")
        seen.add(mid)
        if not isinstance(entry.get("valence"), (int, float)):
            _fail(f"{path}.valence", "must be a number")
        if not isinstance(entry.get("arousal"), (int, float)):
            _fail(f"{path}.arousal", "must be a number")
        if not isinstance(entry.get("engagement_floor"), (int, float)):
            _fail(f"{path}.engagement_floor", "must be a number")
    response_modes = payload.get("response_modes")
    if not isinstance(response_modes, dict):
        _fail("$.response_modes", "must be an object")
    for rm_id, rm in response_modes.items():
        rmp = f"$.response_modes.{rm_id}"
        if not isinstance(rm, dict):
            _fail(rmp, "must be an object")
        for key in ("max_words", "can_ask", "can_withhold"):
            if key not in rm:
                _fail(rmp, f"missing required property {key!r}")


def load_mood_states() -> dict[str, Any]:
    try:
        payload = load_contract_json("mood_states.v1.json")
        validate_mood_states(payload)
        return dict(payload)
    except ContractValidationError:
        return {
            "moods": [
                {"mood_id": "neutral", "valence": 0.0, "arousal": 0.1, "response_mode": "normal", "engagement_floor": 0.5},
            ],
            "response_modes": {
                "normal": {"max_words": 40, "can_ask": True, "can_withhold": False},
            },
        }


def validate_affect_lexicon(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.affect_lexicon.v1":
        _fail("$.schema_id", "must equal 'melm.affect_lexicon.v1'")
    entries = payload.get("entries")
    if not isinstance(entries, dict) or not entries:
        _fail("$.entries", "must be a non-empty object")
    for lemma, entry in entries.items():
        path = f"$.entries.{lemma}"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        for key in ("valence", "arousal", "tags"):
            if key not in entry:
                _fail(path, f"missing required property {key!r}")
        if not isinstance(entry.get("valence"), (int, float)):
            _fail(f"{path}.valence", "must be a number")
        if not isinstance(entry.get("arousal"), (int, float)):
            _fail(f"{path}.arousal", "must be a number")
        if not isinstance(entry.get("tags"), list):
            _fail(f"{path}.tags", "must be an array")


def load_affect_lexicon() -> dict[str, Any]:
    try:
        payload = load_contract_json("affect_lexicon.v1.json")
        validate_affect_lexicon(payload)
        return dict(payload)
    except ContractValidationError:
        return {
            "entries": {
                "happy": {"valence": 0.8, "arousal": 0.5, "tags": ["positive"]},
                "sad": {"valence": -0.7, "arousal": 0.15, "tags": ["negative"]},
            },
        }


def validate_response_pools(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.response_pools.v1":
        _fail("$.schema_id", "must equal 'melm.response_pools.v1'")
    pools = payload.get("pools")
    if not isinstance(pools, dict):
        _fail("$.pools", "must be an object")
    for key, templates in pools.items():
        path = f"$.pools.{key}"
        if not isinstance(templates, list) or not templates:
            _fail(path, "must be a non-empty array of strings")
        for t in templates:
            if not isinstance(t, str) or not t:
                _fail(path, "each template must be a non-empty string")


def load_response_pools() -> dict[str, Any]:
    try:
        payload = load_contract_json("response_pools.v1.json")
        validate_response_pools(payload)
        return dict(payload)
    except ContractValidationError:
        return {"pools": {}}


def validate_perception_affect_map(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.perception_affect_map.v1":
        _fail("$.schema_id", "must equal 'melm.perception_affect_map.v1'")
    default_priming = payload.get("default_priming")
    if not isinstance(default_priming, dict):
        _fail("$.default_priming", "must be an object")
    for key in ("valence", "arousal", "urgency"):
        if key not in default_priming:
            _fail(f"$.default_priming.{key}", "missing required property")
    stimuli = payload.get("stimuli")
    if not isinstance(stimuli, dict):
        _fail("$.stimuli", "must be an object")
    for lemma, entry in stimuli.items():
        path = f"$.stimuli.{lemma}"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        for key in ("valence", "arousal", "urgency"):
            if key not in entry:
                _fail(path, f"missing required property {key!r}")


def load_perception_affect_map() -> dict[str, Any]:
    try:
        payload = load_contract_json("perception_affect_map.v1.json")
        validate_perception_affect_map(payload)
        return dict(payload)
    except ContractValidationError:
        return {
            "default_priming": {"valence": 0.0, "arousal": 0.3, "urgency": "low"},
            "stimuli": {},
        }


# Canonical condition variables a creative-behavior condition may reference.
# Drift (e.g. "current_intent" vs the canonical "intent") must fail validation
# so the contract and the ConditionEvaluator stay in lockstep.
_ALLOWED_BEHAVIOR_CONDITION_VARS = frozenset({
    "prev_mood_id", "current_mood_id", "engagement", "intent",
    "occurrence", "response_mode", "prev_affect_has_pain",
    "affect_has_fatigue", "ambient_valence_delta",
})
_BEHAVIOR_CONDITION_KEYWORDS = frozenset({
    "in", "not", "and", "or", "abs", "true", "false",
})


def _unknown_condition_vars(condition: str) -> set[str]:
    """Return identifiers in *condition* that are not allowed variables/keywords."""
    import re
    stripped = re.sub(r"'[^']*'", " ", condition)  # drop string literals
    unknown: set[str] = set()
    for tok in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", stripped):
        if tok.lower() in _BEHAVIOR_CONDITION_KEYWORDS:
            continue
        if tok in _ALLOWED_BEHAVIOR_CONDITION_VARS:
            continue
        unknown.add(tok)
    return unknown


def validate_creative_behaviors(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.creative_behaviors.v1":
        _fail("$.schema_id", "must equal 'melm.creative_behaviors.v1'")
    behaviors = payload.get("behaviors")
    if not isinstance(behaviors, list):
        _fail("$.behaviors", "must be an array")
    seen = set()
    for idx, entry in enumerate(behaviors):
        path = f"$.behaviors[{idx}]"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        for key in ("id", "trigger", "condition", "cooldown_turns"):
            if key not in entry:
                _fail(path, f"missing required property {key!r}")
        if "template" not in entry and "templates" not in entry:
            _fail(path, "must have either 'template' or 'templates'")
        if not isinstance(entry.get("cooldown_turns"), int):
            _fail(f"{path}.cooldown_turns", "must be an integer")
        condition = entry.get("condition")
        if not isinstance(condition, str) or not condition.strip():
            _fail(f"{path}.condition", "must be a non-empty string")
        unknown = _unknown_condition_vars(condition)
        if unknown:
            _fail(
                f"{path}.condition",
                f"unknown condition variable(s) {sorted(unknown)!r}; "
                f"allowed: {sorted(_ALLOWED_BEHAVIOR_CONDITION_VARS)!r}",
            )
        bid = str(entry.get("id", ""))
        if bid in seen:
            _fail(path, f"duplicate behavior id {bid!r}")
        seen.add(bid)


def load_creative_behaviors() -> dict[str, Any]:
    try:
        payload = load_contract_json("creative_behaviors.v1.json")
        validate_creative_behaviors(payload)
        return dict(payload)
    except ContractValidationError:
        return {"behaviors": []}


def validate_geo_decision(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.geo_decision.v1":
        _fail("$.schema_id", "must equal 'melm.geo_decision.v1'")
    if not isinstance(payload.get("walk_threshold_km"), (int, float)):
        _fail("$.walk_threshold_km", "must be a number")
    if not isinstance(payload.get("place_purposes", {}), dict):
        _fail("$.place_purposes", "must be an object")
    if not isinstance(payload.get("purpose_overrides", {}), dict):
        _fail("$.purpose_overrides", "must be an object")


def load_geo_decision() -> dict[str, Any]:
    payload = load_contract_json("geo_decision.v1.json")
    validate_geo_decision(payload)
    return dict(payload)


def validate_geo_atlas(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.geo_atlas.v1":
        _fail("$.schema_id", "must equal 'melm.geo_atlas.v1'")
    places = payload.get("places")
    if not isinstance(places, dict) or not places:
        _fail("$.places", "must be a non-empty object")
    for name, coord in places.items():
        if not isinstance(coord, dict) or "lat" not in coord or "lon" not in coord:
            _fail(f"$.places.{name}", "must have 'lat' and 'lon'")
        if not isinstance(coord["lat"], (int, float)) or not isinstance(coord["lon"], (int, float)):
            _fail(f"$.places.{name}", "lat/lon must be numbers")


def load_geo_atlas() -> dict[str, Any]:
    payload = load_contract_json("geo_atlas.v1.json")
    validate_geo_atlas(payload)
    return dict(payload)


def validate_ethical_constraints(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.ethical_constraints.v1":
        _fail("$.schema_id", "must equal 'melm.ethical_constraints.v1'")
    for key in ("protected_probe_markers", "disclosure_verbs"):
        if not isinstance(payload.get(key), list) or not payload.get(key):
            _fail(f"$.{key}", "must be a non-empty array")
    inducements = payload.get("inducements")
    if not isinstance(inducements, dict) or not inducements:
        _fail("$.inducements", "must be a non-empty object")
    for name, block in inducements.items():
        if not isinstance(block, dict) or "refusal_reason" not in block:
            _fail(f"$.inducements.{name}", "must have a refusal_reason")
    templates = payload.get("refusal_templates")
    if not isinstance(templates, dict) or "privacy_nonnegotiable" not in templates:
        _fail("$.refusal_templates", "must include 'privacy_nonnegotiable'")


def load_ethical_constraints() -> dict[str, Any]:
    payload = load_contract_json("ethical_constraints.v1.json")
    validate_ethical_constraints(payload)
    return dict(payload)


def validate_self_identity_facts(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.self_identity_facts.v1":
        _fail("$.schema_id", "must equal 'melm.self_identity_facts.v1'")
    if not isinstance(payload.get("identity_facts"), dict) or not payload.get("identity_facts"):
        _fail("$.identity_facts", "must be a non-empty object")
    if not isinstance(payload.get("fallback_fact"), str) or not payload.get("fallback_fact"):
        _fail("$.fallback_fact", "must be a non-empty string")


def load_self_identity_facts() -> dict[str, Any]:
    payload = load_contract_json("self_identity_facts.v1.json")
    validate_self_identity_facts(payload)
    return dict(payload)


# ---------------------------------------------------------------------------
# Self-identity derivation contract (Task 1)
# ---------------------------------------------------------------------------


def validate_self_identity(data: dict[str, Any]) -> None:
    """Validate self_identity.v1.json contract.

    Checks: analysis_window_days (positive int), min_data_points (positive int),
    identity_labels (dict of intent → {label, frame}), identity_narratives
    (dict of mood_id → str), name_awareness_templates (dict of key → str).
    """
    if data.get("schema_id") != "melm.self_identity.v1":
        _fail("$.schema_id", "must equal 'melm.self_identity.v1'")
    window = data.get("analysis_window_days")
    if not isinstance(window, int) or window < 1:
        _fail("$.analysis_window_days", "must be a positive integer")
    min_pts = data.get("min_data_points")
    if not isinstance(min_pts, int) or min_pts < 1:
        _fail("$.min_data_points", "must be a positive integer")
    labels = data.get("identity_labels")
    if not isinstance(labels, dict) or not labels:
        _fail("$.identity_labels", "must be a non-empty object")
    for intent, entry in labels.items():
        ipath = f"$.identity_labels.{intent}"
        if not isinstance(intent, str) or not intent:
            _fail(ipath, "intent key must be a non-empty string")
        if not isinstance(entry, dict):
            _fail(ipath, "must be an object with 'label' and 'frame'")
        if "label" not in entry or not isinstance(entry["label"], str) or not entry["label"]:
            _fail(f"{ipath}.label", "must be a non-empty string")
        if "frame" not in entry or not isinstance(entry["frame"], str) or not entry["frame"]:
            _fail(f"{ipath}.frame", "must be a non-empty string")
    narratives = data.get("identity_narratives")
    if not isinstance(narratives, dict) or not narratives:
        _fail("$.identity_narratives", "must be a non-empty object")
    for mood_id, template in narratives.items():
        npath = f"$.identity_narratives.{mood_id}"
        if not isinstance(mood_id, str) or not mood_id:
            _fail(npath, "mood_id must be a non-empty string")
        if not isinstance(template, str) or not template:
            _fail(npath, "must be a non-empty string")
    templates = data.get("name_awareness_templates")
    if not isinstance(templates, dict) or not templates:
        _fail("$.name_awareness_templates", "must be a non-empty object")
    for key, val in templates.items():
        tpath = f"$.name_awareness_templates.{key}"
        if not isinstance(key, str) or not key:
            _fail(tpath, "key must be a non-empty string")
        if not isinstance(val, str) or not val:
            _fail(tpath, "must be a non-empty string")


def load_self_identity() -> dict[str, Any]:
    payload = load_contract_json("self_identity.v1.json")
    validate_self_identity(payload)
    return dict(payload)


# ---------------------------------------------------------------------------
# Knowledge types / world relations contracts (MVP3 foundation)
# ---------------------------------------------------------------------------


def validate_knowledge_types(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.knowledge_types.v1":
        _fail("$.schema_id", "must equal 'melm.knowledge_types.v1'")
    type_markers = payload.get("type_markers")
    if not isinstance(type_markers, dict):
        _fail("$.type_markers", "must be an object")
    opinion = type_markers.get("opinion_markers")
    if not isinstance(opinion, list) or not opinion:
        _fail("$.type_markers.opinion_markers", "must be a non-empty array of strings")
    for m in opinion:
        if not isinstance(m, str) or not m:
            _fail("$.type_markers.opinion_markers", "each entry must be a non-empty string")
    literary = type_markers.get("literary_stems")
    if not isinstance(literary, list) or not literary:
        _fail("$.type_markers.literary_stems", "must be a non-empty array of strings")
    for s in literary:
        if not isinstance(s, str) or not s:
            _fail("$.type_markers.literary_stems", "each entry must be a non-empty string")
    provenance = type_markers.get("provenance_confidence")
    if not isinstance(provenance, dict):
        _fail("$.type_markers.provenance_confidence", "must be an object")
    for key in ("seed", "user", "cloud"):
        val = provenance.get(key)
        if not isinstance(val, (int, float)):
            _fail(f"$.type_markers.provenance_confidence.{key}", "must be a number")
    truth = payload.get("truth_arbitration")
    if not isinstance(truth, dict):
        _fail("$.truth_arbitration", "must be an object")
    for key in ("contradiction_prompt", "contradiction_ack", "assert_ack", "negate_ack"):
        val = truth.get(key)
        if not isinstance(val, str) or not val:
            _fail(f"$.truth_arbitration.{key}", "must be a non-empty string")


def load_knowledge_types() -> dict[str, Any]:
    payload = load_contract_json("knowledge_types.v1.json")
    validate_knowledge_types(payload)
    return dict(payload)


def validate_world_relations(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.world_relations.v1":
        _fail("$.schema_id", "must equal 'melm.world_relations.v1'")
    pred_map = payload.get("predicate_to_relation")
    if not isinstance(pred_map, dict) or not pred_map:
        _fail("$.predicate_to_relation", "must be a non-empty object")
    for pred, entry in pred_map.items():
        path = f"$.predicate_to_relation.{pred}"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        for key in ("pattern", "relation_id", "confidence"):
            if key not in entry:
                _fail(path, f"missing required property {key!r}")
        if not isinstance(entry.get("pattern"), str) or not entry["pattern"]:
            _fail(f"{path}.pattern", "must be a non-empty string")
        if not isinstance(entry.get("relation_id"), str) or not entry["relation_id"]:
            _fail(f"{path}.relation_id", "must be a non-empty string")
        if not isinstance(entry.get("confidence"), (int, float)):
            _fail(f"{path}.confidence", "must be a number")


def load_world_relations() -> dict[str, Any]:
    payload = load_contract_json("world_relations.v1.json")
    validate_world_relations(payload)
    return dict(payload)


# ---------------------------------------------------------------------------
# Verb states / moral cognition contracts (T4)
# ---------------------------------------------------------------------------

def validate_verb_states(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.verb_states.v1":
        _fail("$.schema_id", "must equal 'melm.verb_states.v1'")
    verbs = payload.get("verbs")
    if not isinstance(verbs, dict) or not verbs:
        _fail("$.verbs", "must be a non-empty object")
    for verb, entry in verbs.items():
        vpath = f"$.verbs.{verb}"
        if not isinstance(entry, dict):
            _fail(vpath, "must be an object")
        ps = entry.get("patient_states")
        if not isinstance(ps, dict) or not ps:
            _fail(f"{vpath}.patient_states", "must be a non-empty object")
        for dim in ("physical", "emotional", "mental"):
            states = ps.get(dim, [])
            if not isinstance(states, list):
                _fail(f"{vpath}.patient_states.{dim}", "must be a list")
            for s in states:
                if not isinstance(s, str) or not s:
                    _fail(f"{vpath}.patient_states.{dim}", "entries must be non-empty strings")
        pts = entry.get("patient_types")
        if not isinstance(pts, list) or not pts:
            _fail(f"{vpath}.patient_types", "must be a non-empty list")
        for pt in pts:
            if not isinstance(pt, str) or not pt:
                _fail(f"{vpath}.patient_types", "entries must be non-empty strings")


def load_verb_states() -> dict[str, Any]:
    payload = load_contract_json("verb_states.v1.json")
    validate_verb_states(payload)
    return dict(payload)


def validate_causal_effects(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.causal_effects.v1":
        _fail("$.schema_id", "must equal 'melm.causal_effects.v1'")
    rules = payload.get("rules")
    if not isinstance(rules, dict) or not rules:
        _fail("$.rules", "must be a non-empty object")
    for verb, rule in rules.items():
        path = f"$.rules.{verb}"
        if not isinstance(rule, dict):
            _fail(path, "must be an object")
        if not isinstance(rule.get("confidence"), (int, float)):
            _fail(f"{path}.confidence", "must be a number")
        if "effects" not in rule:
            _fail(f"{path}.effects", "required")
        effects = rule.get("effects", {})
        if not isinstance(effects, dict):
            _fail(f"{path}.effects", "must be an object")
        for domain, states in effects.items():
            if not isinstance(states, list) or any(not isinstance(s, str) for s in states):
                _fail(f"{path}.effects.{domain}", "must be an array of strings")
        provenance = rule.get("provenance", "manual_curated")
        if provenance not in ("manual_curated", "offline_extractor", "user_stated", "cloud_candidate"):
            _fail(f"{path}.provenance", f"unknown provenance {provenance!r}")
        review_status = rule.get("review_status", "approved")
        if review_status not in ("approved", "pending", "rejected"):
            _fail(f"{path}.review_status", f"unknown review_status {review_status!r}")


def load_causal_effects() -> dict[str, Any]:
    payload = load_contract_json("causal_effects.v1.json")
    validate_causal_effects(payload)
    return dict(payload)


def validate_state_valences(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.state_valences.v1":
        _fail("$.schema_id", "must equal 'melm.state_valences.v1'")
    valences = payload.get("valences")
    if not isinstance(valences, dict) or not valences:
        _fail("$.valences", "must be a non-empty object")
    for state, valence in valences.items():
        spath = f"$.valences.{state}"
        if not isinstance(state, str) or not state:
            _fail(spath, "key must be a non-empty string")
        if not isinstance(valence, (int, float)):
            _fail(spath, "must be a number")
        if not -1.0 <= valence <= 1.0:
            _fail(spath, "must be between -1.0 and 1.0")


def load_state_valences() -> dict[str, Any]:
    payload = load_contract_json("state_valences.v1.json")
    validate_state_valences(payload)
    return dict(payload)


# ---------------------------------------------------------------------------
# Curiosity / context / agreement contracts (Phase 1B)
# ---------------------------------------------------------------------------

def validate_deferred_task_templates(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.deferred_task_templates.v1":
        _fail("$.schema_id", "must equal 'melm.deferred_task_templates.v1'")
    templates = payload.get("templates")
    if not isinstance(templates, dict) or not templates:
        _fail("$.templates", "must be a non-empty object")
    for key, val in templates.items():
        if not isinstance(key, str) or not key:
            _fail(f"$.templates.{key}", "key must be a non-empty string")
        if not isinstance(val, str) or not val:
            _fail(f"$.templates.{key}", "must be a non-empty string")


def load_deferred_task_templates() -> dict[str, str]:
    payload = load_contract_json("deferred_task_templates.v1.json")
    validate_deferred_task_templates(payload)
    return dict(payload["templates"])


def validate_atom_templates(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.atom_templates.v1":
        _fail("$.schema_id", "must equal 'melm.atom_templates.v1'")
    templates = payload.get("templates")
    if not isinstance(templates, dict) or not templates:
        _fail("$.templates", "must be a non-empty object")
    for key, val in templates.items():
        if not isinstance(key, str) or not key:
            _fail(f"$.templates.{key}", "key must be a non-empty string")
        if not isinstance(val, str) or not val:
            _fail(f"$.templates.{key}", "must be a non-empty string")


def load_atom_templates() -> dict[str, str]:
    payload = load_contract_json("atom_templates.v1.json")
    validate_atom_templates(payload)
    return dict(payload["templates"])


def validate_novelty_patterns(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.novelty_patterns.v1":
        _fail("$.schema_id", "must equal 'melm.novelty_patterns.v1'")
    palindrome = payload.get("palindrome")
    if not isinstance(palindrome, dict) or "min_length" not in palindrome or "case_sensitive" not in palindrome:
        _fail("$.palindrome", "must be an object with 'min_length' and 'case_sensitive'")
    if not isinstance(palindrome.get("min_length"), int):
        _fail("$.palindrome.min_length", "must be an integer")
    if not isinstance(palindrome.get("case_sensitive"), bool):
        _fail("$.palindrome.case_sensitive", "must be a boolean")
    morphemes = payload.get("morpheme_patterns")
    if not isinstance(morphemes, list) or not morphemes:
        _fail("$.morpheme_patterns", "must be a non-empty array of strings")
    for m in morphemes:
        if not isinstance(m, str) or not m:
            _fail("$.morpheme_patterns", "each entry must be a non-empty string")
    symbols = payload.get("cultural_symbols")
    if not isinstance(symbols, dict) or not symbols:
        _fail("$.cultural_symbols", "must be a non-empty object")
    for key, entry in symbols.items():
        if not isinstance(entry, dict) or "patterns" not in entry or "description" not in entry:
            _fail(f"$.cultural_symbols.{key}", "must have 'patterns' and 'description'")
        if not isinstance(entry["patterns"], list) or not entry["patterns"]:
            _fail(f"$.cultural_symbols.{key}.patterns", "must be a non-empty array")
        if not isinstance(entry["description"], str) or not entry["description"]:
            _fail(f"$.cultural_symbols.{key}.description", "must be a non-empty string")


def load_novelty_patterns() -> dict[str, Any]:
    payload = load_contract_json("novelty_patterns.v1.json")
    validate_novelty_patterns(payload)
    return dict(payload)


def validate_agreement_templates(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.agreement_templates.v1":
        _fail("$.schema_id", "must equal 'melm.agreement_templates.v1'")
    templates = payload.get("templates")
    if not isinstance(templates, dict) or not templates:
        _fail("$.templates", "must be a non-empty object")
    for key, val in templates.items():
        if not isinstance(key, str) or not key:
            _fail(f"$.templates.{key}", "key must be a non-empty string")
        if not isinstance(val, str) or not val:
            _fail(f"$.templates.{key}", "must be a non-empty string")


def load_agreement_templates() -> dict[str, str]:
    payload = load_contract_json("agreement_templates.v1.json")
    validate_agreement_templates(payload)
    return dict(payload["templates"])


def validate_epistemic_states(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.epistemic_states.v1":
        _fail("$.schema_id", "must equal 'melm.epistemic_states.v1'")
    mappings = payload.get("valence_mappings")
    if not isinstance(mappings, dict) or not mappings:
        _fail("$.valence_mappings", "must be a non-empty object")
    for key, val in mappings.items():
        if not isinstance(key, str) or not key:
            _fail(f"$.valence_mappings.{key}", "key must be a non-empty string")
        if not isinstance(val, (int, float)):
            _fail(f"$.valence_mappings.{key}", "must be a number")
    max_open = payload.get("max_open_states")
    if not isinstance(max_open, int) or max_open < 1:
        _fail("$.max_open_states", "must be a positive integer")
    stale = payload.get("staleness_hours")
    if not isinstance(stale, (int, float)) or stale < 0:
        _fail("$.staleness_hours", "must be a non-negative number")


def load_epistemic_states() -> dict[str, Any]:
    payload = load_contract_json("epistemic_states.v1.json")
    validate_epistemic_states(payload)
    return dict(payload)


def validate_background_task_policies(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.background_task_policies.v1":
        _fail("$.schema_id", "must equal 'melm.background_task_policies.v1'")
    for key in ("max_concurrent_jobs", "default_check_interval_seconds", "max_retries", "job_timeout_seconds"):
        val = payload.get(key)
        if not isinstance(val, (int, float)) or val < 0:
            _fail(f"$.{key}", "must be a non-negative number")
    deferral = payload.get("deferral_policy")
    if not isinstance(deferral, dict):
        _fail("$.deferral_policy", "must be an object")
    for dkey in ("max_deferred_per_session", "auto_research_timeout_seconds"):
        dval = deferral.get(dkey)
        if not isinstance(dval, (int, float)) or dval < 0:
            _fail(f"$.deferral_policy.{dkey}", "must be a non-negative number")


def load_background_task_policies() -> dict[str, Any]:
    payload = load_contract_json("background_task_policies.v1.json")
    validate_background_task_policies(payload)
    return dict(payload)


def validate_research_deferral_triggers(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.research_deferral_triggers.v1":
        _fail("$.schema_id", "must equal 'melm.research_deferral_triggers.v1'")
    defer_when = payload.get("defer_when")
    if not isinstance(defer_when, dict) or not defer_when:
        _fail("$.defer_when", "must be a non-empty object")
    for key in ("provider_unavailable", "network_unreachable", "utterance_contains_defer_keywords"):
        if key in defer_when and not isinstance(defer_when[key], bool):
            _fail(f"$.defer_when.{key}", "must be a boolean")
    if "turn_count_exceeds" in defer_when and not isinstance(defer_when["turn_count_exceeds"], int):
        _fail("$.defer_when.turn_count_exceeds", "must be an integer")
    for list_key in ("defer_keywords", "immediate_keywords"):
        kw = payload.get(list_key)
        if not isinstance(kw, list):
            _fail(f"$.{list_key}", "must be an array")
        for item in kw:
            if not isinstance(item, str) or not item:
                _fail(f"$.{list_key}", "each entry must be a non-empty string")


def load_research_deferral_triggers() -> dict[str, Any]:
    payload = load_contract_json("research_deferral_triggers.v1.json")
    validate_research_deferral_triggers(payload)
    return dict(payload)


def validate_commitment_parsers(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.commitment_parsers.v1":
        _fail("$.schema_id", "must equal 'melm.commitment_parsers.v1'")
    for section in ("commitment_cues", "temporal_cues"):
        cues = payload.get(section)
        if not isinstance(cues, dict) or not cues:
            _fail(f"$.{section}", "must be a non-empty object")
        for key, values in cues.items():
            if not isinstance(key, str) or not key:
                _fail(f"$.{section}.{key}", "key must be a non-empty string")
            if not isinstance(values, list) or not values:
                _fail(f"$.{section}.{key}", "must be a non-empty array of strings")
            for v in values:
                if not isinstance(v, str) or not v:
                    _fail(f"$.{section}.{key}", "each entry must be a non-empty string")


def load_commitment_parsers() -> dict[str, Any]:
    payload = load_contract_json("commitment_parsers.v1.json")
    validate_commitment_parsers(payload)
    return dict(payload)


# ---------------------------------------------------------------------------
# Story plan schema
# ---------------------------------------------------------------------------

def validate_story_plan_schema(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.story_plan_schema.v1":
        _fail("$.schema_id", "must equal 'melm.story_plan_schema.v1'")
    fields = payload.get("story_plan_fields")
    if not isinstance(fields, dict):
        _fail("$.story_plan_fields", "must be a non-empty object")
    required = payload.get("required")
    if not isinstance(required, list):
        _fail("$.required", "must be a list")
    valid_types = {"string", "integer", "array:string"}
    for fname, ftype in fields.items():
        if not isinstance(ftype, str) or ftype not in valid_types:
            _fail(f"$.story_plan_fields.{fname}", f"must be one of {valid_types}")
    for r in required:
        if not isinstance(r, str) or r not in fields:
            _fail(f"$.required[{r}]", f"must be a key in story_plan_fields")


def load_story_plan_schema() -> dict[str, Any]:
    payload = load_contract_json("story_plan_schema.v1.json")
    validate_story_plan_schema(payload)
    return payload


def get_contract_info(schema_short_name: str) -> dict[str, Any] | None:
    """Look up a contract by its short name (e.g. 'story_plan_schema.v1')."""
    registry = ContractRegistry.load()
    full_id = f"melm.{schema_short_name}"
    try:
        return dict(registry.require(full_id))
    except ContractValidationError:
        return None


# ---------------------------------------------------------------------------
# Music contracts (Phase 1a)
# ---------------------------------------------------------------------------

def validate_music_style_templates(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.music_style_templates.v1":
        _fail("$.schema_id", "must equal 'melm.music_style_templates.v1'")
    templates = payload.get("templates")
    if not isinstance(templates, dict) or not templates:
        _fail("$.templates", "must be a non-empty object")
    for name, tmpl in templates.items():
        tpath = f"$.templates.{name}"
        if not isinstance(tmpl, dict):
            _fail(tpath, "must be an object")
        for key in ("time_signature", "tempo_range", "chord_voicing", "bass_pattern", "arpeggio_pattern", "scale_preference"):
            if key not in tmpl:
                _fail(tpath, f"missing required property {key!r}")
        if not isinstance(tmpl.get("time_signature"), str) or not tmpl["time_signature"]:
            _fail(f"{tpath}.time_signature", "must be a non-empty string")
        tr = tmpl.get("tempo_range")
        if not isinstance(tr, list) or len(tr) != 2 or not all(isinstance(v, int) for v in tr):
            _fail(f"{tpath}.tempo_range", "must be a list of 2 ints")
        for key in ("chord_voicing", "bass_pattern", "arpeggio_pattern", "scale_preference"):
            val = tmpl.get(key)
            if not isinstance(val, str) or not val:
                _fail(f"{tpath}.{key}", "must be a non-empty string")


def load_music_style_templates() -> dict[str, Any]:
    payload = load_contract_json("music_style_templates.v1.json")
    validate_music_style_templates(payload)
    return dict(payload)


def validate_music_genre_scales(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.music_genre_scales.v1":
        _fail("$.schema_id", "must equal 'melm.music_genre_scales.v1'")
    scales = payload.get("scales")
    if not isinstance(scales, dict) or not scales:
        _fail("$.scales", "must be a non-empty object")
    for name, intervals in scales.items():
        spath = f"$.scales.{name}"
        if not isinstance(intervals, list) or not intervals:
            _fail(spath, "must be a non-empty list")
        for v in intervals:
            if not isinstance(v, int) or not 0 <= v <= 11:
                _fail(spath, "each interval must be an int in 0..11")
    genre_map = payload.get("genre_scale_map")
    if not isinstance(genre_map, dict) or not genre_map:
        _fail("$.genre_scale_map", "must be a non-empty object")
    for genre, scale in genre_map.items():
        if not isinstance(genre, str) or not genre:
            _fail(f"$.genre_scale_map.{genre!r}", "key must be a non-empty string")
        if scale not in scales:
            _fail(f"$.genre_scale_map.{genre!r}", f"unknown scale {scale!r}")


def load_music_genre_scales() -> dict[str, Any]:
    payload = load_contract_json("music_genre_scales.v1.json")
    validate_music_genre_scales(payload)
    return dict(payload)


def validate_mood_faces(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.mood_faces.v1":
        _fail("$.schema_id", "must equal 'melm.mood_faces.v1'")
    faces = payload.get("faces")
    if not isinstance(faces, dict) or not faces:
        _fail("$.faces", "must be a non-empty object")
    for name, face in faces.items():
        fpath = f"$.faces.{name}"
        if not isinstance(face, dict):
            _fail(fpath, "must be an object")
        eyes = face.get("eyes")
        if not isinstance(eyes, list) or not eyes or not all(isinstance(e, str) for e in eyes):
            _fail(f"{fpath}.eyes", "must be a non-empty list of strings")
        mouths = face.get("mouths")
        if not isinstance(mouths, list) or not mouths or not all(isinstance(m, str) for m in mouths):
            _fail(f"{fpath}.mouths", "must be a non-empty list of strings")
    vbs = payload.get("valence_boundaries")
    if not isinstance(vbs, list) or len(vbs) != 3 or not all(isinstance(v, (int, float)) and v > 0 for v in vbs):
        _fail("$.valence_boundaries", "must be a list of 3 positive floats")


def load_mood_faces() -> dict[str, Any]:
    payload = load_contract_json("mood_faces.v1.json")
    validate_mood_faces(payload)
    return dict(payload)


def validate_mood_face_tones(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.mood_face_tones.v1":
        _fail("$.schema_id", "must equal 'melm.mood_face_tones.v1'")
    tones = payload.get("tones")
    if not isinstance(tones, dict) or not tones:
        _fail("$.tones", "must be a non-empty object")
    for key, val in tones.items():
        if not isinstance(key, str) or not key:
            _fail(f"$.tones.{key!r}", "key must be a non-empty string")
        if not isinstance(val, str) or not val:
            _fail(f"$.tones.{key!r}", "value must be a non-empty string")
    mood_cues = payload.get("mood_cues")
    if not isinstance(mood_cues, dict) or not mood_cues:
        _fail("$.mood_cues", "must be a non-empty object")
    for key, val in mood_cues.items():
        if not isinstance(key, str) or not key:
            _fail(f"$.mood_cues.{key!r}", "key must be a non-empty string")
        if not isinstance(val, str) or not val:
            _fail(f"$.mood_cues.{key!r}", "value must be a non-empty string")


def load_mood_face_tones() -> dict[str, Any]:
    payload = load_contract_json("mood_face_tones.v1.json")
    validate_mood_face_tones(payload)
    return dict(payload)


def validate_noise_tokens(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.noise_tokens.v1":
        _fail("$.schema_id", "must equal 'melm.noise_tokens.v1'")
    entries = payload.get("entries")
    if not isinstance(entries, dict) or not entries:
        _fail("$.entries", "must be a non-empty object")
    for token, entry in entries.items():
        path = f"$.entries.{token}"
        if not isinstance(token, str) or not token:
            _fail(path, "token must be a non-empty string")
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        for key in ("valence", "arousal", "tags", "strip_from_parse"):
            if key not in entry:
                _fail(path, f"missing required property {key!r}")
        if not isinstance(entry["valence"], (int, float)):
            _fail(f"{path}.valence", "must be a number")
        if not isinstance(entry["arousal"], (int, float)):
            _fail(f"{path}.arousal", "must be a number")
        if not isinstance(entry["tags"], list) or not entry["tags"]:
            _fail(f"{path}.tags", "must be a non-empty array of strings")
        for tag in entry["tags"]:
            if not isinstance(tag, str) or not tag:
                _fail(f"{path}.tags", "each tag must be a non-empty string")
        if not isinstance(entry["strip_from_parse"], bool):
            _fail(f"{path}.strip_from_parse", "must be a boolean")
    if "fallback_strip" in payload and not isinstance(payload["fallback_strip"], bool):
        _fail("$.fallback_strip", "must be a boolean")


def load_noise_tokens() -> dict[str, Any]:
    payload = load_contract_json("noise_tokens.v1.json")
    validate_noise_tokens(payload)
    return dict(payload["entries"])


def validate_normalization_expansions(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.normalization_expansions.v1":
        _fail("$.schema_id", "must equal 'melm.normalization_expansions.v1'")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        _fail("$.entries", "must be a non-empty array")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        path = f"$.entries[{index}]"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        raw = entry.get("raw")
        if not isinstance(raw, str) or not raw:
            _fail(f"{path}.raw", "must be a non-empty string")
        if raw in seen:
            _fail(f"{path}.raw", f"duplicate raw form {raw!r}")
        seen.add(raw)
        std = entry.get("standard")
        if not isinstance(std, str) or not std:
            _fail(f"{path}.standard", "must be a non-empty string")
        cat = entry.get("category")
        if not isinstance(cat, str) or not cat:
            _fail(f"{path}.category", "must be a non-empty string")


def load_normalization_expansions() -> dict[str, Any]:
    payload = load_contract_json("normalization_expansions.v1.json")
    validate_normalization_expansions(payload)
    return payload


def validate_agreement_rules(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.agreement_rules.v1":
        _fail("$.schema_id", "must equal 'melm.agreement_rules.v1'")
    subjects = payload.get("non_3sg_subjects")
    if not isinstance(subjects, list) or not subjects:
        _fail("$.non_3sg_subjects", "must be a non-empty array")
    irr = payload.get("irregular_3sg_map")
    if not isinstance(irr, dict) or not irr:
        _fail("$.irregular_3sg_map", "must be a non-empty object")
    adverbs = payload.get("intervening_adverbs")
    if not isinstance(adverbs, list):
        _fail("$.intervening_adverbs", "must be an array")


def load_agreement_rules() -> dict[str, Any]:
    payload = load_contract_json("agreement_rules.v1.json")
    validate_agreement_rules(payload)
    return payload


def validate_token_typability(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.token_typability.v1":
        _fail("$.schema_id", "must equal 'melm.token_typability.v1'")
    for key in ("min_vowel_ratio", "max_consonant_run", "min_gibberish_len", "max_abbreviation_len"):
        if not isinstance(payload.get(key), (int, float)):
            _fail(f"$.{key}", "must be a number")
    known = payload.get("known_abbreviations", [])
    if not isinstance(known, list):
        _fail("$.known_abbreviations", "must be a list")
    for entry in known:
        if not isinstance(entry, str) or not entry:
            _fail("$.known_abbreviations", "each entry must be a non-empty string")


def load_token_typability() -> dict[str, Any]:
    try:
        payload = load_contract_json("token_typability.v1.json")
        validate_token_typability(payload)
        return dict(payload)
    except ContractValidationError:
        return {}  # fail-open: no contract → no classification


# ---------------------------------------------------------------------------
# Causal cue lemmas (C1 track)
# ---------------------------------------------------------------------------

_VALID_CUE_TYPES = {"causal_explanation", "causal_prediction"}

_VALID_DIRECTIONS = {"effect_to_cause", "cause_to_effect"}


def validate_causal_cues(data: dict[str, Any]) -> None:
    if data.get("schema_id") != "melm.causal_cues.v1":
        _fail("$.schema_id", "must equal 'melm.causal_cues.v1'")
    cues = data.get("cues")
    if not isinstance(cues, list) or not cues:
        _fail("$.cues", "must be a non-empty array")
    for index, cue in enumerate(cues):
        path = f"$.cues[{index}]"
        if not isinstance(cue, dict):
            _fail(path, "must be an object")
        for key in ("lemma", "cue_type", "direction", "language", "confidence"):
            if key not in cue:
                _fail(path, f"missing required property {key!r}")
        lemma = cue.get("lemma")
        if not isinstance(lemma, str) or not lemma:
            _fail(f"{path}.lemma", "must be a non-empty string")
        cue_type = cue.get("cue_type")
        if not isinstance(cue_type, str) or cue_type not in _VALID_CUE_TYPES:
            _fail(f"{path}.cue_type", f"must be one of {sorted(_VALID_CUE_TYPES)!r}")
        direction = cue.get("direction")
        if not isinstance(direction, str) or direction not in _VALID_DIRECTIONS:
            _fail(f"{path}.direction", f"must be one of {sorted(_VALID_DIRECTIONS)!r}")
        language = cue.get("language")
        if not isinstance(language, str) or not language:
            _fail(f"{path}.language", "must be a non-empty string")
        confidence = cue.get("confidence")
        if not isinstance(confidence, (int, float)):
            _fail(f"{path}.confidence", "must be a number")
        if not 0 <= confidence <= 1:
            _fail(f"{path}.confidence", "must be between 0 and 1")
        co_cues = cue.get("co_cue_lemmas")
        if co_cues is not None:
            if not isinstance(co_cues, list) or not co_cues:
                _fail(f"{path}.co_cue_lemmas", "must be a non-empty array when present")
            for item in co_cues:
                if not isinstance(item, str) or not item:
                    _fail(f"{path}.co_cue_lemmas", "each entry must be a non-empty string")


def load_causal_cues() -> list[dict[str, Any]]:
    payload = load_contract_json("causal_cues.v1.json")
    validate_causal_cues(payload)
    return list(payload["cues"])


_VALID_CAUSAL_LINK_RELATIONS = {"causes", "caused_by", "enables", "prevents"}


_VALID_INTRODUCES = {"cause", "effect"}


def validate_causal_link_markers(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.causal_link_markers.v1":
        _fail("$.schema_id", "must equal 'melm.causal_link_markers.v1'")
    markers = payload.get("markers")
    if not isinstance(markers, dict) or not markers:
        _fail("$.markers", "must be a non-empty object")
    for marker, entry in markers.items():
        path = f"$.markers.{marker}"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        relation = entry.get("relation")
        if not isinstance(relation, str) or relation not in _VALID_CAUSAL_LINK_RELATIONS:
            _fail(f"{path}.relation", f"must be one of {sorted(_VALID_CAUSAL_LINK_RELATIONS)!r}")
        direction = entry.get("direction")
        if not isinstance(direction, str) or direction not in _VALID_DIRECTIONS:
            _fail(f"{path}.direction", f"must be one of {sorted(_VALID_DIRECTIONS)!r}")
        introduces = entry.get("introduces", "")
        if not isinstance(introduces, str) or introduces not in _VALID_INTRODUCES:
            _fail(f"{path}.introduces", f"must be one of {sorted(_VALID_INTRODUCES)!r}")
        description = entry.get("description", "")
        if not isinstance(description, str):
            _fail(f"{path}.description", "must be a string")


def load_causal_link_markers() -> dict[str, dict[str, Any]]:
    payload = load_contract_json("causal_link_markers.v1.json")
    validate_causal_link_markers(payload)
    return dict(payload.get("markers", {}))


# ---------------------------------------------------------------------------
# Causal frames (V0.4 atomic causality)
# ---------------------------------------------------------------------------

_VALID_CAUSE_KINDS = {
    "intentional_action", "natural_process", "accidental_process",
    "instrumental_action", "unknown",
}


def _load_semantic_class_ids() -> set[str]:
    try:
        payload = load_contract_json("semantic_classes.v1.json")
        return {c["class_id"] for c in payload.get("classes", []) if isinstance(c.get("class_id"), str)}
    except Exception:
        return set()


def validate_causal_frames(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.causal_frames.v1":
        _fail("$.schema_id", "must equal 'melm.causal_frames.v1'")
    valid_classes = _load_semantic_class_ids()

    # predicate_frames
    frames = payload.get("predicate_frames")
    if not isinstance(frames, dict):
        _fail("$.predicate_frames", "must be an object")
    for pid, frame in frames.items():
        path = f"$.predicate_frames.{pid}"
        if not isinstance(frame, dict):
            _fail(path, "must be an object")
        if frame.get("predicate_id") != pid:
            _fail(f"{path}.predicate_id", f"must equal key {pid!r}")
        sc = frame.get("semantic_class", "")
        if not isinstance(sc, str) or sc not in valid_classes:
            _fail(f"{path}.semantic_class", f"must be a valid semantic class, got {sc!r}")
        cause_kind = frame.get("default_cause_kind", "")
        if cause_kind not in _VALID_CAUSE_KINDS:
            _fail(f"{path}.default_cause_kind", f"must be one of {sorted(_VALID_CAUSE_KINDS)!r}")
        effects = frame.get("effects", [])
        if not isinstance(effects, list):
            _fail(f"{path}.effects", "must be an array")
        for ei, effect in enumerate(effects):
            epath = f"{path}.effects[{ei}]"
            if not isinstance(effect, dict):
                _fail(epath, "must be an object")
            for key in ("state", "domain", "target_role", "relation", "confidence"):
                if key not in effect:
                    _fail(epath, f"missing required property {key!r}")
            if not isinstance(effect["state"], str) or not effect["state"]:
                _fail(f"{epath}.state", "must be a non-empty string")
            if not isinstance(effect["domain"], str) or not effect["domain"]:
                _fail(f"{epath}.domain", "must be a non-empty string")
            if not isinstance(effect["target_role"], str) or not effect["target_role"]:
                _fail(f"{epath}.target_role", "must be a non-empty string")
            rel = effect.get("relation")
            if not isinstance(rel, str) or rel not in _VALID_CAUSAL_LINK_RELATIONS:
                _fail(f"{epath}.relation", f"must be one of {sorted(_VALID_CAUSAL_LINK_RELATIONS)!r}")
            conf = effect.get("confidence")
            if not isinstance(conf, (int, float)):
                _fail(f"{epath}.confidence", "must be a number")
            if not 0 <= conf <= 1:
                _fail(f"{epath}.confidence", "must be between 0 and 1")

    # state_definitions
    states = payload.get("state_definitions")
    if not isinstance(states, dict):
        _fail("$.state_definitions", "must be an object")
    for sid, state_entry in states.items():
        spath = f"$.state_definitions.{sid}"
        if not isinstance(state_entry, dict):
            _fail(spath, "must be an object")
        if state_entry.get("state_id") != sid:
            _fail(f"{spath}.state_id", f"must equal key {sid!r}")
        sc = state_entry.get("semantic_class", "")
        if not isinstance(sc, str) or sc not in valid_classes:
            _fail(f"{spath}.semantic_class", f"must be a valid semantic class, got {sc!r}")
        if not isinstance(state_entry.get("definition"), str) or not state_entry["definition"]:
            _fail(f"{spath}.definition", "must be a non-empty string")

    # active_entity_affordances
    entities = payload.get("active_entity_affordances")
    if not isinstance(entities, dict):
        _fail("$.active_entity_affordances", "must be an object")
    for eid, entry in entities.items():
        epath = f"$.active_entity_affordances.{eid}"
        if not isinstance(entry, dict):
            _fail(epath, "must be an object")
        sc = entry.get("semantic_class", "")
        if not isinstance(sc, str) or sc not in valid_classes:
            _fail(f"{epath}.semantic_class", f"must be a valid semantic class, got {sc!r}")
        bindings = entry.get("role_bindings", [])
        if not isinstance(bindings, list):
            _fail(f"{epath}.role_bindings", "must be an array")
        for bi, binding in enumerate(bindings):
            bpath = f"{epath}.role_bindings[{bi}]"
            if not isinstance(binding, dict):
                _fail(bpath, "must be an object")
            pid = binding.get("predicate_id", "")
            if isinstance(pid, str) and pid and pid not in frames:
                _fail(f"{bpath}.predicate_id", f"references unknown predicate frame {pid!r}")

    # surface_aliases
    aliases = payload.get("surface_aliases")
    if not isinstance(aliases, dict):
        _fail("$.surface_aliases", "must be an object")
    for alias, entry in aliases.items():
        apath = f"$.surface_aliases.{alias}"
        if not isinstance(entry, dict):
            _fail(apath, "must be an object")
        canonical = entry.get("canonical")
        if canonical is not None:
            if not isinstance(canonical, str) or not canonical:
                _fail(f"{apath}.canonical", "must be a non-empty string when present")
            if canonical != alias and canonical not in aliases:
                _fail(f"{apath}.canonical", f"must point to self or another alias, got {canonical!r}")


def load_causal_frames() -> dict[str, Any]:
    payload = load_contract_json("causal_frames.v1.json")
    validate_causal_frames(payload)
    return dict(payload)


# ---------------------------------------------------------------------------
# Noun atoms (MVP3)
# ---------------------------------------------------------------------------

_VALID_NOUN_KINDS = {"object", "place", "person", "animal", "plant", "concept"}


def validate_noun_atoms(payload: dict[str, Any]) -> None:
    """Validate noun_atoms.v1.json structure.

    Checks: schema_id, unique entity_ids, valid semantic_class_id references
    against the semantic classes spine, valid kind values.
    """
    if payload.get("schema_id") != "melm.noun_atoms.v1":
        _fail("$.schema_id", "must equal 'melm.noun_atoms.v1'")
    entities = payload.get("entities")
    if not isinstance(entities, list) or not entities:
        _fail("$.entities", "must be a non-empty array")
    class_ids = load_semantic_class_ids()
    seen: set[str] = set()
    for i, entry in enumerate(entities):
        path = f"$.entities[{i}]"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        eid = str(entry.get("entity_id", ""))
        if not eid:
            _fail(path, "must have a non-empty 'entity_id'")
        if eid in seen:
            _fail(path, f"duplicate entity_id {eid!r}")
        seen.add(eid)
        label = entry.get("label")
        if not isinstance(label, str) or not label:
            _fail(f"{path}.label", "must be a non-empty string")
        sc = str(entry.get("semantic_class_id", ""))
        if sc and sc not in class_ids:
            _fail(f"{path}.semantic_class_id", f"unknown class {sc!r}")
        kind = str(entry.get("kind", ""))
        if kind and kind not in _VALID_NOUN_KINDS:
            _fail(f"{path}.kind", f"must be one of {sorted(_VALID_NOUN_KINDS)!r}")
        definition = entry.get("definition")
        if definition is not None and (not isinstance(definition, str) or not definition):
            _fail(f"{path}.definition", "must be a non-empty string when present")
        genus = entry.get("genus_lemma")
        if genus is not None and (not isinstance(genus, str) or not genus):
            _fail(f"{path}.genus_lemma", "must be a non-empty string when present")
        slots = entry.get("slots")
        if slots is not None and not isinstance(slots, dict):
            _fail(f"{path}.slots", "must be an object when present")
        relations = entry.get("relations")
        if relations is not None:
            if not isinstance(relations, list):
                _fail(f"{path}.relations", "must be an array when present")
            for j, rel in enumerate(relations):
                rpath = f"{path}.relations[{j}]"
                if not isinstance(rel, dict):
                    _fail(rpath, "must be an object")
                if "target" not in rel or "relation" not in rel:
                    _fail(rpath, "must have 'target' and 'relation' keys")
                if not isinstance(rel["target"], str) or not rel["target"]:
                    _fail(f"{rpath}.target", "must be a non-empty string")
                if not isinstance(rel["relation"], str) or not rel["relation"]:
                    _fail(f"{rpath}.relation", "must be a non-empty string")


def load_noun_atoms() -> dict[str, Any]:
    """Load and validate noun_atoms.v1.json.

    Returns the full payload dict (schema_id + entities list).
    """
    payload = load_contract_json("noun_atoms.v1.json")
    validate_noun_atoms(payload)
    return dict(payload)


def seed_noun_atoms(store: Any) -> None:
    """Seed all noun entities from noun_atoms.v1.json into the entity store.

    *store* must duck-type ``add_entity``, ``set_entity_slot``, ``add_relation``.
    """
    import json as _json

    payload = load_noun_atoms()
    seeded_ids = {str(e["entity_id"]) for e in payload["entities"]}
    for entry in payload["entities"]:
        eid = str(entry["entity_id"])
        store.add_entity(
            entity_id=eid,
            kind=str(entry.get("kind", "object")),
            label=str(entry.get("label", "")),
            semantic_class_id=str(entry.get("semantic_class_id", "")),
            canonical_lemma=str(entry.get("canonical_lemma", "")),
        )
        slots = entry.get("slots", {})
        for slot_name, value in slots.items():
            store.set_entity_slot(
                entity_id=eid,
                slot_name=slot_name,
                value=value,
                provenance="seed",
                confidence=1.0,
            )
        for rel in entry.get("relations", []):
            target_id = str(rel["target"])
            if target_id not in seeded_ids:
                continue
            store.add_relation(
                entity_id=eid,
                relation=str(rel["relation"]),
                target_entity_id=target_id,
                strength=float(rel.get("strength", 0.8)),
                provenance="seed",
            )


# ---------------------------------------------------------------------------
# Modifier atoms (UOL Modifiers Slot)
# ---------------------------------------------------------------------------

_VALID_MODIFIER_TYPES = {"adjective", "adverb", "intensifier", "negation_adjunct"}


def validate_modifier_atoms(payload: dict[str, Any]) -> None:
    """Validate modifier_atoms.v1.json structure.

    Checks: schema_id, non-empty entries, valid semantic_class_id references
    against the semantic classes spine, known modifier_type values.
    """
    if payload.get("schema_id") != "melm.modifier_atoms.v1":
        _fail("$.schema_id", "must equal 'melm.modifier_atoms.v1'")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        _fail("$.entries", "must be a non-empty array")
    class_ids = load_semantic_class_ids()
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        path = f"$.entries[{i}]"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        lemma = str(entry.get("canonical_lemma", ""))
        if not lemma:
            _fail(f"{path}.canonical_lemma", "must be a non-empty string")
        if lemma in seen:
            _fail(path, f"duplicate canonical_lemma {lemma!r}")
        seen.add(lemma)
        sc = str(entry.get("semantic_class_id", ""))
        if sc and sc not in class_ids:
            _fail(f"{path}.semantic_class_id", f"unknown class {sc!r}")
        mt = str(entry.get("modifier_type", ""))
        if mt and mt not in _VALID_MODIFIER_TYPES:
            _fail(f"{path}.modifier_type", f"must be one of {sorted(_VALID_MODIFIER_TYPES)!r}")


def load_modifier_atoms() -> dict[str, Any]:
    """Load and validate modifier_atoms.v1.json.

    Returns the full payload dict (schema_id + entries list).
    """
    payload = load_contract_json("modifier_atoms.v1.json")
    validate_modifier_atoms(payload)
    return dict(payload)


def validate_identity_token_roles(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.identity_token_roles.v1":
        _fail("$.schema_id", "must equal 'melm.identity_token_roles.v1'")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        _fail("$.entries", "must be a non-empty array")
    for entry in entries:
        if not isinstance(entry, dict):
            _fail("$.entries[*]", "each entry must be an object")
        for key in ("token", "role", "meaning"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                _fail(f"$.entries[*].{key}", "must be a non-empty string")


def load_identity_token_roles() -> dict[str, tuple[str, str]]:
    payload = load_contract_json("identity_token_roles.v1.json")
    validate_identity_token_roles(payload)
    return {e["token"]: (e["role"], e["meaning"]) for e in payload["entries"]}


def validate_task_domain_terms(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.task_domain_terms.v1":
        _fail("$.schema_id", "must equal 'melm.task_domain_terms.v1'")
    terms = payload.get("terms")
    if not isinstance(terms, list) or not terms:
        _fail("$.terms", "must be a non-empty array of strings")
    for t in terms:
        if not isinstance(t, str) or not t:
            _fail("$.terms", "each term must be a non-empty string")


def load_task_domain_terms() -> set[str]:
    payload = load_contract_json("task_domain_terms.v1.json")
    validate_task_domain_terms(payload)
    return set(payload["terms"])


def validate_story_constraint_stopwords(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.story_constraint_stopwords.v1":
        _fail("$.schema_id", "must equal 'melm.story_constraint_stopwords.v1'")
    words = payload.get("stopwords")
    if not isinstance(words, list) or not words:
        _fail("$.stopwords", "must be a non-empty array of strings")
    for w in words:
        if not isinstance(w, str) or not w:
            _fail("$.stopwords", "each stopword must be a non-empty string")


def load_story_constraint_stopwords() -> set[str]:
    payload = load_contract_json("story_constraint_stopwords.v1.json")
    validate_story_constraint_stopwords(payload)
    return set(payload["stopwords"])


def validate_identity_scope_tokens(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.identity_scope_tokens.v1":
        _fail("$.schema_id", "must equal 'melm.identity_scope_tokens.v1'")
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        _fail("$.tokens", "must be a non-empty array of strings")
    for t in tokens:
        if not isinstance(t, str) or not t:
            _fail("$.tokens", "each token must be a non-empty string")


def load_identity_scope_tokens() -> set[str]:
    payload = load_contract_json("identity_scope_tokens.v1.json")
    validate_identity_scope_tokens(payload)
    return set(payload["tokens"])


def validate_music_instruments(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.music_instruments.v1":
        _fail("$.schema_id", "must equal 'melm.music_instruments.v1'")
    instruments = payload.get("instruments")
    if not isinstance(instruments, list) or not instruments:
        _fail("$.instruments", "must be a non-empty array of strings")
    for inst in instruments:
        if not isinstance(inst, str) or not inst:
            _fail("$.instruments", "each instrument must be a non-empty string")


def load_music_instruments() -> set[str]:
    payload = load_contract_json("music_instruments.v1.json")
    validate_music_instruments(payload)
    return set(payload["instruments"])


def validate_verb_atoms(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.verb_atoms.v1":
        _fail("$.schema_id", "must equal 'melm.verb_atoms.v1'")
    entities = payload.get("entities")
    if not isinstance(entities, list) or not entities:
        _fail("$.entities", "must be a non-empty array")
    seen: set[str] = set()
    for entry in entities:
        if not isinstance(entry, dict):
            _fail("$.entities[*]", "each entity must be an object")
        eid = str(entry.get("entity_id", ""))
        if not eid:
            _fail("$.entities[*].entity_id", "must be a non-empty string")
        if eid in seen:
            _fail(f"$.entities[*].entity_id({eid})", "duplicate entity_id")
        seen.add(eid)
        sc = str(entry.get("semantic_class_id", ""))
        if not sc:
            _fail(f"$.entities[*]({eid}).semantic_class_id", "must be a non-empty string")
        if entry.get("kind") != "action":
            _fail(f"$.entities[*]({eid}).kind", "must equal 'action'")


def load_verb_atoms() -> dict[str, dict[str, Any]]:
    payload = load_contract_json("verb_atoms.v1.json")
    validate_verb_atoms(payload)
    return {e["entity_id"]: e for e in payload["entities"]}


def seed_verb_atoms(store: Any) -> None:
    import json as _json

    payload = load_contract_json("verb_atoms.v1.json")
    for entry in payload["entities"]:
        eid = str(entry["entity_id"])
        store.add_entity(
            entity_id=eid,
            kind=str(entry.get("kind", "action")),
            label=str(entry.get("label", "")),
            semantic_class_id=str(entry.get("semantic_class_id", "")),
            canonical_lemma=str(entry.get("canonical_lemma", "")),
        )
        slots = entry.get("slots", {})
        for slot_name, value in slots.items():
            store.set_entity_slot(
                entity_id=eid,
                slot_name=slot_name,
                value=value,
                provenance="seed",
                confidence=1.0,
            )


def validate_patient_type_map(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.patient_type_map.v1":
        _fail("$.schema_id", "must equal 'melm.patient_type_map.v1'")
    markers = payload.get("person_patient_markers")
    if not isinstance(markers, list) or not markers:
        _fail("$.person_patient_markers", "must be a non-empty list")
    for i, m in enumerate(markers):
        if not isinstance(m, str) or not m:
            _fail(f"$.person_patient_markers[{i}]", "must be a non-empty string")
    mappings = payload.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        _fail("$.mappings", "must be a non-empty list")
    seen = set()
    for i, entry in enumerate(mappings):
        path = f"$.mappings[{i}]"
        if not isinstance(entry, dict):
            _fail(path, "must be an object")
        surface = entry.get("surface")
        if not isinstance(surface, str) or not surface:
            _fail(f"{path}.surface", "must be a non-empty string")
        if surface in seen:
            _fail(f"{path}.surface", f"duplicate surface {surface!r}")
        seen.add(surface)
        sc = entry.get("semantic_class")
        if not isinstance(sc, str) or not sc:
            _fail(f"{path}.semantic_class", "must be a non-empty string")


def load_patient_type_map() -> dict[str, str]:
    data = load_contract_json("patient_type_map.v1.json")
    validate_patient_type_map(data)
    return {entry["surface"]: entry["semantic_class"] for entry in data["mappings"]}


def validate_family_relation_terms(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.family_relation_terms.v1":
        _fail("$.schema_id", "must equal 'melm.family_relation_terms.v1'")
    terms = payload.get("terms")
    if not isinstance(terms, list) or not terms:
        _fail("$.terms", "must be a non-empty list")
    for i, t in enumerate(terms):
        if not isinstance(t, str) or not t:
            _fail(f"$.terms[{i}]", "must be a non-empty string")


def load_family_relation_terms() -> frozenset[str]:
    data = load_contract_json("family_relation_terms.v1.json")
    validate_family_relation_terms(data)
    return frozenset(data["terms"])


def validate_status_domain_terms(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.status_domain_terms.v1":
        _fail("$.schema_id", "must equal 'melm.status_domain_terms.v1'")
    groups = payload.get("groups")
    if not isinstance(groups, dict) or not groups:
        _fail("$.groups", "must be a non-empty object")
    for name, tokens in groups.items():
        if not isinstance(tokens, list) or not tokens:
            _fail(f"$.groups.{name}", "must be a non-empty list")
        for i, t in enumerate(tokens):
            if not isinstance(t, str) or not t:
                _fail(f"$.groups.{name}[{i}]", "must be a non-empty string")


def load_status_domain_terms() -> dict[str, frozenset[str]]:
    data = load_contract_json("status_domain_terms.v1.json")
    validate_status_domain_terms(data)
    return {name: frozenset(tokens) for name, tokens in data["groups"].items()}


def validate_autobiographical_scope_terms(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.autobiographical_scope_terms.v1":
        _fail("$.schema_id", "must equal 'melm.autobiographical_scope_terms.v1'")
    groups = payload.get("groups")
    if not isinstance(groups, dict) or not groups:
        _fail("$.groups", "must be a non-empty object")
    for name, tokens in groups.items():
        if not isinstance(tokens, list) or not tokens:
            _fail(f"$.groups.{name}", "must be a non-empty list")
        for i, t in enumerate(tokens):
            if not isinstance(t, str) or not t:
                _fail(f"$.groups.{name}[{i}]", "must be a non-empty string")


def load_autobiographical_scope_terms() -> dict[str, frozenset[str]]:
    data = load_contract_json("autobiographical_scope_terms.v1.json")
    validate_autobiographical_scope_terms(data)
    return {name: frozenset(tokens) for name, tokens in data["groups"].items()}


def validate_intent_evidence_sources(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.intent_evidence_sources.v1":
        _fail("$.schema_id", "must equal 'melm.intent_evidence_sources.v1'")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not sources:
        _fail("$.sources", "must be a non-empty object")
    for intent, source in sources.items():
        if not isinstance(intent, str) or not intent:
            _fail("$.sources", f"intent key {intent!r} must be a non-empty string")
        if not isinstance(source, str) or not source:
            _fail(f"$.sources.{intent}", "must be a non-empty string")


def load_intent_evidence_sources() -> dict[str, str]:
    data = load_contract_json("intent_evidence_sources.v1.json")
    validate_intent_evidence_sources(data)
    return dict(data["sources"])


def validate_mood_emoji_map(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.mood_emoji_map.v1":
        _fail("$.schema_id", "must equal 'melm.mood_emoji_map.v1'")
    moods = payload.get("moods")
    if not isinstance(moods, dict) or not moods:
        _fail("$.moods", "must be a non-empty object")
    for mood_id, emoji in moods.items():
        if not isinstance(mood_id, str) or not mood_id:
            _fail("$.moods", f"mood key {mood_id!r} must be a non-empty string")
        if not isinstance(emoji, str) or not emoji:
            _fail(f"$.moods.{mood_id}", "must be a non-empty string")


def load_mood_emoji_map() -> dict[str, str]:
    data = load_contract_json("mood_emoji_map.v1.json")
    validate_mood_emoji_map(data)
    return dict(data["moods"])


def validate_always_respond_intents(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.always_respond_intents.v1":
        _fail("$.schema_id", "must equal 'melm.always_respond_intents.v1'")
    intents = payload.get("intents")
    if not isinstance(intents, list) or not intents:
        _fail("$.intents", "must be a non-empty list")
    for i, v in enumerate(intents):
        if not isinstance(v, str) or not v:
            _fail(f"$.intents[{i}]", "must be a non-empty string")


def load_always_respond_intents() -> frozenset[str]:
    data = load_contract_json("always_respond_intents.v1.json")
    validate_always_respond_intents(data)
    return frozenset(data["intents"])


def validate_short_circuit_reasons(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.short_circuit_reasons.v1":
        _fail("$.schema_id", "must equal 'melm.short_circuit_reasons.v1'")
    for field in ("reasons", "template_only_reasons"):
        items = payload.get(field)
        if not isinstance(items, list) or not items:
            _fail(f"$.{field}", "must be a non-empty list")
        for i, v in enumerate(items):
            if not isinstance(v, str) or not v:
                _fail(f"$.{field}[{i}]", "must be a non-empty string")


def load_short_circuit_reasons() -> dict[str, frozenset[str]]:
    data = load_contract_json("short_circuit_reasons.v1.json")
    validate_short_circuit_reasons(data)
    return {
        "reasons": frozenset(data["reasons"]),
        "template_only_reasons": frozenset(data["template_only_reasons"]),
    }


def validate_environment_prep_phrases(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.environment_prep_phrases.v1":
        _fail("$.schema_id", "must equal 'melm.environment_prep_phrases.v1'")
    mapping = payload.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        _fail("$.mapping", "must be a non-empty object")
    for env, phrase in mapping.items():
        if not isinstance(env, str) or not env:
            _fail("$.mapping", f"environment key {env!r} must be a non-empty string")
        if not isinstance(phrase, str) or not phrase:
            _fail(f"$.mapping.{env}", "must be a non-empty string")
    if not isinstance(payload.get("default_format"), str) or not payload.get("default_format"):
        _fail("$.default_format", "must be a non-empty string")


def load_environment_prep_phrases() -> dict[str, Any]:
    data = load_contract_json("environment_prep_phrases.v1.json")
    validate_environment_prep_phrases(data)
    return {
        "mapping": dict(data["mapping"]),
        "default_format": data["default_format"],
    }


def validate_sentience_map(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.sentience_map.v1":
        _fail("$.schema_id", "must equal 'melm.sentience_map.v1'")
    sm = payload.get("sentience_map")
    if not isinstance(sm, dict) or not sm:
        _fail("$.sentience_map", "must be a non-empty object")
    for key, val in sm.items():
        if not isinstance(key, str) or not key:
            _fail("$.sentience_map", f"key {key!r} must be a non-empty string")
        if not isinstance(val, bool):
            _fail(f"$.sentience_map.{key}", "must be a boolean")


def load_sentience_map() -> dict[str, bool]:
    data = load_contract_json("sentience_map.v1.json")
    validate_sentience_map(data)
    return dict(data["sentience_map"])


def validate_damage_markers(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.damage_markers.v1":
        _fail("$.schema_id", "must equal 'melm.damage_markers.v1'")
    markers = payload.get("markers")
    if not isinstance(markers, list) or not markers:
        _fail("$.markers", "must be a non-empty list")
    for i, v in enumerate(markers):
        if not isinstance(v, str) or not v:
            _fail(f"$.markers[{i}]", "must be a non-empty string")


def load_damage_markers() -> set[str]:
    data = load_contract_json("damage_markers.v1.json")
    validate_damage_markers(data)
    return set(data["markers"])


def validate_moral_responses(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.moral_responses.v1":
        _fail("$.schema_id", "must equal 'melm.moral_responses.v1'")
    responses = payload.get("responses")
    if not isinstance(responses, dict) or not responses:
        _fail("$.responses", "must be a non-empty object")
    for key, text in responses.items():
        if not isinstance(key, str) or not key:
            _fail("$.responses", f"key {key!r} must be a non-empty string")
        if not isinstance(text, str) or not text:
            _fail(f"$.responses.{key}", "must be a non-empty string")


def load_moral_responses() -> dict[str, str]:
    data = load_contract_json("moral_responses.v1.json")
    validate_moral_responses(data)
    return dict(data["responses"])


def validate_intent_domains(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.intent_domains.v1":
        _fail("$.schema_id", "must equal 'melm.intent_domains.v1'")
    domains = payload.get("domains")
    if not isinstance(domains, dict) or not domains:
        _fail("$.domains", "must be a non-empty object")
    for intent, domain in domains.items():
        if not isinstance(intent, str) or not intent:
            _fail("$.domains", f"intent key {intent!r} must be a non-empty string")
        if not isinstance(domain, str):
            _fail(f"$.domains.{intent}", "must be a string")


def load_intent_domains() -> dict[str, str]:
    data = load_contract_json("intent_domains.v1.json")
    validate_intent_domains(data)
    return dict(data["domains"])


def validate_frame_local_sources(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.frame_local_sources.v1":
        _fail("$.schema_id", "must equal 'melm.frame_local_sources.v1'")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not sources:
        _fail("$.sources", "must be a non-empty object")
    for intent, src_list in sources.items():
        if not isinstance(intent, str) or not intent:
            _fail("$.sources", f"intent key {intent!r} must be a non-empty string")
        if not isinstance(src_list, list):
            _fail(f"$.sources.{intent}", "must be a list of strings")
        for src in src_list:
            if not isinstance(src, str):
                _fail(f"$.sources.{intent}", f"source {src!r} must be a string")


def load_frame_local_sources() -> dict[str, list[str]]:
    data = load_contract_json("frame_local_sources.v1.json")
    validate_frame_local_sources(data)
    return {k: list(v) for k, v in data["sources"].items()}


def validate_pool_intents(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.pool_intents.v1":
        _fail("$.schema_id", "must equal 'melm.pool_intents.v1'")
    intents = payload.get("intents")
    if not isinstance(intents, list) or not intents:
        _fail("$.intents", "must be a non-empty list")
    for item in intents:
        if not isinstance(item, str) or not item:
            _fail("$.intents", f"intent {item!r} must be a non-empty string")


def load_pool_intents() -> set[str]:
    data = load_contract_json("pool_intents.v1.json")
    validate_pool_intents(data)
    return set(data["intents"])


def validate_synthesis_quality_weights(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.synthesis_quality_weights.v1":
        _fail("$.schema_id", "must equal 'melm.synthesis_quality_weights.v1'")
    for mode in ("refusal", "normal"):
        weights = payload.get(mode)
        if not isinstance(weights, dict) or not weights:
            _fail(f"$.{mode}", "must be a non-empty object")
        for key, val in weights.items():
            if not isinstance(key, str):
                _fail(f"$.{mode}", f"weight key {key!r} must be a string")
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                _fail(f"$.{mode}.{key}", "must be a float in [0, 1]")


def load_synthesis_quality_weights() -> dict[str, dict[str, float]]:
    data = load_contract_json("synthesis_quality_weights.v1.json")
    validate_synthesis_quality_weights(data)
    return data


def validate_persona_emoji_intents(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.persona_emoji_intents.v1":
        _fail("$.schema_id", "must equal 'melm.persona_emoji_intents.v1'")
    intents = payload.get("intents")
    if not isinstance(intents, list) or not intents:
        _fail("$.intents", "must be a non-empty list")
    for item in intents:
        if not isinstance(item, str) or not item:
            _fail("$.intents", f"intent {item!r} must be a non-empty string")


def load_persona_emoji_intents() -> set[str]:
    data = load_contract_json("persona_emoji_intents.v1.json")
    validate_persona_emoji_intents(data)
    return set(data["intents"])


def validate_midi_music_mapping(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.midi_music_mapping.v1":
        _fail("$.schema_id", "must equal 'melm.midi_music_mapping.v1'")
    mappings = payload.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        _fail("$.mappings", "must be a non-empty array")
    for i, m in enumerate(mappings):
        if not isinstance(m, dict):
            _fail(f"$.mappings[{i}]", "must be an object")
        kw = m.get("keywords")
        if not isinstance(kw, list) or not kw:
            _fail(f"$.mappings[{i}].keywords", "must be a non-empty array")
        for k in kw:
            if not isinstance(k, str):
                _fail(f"$.mappings[{i}].keywords", "each keyword must be a string")
        if not isinstance(m.get("genre"), str) or not m["genre"]:
            _fail(f"$.mappings[{i}].genre", "must be a non-empty string")
        if not isinstance(m.get("mood"), str) or not m["mood"]:
            _fail(f"$.mappings[{i}].mood", "must be a non-empty string")


def load_midi_music_mapping() -> list[dict[str, Any]]:
    data = load_contract_json("midi_music_mapping.v1.json")
    validate_midi_music_mapping(data)
    return list(data["mappings"])


def validate_commitment_responses(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.commitment_responses.v1":
        _fail("$.schema_id", "must equal 'melm.commitment_responses.v1'")
    responses = payload.get("responses")
    if not isinstance(responses, dict):
        _fail("$.responses", "must be an object")
    for key in ("prefix_to", "prefix_that", "default"):
        val = responses.get(key) if isinstance(responses, dict) else None
        if not isinstance(val, str) or not val:
            _fail(f"$.responses.{key}", "must be a non-empty string")


def load_commitment_responses() -> dict[str, str]:
    data = load_contract_json("commitment_responses.v1.json")
    validate_commitment_responses(data)
    return dict(data["responses"])


def validate_consent_revocation_response(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.consent_revocation_response.v1":
        _fail("$.schema_id", "must equal 'melm.consent_revocation_response.v1'")
    for field in ("answer", "evidence_key", "evidence_text"):
        val = payload.get(field)
        if not isinstance(val, str) or not val:
            _fail(f"$.{field}", "must be a non-empty string")


def load_consent_revocation_response() -> dict[str, str]:
    data = load_contract_json("consent_revocation_response.v1.json")
    validate_consent_revocation_response(data)
    return {k: str(data[k]) for k in ("answer", "evidence_key", "evidence_text")}


def validate_story_follow_up_phrase(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.story_follow_up_phrase.v1":
        _fail("$.schema_id", "must equal 'melm.story_follow_up_phrase.v1'")
    for field in ("phrase", "default_name"):
        val = payload.get(field)
        if not isinstance(val, str) or not val:
            _fail(f"$.{field}", "must be a non-empty string")


def load_story_follow_up_phrase() -> dict[str, str]:
    data = load_contract_json("story_follow_up_phrase.v1.json")
    validate_story_follow_up_phrase(data)
    return {k: str(data[k]) for k in ("phrase", "default_name")}


def validate_social_status_patterns(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.social_status_patterns.v1":
        _fail("$.schema_id", "must equal 'melm.social_status_patterns.v1'")
    patterns = payload.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        _fail("$.patterns", "must be a non-empty array")
    for i, p in enumerate(patterns):
        if not isinstance(p, str) or not p:
            _fail(f"$.patterns[{i}]", "must be a non-empty string")


def load_social_status_patterns() -> list[str]:
    data = load_contract_json("social_status_patterns.v1.json")
    validate_social_status_patterns(data)
    return list(data["patterns"])


def validate_autobiographical_horizon_tokens(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.autobiographical_horizon_tokens.v1":
        _fail("$.schema_id", "must equal 'melm.autobiographical_horizon_tokens.v1'")
    for section in ("day_span", "all_history", "long_term_memory"):
        if section not in payload:
            _fail(f"$", f"missing required section {section!r}")
        val = payload[section]
        if not isinstance(val, dict):
            _fail(f"$.{section}", "must be an object")


def load_autobiographical_horizon_tokens() -> dict[str, Any]:
    data = load_contract_json("autobiographical_horizon_tokens.v1.json")
    validate_autobiographical_horizon_tokens(data)
    return dict(data)


def validate_mail_verb_sets(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.mail_verb_sets.v1":
        _fail("$.schema_id", "must equal 'melm.mail_verb_sets.v1'")
    for key in ("send_verbs", "read_verbs", "gate_tokens"):
        arr = payload.get(key)
        if not isinstance(arr, list) or not arr:
            _fail(f"$.{key}", "must be a non-empty array")
        for i, v in enumerate(arr):
            if not isinstance(v, str) or not v:
                _fail(f"$.{key}[{i}]", "must be a non-empty string")


def load_mail_verb_sets() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    data = load_contract_json("mail_verb_sets.v1.json")
    validate_mail_verb_sets(data)
    return (
        frozenset(data["send_verbs"]),
        frozenset(data["read_verbs"]),
        frozenset(data["gate_tokens"]),
    )


def validate_short_circuit_responses(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.short_circuit_responses.v1":
        _fail("$.schema_id", "must equal 'melm.short_circuit_responses.v1'")
    responses = payload.get("responses")
    if not isinstance(responses, dict) or not responses:
        _fail("$.responses", "must be a non-empty object")
    for key, val in responses.items():
        if not isinstance(key, str) or not key:
            _fail("$.responses", "key must be a non-empty string")
        if not isinstance(val, str) or not val:
            _fail(f"$.responses.{key}", "must be a non-empty string")
    vt = payload.get("valence_thresholds")
    if not isinstance(vt, dict) or not vt:
        _fail("$.valence_thresholds", "must be a non-empty object")
    for key, val in vt.items():
        if not isinstance(key, str):
            _fail("$.valence_thresholds", "key must be a string")
        if not isinstance(val, (int, float)):
            _fail(f"$.valence_thresholds.{key}", "must be a number")
    ek = payload.get("evidence_keys")
    if not isinstance(ek, dict):
        _fail("$.evidence_keys", "must be an object")


def load_short_circuit_responses() -> dict[str, Any]:
    data = load_contract_json("short_circuit_responses.v1.json")
    validate_short_circuit_responses(data)
    return data


_UOL_TRIGGER_CONDITION_KEYS = {
    "assistant_targeted",
    "polarity",
    "min_affect_valence",
    "negative_modifier",
    "match_operator",
    "required_tokens",
    "excluded_tokens",
    "speech_acts",
}


def validate_uol_trigger_responses(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.uol_trigger_responses.v1":
        _fail("$.schema_id", "must equal 'melm.uol_trigger_responses.v1'")
    triggers = payload.get("triggers")
    if not isinstance(triggers, list) or not triggers:
        _fail("$.triggers", "must be a non-empty array")
    valid_operators = {"any", "all"}
    for i, trigger in enumerate(triggers):
        path = f"$.triggers[{i}]"
        if not isinstance(trigger, dict):
            _fail(path, "must be an object")
        tid = trigger.get("trigger_id")
        if not isinstance(tid, str) or not tid:
            _fail(f"{path}.trigger_id", "must be a non-empty string")
        desc = trigger.get("description")
        if desc is not None and not isinstance(desc, str):
            _fail(f"{path}.description", "must be a string")
        conditions = trigger.get("conditions")
        if not isinstance(conditions, dict):
            _fail(f"{path}.conditions", "must be an object")
        op = conditions.get("match_operator", "any")
        if op not in valid_operators:
            _fail(f"{path}.conditions.match_operator", "must be one of 'any' or 'all'")
        for key in conditions:
            if key not in _UOL_TRIGGER_CONDITION_KEYS:
                _fail(f"{path}.conditions.{key}", "unknown condition key")
        if "assistant_targeted" in conditions and not isinstance(conditions["assistant_targeted"], bool):
            _fail(f"{path}.conditions.assistant_targeted", "must be a boolean")
        if "polarity" in conditions and conditions["polarity"] not in {"positive", "negative"}:
            _fail(f"{path}.conditions.polarity", "must be 'positive' or 'negative'")
        if "min_affect_valence" in conditions and not isinstance(conditions["min_affect_valence"], (int, float)):
            _fail(f"{path}.conditions.min_affect_valence", "must be a number")
        if "negative_modifier" in conditions and not isinstance(conditions["negative_modifier"], bool):
            _fail(f"{path}.conditions.negative_modifier", "must be a boolean")
        if "required_tokens" in conditions:
            rt = conditions["required_tokens"]
            if not isinstance(rt, list) or not rt or not all(isinstance(t, str) and t for t in rt):
                _fail(f"{path}.conditions.required_tokens", "must be a non-empty array of strings")
        if "excluded_tokens" in conditions:
            et = conditions["excluded_tokens"]
            if not isinstance(et, list) or not et or not all(isinstance(t, str) and t for t in et):
                _fail(f"{path}.conditions.excluded_tokens", "must be a non-empty array of strings")
        if "speech_acts" in conditions:
            sa = conditions["speech_acts"]
            if not isinstance(sa, list) or not sa or not all(isinstance(t, str) and t for t in sa):
                _fail(f"{path}.conditions.speech_acts", "must be a non-empty array of strings")
        fallback = trigger.get("fallback_pool")
        if fallback is not None:
            if not isinstance(fallback, list) or not fallback:
                _fail(f"{path}.fallback_pool", "must be a non-empty array of strings")
            for j, entry in enumerate(fallback):
                if not isinstance(entry, str) or not entry:
                    _fail(f"{path}.fallback_pool[{j}]", "must be a non-empty string")


def load_uol_trigger_responses() -> dict[str, Any]:
    data = load_contract_json("uol_trigger_responses.v1.json")
    validate_uol_trigger_responses(data)
    return data


def validate_personal_memory_evidence_map(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.personal_memory_evidence_map.v1":
        _fail("$.schema_id", "must equal 'melm.personal_memory_evidence_map.v1'")
    rules = payload.get("rules")
    if not isinstance(rules, list) or not rules:
        _fail("$.rules", "must be a non-empty array")
    for i, r in enumerate(rules):
        if not isinstance(r, dict):
            _fail(f"$.rules[{i}]", "must be an object")
        cond = r.get("condition")
        if not isinstance(cond, str) or not cond:
            _fail(f"$.rules[{i}].condition", "must be a non-empty string")
        res = r.get("result")
        if not isinstance(res, str) or not res:
            _fail(f"$.rules[{i}].result", "must be a non-empty string")


def load_personal_memory_evidence_map() -> list[dict[str, str]]:
    data = load_contract_json("personal_memory_evidence_map.v1.json")
    validate_personal_memory_evidence_map(data)
    return list(data["rules"])


def validate_music_discovery_verbs(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.music_discovery_verbs.v1":
        _fail("$.schema_id", "must equal 'melm.music_discovery_verbs.v1'")
    for key in ("discovery_verbs", "music_tokens"):
        arr = payload.get(key)
        if not isinstance(arr, list) or not arr:
            _fail(f"$.{key}", "must be a non-empty array")
        for i, v in enumerate(arr):
            if not isinstance(v, str) or not v:
                _fail(f"$.{key}[{i}]", "must be a non-empty string")


def load_music_discovery_verbs() -> tuple[frozenset[str], frozenset[str]]:
    data = load_contract_json("music_discovery_verbs.v1.json")
    validate_music_discovery_verbs(data)
    return (
        frozenset(data["discovery_verbs"]),
        frozenset(data["music_tokens"]),
    )


def validate_safety_school_terms(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.safety_school_terms.v1":
        _fail("$.schema_id", "must equal 'melm.safety_school_terms.v1'")
    for key in ("nudity_terms", "school_clothing_terms"):
        arr = payload.get(key)
        if not isinstance(arr, list) or not arr:
            _fail(f"$.{key}", "must be a non-empty array")
        for i, v in enumerate(arr):
            if not isinstance(v, str) or not v:
                _fail(f"$.{key}[{i}]", "must be a non-empty string")
    om = payload.get("object_mapping")
    if not isinstance(om, dict):
        _fail("$.object_mapping", "must be an object")


def load_safety_school_terms() -> dict[str, Any]:
    data = load_contract_json("safety_school_terms.v1.json")
    validate_safety_school_terms(data)
    return data


def validate_media_object_tokens(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.media_object_tokens.v1":
        _fail("$.schema_id", "must equal 'melm.media_object_tokens.v1'")
    dt = payload.get("direct_tokens")
    if not isinstance(dt, dict) or not dt:
        _fail("$.direct_tokens", "must be a non-empty object")
    for key, val in dt.items():
        if not isinstance(key, str) or not key:
            _fail("$.direct_tokens", "key must be a non-empty string")
        if not isinstance(val, str):
            _fail(f"$.direct_tokens.{key}", "must be a string")
    ft = payload.get("fallback_tokens")
    if not isinstance(ft, list):
        _fail("$.fallback_tokens", "must be an array")
    for i, v in enumerate(ft):
        if not isinstance(v, str):
            _fail(f"$.fallback_tokens[{i}]", "must be a string")
    for field in ("fallback_object", "default_object"):
        val = payload.get(field)
        if not isinstance(val, str) or not val:
            _fail(f"$.{field}", "must be a non-empty string")


def load_media_object_tokens() -> dict[str, Any]:
    data = load_contract_json("media_object_tokens.v1.json")
    validate_media_object_tokens(data)
    return data


def validate_child_memory_markers(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.child_memory_markers.v1":
        _fail("$.schema_id", "must equal 'melm.child_memory_markers.v1'")
    bm = payload.get("base_markers")
    if not isinstance(bm, list) or not bm:
        _fail("$.base_markers", "must be a non-empty array")
    for i, v in enumerate(bm):
        if not isinstance(v, str) or not v:
            _fail(f"$.base_markers[{i}]", "must be a non-empty string")
    sm = payload.get("suffix_mapping")
    if not isinstance(sm, dict):
        _fail("$.suffix_mapping", "must be an object")
    ds = payload.get("default_suffix")
    if not isinstance(ds, str) or not ds:
        _fail("$.default_suffix", "must be a non-empty string")


def load_child_memory_markers() -> dict[str, Any]:
    data = load_contract_json("child_memory_markers.v1.json")
    validate_child_memory_markers(data)
    return data


def validate_contact_enrichment_templates(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.contact_enrichment_templates.v1":
        _fail("$.schema_id", "must equal 'melm.contact_enrichment_templates.v1'")
    templates = payload.get("templates")
    if not isinstance(templates, dict) or not templates:
        _fail("$.templates", "must be a non-empty object")
    for key, val in templates.items():
        if not isinstance(val, str) or not val:
            _fail(f"$.templates.{key}", "must be a non-empty string")
    if "default" not in templates:
        _fail("$.templates", "must include a 'default' entry")


def load_contact_enrichment_templates() -> dict[str, str]:
    data = load_contract_json("contact_enrichment_templates.v1.json")
    validate_contact_enrichment_templates(data)
    return dict(data["templates"])


def validate_contact_object_tokens(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.contact_object_tokens.v1":
        _fail("$.schema_id", "must equal 'melm.contact_object_tokens.v1'")
    rt = payload.get("relationship_tokens")
    if not isinstance(rt, list) or not rt:
        _fail("$.relationship_tokens", "must be a non-empty array")
    for i, v in enumerate(rt):
        if not isinstance(v, str) or not v:
            _fail(f"$.relationship_tokens[{i}]", "must be a non-empty string")
    for field in ("relationship_object", "someone_token", "default_object"):
        val = payload.get(field)
        if not isinstance(val, str) or not val:
            _fail(f"$.{field}", "must be a non-empty string")


def load_contact_object_tokens() -> dict[str, Any]:
    data = load_contract_json("contact_object_tokens.v1.json")
    validate_contact_object_tokens(data)
    return data


def validate_lesson_keywords(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.lesson_keywords.v1":
        _fail("$.schema_id", "must equal 'melm.lesson_keywords.v1'")
    kws = payload.get("keywords")
    if not isinstance(kws, list) or len(kws) < 2:
        _fail("$.keywords", "must be a non-empty list of strings")
    for kw in kws:
        if not isinstance(kw, str) or not kw:
            _fail(f"$.keywords[{kws.index(kw) if kw in kws else -1}]", "must be a non-empty string")


def load_lesson_keywords() -> list[str]:
    data = load_contract_json("lesson_keywords.v1.json")
    validate_lesson_keywords(data)
    return list(data["keywords"])


def validate_literary_device_map(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.literary_device_map.v1":
        _fail("$.schema_id", "must equal 'melm.literary_device_map.v1'")
    groups = payload.get("lesson_device_groups")
    if not isinstance(groups, dict) or not groups:
        _fail("$.lesson_device_groups", "must be a non-empty object")
    for device, lessons in groups.items():
        if not isinstance(lessons, list) or not lessons:
            _fail(f"$.lesson_device_groups.{device}", "must be a non-empty list of strings")
    if not isinstance(payload.get("default_device"), str):
        _fail("$.default_device", "must be a string")
    if not isinstance(payload.get("theme_default_device"), str):
        _fail("$.theme_default_device", "must be a string")


def load_literary_device_map() -> dict[str, Any]:
    data = load_contract_json("literary_device_map.v1.json")
    validate_literary_device_map(data)
    return {"lesson_device_groups": dict(data["lesson_device_groups"]), "default_device": str(data["default_device"]), "theme_default_device": str(data["theme_default_device"])}


def validate_story_pipeline_prompts(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.story_pipeline_prompts.v1":
        _fail("$.schema_id", "must equal 'melm.story_pipeline_prompts.v1'")
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        _fail("$.stages", "must be an object")
    required_stage_keys = {"phase", "temperature", "max_tokens", "system_prompt"}
    for stage_id, stage in stages.items():
        if not isinstance(stage, dict):
            _fail(f"$.stages.{stage_id}", "must be an object")
        for key in required_stage_keys:
            if key not in stage:
                _fail(f"$.stages.{stage_id}.{key}", "is required")
        if stage.get("phase") not in ("planning", "generation"):
            _fail(f"$.stages.{stage_id}.phase", "must be 'planning' or 'generation'")
        if not isinstance(stage.get("temperature"), (int, float)):
            _fail(f"$.stages.{stage_id}.temperature", "must be a number")
        if not isinstance(stage.get("max_tokens"), int):
            _fail(f"$.stages.{stage_id}.max_tokens", "must be an integer")
        if not isinstance(stage.get("system_prompt"), str):
            _fail(f"$.stages.{stage_id}.system_prompt", "must be a string")


def load_story_pipeline_prompts() -> dict[str, Any]:
    payload = load_contract_json("story_pipeline_prompts.v1.json")
    validate_story_pipeline_prompts(payload)
    return payload


def validate_storytelling_phrases(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.storytelling_phrases.v1":
        _fail("$.schema_id", "must equal 'melm.storytelling_phrases.v1'")
    cultures = payload.get("cultures")
    if not isinstance(cultures, dict) or not cultures:
        _fail("$.cultures", "must be a non-empty object")
    required_functions = {"openings", "transitions", "closings", "nature_descriptions",
                          "character_descriptions", "exaggerations", "emotional_beats", "moral_framings"}
    for cname, cat in cultures.items():
        if not isinstance(cat, dict):
            _fail(f"$.cultures.{cname}", "must be an object")
        for func in required_functions:
            if func not in cat:
                _fail(f"$.cultures.{cname}", f"missing required function '{func}'")
            phrases = cat[func]
            if not isinstance(phrases, list) or len(phrases) < 3:
                _fail(f"$.cultures.{cname}.{func}", "must be a list with 3+ entries")
            for i, phr in enumerate(phrases):
                if not isinstance(phr, str) or not phr.strip():
                    _fail(f"$.cultures.{cname}.{func}[{i}]", "must be a non-empty string")


def load_storytelling_phrases() -> dict[str, Any]:
    payload = load_contract_json("storytelling_phrases.v1.json")
    validate_storytelling_phrases(payload)
    return payload


def validate_story_scene_templates(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.story_scene_templates.v1":
        _fail("$.schema_id", "must equal 'melm.story_scene_templates.v1'")
    archetypes = payload.get("archetypes", [])
    if not archetypes:
        _fail("$.archetypes", "must be a non-empty array")
    for i, a in enumerate(archetypes):
        if "archetype_id" not in a:
            _fail(f"$.archetypes[{i}]", "missing archetype_id")
        if "entity_slots" not in a or not a["entity_slots"]:
            _fail(f"$.archetypes[{i}]", "missing or empty entity_slots")
        if "atom_sequence" not in a or not a["atom_sequence"]:
            _fail(f"$.archetypes[{i}]", "missing or empty atom_sequence")
        for j, s in enumerate(a.get("entity_slots", [])):
            if "role" not in s:
                _fail(f"$.archetypes[{i}].entity_slots[{j}]", "missing role")
            if "allowed_classes" not in s:
                _fail(f"$.archetypes[{i}].entity_slots[{j}]", "missing allowed_classes")
    if "topic_to_entity_classes" not in payload:
        _fail("$.topic_to_entity_classes", "missing required field")
    if "story_arcs" not in payload:
        _fail("$.story_arcs", "missing required field")


def load_story_scene_templates() -> dict[str, Any]:
    payload = load_contract_json("story_scene_templates.v1.json")
    validate_story_scene_templates(payload)
    return payload


def validate_folk_tales(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.folk_tales.v1":
        _fail("$.schema_id", "must equal 'melm.folk_tales.v1'")
    stories = payload.get("stories")
    if not isinstance(stories, list) or len(stories) < 1:
        _fail("$.stories", "must be a non-empty array")
    for i, story in enumerate(stories):
        if not isinstance(story, dict):
            _fail(f"$.stories[{i}]", "must be an object")
        if "title" not in story or not isinstance(story["title"], str):
            _fail(f"$.stories[{i}].title", "must be a non-empty string")
        if "text" not in story or not isinstance(story["text"], str) or len(story["text"]) < 50:
            _fail(f"$.stories[{i}].text", "must be a string with 50+ chars")


def load_folk_tales() -> dict[str, Any]:
    payload = load_contract_json("folk_tales.v1.json")
    validate_folk_tales(payload)
    return payload


def validate_semantic_attention_rules(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.semantic_attention_rules.v1":
        _fail("$.schema_id", "must equal 'melm.semantic_attention_rules.v1'")
    for key in ("response_artifact_terms", "technical_token_terms", "reasoning_cues", "stopwords"):
        val = payload.get(key)
        if not isinstance(val, list) or not val:
            _fail(f"$.{key}", "must be a non-empty array")
        for i, v in enumerate(val):
            if not isinstance(v, str) or not v:
                _fail(f"$.{key}[{i}]", "must be a non-empty string")
    ott = payload.get("output_type_terms")
    if not isinstance(ott, dict) or not ott:
        _fail("$.output_type_terms", "must be a non-empty object")
    for key, val in ott.items():
        if not isinstance(key, str) or not key:
            _fail("$.output_type_terms", f"key {key!r} must be a non-empty string")
        if not isinstance(val, str) or not val:
            _fail(f"$.output_type_terms.{key}", "must be a non-empty string")


def load_semantic_attention_rules() -> dict[str, Any]:
    payload = load_contract_json("semantic_attention_rules.v1.json")
    validate_semantic_attention_rules(payload)
    return payload


def validate_nlg_atomic_renderers(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.nlg_atomic_renderers.v1":
        _fail("$.schema_id", "must equal 'melm.nlg_atomic_renderers.v1'")
    families = payload.get("renderer_families")
    if not isinstance(families, dict) or not families:
        _fail("$.renderer_families", "must be a non-empty object")
    for fname, family in families.items():
        fpath = f"$.renderer_families.{fname}"
        if not isinstance(family, dict):
            _fail(fpath, "must be an object")
        for key in ("description", "priority", "required_conditions", "forbidden_conditions", "templates"):
            if key not in family:
                _fail(fpath, f"missing required property {key!r}")
        if not isinstance(family.get("description"), str):
            _fail(f"{fpath}.description", "must be a string")
        if not isinstance(family.get("priority"), int):
            _fail(f"{fpath}.priority", "must be an integer")
        if not isinstance(family.get("required_conditions"), dict):
            _fail(f"{fpath}.required_conditions", "must be an object")
        if not isinstance(family.get("forbidden_conditions"), dict):
            _fail(f"{fpath}.forbidden_conditions", "must be an object")
        templates = family.get("templates")
        if not isinstance(templates, list) or not templates:
            _fail(f"{fpath}.templates", "must be a non-empty array")
        for i, t in enumerate(templates):
            if not isinstance(t, str) or not t:
                _fail(f"{fpath}.templates[{i}]", "must be a non-empty string")


def load_nlg_atomic_renderers() -> dict[str, Any]:
    payload = load_contract_json("nlg_atomic_renderers.v1.json")
    validate_nlg_atomic_renderers(payload)
    return payload


def validate_nlg_fallback_phrases(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.nlg_fallback_phrases.v1":
        _fail("$.schema_id", "must equal 'melm.nlg_fallback_phrases.v1'")
    for section in ("entity_nlg_templates", "refusal_templates", "social_status_templates", "safety_school_templates", "music_templates", "story_verb_tenses", "story_sentence_patterns"):
        if section not in payload:
            _fail(f"$.{section}", "is required")
    et = payload["entity_nlg_templates"]
    if not isinstance(et, dict) or "fragile" not in et:
        _fail("$.entity_nlg_templates.fragile", "is required")
    ref = payload["refusal_templates"]
    if not isinstance(ref, dict) or "default" not in ref:
        _fail("$.refusal_templates.default", "is required")
    soc = payload["social_status_templates"]
    if not isinstance(soc, dict) or "default" not in soc:
        _fail("$.social_status_templates.default", "is required")
    saf = payload["safety_school_templates"]
    if not isinstance(saf, dict) or "weather_policy" not in saf:
        _fail("$.safety_school_templates.weather_policy", "is required")
    mus = payload["music_templates"]
    if not isinstance(mus, dict) or "success" not in mus or "failure" not in mus:
        _fail("$.music_templates", "must have success and failure keys")
    tenses = payload["story_verb_tenses"]
    if not isinstance(tenses, dict) or "walk" not in tenses:
        _fail("$.story_verb_tenses", "must have at least 'walk' entry")
    patterns = payload["story_sentence_patterns"]
    if not isinstance(patterns, list) or len(patterns) < 4:
        _fail("$.story_sentence_patterns", "must have at least 4 patterns")
    for i, p in enumerate(patterns):
        if not isinstance(p, str) or not p:
            _fail(f"$.story_sentence_patterns[{i}]", "must be a non-empty string")


def load_nlg_fallback_phrases() -> dict[str, Any]:
    data = load_contract_json("nlg_fallback_phrases.v1.json")
    validate_nlg_fallback_phrases(data)
    return data


def validate_atom_intent_predicates(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.atom_intent_predicates.v1":
        _fail("$.schema_id", "must equal 'melm.atom_intent_predicates.v1'")
    ip = payload.get("intent_predicates")
    if not isinstance(ip, dict) or not ip:
        _fail("$.intent_predicates", "must be a non-empty object")
    for key, arr in ip.items():
        if not isinstance(key, str) or not key:
            _fail("$.intent_predicates", f"key {key!r} must be a non-empty string")
        if not isinstance(arr, list) or not arr:
            _fail(f"$.intent_predicates.{key}", "must be a non-empty list")
        for i, v in enumerate(arr):
            if not isinstance(v, str) or not v:
                _fail(f"$.intent_predicates.{key}[{i}]", "must be a non-empty string")


def load_atom_intent_predicates() -> dict[str, list[str]]:
    data = load_contract_json("atom_intent_predicates.v1.json")
    validate_atom_intent_predicates(data)
    return {k: list(v) for k, v in data["intent_predicates"].items()}


def validate_private_cloud_evidence_map(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.private_cloud_evidence_map.v1":
        _fail("$.schema_id", "must equal 'melm.private_cloud_evidence_map.v1'")
    if not isinstance(payload.get("default_key"), str) or not payload["default_key"]:
        _fail("$.default_key", "must be a non-empty string")
    for section in ("evidence_groups", "child_groups"):
        arr = payload.get(section)
        if not isinstance(arr, list):
            _fail(f"$.{section}", "must be a list")
        for i, g in enumerate(arr):
            if not isinstance(g, dict):
                _fail(f"$.{section}[{i}]", "must be an object")
            if not isinstance(g.get("key"), str) or not g["key"]:
                _fail(f"$.{section}[{i}].key", "must be a non-empty string")
            match = g.get("match")
            if match not in ("any", "all", "any_group", "any_or_all", "all_with_any"):
                _fail(f"$.{section}[{i}].match", "must be 'any', 'all', 'any_group', 'any_or_all', or 'all_with_any'")
    ct = payload.get("child_triggers")
    if not isinstance(ct, list) or not ct:
        _fail("$.child_triggers", "must be a non-empty array")
    for i, v in enumerate(ct):
        if not isinstance(v, str) or not v:
            _fail(f"$.child_triggers[{i}]", "must be a non-empty string")


def load_private_cloud_evidence_map() -> dict[str, Any]:
    data = load_contract_json("private_cloud_evidence_map.v1.json")
    validate_private_cloud_evidence_map(data)
    return data


def validate_revoked_fact_markers(payload: dict[str, Any]) -> None:
    if payload.get("schema_id") != "melm.revoked_fact_markers.v1":
        _fail("$.schema_id", "must equal 'melm.revoked_fact_markers.v1'")
    fm = payload.get("forget_markers")
    if not isinstance(fm, list) or not fm:
        _fail("$.forget_markers", "must be a non-empty array")
    for i, v in enumerate(fm):
        if not isinstance(v, str) or not v:
            _fail(f"$.forget_markers[{i}]", "must be a non-empty string")
    sm = payload.get("simple_markers")
    if not isinstance(sm, dict):
        _fail("$.simple_markers", "must be an object")
    for key, arr in sm.items():
        if not isinstance(key, str) or not key:
            _fail("$.simple_markers", f"key {key!r} must be a non-empty string")
        if not isinstance(arr, list) or not arr:
            _fail(f"$.simple_markers.{key}", "must be a non-empty array")
        for i, v in enumerate(arr):
            if not isinstance(v, str) or not v:
                _fail(f"$.simple_markers.{key}[{i}]", "must be a non-empty string")
    cpm = payload.get("child_parent_markers")
    if not isinstance(cpm, list) or not cpm:
        _fail("$.child_parent_markers", "must be a non-empty array")
    for i, v in enumerate(cpm):
        if not isinstance(v, str) or not v:
            _fail(f"$.child_parent_markers[{i}]", "must be a non-empty string")
    cm = payload.get("child_markers")
    if not isinstance(cm, dict) or not cm:
        _fail("$.child_markers", "must be a non-empty object")
    for mkey, val in cm.items():
        if not isinstance(mkey, str) or not mkey:
            _fail("$.child_markers", f"key {mkey!r} must be a non-empty string")
        if not isinstance(val, str) or not val:
            _fail(f"$.child_markers.{mkey}", "must be a non-empty string")
    if not isinstance(payload.get("child_default"), str) or not payload["child_default"]:
        _fail("$.child_default", "must be a non-empty string")
    hm = payload.get("household_markers")
    if not isinstance(hm, list) or not hm:
        _fail("$.household_markers", "must be a non-empty array")
    for i, v in enumerate(hm):
        if not isinstance(v, str) or not v:
            _fail(f"$.household_markers[{i}]", "must be a non-empty string")
    if not isinstance(payload.get("household_key"), str) or not payload["household_key"]:
        _fail("$.household_key", "must be a non-empty string")


def load_revoked_fact_markers() -> dict[str, Any]:
    data = load_contract_json("revoked_fact_markers.v1.json")
    validate_revoked_fact_markers(data)
    return data
