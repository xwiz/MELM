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
    "personal_goal_advice", "open_domain",
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
                    full_content = (CONTRACT_ROOT / artifact_path).read_bytes()
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


def load_story_components() -> dict[str, dict[str, str]]:
    payload = load_contract_json("story_components.v1.json")
    validate_story_components(payload)
    return {
        "images": dict(payload["images"]),
        "challenges": dict(payload["challenges"]),
        "lessons": dict(payload["lessons"]),
    }


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
    return dict(payload["identity_templates"])


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


def load_answer_templates() -> dict[str, Any]:
    payload = load_contract_json("answer_templates.v1.json")
    validate_answer_templates(payload)
    return dict(payload["intents"])


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


def load_open_domain_templates() -> dict[str, Any]:
    payload = load_contract_json("open_domain_templates.v1.json")
    validate_open_domain_templates(payload)
    return dict(payload["intents"])


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
    "equivalence", "politeness", "discourse_particle", "pronoun",
}

_VALID_SUBROLES = {
    "manner", "theme", "time", "location", "selection", "agent", "reason",
    "possibility", "necessity", "obligation", "future",
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
