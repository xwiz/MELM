from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from melm.contracts import load_affect_lexicon, load_uol_trigger_responses

from .uol_types import AffectSignal


_VARIABLE_RE = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class UolTriggerMatch:
    trigger_id: str
    fallback_pool: tuple[str, ...]
    variables: dict[str, str]


def _uol_trigger_responses() -> dict[str, Any]:
    try:
        return load_uol_trigger_responses()
    except Exception:
        return {"triggers": []}


def _load_learned_responses(store: Any, trigger_id: str) -> list[dict[str, Any]]:
    if store is None:
        return []
    try:
        entities = store.find_entities(kind="learned_trigger_response")
        results = []
        for ent in entities:
            tid_slot = store.get_entity_slot(ent.entity_id, "trigger_id")
            if tid_slot is not None and tid_slot.value_json:
                try:
                    stored_tid = json.loads(tid_slot.value_json)
                except (json.JSONDecodeError, TypeError):
                    continue
                if stored_tid == trigger_id:
                    row = {"entity_id": ent.entity_id}
                    for slot_name in ("response_text", "variables", "source", "context_uol", "confidence", "use_count", "learned_at"):
                        slot = store.get_entity_slot(ent.entity_id, slot_name)
                        if slot is not None and slot.value_json:
                            try:
                                row[slot_name] = json.loads(slot.value_json)
                            except (json.JSONDecodeError, TypeError):
                                row[slot_name] = None
                        else:
                            row[slot_name] = None
                    results.append(row)
        return results
    except Exception:
        return []


def _seed_for_response(trigger_id: str, utterance: str, response_text: str) -> int:
    raw = f"{trigger_id}:{utterance}:{response_text}"
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def _is_assistant_targeted(uol_act: dict[str, Any] | None) -> bool:
    if uol_act is None:
        return False
    if uol_act.get("speaker") == "assistant":
        return False
    for atom in uol_act.get("content", []):
        if not isinstance(atom, dict):
            continue
        pred = atom.get("predicate", {}) or {}
        if pred.get("id") == "you" or pred.get("lemma") == "you":
            return True
        for role in atom.get("roles", []):
            if isinstance(role, dict) and role.get("value") == "assistant":
                return True
    return False


def _atom_variables(atom: dict[str, Any]) -> dict[str, str]:
    variables: dict[str, str] = {}
    pred = atom.get("predicate", {}) or {}
    if pred.get("id"):
        variables["predicate"] = str(pred["id"])
    if pred.get("lemma"):
        variables["predicate_lemma"] = str(pred["lemma"])
    for role in atom.get("roles", []):
        if not isinstance(role, dict):
            continue
        role_name = role.get("role")
        value = role.get("value")
        if role_name and value is not None:
            variables[role_name] = str(value)
    for modifier in atom.get("modifiers", []):
        if isinstance(modifier, dict) and modifier.get("lemma"):
            variables.setdefault("modifier", str(modifier["lemma"]))
            break
    return variables


def _merge_variables(uol_act: dict[str, Any] | None) -> dict[str, str]:
    if uol_act is None:
        return {}
    merged: dict[str, str] = {}
    for atom in uol_act.get("content", []):
        if isinstance(atom, dict):
            merged.update(_atom_variables(atom))
    return merged


def _affect_lexicon() -> dict[str, Any]:
    try:
        return load_affect_lexicon().get("entries", {})
    except Exception:
        return {}


def _modifier_valence(lemma: str) -> float:
    entry = _affect_lexicon().get(lemma.lower())
    if isinstance(entry, dict):
        return float(entry.get("valence", 0.0))
    return 0.0


def _has_negative_modifier(uol_act: dict[str, Any] | None) -> bool:
    if uol_act is None:
        return False
    for atom in uol_act.get("content", []):
        if not isinstance(atom, dict):
            continue
        for modifier in atom.get("modifiers", []):
            if isinstance(modifier, dict):
                lemma = str(modifier.get("lemma", "")).lower()
                if lemma and _modifier_valence(lemma) < 0:
                    return True
    return False


def _polarity_of_atom(uol_act: dict[str, Any] | None) -> str:
    if uol_act is None:
        return "positive"
    for atom in uol_act.get("content", []):
        if not isinstance(atom, dict):
            continue
        ctx = atom.get("context", {}) or {}
        if ctx.get("polarity") == "negative":
            return "negative"
        for role in atom.get("roles", []):
            if isinstance(role, dict) and role.get("role") == "polarity" and role.get("value") == "negative":
                return "negative"
    return "positive"


def _affect_valence(affect: AffectSignal | None) -> float:
    if affect is None:
        return 0.0
    return float(getattr(affect, "valence", 0.0))


def _condition_matches(
    conditions: dict[str, Any],
    uol_act: dict[str, Any] | None,
    affect: AffectSignal | None,
    tokens: tuple[str, ...],
) -> bool:
    if "assistant_targeted" in conditions:
        targeted = _is_assistant_targeted(uol_act)
        if targeted != bool(conditions["assistant_targeted"]):
            return False
    if "speech_acts" in conditions:
        act = str(uol_act.get("act", "")).strip().lower() if uol_act else ""
        if act not in {str(a).lower() for a in conditions["speech_acts"]}:
            return False
    if "required_tokens" in conditions:
        required = {str(t).lower() for t in conditions["required_tokens"]}
        token_set = {str(t).lower() for t in tokens}
        if not (required & token_set):
            return False
    if "excluded_tokens" in conditions:
        excluded = {str(t).lower() for t in conditions["excluded_tokens"]}
        token_set = {str(t).lower() for t in tokens}
        if excluded & token_set:
            return False

    checks: list[bool] = []
    if "polarity" in conditions:
        checks.append(_polarity_of_atom(uol_act) == str(conditions["polarity"]))
    if "min_affect_valence" in conditions:
        checks.append(_affect_valence(affect) <= float(conditions["min_affect_valence"]))
    if "negative_modifier" in conditions and conditions["negative_modifier"]:
        checks.append(_has_negative_modifier(uol_act))

    if not checks:
        return True
    operator = str(conditions.get("match_operator", "any")).lower()
    if operator == "all":
        return all(checks)
    return any(checks)


def _fill_template(template: str, variables: dict[str, str]) -> str:
    def repl(match: re.Match) -> str:
        return variables.get(match.group(1), "")
    return _VARIABLE_RE.sub(repl, template)


def detect_uol_trigger(
    uol_act: dict[str, Any] | None,
    affect: AffectSignal | None,
    tokens: tuple[str, ...],
) -> UolTriggerMatch | None:
    payload = _uol_trigger_responses()
    triggers = payload.get("triggers", [])
    if not triggers:
        return None
    variables = _merge_variables(uol_act)
    for trigger in triggers:
        if not isinstance(trigger, dict):
            continue
        conditions = trigger.get("conditions", {})
        if not isinstance(conditions, dict):
            continue
        if not _condition_matches(conditions, uol_act, affect, tokens):
            continue
        fallback = trigger.get("fallback_pool", [])
        fallback_pool = tuple(str(t) for t in fallback if t)
        return UolTriggerMatch(
            trigger_id=str(trigger.get("trigger_id", "")),
            fallback_pool=fallback_pool,
            variables=variables,
        )
    return None


def render_trigger_response(
    match: UolTriggerMatch,
    utterance: str,
    affect: AffectSignal | None,
    profile: Any,
    store: Any = None,
) -> str:
    learned = _load_learned_responses(store, match.trigger_id)
    if learned:
        for entry in learned:
            text = entry.get("response_text") or ""
            if text:
                return _fill_template(text, match.variables)

    if match.fallback_pool:
        seed = _seed_for_response(match.trigger_id, utterance, "fallback")
        idx = seed % len(match.fallback_pool)
        return _fill_template(match.fallback_pool[idx], match.variables)

    return "I disagree."
