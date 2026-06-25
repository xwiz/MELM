"""UOL-pattern trigger responses.

Detects UOL atom patterns that call for a randomized, mood-aware response
(e.g. defiant denials when the user asserts something negative about the
assistant). All trigger patterns and response pools live in the contract
``uol_trigger_responses.v1.json``; this module is a generic consumer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from melm.contracts import load_affect_lexicon, load_uol_trigger_responses

from .uol_types import AffectSignal


_VARIABLE_RE = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class UolTriggerMatch:
    """A UOL trigger that fired together with its matched atom variables."""

    trigger_id: str
    response_pool: tuple[str, ...]
    variables: dict[str, str]


def _uol_trigger_responses() -> dict[str, Any]:
    try:
        return load_uol_trigger_responses()
    except Exception:
        return {"triggers": []}


def _is_assistant_targeted(uol_act: dict[str, Any] | None) -> bool:
    """Return True when the assistant is the subject/target of the utterance.

    A default addressee of "assistant" is not enough — the UOL atom must
    reference the assistant as agent, theme, patient, or second-person predicate.
    """
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
    """Extract template variables from a single UOL atom dict."""
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
    """Return True if a UOL act satisfies a trigger's conditions.

    Gate conditions (``assistant_targeted``, ``speech_acts``,
    ``required_tokens``, ``excluded_tokens``) must all pass. After that, any
    matching condition (``polarity``, ``min_affect_valence``,
    ``negative_modifier``) is sufficient by default. Use ``all`` to require
    every matching condition.
    """
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
    """Fill placeholders, leaving missing variables as empty strings."""
    def repl(match: re.Match) -> str:
        return variables.get(match.group(1), "")
    return _VARIABLE_RE.sub(repl, template)


def _template_variables(template: str) -> frozenset[str]:
    return frozenset(_VARIABLE_RE.findall(template))


def _select_response(
    pool: tuple[str, ...],
    variables: dict[str, str],
    seed: int,
    *,
    nlg_template: str | None = None,
    nlg_enabled: bool = False,
) -> str:
    """Pick a deterministic-but-randomized response that can be filled.

    Templates that reference missing variables are skipped unless they are
    the only option, in which case missing variables are replaced with
    empty strings.
    """
    candidates: list[str] = []
    for template in pool:
        needed = _template_variables(template)
        if needed.issubset(variables.keys()):
            candidates.append(template)
    if not candidates:
        candidates = list(pool)
    if nlg_enabled and nlg_template:
        needed = _template_variables(nlg_template)
        if needed.issubset(variables.keys()):
            candidates.append(nlg_template)
    index = seed % len(candidates)
    return _fill_template(candidates[index], variables)


def _response_seed(trigger_id: str, utterance: str) -> int:
    raw = f"{trigger_id}:{utterance}"
    h = 0
    for ch in raw.encode("utf-8"):
        h = ((h << 5) - h) + ch
        h &= 0xFFFFFFFF
    return abs(h)


def detect_uol_trigger(
    uol_act: dict[str, Any] | None,
    affect: AffectSignal | None,
    tokens: tuple[str, ...],
) -> UolTriggerMatch | None:
    """Find the first trigger whose conditions match the UOL act.

    Triggers are evaluated in contract order. Only the first match is
    returned so the behavior is deterministic and easy to reason about.
    """
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
        responses = trigger.get("responses", {})
        pool = tuple(str(t) for t in responses.get("pool", []) if t)
        if not pool:
            continue
        return UolTriggerMatch(
            trigger_id=str(trigger.get("trigger_id", "")),
            response_pool=pool,
            variables=variables,
        )
    return None


def render_trigger_response(
    match: UolTriggerMatch,
    utterance: str,
    affect: AffectSignal | None,
    profile: Any,
) -> str:
    """Render a randomized response from the matched trigger pool."""
    seed = _response_seed(match.trigger_id, utterance)
    return _select_response(
        match.response_pool,
        match.variables,
        seed,
    )
