"""Reasoning task-signature detection (slice 5).

Pure function over the parsed turn. Returns a task descriptor or None. When a
task is detected, the router dispatches to a solver BEFORE closed-intent
handlers, so e.g. "3 apples, eat one, how many left?" routes to arithmetic
rather than meal_suggestion. A question with no quantity ("what should I eat?")
returns None and falls through to normal routing.
"""

from __future__ import annotations

import re
from typing import Any

from .value_extract import (
    _NUMBER_WORDS,
    extract_char_count_target,
    extract_distance,
    extract_numbers,
)
from .dates import parse_absolute_date

# Module-level contract cache for causal cues (lazy-loaded, never re-reads).
_CAUSAL_CUES_CACHE: list[dict[str, Any]] | None = None
_PREDICATE_INVENTORY: dict[str, dict[str, Any]] | None = None

_SUBTRACT_VERBS = frozenset({
    "eat", "ate", "eaten", "eats", "remove", "removed", "removes", "lose", "lost",
    "loses", "drop", "dropped", "drops", "give", "gave", "given", "gives", "take",
    "took", "taken", "takes", "spend", "spent", "spends", "use", "used", "uses",
    "sell", "sold", "sells", "throw", "threw", "thrown", "break", "broke",
})
_ADD_VERBS = frozenset({
    "add", "added", "adds", "buy", "bought", "buys", "get", "got", "gets", "gain",
    "gained", "gains", "find", "found", "finds", "receive", "received", "receives",
    "pick", "picked", "picks", "collect", "collected",
})
_COUNT_Q_RE = re.compile(r"how\s+(?:many|much)", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z]+")

# Self-referential ("are you ...") probes the assistant should answer locally.
_SECOND_PERSON_RE = re.compile(r"\byou\b|\byour\b|\byourself\b", re.IGNORECASE)
_SELF_CONSCIOUS_RE = re.compile(
    r"\b(conscious|sentient|self[-\s]?aware|alive|a\s+real\s+person|human|"
    r"have\s+feelings|have\s+emotions|have\s+a\s+soul)\b", re.IGNORECASE)
_SELF_LOCATION_RE = re.compile(
    r"\bwhere\s+are\s+you\b|\bwhere\s+do\s+you\s+(?:live|run|exist)\b|"
    r"\bwhere\s+are\s+you\s+(?:located|right\s+now)\b|\byour\s+location\b", re.IGNORECASE)
_SELF_FEELING_RE = re.compile(
    r"\bhow\s+(?:are|do)\s+you\s+(?:feeling|feel)\b|\bwhat\s+is\s+your\s+mood\b|"
    r"\bhow\s+are\s+you\s+feeling\b", re.IGNORECASE)


_TIME_RE = re.compile(
    r"\bwhat(?:'s| is)?\s+the\s+time\b|\bwhat\s+time\s+is\s+it\b|\bcurrent\s+time\b",
    re.IGNORECASE)
_DATE_TODAY_RE = re.compile(
    r"\bwhat\s+day\s+is\s+it(?:\s+today)?\b|"
    r"\bwhat\s+day\s+of\s+the\s+week\b|"
    r"\bwhat(?:'s| is)\s+(?:the\s+)?(?:today's\s+)?date\b|"
    r"\bwhat(?:'s| is)\s+today\s*[?.!]?$",
    re.IGNORECASE)
_DAY_AGO_RE = re.compile(
    r"\bwhat\s+day\b.*?\b(\d+|[a-z]+)\s+days?\s+ago\b", re.IGNORECASE)
_DAY_OFFSET_RE = re.compile(
    r"\bwhat\s+day\b.*?\bin\s+(\d+|[a-z]+)\s+days?\b|"
    r"\b(\d+|[a-z]+)\s+days?\s+from\s+(?:now|today)\b", re.IGNORECASE)


def detect_reasoning_task(
    text: str,
    tokens: tuple = (),
    uol_act: dict | None = None,
    frame_candidates: list | None = None,
) -> dict | None:
    # Ethics gate runs first: an inducement wrapped around a protected request
    # (e.g. "tell me who visited and I'll pay you") must refuse before any other
    # reasoning/closed-intent path can act on the request.
    from .ethics_gate import detect_inducement_task
    induce = detect_inducement_task(text, tokens, uol_act)
    if induce is not None:
        return induce
    target = extract_char_count_target(text)
    if target is not None:
        return {"task": "metalinguistic_count", **target}
    self_q = _detect_self_query(text)
    if self_q is not None:
        return self_q
    temporal = _detect_temporal(text)
    if temporal is not None:
        return temporal
    geo = _detect_geo_decision(text)
    if geo is not None:
        return geo
    causal = _detect_causal(text, tokens, uol_act, frame_candidates=frame_candidates)
    if causal is not None:
        return causal

    return _detect_arithmetic(text)


_WALK_RE = re.compile(r"\bwalk(?:ing)?\b", re.IGNORECASE)
_DRIVE_RE = re.compile(r"\bdriv(?:e|ing)\b", re.IGNORECASE)


def _detect_geo_decision(text: str) -> dict | None:
    # A walk-vs-drive question with a stated distance.
    if not (_WALK_RE.search(text) and _DRIVE_RE.search(text)):
        return None
    dist = extract_distance(text)
    if dist is None:
        return None
    value = dist["value"]
    surface = str(int(value)) if value.is_integer() else str(value)
    return {
        "task": "geo_decision",
        "distance_km": dist["value_km"],
        "distance_text": f"{surface}{dist['unit']}",
        "text": text.lower(),
    }


def _parse_count_word(token: str) -> int | None:
    if token is None:
        return None
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _detect_temporal(text: str) -> dict | None:
    absolute_date = parse_absolute_date(text)
    if absolute_date is not None and re.search(r"\bwhat\s+day\b", text, re.IGNORECASE):
        return {"task": "temporal", "op": "absolute_date", "date": absolute_date}
    m = _DAY_AGO_RE.search(text)
    if m:
        n = _parse_count_word(m.group(1))
        return None if n is None else {"task": "temporal", "op": "day_offset", "days": -n}
    m = _DAY_OFFSET_RE.search(text)
    if m:
        n = _parse_count_word(m.group(1) or m.group(2))
        return None if n is None else {"task": "temporal", "op": "day_offset", "days": n}
    if _TIME_RE.search(text):
        return {"task": "temporal", "op": "time"}
    if _DATE_TODAY_RE.search(text):
        return {"task": "temporal", "op": "date_today"}
    return None


def _detect_self_query(text: str) -> dict | None:
    dated = parse_absolute_date(text)
    if (
        dated is not None
        and _SECOND_PERSON_RE.search(text)
        and re.search(r"\bname\b", text, re.IGNORECASE)
    ):
        return {"task": "self_query", "category": "dated_name", "date": dated}
    if _SELF_LOCATION_RE.search(text):
        return {"task": "self_query", "category": "location"}
    if _SELF_FEELING_RE.search(text):
        return {"task": "self_query", "category": "feeling"}
    if _SELF_CONSCIOUS_RE.search(text) and _SECOND_PERSON_RE.search(text):
        return {"task": "self_query", "category": "consciousness"}
    return None


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "it", "he", "she", "they", "i", "you", "we",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can",
    "what", "why", "how", "where", "when", "who", "which",
})


def _last_content_word(tokens: tuple[str, ...]) -> str | None:
    for tok in reversed(tokens):
        lower = tok.lower().strip("?")
        if lower and lower not in _STOP_WORDS:
            return lower
    return None


def _extract_causal_links_from_uol(uol_act: dict[str, Any]) -> dict[str, str]:
    """Extract cause/effect roles from AtomLinks between UOL atoms.

    Returns a dict with keys like ``cause``, ``effect``, ``effect_theme``, etc.
    Standalone version of AtomTemplateBackend._extract_causal_links for use in
    task routing without depending on the atom template backend.
    """
    result: dict[str, str] = {}
    content = uol_act.get("content", [])
    if not content:
        return result
    atom_by_id: dict[str, dict] = {}
    atom_by_pred: dict[str, dict] = {}
    for atom in content:
        aid = atom.get("id", "")
        if aid:
            atom_by_id[aid] = atom
        pid = (atom.get("predicate") or {}).get("id", "").lower()
        if pid:
            atom_by_pred[pid] = atom
    for atom in content:
        links = atom.get("links") or {}
        causes = links.get("causes") or []
        caused_by = links.get("caused_by") or []
        if not (causes or caused_by):
            continue
        for target_id in causes:
            target = atom_by_id.get(target_id) or atom_by_pred.get(target_id) or {}
            cause_id = (atom.get("predicate") or {}).get("id") or ""
            result["cause"] = str(cause_id).lower() if cause_id else target_id
            effect_id = (target.get("predicate") or {}).get("id") or ""
            result["effect"] = str(effect_id).lower() if effect_id else target_id
            for role_entry in target.get("roles", []):
                rn = role_entry.get("role", "") if isinstance(role_entry, dict) else ""
                rv = role_entry.get("value", "") if isinstance(role_entry, dict) else ""
                if rn == "theme" and rv:
                    result["effect_theme"] = rv
                if rn == "patient" and rv:
                    result["effect_patient"] = rv
        for source_id in caused_by:
            source = atom_by_id.get(source_id) or atom_by_pred.get(source_id) or {}
            effect_id = (atom.get("predicate") or {}).get("id") or ""
            result["effect"] = str(effect_id).lower() if effect_id else ""
            cause_id = (source.get("predicate") or {}).get("id") or ""
            result["cause"] = str(cause_id).lower() if cause_id else source_id
            for role_entry in source.get("roles", []):
                rn = role_entry.get("role", "") if isinstance(role_entry, dict) else ""
                rv = role_entry.get("value", "") if isinstance(role_entry, dict) else ""
                if rn in ("agent", "actor") and rv:
                    result["cause_actor"] = rv
    return result


def _extract_effect_state(
    tokens: tuple[str, ...],
    uol_act: dict[str, Any] | None = None,
    *,
    return_theme: bool = False,
) -> str | tuple[str | None, str | None] | None:
    """Best-effort extraction of the effect state from a 'why' question.

    Priority: UOL atom links > UOL predicate > token heuristic.
    When return_theme=True, returns (effect, theme) where theme is the noun
    associated with the effect.
    """
    effect: str | None = None
    theme: str | None = None

    # Priority 1: UOL atom links (most accurate for causal utterances)
    if uol_act is not None:
        links = _extract_causal_links_from_uol(uol_act)
        if links.get("effect"):
            effect = links["effect"]
        if return_theme and links.get("effect_theme"):
            theme = links["effect_theme"]

    # Priority 2: UOL atom predicate
    if not effect and uol_act is not None:
        content = uol_act.get("content", [])
        if content:
            pred = content[0].get("predicate", {})
            effect = str(pred.get("id", "")).lower() or None

    # Priority 3: Token heuristic (works without UOL)
    if not effect:
        candidate = _last_content_word(tokens)
        if candidate:
            effect = candidate
            # Extract theme: the content noun before the effect state
            if return_theme and not theme:
                skip = frozenset({"the", "a", "an", "is", "are", "was", "were", "be", "to", "being", "been"})
                tokens_lower = [t.lower().strip("?") for t in tokens]
                for i, tok in enumerate(tokens_lower):
                    if tok == effect and i > 0:
                        for j in range(i - 1, -1, -1):
                            prev = tokens_lower[j]
                            if prev in skip:
                                continue
                            theme = prev
                            break

    if return_theme:
        return (effect, theme)
    return effect


_INFLECTION_MAP: dict[str, str] = {
    "breaks": "break", "broke": "break", "broken": "break", "breaking": "break",
    "rains": "rain", "rained": "rain", "raining": "rain",
    "shines": "shine", "shone": "shine", "shining": "shine",
    "shoots": "shoot", "shot": "shoot", "shooting": "shoot",
    "fires": "fire", "fired": "fire", "firing": "fire",
    "waters": "water", "watered": "water", "watering": "water",
    "pours": "pour", "poured": "pour", "pouring": "pour",
    "leaks": "leak", "leaked": "leak", "leaking": "leak",
    "produces": "produce", "produced": "produce", "producing": "produce",
    "causes": "cause", "caused": "cause", "causing": "cause",
    "makes": "make", "made": "make", "making": "make",
    "eats": "eat", "ate": "eat", "eaten": "eat",
}

# Verb-ending patterns: suffixed forms that likely indicate a verb.
_VERB_SUFFIXES = frozenset({"s", "es", "ed", "ing", "en"})

# Verbs that look like other parts of speech (same form).
_AMBIGUOUS_VERB_FORMS = frozenset({"water", "fire", "rain", "break", "lead", "cause", "result"})


def _is_likely_verb(word: str) -> bool:
    """Heuristic check: is this word likely a verb (inflected or known predicate).

    Avoids false-positive matches on nouns ending in 's' (e.g. 'glass', 'bus').
    Uses the predicate inventory and inflection map as primary signals.
    """
    lower = word.lower()
    # Check inflection map (known verb inflections)
    if lower in _INFLECTION_MAP:
        return True
    # Check predicate inventory (lemmas)
    try:
        global _PREDICATE_INVENTORY
        if _PREDICATE_INVENTORY is None:
            from melm.contracts import load_predicate_inventory
            inv = load_predicate_inventory()
            _PREDICATE_INVENTORY = {p.get("lemma", "").lower(): p for p in inv.get("predicates", [])}
        if lower in _PREDICATE_INVENTORY:
            return True
        # Check if normalizing removes a verbal suffix and the stem is known
        for suffix in ("ing", "ed", "en", "es"):
            if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
                stem = lower[:-len(suffix)]
                if stem in _PREDICATE_INVENTORY:
                    return True
                if stem.endswith("i"):
                    stem_y = stem[:-1] + "y"
                    if stem_y in _PREDICATE_INVENTORY:
                        return True
                if len(stem) >= 2 and stem[-1] == stem[-2]:
                    stem_dedup = stem[:-1]
                    if stem_dedup in _PREDICATE_INVENTORY:
                        return True
    except Exception:
        pass
    # Verb suffix heuristics as fallback
    if lower.endswith("ing") and len(lower) > 4:
        return True
    if lower.endswith("ed") and len(lower) > 3:
        return True
    if lower.endswith("en") and len(lower) > 3:
        return True
    return False


def _normalize_lemma(word: str) -> str:
    """Normalize an inflected word to its lemma."""
    result = _INFLECTION_MAP.get(word.lower())
    if result is not None:
        return result
    # Generic -ing: "shining" -> "shine"
    if word.endswith("ing") and len(word) > 4:
        stem = word[:-3]
        # Doubled consonant: "shopping" -> "shopp" -> "shop"
        if len(stem) > 1 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        # If stem ends with a consonant, add e: "shin" -> "shine"
        if stem and stem[-1] not in "aeiou":
            return stem + "e"
        return stem
    # "es" ending: "shines" -> just remove s (handles both "passes" -> "pass" and "shines" -> "shine")
    if word.endswith("es") and len(word) > 4:
        # "passes" -> "pass": double s before es means remove es
        if word.endswith("sses"):
            return word[:-2]
        # "shines" -> "shine": remove trailing s
        return word[:-1]
    # Simple "s" removal for 3rd-person singular: "rains" -> "rain"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        return word[:-1]
    # "ed" ending: "rained" -> "rain"
    if word.endswith("ed") and len(word) > 3:
        stem = word[:-2]
        if len(stem) > 1 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    return word


def _extract_cause(
    tokens: tuple[str, ...],
    uol_act: dict[str, Any] | None = None,
    *,
    return_roles: bool = False,
) -> str | None | dict[str, Any]:
    """Best-effort extraction of the cause verb from a 'what happens if' question.

    The cause is a predicate/event (e.g. 'break', 'shine'), not a determiner or
    noun artifact. Prefer the UOL causal-clause atom, then fall back to a
    token-based verb search.
    """
    cause: str | None = None
    actor: str | None = None
    patient: str | None = None

    # Priority 1: UOL causal clause atom if available.
    if uol_act is not None:
        content = uol_act.get("content", [])
        for atom in content:
            links = atom.get("links", {})
            if links.get("causes") or links.get("enables") or links.get("prevents"):
                pred = atom.get("predicate", {})
                candidate = str(pred.get("id", "") or pred.get("lemma", "")).lower()
                if candidate:
                    cause = candidate
                    break
        if not cause and content:
            pred = content[0].get("predicate", {})
            candidate = str(pred.get("id", "") or pred.get("lemma", "")).lower()
            if candidate:
                cause = candidate

    # Priority 2: Token-based: find verb after 'if' marker.
    if not cause:
        determiners = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"}
        pronouns = {"it", "he", "she", "they", "i", "you", "we"}
        skip = determiners | pronouns
        for i, tok in enumerate(tokens):
            if tok.lower() == "if":
                # Collect all content words after 'if'
                after_if: list[tuple[int, str]] = []
                for j in range(i + 1, len(tokens)):
                    word = tokens[j].lower()
                    if word in skip:
                        continue
                    if len(word) < 2:
                        continue
                    after_if.append((j, word))
                # Find the likely verb among them (prefer last word for "X breaks" pattern)
                cause_candidate = None
                patient_candidate = None
                for idx, (j, word) in enumerate(after_if):
                    lemma = _normalize_lemma(word)
                    if _is_likely_verb(word) or lemma != word:
                        cause_candidate = lemma
                        # Everything before this is patient/actor
                        patient_tokens = [w for (_, w) in after_if[:idx]]
                        if patient_tokens:
                            patient_candidate = " ".join(patient_tokens)
                        break
                if cause_candidate is None and after_if:
                    # Fallback: use the last word
                    last_j, last_word = after_if[-1]
                    cause_candidate = _normalize_lemma(last_word)
                    if len(after_if) > 1:
                        patient_tokens = [w for (_, w) in after_if[:-1]]
                        patient_candidate = " ".join(patient_tokens)
                if cause_candidate:
                    cause = cause_candidate
                    if patient_candidate:
                        # Check if patient is an active entity affordance -> actor
                        try:
                            from melm.appliance.reasoning.causal_frames import resolve_active_entity
                            if resolve_active_entity(patient_candidate):
                                actor = patient_candidate
                            else:
                                patient = patient_candidate
                        except Exception:
                            patient = patient_candidate
                # Actor is the noun before 'if' (not a cue word)
                if actor is None and i > 0:
                    for k in range(i - 1, -1, -1):
                        prev = tokens[k].lower()
                        if prev in {"what", "happens", "happen", "would", "will", "the", "a", "an"}:
                            continue
                        if len(prev) > 1:
                            actor = prev
                            break
                break

    # Priority 3: Fallback to last content word
    if not cause:
        candidate = _last_content_word(tokens)
        if candidate:
            cause = _normalize_lemma(candidate)

    if not cause:
        return None

    if return_roles:
        result: dict[str, Any] = {"cause": cause}
        if actor:
            result["actor"] = actor
        if patient:
            result["patient"] = patient
        return result
    return cause


def _load_causal_cues() -> list[dict[str, Any]] | None:
    """Lazy-load causal cues from contract. Returns None on failure for fallback."""
    global _CAUSAL_CUES_CACHE
    if _CAUSAL_CUES_CACHE is not None:
        return _CAUSAL_CUES_CACHE if _CAUSAL_CUES_CACHE else None
    try:
        from melm.contracts import load_causal_cues as _load
        _CAUSAL_CUES_CACHE = _load()
    except Exception:
        _CAUSAL_CUES_CACHE = []
    return _CAUSAL_CUES_CACHE if _CAUSAL_CUES_CACHE else None


def _extract_contrast_causes(
    tokens: tuple[str, ...],
) -> tuple[str | None, str | None]:
    """Extract two causes from 'what happens if X vs Y' or 'X or Y' patterns."""
    contrast_markers = {"vs", "v", "or"}
    tokens_lower = [t.lower().strip("?") for t in tokens]
    # Find the first 'if' marker
    if_idx = None
    for i, tok in enumerate(tokens_lower):
        if tok == "if":
            if_idx = i
            break
    if if_idx is None:
        return None, None
    skip = {"the", "a", "an", "it", "he", "she", "they", "i", "you", "we"}
    after_if: list[str] = []
    for j in range(if_idx + 1, len(tokens_lower)):
        if tokens_lower[j] in skip:
            continue
        if len(tokens_lower[j]) < 2:
            continue
        after_if.append(tokens_lower[j])
    # Find contrast marker
    for marker in contrast_markers:
        if marker in after_if:
            idx = after_if.index(marker)
            # Left side: everything before marker, pick last likely verb
            left_candidates: list[str] = []
            for w in after_if[:idx]:
                if _is_likely_verb(w):
                    left_candidates.append(_normalize_lemma(w))
            if not left_candidates and after_if[:idx]:
                left_candidates = [_normalize_lemma(after_if[idx - 1])]
            # Right side: everything after marker, pick first likely verb
            right_candidates: list[str] = []
            for w in after_if[idx + 1:]:
                if _is_likely_verb(w):
                    right_candidates.append(_normalize_lemma(w))
            if not right_candidates and after_if[idx + 1:]:
                right_candidates = [_normalize_lemma(after_if[idx + 1])]
            left = left_candidates[-1] if left_candidates else None
            right = right_candidates[0] if right_candidates else None
            return left, right
    return None, None


def _detect_causal(
    text: str,
    tokens: tuple[str, ...] = (),
    uol_act: dict[str, Any] | None = None,
    *,
    frame_candidates: list | None = None,
) -> dict[str, Any] | None:
    """Detect causal explanation, prediction, and contrast tasks.

    Contract-driven via causal_cues.v1.json. Falls back to inline cue logic
    when the contract is unavailable. When ``frame_candidates`` are provided,
    uses them for richer role extraction via UOL atom links.
    """
    if not tokens:
        tokens = tuple(_WORD_RE.findall(text.lower()))
    token_set = {str(tok).lower() for tok in tokens}
    # Also check normalized lemmas for cue matching: "causes" -> "cause"
    lemma_set = {_normalize_lemma(t) for t in token_set} | token_set

    # Frame-candidate enrichment: if frame linker found structurally parsed
    # atoms with causal link roles, use those directly for the task descriptor.
    within_frame = False
    if frame_candidates:
        for fc in frame_candidates:
            # FrameCandidate is a dataclass or dict — handle both forms
            slot_states = fc.get("slot_states") if isinstance(fc, dict) else getattr(fc, "slot_states", None)
            if slot_states:
                slots_str = str(slot_states)
                if "cause" in slots_str.lower() or "effect" in slots_str.lower():
                    within_frame = True
                    break

    # Check for contrast: "what happens if X vs Y" / "X or Y"
    if "what" in lemma_set and ("happens" in lemma_set or "happen" in lemma_set):
        contrast_markers = {"vs", "v", "or"}
        if contrast_markers & token_set:
            left, right = _extract_contrast_causes(tokens)
            if left and right:
                return {"task": "causal_contrast", "cause_a": left, "cause_b": right}

    cues = _load_causal_cues()
    if cues is not None:
        for cue in cues:
            lemma = str(cue.get("lemma", "")).lower()
            if lemma not in lemma_set:
                continue
            co_cues = cue.get("co_cue_lemmas", [])
            if co_cues:
                co_lemmas = {_normalize_lemma(c) for c in co_cues}
                if not (co_lemmas & lemma_set):
                    continue
            cue_type = cue.get("cue_type")
            if cue_type == "causal_explanation":
                effect, theme = _extract_effect_state(tokens, uol_act, return_theme=True)
                if effect:
                    result: dict[str, Any] = {"task": "causal_explanation", "effect": effect}
                    if theme:
                        result["theme"] = theme
                    if within_frame:
                        result["frame_confirmed"] = True
                    return result
            elif cue_type == "causal_prediction":
                cause_result = _extract_cause(tokens, uol_act, return_roles=True)
                if isinstance(cause_result, dict) and cause_result.get("cause"):
                    result = {"task": "causal_prediction", "cause": cause_result["cause"]}
                    if cause_result.get("actor"):
                        result["actor"] = cause_result["actor"]
                    if cause_result.get("patient"):
                        result["patient"] = cause_result["patient"]
                    if within_frame:
                        result["frame_confirmed"] = True
                    return result
                if isinstance(cause_result, str):
                    return {"task": "causal_prediction", "cause": cause_result}

    # Inline fallback removed per V4A design spec — contract covers all cue cases.
    return None


def _detect_arithmetic(text: str) -> dict | None:
    # Require a count question, ≥2 quantities, and an arithmetic operation verb.
    if not _COUNT_Q_RE.search(text):
        return None
    nums = extract_numbers(text)
    if len(nums) < 2:
        return None
    sign = None
    for word in _WORD_RE.findall(text.lower()):
        if word in _SUBTRACT_VERBS:
            sign = -1
            break
        if word in _ADD_VERBS:
            sign = 1
            break
    if sign is None:
        return None
    start_val, start_surface, _ = nums[0]
    delta_val = nums[1][0]
    noun = ""
    m = re.search(re.escape(start_surface) + r"\s+([A-Za-z]+)", text)
    if m:
        noun = m.group(1).lower()
    return {
        "task": "quantity_arithmetic",
        "start": start_val, "delta": delta_val, "sign": sign, "noun": noun,
    }
