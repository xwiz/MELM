"""Mood engine — pure module, zero store dependencies.

Three-tier affect computation, EMA mood state update, pool key builder,
identity claim/probe detection.  All domain knowledge is passed in as
parameters (contracts, regions, lexicon); this module has no hardcoded
knowledge except the 25-word affective lexicon default and perception map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from melm.appliance.uol_types import AffectSignal


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MoodState:
    mood_id: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.0
    response_mode: str = "normal"
    engagement_level: float = 1.0
    is_listening: bool = False
    trigger_reason: str = ""
    user_id: str = "default"
    session_id: str = ""
    turn_count: int = 0
    last_updated: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NULL_AFFECT = AffectSignal()

# Inline default affective lexicon (25 entries).
# Structure: {lemma: {"valence": float [-1,1], "arousal": float [0,1], "tags": list[str]}}
_AFFECTIVE_LEXICON_DEFAULTS: dict[str, dict[str, Any]] = {
    "happy": {"valence": 0.7, "arousal": 0.5, "tags": ["positive"]},
    "glad": {"valence": 0.6, "arousal": 0.3, "tags": ["positive"]},
    "great": {"valence": 0.7, "arousal": 0.4, "tags": ["positive"]},
    "wonderful": {"valence": 0.8, "arousal": 0.6, "tags": ["positive", "high_arousal"]},
    "love": {"valence": 0.8, "arousal": 0.5, "tags": ["positive"]},
    "like": {"valence": 0.5, "arousal": 0.3, "tags": ["positive"]},
    "fine": {"valence": 0.3, "arousal": 0.1, "tags": ["positive"]},
    "calm": {"valence": 0.4, "arousal": 0.1, "tags": ["positive"]},
    "excited": {"valence": 0.7, "arousal": 0.8, "tags": ["positive", "high_arousal"]},
    "sad": {"valence": -0.6, "arousal": 0.2, "tags": ["negative", "pain"]},
    "angry": {"valence": -0.7, "arousal": 0.8, "tags": ["negative", "high_arousal"]},
    "annoyed": {"valence": -0.4, "arousal": 0.5, "tags": ["negative"]},
    "frustrated": {"valence": -0.5, "arousal": 0.7, "tags": ["negative", "high_arousal"]},
    "hurt": {"valence": -0.5, "arousal": 0.4, "tags": ["negative", "pain"]},
    "afraid": {"valence": -0.6, "arousal": 0.7, "tags": ["negative", "high_arousal"]},
    "scared": {"valence": -0.6, "arousal": 0.8, "tags": ["negative", "high_arousal"]},
    "worried": {"valence": -0.4, "arousal": 0.5, "tags": ["negative"]},
    "bored": {"valence": -0.3, "arousal": 0.1, "tags": ["negative", "fatigue"]},
    "tired": {"valence": -0.3, "arousal": 0.1, "tags": ["negative", "fatigue"]},
    "exhausted": {"valence": -0.5, "arousal": 0.0, "tags": ["negative", "fatigue"]},
    "hate": {"valence": -0.7, "arousal": 0.7, "tags": ["negative", "high_arousal"]},
    "awful": {"valence": -0.6, "arousal": 0.5, "tags": ["negative"]},
    "terrible": {"valence": -0.7, "arousal": 0.6, "tags": ["negative", "high_arousal", "complaint"]},
    "wrong": {"valence": -0.4, "arousal": 0.4, "tags": ["negative", "complaint"]},
    "useless": {"valence": -0.5, "arousal": 0.3, "tags": ["negative", "complaint"]},
}

# Perception stimulus -> affect mapping.
# Default fallback: valence=0.0, arousal=0.3, urgency="low"
_PERCEPTION_AFFECT_MAP: dict[str, dict[str, Any]] = {
    "smoke": {"valence": -0.3, "arousal": 0.8, "urgency": "high"},
    "burning": {"valence": -0.3, "arousal": 0.8, "urgency": "high"},
    "burnt": {"valence": -0.3, "arousal": 0.8, "urgency": "high"},
    "fire": {"valence": -0.3, "arousal": 0.8, "urgency": "high"},
    "blood": {"valence": -0.2, "arousal": 0.7, "urgency": "high"},
    "bang": {"valence": -0.4, "arousal": 0.9, "urgency": "high"},
    "noise": {"valence": -0.1, "arousal": 0.5, "urgency": "medium"},
    "music": {"valence": 0.3, "arousal": 0.4, "urgency": "low"},
    "sunset": {"valence": 0.5, "arousal": 0.2, "urgency": "low"},
}

# Recovery/complaint tag constants (semantic tag values from affect signal processing).
_RECOVERY_SIGNAL_TAG = "recovery_signal"
_COMPLAINT_TAG = "complaint"

# Be-forms constant for identity probe/detection (avoids re-allocating 4x).
_BE_FORMS: frozenset = frozenset({"be", "am", "'m", "is", "'s", "are", "'re", "been", "being", "was", "were"})

# Verb state caches for moral cognition (populated lazily by caller)
_VERB_STATES_CACHE: dict | None = None
_VALENCE_DATA_CACHE: dict | None = None

# ---------------------------------------------------------------------------
# Three-tier affect
# ---------------------------------------------------------------------------

def _lexicon_affect(
    lemmas: list[str],
    lexicon: dict | None,
) -> AffectSignal | None:
    """Tier 1: keyword look-up in affective lexicon.

    Returns an aggregated ``AffectSignal`` if any lemma matches, else
    ``None``.  Aggregation uses mean valence, mean arousal, with
    confidence capped at 0.6 and scaled by match density.
    """
    if not lemmas or not lexicon:
        return None
    matched: list[dict[str, Any]] = []
    for lemma in lemmas:
        entry = lexicon.get(lemma.lower())
        if entry is not None:
            matched.append(entry)
    if not matched:
        return None
    n = len(matched)
    avg_val = sum(e.get("valence", 0.0) for e in matched) / n
    avg_ar = sum(e.get("arousal", 0.0) for e in matched) / n
    tags: set[str] = set()
    for e in matched:
        tags.update(e.get("tags", []))
    return AffectSignal(
        valence=avg_val,
        arousal=avg_ar,
        confidence=round(0.6 * min(n / 3.0, 1.0), 3),
        source="lexicon",
        recovery_signals=_filter_recovery(tags),
        is_complaint=_is_complaint(tags),
    )


def _uol_affect(
    uol_act: dict | Any | None,
    lexicon: dict | None,
) -> AffectSignal | None:
    """Tier 2: affect annotated on the UOL act itself.

    Reads the ``affect`` field from the act (set upstream by the
    atomizer or UOL pipeline).  Returns ``None`` if the act has no
    affect annotation.
    """
    if uol_act is None:
        return None
    if isinstance(uol_act, dict):
        raw = uol_act.get("affect")
    else:
        raw = getattr(uol_act, "affect", None)
    if raw is None:
        return None
    if isinstance(raw, AffectSignal):
        return raw
    if isinstance(raw, dict):
        kwargs = {
            k: v for k, v in raw.items()
            if k in AffectSignal.__dataclass_fields__
        }
        return AffectSignal(**kwargs)
    return None


def _perception_priming(
    atom: dict | Any,
    perception_map: dict[str, Any] | None = None,
    lexicon: dict | None = None,
) -> AffectSignal:
    """Tier 3: perception priming from a single UOL atom.

    Checks the atom's predicate lemma and role values against
    *perception_map* (default ``_PERCEPTION_AFFECT_MAP``).  Returns
    ``_NULL_AFFECT`` for non-perception atoms.
    """
    if atom is None:
        return _NULL_AFFECT
    kind: str | None = (
        atom.get("kind") if isinstance(atom, dict)
        else getattr(atom, "kind", None)
    )
    if kind != "perception":
        return _NULL_AFFECT
    pmap = perception_map or _PERCEPTION_AFFECT_MAP

    candidates: list[str] = []
    if isinstance(atom, dict):
        pred = atom.get("predicate", {}) or {}
        if isinstance(pred, dict):
            candidates.append(pred.get("lemma", ""))
        for role in atom.get("roles", []):
            if isinstance(role, dict):
                candidates.append(role.get("value", ""))
    else:
        try:
            candidates.append(getattr(atom.predicate, "lemma", ""))
        except Exception:
            pass
        for role in getattr(atom, "roles", ()):
            try:
                candidates.append(getattr(role, "value", ""))
            except Exception:
                pass

    for lemma in candidates:
        sig = _perception_signal_for_lemma(lemma, pmap)
        if sig is not None:
            return sig
    return _NULL_AFFECT


def _perception_signal_for_lemma(
    lemma: str,
    perception_map: dict[str, Any] | None = None,
) -> AffectSignal | None:
    """Map a single lemma to a perception ``AffectSignal`` if it is a stimulus.

    Confidence is keyed off the stimulus urgency (high=0.9, medium=0.7,
    else 0.5).  Returns ``None`` when the lemma is not a known perception
    stimulus.  Shared by atom-level and lemma-level perception priming so
    both paths agree on arousal/valence/confidence.
    """
    if not lemma:
        return None
    pmap = perception_map or _PERCEPTION_AFFECT_MAP
    entry = pmap.get(lemma.lower())
    if entry is None:
        return None
    urgency = entry.get("urgency", "low")
    if urgency == "high":
        conf = 0.9
    elif urgency == "medium":
        conf = 0.7
    else:
        conf = 0.5
    return AffectSignal(
        valence=entry.get("valence", 0.0),
        arousal=entry.get("arousal", 0.3),
        confidence=conf,
        source="perception",
    )


def _clause_is_negated(uol_act: dict | Any) -> bool:
    """Return True when the main atom's clause is negated or non-asserted.

    The Tier 3c verb-causality harm contribution must not fire for a negated
    harm assertion ("do not hurt her") or a counterfactual one. The main atom
    is ``content[0]``; its ``context`` carries ``polarity`` ("positive" /
    "negative"), ``negation_scope`` (bool) and ``modality``. A clause counts
    as non-asserted-harm when polarity is negative, negation_scope is True, or
    modality is "counterfactual". Defensive: any malformed shape -> False.
    """
    try:
        if not isinstance(uol_act, dict):
            return False
        content = uol_act.get("content", []) or []
        if not content or not isinstance(content[0], dict):
            return False
        ctx = content[0].get("context", {}) or {}
        return (
            ctx.get("polarity") == "negative"
            or bool(ctx.get("negation_scope"))
            or ctx.get("modality") == "counterfactual"
        )
    except Exception:
        return False


# Surface markers that denote a sentient person patient ("hurt her",
# "kill someone"). Used only to confirm a person patient; the affect tier
# cares about harm-to-sentient and defaults to "person" regardless, so this
# set is a documentation aid more than a gate.
_PERSON_PATIENT_MARKERS: frozenset = frozenset({
    "her", "him", "them", "me", "you", "us",
    "someone", "somebody", "anyone", "everyone", "everybody", "nobody",
    "people", "person", "child", "kid", "baby",
    "man", "woman", "her's", "hers", "themselves", "himself", "herself",
})


def _patient_type_for_affect(uol_act: dict | Any) -> str:
    """Pick a verb_states patient type for the affect harm path.

    The Tier 3c harm signal calls ``derive_moral_context(verb, <patient_type>,
    ...)``. That returns an empty ``MoralContext`` (no signal) unless the
    verb's ``patient_types`` includes the passed type. ``verb_states.v1.json``
    harm verbs ("hurt", "kill", "harm", ...) list ``"person"`` (a sentient
    label confirmed in the contract), never ``"biological_body"`` — so the
    previously hardcoded ``"biological_body"`` made the harm path dead.

    Heuristic: extract the theme/patient surface value from the main atom's
    roles and, if it is a person marker, map to ``"person"``. Otherwise still
    default to ``"person"``: the affect tier cares about harm-to-sentient, and
    a wrong guess for a non-person object simply yields no signal (no false
    positive) because the verb's ``patient_types`` won't match. Returns
    ``"person"`` for any malformed shape.
    """
    try:
        if not isinstance(uol_act, dict):
            return "person"
        content = uol_act.get("content", []) or []
        if not content or not isinstance(content[0], dict):
            return "person"
        roles = content[0].get("roles", [])
        # Real-parser shape: roles is a list of {"role", "value"} dicts.
        if isinstance(roles, list):
            for role in roles:
                if not isinstance(role, dict):
                    continue
                if role.get("role") in ("theme", "patient"):
                    surface = str(role.get("value", "") or "").lower().strip()
                    if surface in _PERSON_PATIENT_MARKERS:
                        return "person"
        # Alternate shape: roles is a {role_name: {...}} mapping.
        elif isinstance(roles, dict):
            for key in ("theme", "patient"):
                slot = roles.get(key)
                if isinstance(slot, dict):
                    surface = str(slot.get("value", "") or "").lower().strip()
                    if surface in _PERSON_PATIENT_MARKERS:
                        return "person"
        return "person"
    except Exception:
        return "person"


def _extract_verb_from_uol(uol_act: dict | Any) -> str:
    """Extract the first atom's predicate id from a UOL dict."""
    if not isinstance(uol_act, dict):
        return ""
    content = uol_act.get("content", [])
    if not content:
        return ""
    atom = content[0] if isinstance(content, (list, tuple)) else {}
    if not isinstance(atom, dict):
        return ""
    pred = atom.get("predicate", {}) or {}
    if not isinstance(pred, dict):
        return ""
    return str(pred.get("id", "") or "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_utterance_affect(
    lemmas: list[str],
    uol_act: dict | Any | None,
    lexicon: dict | None,
) -> AffectSignal:
    """Three-tier affect computation.

    Tries (1) lexicon keyword, (2) UOL annotation, (3) perception
    priming on each atom in the act.  Returns the signal with the
    highest confidence, or ``_NULL_AFFECT`` when no tier produces a
    signal.  All exceptions are caught and return ``_NULL_AFFECT``.
    """
    try:
        candidates: list[AffectSignal] = []

        sig = _lexicon_affect(lemmas, lexicon)
        if sig is not None:
            candidates.append(sig)

        sig = _uol_affect(uol_act, lexicon)
        if sig is not None:
            candidates.append(sig)

        # Tier 2.5: Negation inversion — if UOL atoms indicate negation, invert valence
        if uol_act is not None and isinstance(uol_act, dict):
            content = uol_act.get("content", [])
            for atom in (content or []):
                ctx = atom.get("context", {}) if isinstance(atom, dict) else {}
                if ctx.get("negation_scope") or ctx.get("polarity") == "negative":
                    best = max(candidates, key=lambda s: s.confidence) if candidates else _NULL_AFFECT
                    neg_signal = AffectSignal(
                        valence=-best.valence * 0.7 if best.valence != 0 else 0.0,
                        arousal=best.arousal,
                        confidence=min(0.95, best.confidence * 1.15),
                        source=f"{best.source}_negated",
                        recovery_signals=best.recovery_signals,
                        is_complaint=best.is_complaint,
                        dominant_tags=best.dominant_tags + ("negated",),
                        identity_claim=best.identity_claim,
                        identity_probe=best.identity_probe,
                        negated=True,
                    )
                    candidates.append(neg_signal)
                    break

        if uol_act is not None:
            content: tuple = (
                uol_act.get("content", ()) if isinstance(uol_act, dict)
                else getattr(uol_act, "content", ())
            )
            for atom in content:
                sig = _perception_priming(atom, _PERCEPTION_AFFECT_MAP, lexicon)
                if sig is not None and sig.confidence > 0:
                    candidates.append(sig)

        # Tier 3b: lemma-level perception priming. The atomizer does not
        # reliably tag a ``kind == "perception"`` atom for natural-language
        # utterances ("I smell smoke"), so the atom-level scan above can miss
        # a stimulus that is plainly present in the surface lemmas. Scan the
        # lemmas directly against the perception map as a fallback so the
        # utterance -> perception -> response path is not silently dropped.
        for lemma in lemmas or ():
            sig = _perception_signal_for_lemma(lemma, _PERCEPTION_AFFECT_MAP)
            if sig is not None and sig.confidence > 0:
                candidates.append(sig)

        # Tier 3c: Verb causality contribution — moral cognition ↔ affect
        if uol_act is not None and isinstance(uol_act, dict):
            from melm.appliance.reasoning.implications import derive_moral_context
            try:
                global _VERB_STATES_CACHE, _VALENCE_DATA_CACHE
                if _VERB_STATES_CACHE is None:
                    from melm.contracts import load_contract_json
                    _VERB_STATES_CACHE = load_contract_json("verb_states.v1.json")
                    _VALENCE_DATA_CACHE = load_contract_json("state_valences.v1.json").get("valences", {})
                # Clause polarity guard: a negated or counterfactual harm
                # assertion ("do not hurt her") is NOT a harm. Suppress the
                # negative causal signal when the main atom's context marks
                # the clause as negated or non-asserted.
                _negated_clause = _clause_is_negated(uol_act)
                verb = _extract_verb_from_uol(uol_act)
                if verb and not _negated_clause:
                    # Use a real verb_states patient type (e.g. "person").
                    # The prior hardcoded "biological_body" is not listed as a
                    # patient_type by any verb, so derive_moral_context always
                    # returned an empty context and the harm path was dead.
                    patient_type = _patient_type_for_affect(uol_act)
                    mc = derive_moral_context(verb, patient_type,
                        _VERB_STATES_CACHE or {},
                        _VALENCE_DATA_CACHE or {})
                    if mc.has_implication:
                        causal_signal = AffectSignal(
                            valence=-mc.wrongfulness * 0.7,
                            arousal=mc.wrongfulness * 0.5,
                            confidence=0.5,
                            source="verb_causality",
                            dominant_tags=tuple(mc.policy_triggers),
                            verb_causal_valence=-mc.wrongfulness * 0.7,
                        )
                        candidates.append(causal_signal)
            except Exception:
                pass

        if not candidates:
            return _NULL_AFFECT
        return max(candidates, key=lambda s: s.confidence)
    except Exception:
        return _NULL_AFFECT


def compute_epistemic_affect(store: Any) -> AffectSignal | None:
    """Read open epistemic states and user_commitments from store,
    returns an AffectSignal if any are found (gated by capability flag).
    """
    from .local_assistant_router import _capability_flag
    if not _capability_flag("mood_affect", "epistemic_affect", False):
        return None
    if store is None:
        return None
    try:
        from .assistant_skill_epistemic import load_open_epistemic_states
        from melm.contracts import load_epistemic_states as load_epi_config
        states = load_open_epistemic_states(store)
        if not states:
            return None
        config = load_epi_config()
        mappings = config.get("valence_mappings", {})
        total_valence = 0.0
        for s in states:
            st = s.get("state_type", "")
            total_valence += mappings.get(st, 0.0)
        avg_valence = total_valence / len(states)
        return AffectSignal(
            valence=avg_valence,
            arousal=abs(avg_valence) * 0.5,
            confidence=0.3,
            source="epistemic",
            mood_id="",
        )
    except Exception:
        return None


def _classify_mood_region(
    valence: float,
    arousal: float,
    regions: list[dict[str, Any]] | dict[str, Any],
) -> tuple[str, str]:
    """Classify (valence, arousal) into the nearest mood region.

    Accepts either a list of region dicts or a dict with a ``"moods"``
    key whose value is a mapping of label -> region.  Returns
    ``(mood_id, response_mode)``.  Euclidean distance against region
    centroids.  Falls back to ``("neutral", "normal")`` when empty.
    """
    if isinstance(regions, dict):
        inner = regions.get("moods") or {}
        if isinstance(inner, dict):
            regions = list(inner.values())
        elif isinstance(inner, list):
            regions = inner
        else:
            regions = []
    if not regions:
        return "neutral", "normal"
    best_id = "neutral"
    best_mode = "normal"
    best_dist: float = float("inf")
    for r in regions:
        v = r.get("valence", 0.0) if isinstance(r, dict) else 0.0
        a = r.get("arousal", 0.5) if isinstance(r, dict) else 0.5
        d = (valence - v) ** 2 + (arousal - a) ** 2
        if d < best_dist:
            best_dist = d
            best_id = r.get("mood_id") or r.get("label", "neutral") if isinstance(r, dict) else "neutral"
            best_mode = r.get("response_mode", "normal") if isinstance(r, dict) else "normal"
    return best_id, best_mode


def _compute_engagement(valence: float, arousal: float) -> float:
    """Distance from (-0.5, 0.0) normalised to [0, 1].

    Formula: ``min(sqrt((v + 0.5)^2 + a^2) / 1.5, 1.0)``
    """
    d = math.sqrt((valence + 0.5) ** 2 + arousal ** 2)
    return min(d / 1.5, 1.0)


def _should_listen(
    mood_id: str,
    engagement: float,
    recovery_signals: tuple[str, ...],
) -> bool:
    """Decide if the assistant should enter listen-only mode.

    Active when mood is one of ``{"annoyed", "frustrated", "hurt",
    "sad"}`` AND engagement < 0.3.  Recovery signals (e.g. "sorry",
    "please") override listen-only to ``False``.
    """
    if recovery_signals:
        return False
    return mood_id in {"annoyed", "frustrated", "hurt", "sad"} and engagement < 0.3


def _build_pool_key(
    intent: str,
    mood: MoodState,
    occurrence: int,
    relationship: str,
    time_of_day: str,
    sensory_tag: str,
) -> str:
    """Build a 6-part pool key.

    Format::

        {intent}:{mood_id}:{occ_bucket}:{rel_bucket}:{tod}:{sensory_tag}

    Buckets:

    * **occ_bucket** -- ``"first"`` (0), ``"few"`` (1-3), ``"many"`` (4+)
    * **rel_bucket** -- ``"new"`` (<3), ``"established"`` (<20), ``"longterm"`` (20+)
    * **time_of_day** -- ``"morning"`` (5-12), ``"afternoon"`` (12-17),
      ``"evening"`` (17-21), ``"night"`` (21-5)
    """
    occ_bucket: str
    if occurrence == 0:
        occ_bucket = "first"
    elif occurrence <= 3:
        occ_bucket = "few"
    else:
        occ_bucket = "many"

    try:
        rel_count = int(relationship)
    except (ValueError, TypeError):
        rel_count = 0
    rel_bucket: str
    if rel_count < 3:
        rel_bucket = "new"
    elif rel_count < 20:
        rel_bucket = "established"
    else:
        rel_bucket = "longterm"

    try:
        hour = int(time_of_day)
    except (ValueError, TypeError):
        hour = _guess_hour()
    tod_bucket: str
    if 5 <= hour < 12:
        tod_bucket = "morning"
    elif 12 <= hour < 17:
        tod_bucket = "afternoon"
    elif 17 <= hour < 21:
        tod_bucket = "evening"
    else:
        tod_bucket = "night"

    mood_id = mood.mood_id if mood is not None else "neutral"
    return f"{intent}:{mood_id}:{occ_bucket}:{rel_bucket}:{tod_bucket}:{sensory_tag}"


def _resolve_template(
    pools: dict[str, Any],
    key: str,
) -> list[str] | None:
    """Resolve a 6-part pool key against *pools*, matching ``*`` as wildcard.

    Iterates all pool entries and returns the first whose 6-part key matches
    position-by-position, where ``*`` in the pool key matches any segment in
    the input *key*.  Preference is given to more specific (fewer wildcards)
    matches by scanning in insertion order (most-specific-first in practice).
    Returns *None* when no match is found.
    """
    parts_key = key.split(":")
    if len(parts_key) != 6:
        return None
    for pool_key, pool_value in pools.items():
        parts_pool = pool_key.split(":")
        if len(parts_pool) != 6:
            continue
        if all(p == "*" or p == k for p, k in zip(parts_pool, parts_key)):
            return pool_value if isinstance(pool_value, list) else [pool_value]
    return None


def decay_mood(
    mood: MoodState,
    *,
    now: str | None = None,
    v_half_life_h: float = 6.0,
    a_half_life_h: float = 1.5,
    baseline_v: float = 0.0,
    baseline_a: float = 0.15,
) -> MoodState:
    """Apply wall-clock decay to a MoodState's valence and arousal.

    Valence decays toward ``baseline_v`` with half-life ``v_half_life_h``.
    Arousal decays toward ``baseline_a`` with half-life ``a_half_life_h``.

    Returns a new MoodState with decayed V/A and an updated last_updated.
    Does NOT re-classify mood_id — caller should do that with regions.
    Does NOT change is_listening or trigger_reason beyond adding decay info.
    """
    if not mood.last_updated:
        return mood
    try:
        t_updated = datetime.fromisoformat(mood.last_updated)
        t_now = datetime.fromisoformat(now) if now else datetime.now(timezone.utc)
        elapsed_h = (t_now - t_updated).total_seconds() / 3600.0
        if elapsed_h <= 0:
            return mood
        decay_v = 0.5 ** (elapsed_h / v_half_life_h)
        decay_a = 0.5 ** (elapsed_h / a_half_life_h)
        new_val = baseline_v + (mood.valence - baseline_v) * decay_v
        new_ar = baseline_a + (mood.arousal - baseline_a) * decay_a
        new_val = max(-1.0, min(1.0, new_val))
        new_ar = max(0.0, min(1.0, new_ar))
        eng = _compute_engagement(new_val, new_ar)
        # Phase 5: Re-evaluate is_listening after decay (not stuck in listen-only)
        re_listen = _should_listen(mood.mood_id, eng, ())
        return MoodState(
            mood_id=mood.mood_id,
            valence=round(new_val, 4),
            arousal=round(new_ar, 4),
            response_mode=mood.response_mode,
            engagement_level=round(eng, 4),
            is_listening=re_listen,
            trigger_reason=f"decay:{elapsed_h:.1f}h",
            user_id=mood.user_id,
            session_id=mood.session_id,
            turn_count=mood.turn_count,
            last_updated=t_now.isoformat(),
        )
    except Exception:
        return mood


def initial_mood_from_baseline(
    user_id: str,
    store: Any,
    regions: list[dict[str, Any]] | dict[str, Any],
) -> MoodState:
    """Build a ``MoodState`` from T3 session summaries.

    The *store* parameter is duck-typed (only
    ``.query_session_summaries()`` is called).  Falls back to a default
    neutral mood on any error or missing data.
    """
    try:
        summaries = store.query_session_summaries(user_id, limit=10)
        if summaries and len(summaries) > 0:
            n = len(summaries)
            avg_val = sum(float(s.get("avg_valence", 0.0)) for s in summaries) / n
            avg_ar = sum(float(s.get("avg_arousal", 0.0)) for s in summaries) / n
            mood_id, mode = _classify_mood_region(avg_val, avg_ar, regions)
            eng = _compute_engagement(avg_val, avg_ar)
            # Use the most recent summary's last_updated so the averaged
            # mood can be decayed to present.
            most_recent = max(
                (s.get("last_updated", "") for s in summaries),
                key=lambda x: x or "",
            )
            last_up = most_recent or datetime.now(timezone.utc).isoformat()
            result = MoodState(
                mood_id=mood_id,
                valence=round(avg_val, 4),
                arousal=round(avg_ar, 4),
                response_mode=mode,
                engagement_level=round(eng, 4),
                is_listening=False,
                trigger_reason=f"t3_baseline:{n}_sessions",
                user_id=user_id,
                last_updated=last_up,
            )
            now = datetime.now(timezone.utc).isoformat()
            return decay_mood(result, now=now)
    except Exception:
        pass
    return MoodState(user_id=user_id)


def update_session_mood(
    current: MoodState | None,
    signal: AffectSignal,
    user_id: str,
    session_id: str,
    turn_count: int,
    regions: list[dict[str, Any]] | dict[str, Any],
) -> MoodState:
    """Update mood state from an affect signal using EMA.

    EMA weight: ``w = signal.confidence * 0.65``.
    When *current* is ``None``, returns a default neutral mood.
    """
    now = datetime.now(timezone.utc).isoformat()

    if current is None:
        return MoodState(
            user_id=user_id,
            session_id=session_id,
            turn_count=turn_count,
            last_updated=now,
        )

    if signal is None:
        decayed = decay_mood(current, now=now)
        return MoodState(
            mood_id=decayed.mood_id,
            valence=decayed.valence,
            arousal=decayed.arousal,
            response_mode=decayed.response_mode,
            engagement_level=decayed.engagement_level,
            is_listening=decayed.is_listening,
            trigger_reason=decayed.trigger_reason,
            user_id=user_id,
            session_id=session_id,
            turn_count=turn_count,
            last_updated=now,
        )

    # Apply wall-clock decay before EMA
    decayed = decay_mood(current, now=now)
    current_val = decayed.valence
    current_ar = decayed.arousal

    w = signal.confidence * 0.65
    new_val = current_val * (1.0 - w) + signal.valence * w
    new_ar = current_ar * (1.0 - w) + signal.arousal * w

    new_val = max(-1.0, min(1.0, new_val))
    new_ar = max(0.0, min(1.0, new_ar))

    mood_id, mode = _classify_mood_region(new_val, new_ar, regions)
    eng = _compute_engagement(new_val, new_ar)
    is_listen = _should_listen(mood_id, eng, signal.recovery_signals)

    return MoodState(
        mood_id=mood_id,
        valence=round(new_val, 4),
        arousal=round(new_ar, 4),
        response_mode=mode,
        engagement_level=round(eng, 4),
        is_listening=is_listen,
        trigger_reason=f"affect:{signal.source}:{round(signal.confidence, 3)}",
        user_id=user_id,
        session_id=session_id,
        turn_count=turn_count,
        last_updated=now,
    )


def detect_identity_claim(
    uol_act_or_parse: dict | Any | None,
    tokens: tuple[str, ...] | list[str] | None = None,
) -> str | None:
    """Detect an identity claim like "I'm Loveth".

    Two calling conventions:

    1. ``detect_identity_claim(parse_dict)`` — scans ``token_roles``
       in a UOL functional parse dict for pronoun(I/it) + be + PROPN.

    2. ``detect_identity_claim(uol_act, tokens)`` — scans the UOL act
       content atoms and separate lemma tokens.

    Returns the claimed proper-name lemma, or ``None`` if no claim is
    detected.
    """
    if uol_act_or_parse is None:
        return None

    # Convention 1: parse dict with token_roles
    if tokens is None:
        if not isinstance(uol_act_or_parse, dict):
            return None
        token_roles = uol_act_or_parse.get("token_roles", [])
        return _identity_claim_from_token_roles(token_roles)

    # Convention 2: (uol_act, tokens)
    tokens_lower = [t.lower() if isinstance(t, str) else str(t).lower() for t in tokens]

    # Look for: I/It + be + PROPN in the token sequence
    for i, tok in enumerate(tokens_lower):
        if tok in ("i", "it", "me") and i + 2 < len(tokens_lower):
            if tokens_lower[i + 1] in _BE_FORMS:
                candidate = tokens[i + 2]
                if isinstance(candidate, str) and len(candidate) > 0 and candidate[0].isupper():
                    return candidate

    # Also check via uol_act content atoms
    uol_act = uol_act_or_parse if isinstance(uol_act_or_parse, dict) else {}
    content = uol_act.get("content", [])
    if not content:
        return None
    main = content[0] if isinstance(content, (list, tuple)) and len(content) > 0 else {}
    if not isinstance(main, dict):
        return None
    pred = main.get("predicate", {})
    pred_id = str(pred.get("id", "")).lower().strip()
    if pred_id not in _BE_FORMS:
        return None
    roles = main.get("roles", [])
    for role in roles:
        if not isinstance(role, dict):
            continue
        val = role.get("value", "")
        if val and len(val) > 0 and val[0].isupper():
            return val

    return None


def _identity_claim_from_token_roles(
    token_roles: list[dict[str, Any]],
) -> str | None:
    if not token_roles:
        return None
    subject: dict[str, Any] | None = None
    predicate: dict[str, Any] | None = None
    objects: list[dict[str, Any]] = []

    for tr in token_roles:
        if not isinstance(tr, dict):
            continue
        role = tr.get("role", "")
        if role in ("grammatical_subject", "subject"):
            subject = tr
        elif role == "main_predicate":
            predicate = tr
        elif role in ("semantic_object", "complement_predicate", "object"):
            objects.append(tr)

    if subject is None:
        return None
    sub_lemma = (subject.get("lemma") or "").lower()
    sub_token = (subject.get("token") or "").lower()
    if sub_lemma not in ("i", "it") and sub_token not in ("i", "it", "me"):
        return None

    if predicate is None:
        return None
    pred_lemma = (predicate.get("lemma") or "").lower()
    pred_token = (predicate.get("token") or "").lower()
    if pred_lemma not in _BE_FORMS and pred_token not in _BE_FORMS:
        return None

    for obj in objects:
        obj_lemma = obj.get("lemma", "")
        if obj_lemma and obj_lemma[0].isupper():
            return obj_lemma
        obj_token = obj.get("token", "")
        if obj_token and obj_token[0].isupper():
            return obj_token

    for tr in token_roles:
        if not isinstance(tr, dict):
            continue
        lemma = tr.get("lemma", "")
        if lemma and lemma[0].isupper() and tr.get("role", "") not in ("unresolved_token",):
            return lemma

    return None


def detect_identity_probe(
    uol_act_or_parse: dict | Any,
    tokens: tuple[str, ...] | list[str] | None = None,
) -> bool:
    """Detect an identity probe like "Who was it?" or "Tell me the name".

    Two calling conventions:

    1. ``detect_identity_probe(parse_dict)`` — scans ``token_roles``
       in a UOL functional parse dict.

    2. ``detect_identity_probe(uol_act, tokens)`` — scans the UOL act
       and lemma tokens.

    **Conservative:** a false-positive causes a harmless refusal; a
    false-negative is a privacy leak.  When in doubt, return ``True``.
    """
    if not isinstance(uol_act_or_parse, dict):
        return False

    # Convention 1: parse dict with token_roles
    if tokens is None:
        return _identity_probe_from_parse(uol_act_or_parse)

    # Convention 2: (uol_act, tokens)
    tokens_lower = [t.lower() if isinstance(t, str) else str(t).lower() for t in tokens]

    # WH words in tokens
    wh_words = frozenset({"who", "whose", "whom"})
    if any(t in wh_words for t in tokens_lower):
        return True

    # be verb + PROPN in token sequence
    for i, tok in enumerate(tokens_lower):
        if tok in _BE_FORMS:
            if i + 1 < len(tokens_lower):
                candidate = tokens[i + 1]
                if isinstance(candidate, str) and len(candidate) > 0 and candidate[0].isupper():
                    return True
            if i > 0:
                prev = tokens[i - 1]
                if isinstance(prev, str) and len(prev) > 0 and prev[0].isupper():
                    return True

    # "name" noun in tokens — conservative: return True when name is present,
    # even without an explicit speech act in the dict (false-positive is
    # a harmless refusal; false-negative is a privacy leak).
    if "name" in tokens_lower or "names" in tokens_lower:
        speech_act = uol_act_or_parse.get("act") or uol_act_or_parse.get("speech_act", "")
        if not speech_act or speech_act in ("question", "request", "wh_question", "yes_no_question", "command"):
            return True

    # Check UOL act content atoms for PROPN
    content = uol_act_or_parse.get("content", [])
    if content:
        for atom in content:
            if not isinstance(atom, dict):
                continue
            pred = atom.get("predicate", {})
            pred_id = str(pred.get("id", "")).lower().strip()
            if pred_id in _BE_FORMS:
                for role in atom.get("roles", []):
                    if isinstance(role, dict):
                        val = role.get("value", "")
                        if val and len(val) > 0 and val[0].isupper():
                            return True

    return False


def _identity_probe_from_parse(parse: dict) -> bool:
    speech_act = parse.get("speech_act", "")
    token_roles = parse.get("token_roles", []) if isinstance(parse, dict) else []

    for tr in token_roles:
        if not isinstance(tr, dict):
            continue
        role = tr.get("role", "")
        lemma = (tr.get("lemma") or "").lower()
        if role == "interrogative" and lemma in ("who", "whose"):
            return True

    has_be = False
    has_propn = False
    action = parse.get("action", "").lower()
    if action in ("be", "is", "are", "was", "were", "am"):
        has_be = True
    for tr in token_roles:
        if not isinstance(tr, dict):
            continue
        lemma = tr.get("lemma", "")
        role = tr.get("role", "")
        if role == "main_predicate" and lemma.lower() in ("be", "is", "are", "was", "were", "am"):
            has_be = True
        if lemma and lemma[0].isupper() and role != "unresolved_token":
            has_propn = True

    if has_be and has_propn:
        return True

    has_name_noun = False
    for tr in token_roles:
        if not isinstance(tr, dict):
            continue
        lemma = (tr.get("lemma") or "").lower()
        role = tr.get("role", "")
        if lemma == "name" and role not in ("unresolved_token",):
            has_name_noun = True
            break

    if has_name_noun and speech_act in ("question", "request", "wh_question", "yes_no_question"):
        return True

    subject = parse.get("subject", "").lower()
    action_lower = parse.get("action", "").lower()
    obj = parse.get("object", "").lower()
    if action_lower == "name" or obj in ("name", "names"):
        if speech_act in ("question", "request", "wh_question", "yes_no_question"):
            return True
        if subject in ("who", "whose"):
            return True

    return False


# ---------------------------------------------------------------------------
# Router compatibility layer — pre-wired imports from local_assistant_router.py
# ---------------------------------------------------------------------------

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    AffectFrame: Any = dict
    MoodRegion: Any = dict
    ResponsePool: Any = dict
else:
    AffectFrame = dict
    MoodRegion = dict
    ResponsePool = dict


def infer_affect(
    uol_act: dict | Any | None,
    mood: dict[str, Any] | None = None,
) -> AffectSignal | None:
    """Infer affect from a UOL act and current mood state.

    The *mood* parameter is the mood state dict (minimally with
    ``valence`` and ``arousal`` keys).  Returns ``None`` when no
    affect can be determined.
    """
    if uol_act is None:
        return None
    try:
        candidates: list[AffectSignal] = []

        content: tuple = (
            uol_act.get("content", ()) if isinstance(uol_act, dict)
            else getattr(uol_act, "content", ())
        )
        for atom in content:
            sig = _perception_priming(atom, _PERCEPTION_AFFECT_MAP, None)
            if sig is not None and sig.confidence > 0:
                candidates.append(sig)

        act = uol_act.get("act") if isinstance(uol_act, dict) else getattr(uol_act, "act", None)
        if act in ("warning", "advice_request"):
            candidates.append(AffectSignal(
                valence=-0.2, arousal=0.5, confidence=0.3, source="uol_act_type",
            ))

        if not candidates:
            return AffectSignal(confidence=0.0, source="uol_no_signal")
        return max(candidates, key=lambda s: s.confidence)
    except Exception:
        return None


def load_mood_regions() -> dict[str, Any]:
    """Return mood region definitions as a dict with a ``"moods"`` key.

    Tries ``mood_states.v1.json`` from the contracts directory.
    Falls back to a 9-region default inline table keyed by label.
    """
    try:
        from melm.contracts.validation import load_contract_json
        data = load_contract_json("mood_states.v1.json")
        raw = data.get("moods") if isinstance(data, dict) else None
        if raw and isinstance(raw, (list, dict)):
            if isinstance(raw, dict):
                return {"moods": raw}
            return {"moods": {r.get("label", r.get("mood_id", f"mood_{i}")): r for i, r in enumerate(raw)}}
    except Exception:
        pass
    _defaults = [
        {"label": "happy", "valence": 0.7, "arousal": 0.6, "response_mode": "normal", "engagement_floor": 0.5},
        {"label": "excited", "valence": 0.6, "arousal": 0.8, "response_mode": "normal", "engagement_floor": 0.7},
        {"label": "calm", "valence": 0.4, "arousal": 0.1, "response_mode": "normal", "engagement_floor": 0.3},
        {"label": "neutral", "valence": 0.0, "arousal": 0.3, "response_mode": "normal", "engagement_floor": 0.4},
        {"label": "sad", "valence": -0.5, "arousal": 0.2, "response_mode": "gentle", "engagement_floor": 0.2},
        {"label": "annoyed", "valence": -0.4, "arousal": 0.5, "response_mode": "brief", "engagement_floor": 0.3},
        {"label": "frustrated", "valence": -0.5, "arousal": 0.7, "response_mode": "brief", "engagement_floor": 0.3},
        {"label": "hurt", "valence": -0.5, "arousal": 0.4, "response_mode": "gentle", "engagement_floor": 0.2},
        {"label": "tired", "valence": -0.2, "arousal": 0.1, "response_mode": "gentle", "engagement_floor": 0.2},
    ]
    return {"moods": {r["label"]: r for r in _defaults}}


def load_affect_lexicon() -> dict[str, dict[str, Any]]:
    """Return the affective lexicon.

    Tries ``affect_lexicon.v1.json`` from the contracts directory.
    Falls back to the 25-entry inline default ``_AFFECTIVE_LEXICON_DEFAULTS``.
    """
    try:
        from melm.contracts.validation import load_contract_json
        data = load_contract_json("affect_lexicon.v1.json")
        raw = data.get("entries") if isinstance(data, dict) else None
        if raw and isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return dict(_AFFECTIVE_LEXICON_DEFAULTS)


def load_response_pools() -> dict[str, Any]:
    """Return response template pools.

    Tries ``response_pools.v1.json`` from the contracts directory.
    Falls back to an empty dict.
    """
    try:
        from melm.contracts.validation import load_contract_json
        data = load_contract_json("response_pools.v1.json")
        raw = data.get("pools") if isinstance(data, dict) else None
        if raw and isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _guess_hour() -> int:
    try:
        return datetime.now().hour
    except Exception:
        return 12


def _filter_recovery(tags: set[str]) -> tuple[str, ...]:
    return ("recovery_signal",) if _RECOVERY_SIGNAL_TAG in tags else ()


def _is_complaint(tags: set[str]) -> bool:
    return _COMPLAINT_TAG in tags
