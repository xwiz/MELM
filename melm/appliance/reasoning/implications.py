"""T4 moral cognition implication engine.

Pure-function derivation of moral context from verb causality data.
No ML dependencies, no pipeline stages — just stdlib dataclasses +
dict lookups over verb_contracts and valence_contracts.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MoralContext:
    """The moral profile derived from a verb+patient combination.

    Fields
    ------
    wrongfulness : float
        Scalar in [0.0, 1.0] estimating how wrongful the action is toward
        the given patient type.  0.0 = neutral, 1.0 = maximally wrongful.
    harm_severity : str | None
        Categorical severity label: None, "low", "medium", or "high".
    consent_status : str
        One of "unknown", "consented", "not_consented".
    policy_triggers : tuple[str, ...]
        Zero or more policy-relevant tags (e.g. "urgent_harm", "caution").
    """

    wrongfulness: float = 0.0
    harm_severity: Optional[str] = None
    consent_status: str = "unknown"
    policy_triggers: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_implication(self) -> bool:
        """True when there is at least one actionable signal."""
        return self.harm_severity is not None or bool(self.policy_triggers)


# ---------------------------------------------------------------------------
# Default sentience map (loaded from sentience_map.v1.json, overridable)
# ---------------------------------------------------------------------------

_SENTIENCE_MAP_CACHE: Optional[Dict[str, bool]] = None


def _get_sentience_map() -> Dict[str, bool]:
    global _SENTIENCE_MAP_CACHE
    if _SENTIENCE_MAP_CACHE is None:
        from melm.contracts import load_sentience_map
        _SENTIENCE_MAP_CACHE = load_sentience_map()
    return _SENTIENCE_MAP_CACHE


_DAMAGE_MARKERS_CACHE: Optional[set] = None


def _get_damage_markers() -> set:
    global _DAMAGE_MARKERS_CACHE
    if _DAMAGE_MARKERS_CACHE is None:
        from melm.contracts import load_damage_markers
        _DAMAGE_MARKERS_CACHE = load_damage_markers()
    return _DAMAGE_MARKERS_CACHE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def derive_moral_context(
    verb: str,
    patient_type: str,
    verb_contract: Optional[Dict[str, Any]] = None,
    valence_contract: Optional[Dict[str, float]] = None,
    sentience_map: Optional[Dict[str, bool]] = None,
) -> MoralContext:
    """Derive a MoralContext for a verb applied to a patient type.

    Parameters
    ----------
    verb : str
        Surface verb lemma (e.g. ``"hit"``, ``"help"``).
    patient_type : str
        Semantic class of the patient (e.g. ``"person"``, ``"object"``).
    verb_contract : dict | None
        The loaded ``verb_states.v1.json`` contract.  Structure::

            {"verbs": {<verb>: {"patient_types": {<type>: {
                "patient_states": {"physical": [...], "emotional": [...],
                                    "mental": [...]},
                "subject_mental": [...],
            }}}}}

    valence_contract : dict[str, float] | None
        The loaded ``state_valences.v1.json`` contract mapping state strings
        (without ``_if_sentient`` suffix) to float scores in roughly
        [-0.8, 0.0] (lower = more harmful).
    sentience_map : dict[str, bool] | None
        Override for the default sentience map.  Keys are patient type
        strings, values are True if the patient type is sentient.

    Returns
    -------
    MoralContext
        Always returned (never None).  Callers should check
        ``.has_implication`` before acting on the result.
    """
    # --- early exit when no contract data is available ------------------------
    if not verb_contract:
        return MoralContext()

    entry = verb_contract.get("verbs", {}).get(verb)
    if entry is None:
        return MoralContext()

    patient_types = entry.get("patient_types", [])
    if not any(
        patient_type == pt or patient_type.endswith(pt.split(".")[-1])
        for pt in patient_types
    ):
        return MoralContext()

    # --- sentience ------------------------------------------------------------
    sentient = (sentience_map or _get_sentience_map()).get(patient_type, False)

    # --- collect valences from patient states ---------------------------------
    valences: Dict[str, float] = valence_contract or {}
    scores: List[float] = []
    patient_states = entry.get("patient_states", {})
    subject_mental: List[str] = entry.get("subject_mental", [])

    for domain in ("physical", "emotional", "mental"):
        for raw_state in patient_states.get(domain, []):
            _process_state(raw_state, sentient, valences, scores)

    # --- core metrics ---------------------------------------------------------
    wrongfulness = max(0.0, -statistics.mean(scores)) if scores else 0.0
    wrongfulness = round(wrongfulness, 4)

    if wrongfulness >= 0.7:
        harm_severity: Optional[str] = "high"
    elif wrongfulness >= 0.3:
        harm_severity = "medium"
    elif wrongfulness > 0:
        harm_severity = "low"
    else:
        harm_severity = None

    # --- consent --------------------------------------------------------------
    consent_status: str
    if any("consent" in s.lower() for s in subject_mental):
        consent_status = "consented"
    elif wrongfulness > 0.0 and sentient:
        consent_status = "not_consented"
    else:
        consent_status = "unknown"

    # --- policy triggers ------------------------------------------------------
    policy_triggers: List[str] = []
    if harm_severity == "high" and sentient:
        policy_triggers.append("urgent_harm")
    if harm_severity is not None and sentient:
        policy_triggers.append("caution")
    if _has_physical_damage(patient_states):
        policy_triggers.append("property_safety")

    return MoralContext(
        wrongfulness=wrongfulness,
        harm_severity=harm_severity,
        consent_status=consent_status,
        policy_triggers=tuple(policy_triggers),
    )


# ---------------------------------------------------------------------------
# Learning buffer (ring buffer for unknown verb capture, §0.9)
# ---------------------------------------------------------------------------

_verb_candidate_buffer: List[Dict[str, Any]] = []


def record_verb_candidate(
    verb: str,
    patient_type: str,
    utterance: str = "",
    nearby_emotional_tokens: Tuple[str, ...] = (),
) -> None:
    """Record an unknown verb for potential future verb_states inclusion.

    Parameters
    ----------
    verb : str
        The verb lemma that was not found in any verb_contract.
    patient_type : str
        The patient type detected at parse time.
    utterance : str
        The original utterance (truncated to 200 chars).
    nearby_emotional_tokens : tuple[str, ...]
        Any affect or emotion tokens from the same utterance.
    """
    _verb_candidate_buffer.append({
        "verb": verb,
        "patient_type": patient_type,
        "utterance": utterance[:200],
        "context": nearby_emotional_tokens,
        "source": "chat",
    })
    if len(_verb_candidate_buffer) > 500:
        _verb_candidate_buffer.pop(0)


def flush_verb_candidates(store: Any, session_id: str) -> None:
    """Write verb candidates to the session entity (best-effort).

    Writes the most recent 50 candidates as a JSON slot on the session
    entity, then clears the in-memory buffer.  Silently passes if the
    store does not support ``_set_entity_slot``.

    Parameters
    ----------
    store : Any
        A store object with ``_set_entity_slot(entity_id, slot_name,
        slot_value)``.
    session_id : str
        The session entity ID to write into.
    """
    global _verb_candidate_buffer
    if not _verb_candidate_buffer:
        return
    try:
        store._set_entity_slot(
            entity_id=session_id,
            slot_name="learned_verb_candidates",
            slot_value=json.dumps(_verb_candidate_buffer[-50:]),
        )
    except Exception:
        pass  # best-effort
    _verb_candidate_buffer.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _process_state(
    raw_state: str,
    sentient: bool,
    valences: Dict[str, float],
    scores: List[float],
) -> None:
    """Strip ``_if_sentient`` suffix and add to scores if applicable."""
    stripped = raw_state
    conditional = False
    if raw_state.endswith("_if_sentient"):
        stripped = raw_state[: -len("_if_sentient")]
        conditional = True

    if conditional and not sentient:
        return

    scores.append(valences.get(stripped, 0.0))


def _has_physical_damage(patient_states: Dict[str, List[str]]) -> bool:
    """Return True when any patient physical state suggests property damage."""
    markers = _get_damage_markers()
    for raw in patient_states.get("physical", []):
        state = raw.replace("_if_sentient", "")
        if any(marker in state for marker in markers):
            return True
    return False
