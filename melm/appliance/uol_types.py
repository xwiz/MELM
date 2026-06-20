"""Universal Ontology Language (UOL) atom and act types.

This module defines the semantic graph primitives used by the atomizer
and downstream frame linker.  Every field is language-neutral; surface
language is handled entirely by the Language Adapter layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class AffectSignal:
    """Affective signal from utterance-level affect detection.

    Produced by the mood engine and carried through the synthesis pipeline.
    """
    valence: float = 0.0
    arousal: float = 0.0
    confidence: float = 0.0
    source: str = "default"
    recovery_signals: tuple[str, ...] = ()
    is_complaint: bool = False
    mood_id: str = ""
    dominant_tags: tuple[str, ...] = ()
    identity_claim: bool = False
    identity_probe: bool = False
    negated: bool = False
    verb_causal_valence: float = 0.0

# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------

AtomKind = Literal[
    "state",
    "relation",
    "event",
    "change",
    "perception",
    "mental",
    "implication",
]

SpeechAct = Literal[
    "claim",
    "question",
    "command",
    "request",
    "answer",
    "warning",
    "advice_request",
    "greeting",
    "farewell",
    "unknown",
]

RoleStatus = Literal["asserted", "unresolved", "inferred", "negated"]

Polarity = Literal["positive", "negative"]
Modality = Literal[
    "assertive",
    "possibility",
    "necessity",
    "obligation",
    "counterfactual",
]


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredicateRef:
    """Canonical predicate from the predicate inventory contract."""

    id: str                      # e.g. "eat", "be", "know"
    semantic_class: str          # e.g. "verb.consume", "verb.stative"
    lemma: str = ""              # surface lemma (language-specific)
    language: str = "en"


@dataclass(frozen=True)
class RoleAssignment:
    """One semantic role filled by a surface expression."""

    role: str                    # agent, patient, theme, experiencer, beneficiary, instrument, location, time, manner, cause, purpose, result
    value: str                   # canonical referent or surface lemma
    status: RoleStatus = "asserted"
    confidence: float = 1.0


@dataclass(frozen=True)
class TimeRef:
    """Temporal interval or deictic reference for an atom.

    Minimal representation aligned with ISO-TimeML and Allen's interval model.
    Full ISO 8601 anchoring is deferred until the resolver layer is active;
    for now the atomizer populates ``tense`` and ``text`` at minimum.
    """

    text: str = ""          # surface span, e.g. "3 days ago", "tomorrow"
    tense: str = ""         # past / present / future (mirrors AtomContext.tense)
    relation: str = ""      # before / after / on / during / since / until
    granularity: str = ""   # year / month / day / hour / minute
    anchor: str = ""        # "utterance_time" or ISO 8601 string when resolved


@dataclass(frozen=True)
class AtomContext:
    """Situation context for an atom (polarity, modality, tense, aspect)."""

    polarity: Polarity = "positive"
    modality: Modality = "assertive"
    negation_scope: bool = False
    tense: str = "present"
    aspect: str = "simple"
    certainty: float = 1.0
    time: Optional[TimeRef] = None
    affect: Optional[AffectSignal] = None


@dataclass(frozen=True)
class AtomLinks:
    """Links to other atoms or entities in the discourse graph."""

    subordinate_atoms: tuple[str, ...] = ()   # atom IDs of ccomp/xcomp/advcl
    coreferent_atoms: tuple[str, ...] = ()
    entity_refs: tuple[str, ...] = ()
    temporal_anchor: str = ""


@dataclass(frozen=True)
class UolAtom:
    """Single semantic atom — the building block of UOL meaning."""

    id: str
    kind: AtomKind
    predicate: PredicateRef
    roles: tuple[RoleAssignment, ...] = ()
    context: AtomContext = field(default_factory=AtomContext)
    links: AtomLinks = field(default_factory=AtomLinks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "predicate": {
                "id": self.predicate.id,
                "semantic_class": self.predicate.semantic_class,
                "lemma": self.predicate.lemma,
                "language": self.predicate.language,
            },
            "roles": [
                {
                    "role": r.role,
                    "value": r.value,
                    "status": r.status,
                    "confidence": r.confidence,
                }
                for r in self.roles
            ],
            "context": {
                "polarity": self.context.polarity,
                "modality": self.context.modality,
                "negation_scope": self.context.negation_scope,
                "tense": self.context.tense,
                "aspect": self.context.aspect,
                "certainty": self.context.certainty,
                "time": {
                    "text": self.context.time.text,
                    "tense": self.context.time.tense,
                    "relation": self.context.time.relation,
                    "granularity": self.context.time.granularity,
                    "anchor": self.context.time.anchor,
                } if self.context.time is not None else None,
            },
            "links": {
                "subordinate_atoms": list(self.links.subordinate_atoms),
                "coreferent_atoms": list(self.links.coreferent_atoms),
                "entity_refs": list(self.links.entity_refs),
                "temporal_anchor": self.links.temporal_anchor,
            },
        }


# ---------------------------------------------------------------------------
# Speech Act wrapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UolAct:
    """Speech act wrapping a sequence of atoms."""

    id: str
    act: SpeechAct
    speaker: str = "user"
    addressee: str | None = "assistant"
    content: tuple[UolAtom, ...] = ()
    expected_answer_type: str | None = None
    urgency: str = "normal"
    policy_domain: str = "general"
    affect: Optional[AffectSignal] = None
    repetition_context: int = 0
    is_identity_probe: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "act": self.act,
            "speaker": self.speaker,
            "addressee": self.addressee,
            "content": [a.to_dict() for a in self.content],
            "expected_answer_type": self.expected_answer_type,
            "urgency": self.urgency,
            "policy_domain": self.policy_domain,
            "affect": {
                "valence": self.affect.valence,
                "arousal": self.affect.arousal,
                "confidence": self.affect.confidence,
                "source": self.affect.source,
                "recovery_signals": list(self.affect.recovery_signals),
                "is_complaint": self.affect.is_complaint,
                "mood_id": self.affect.mood_id,
                "dominant_tags": list(self.affect.dominant_tags),
                "identity_claim": self.affect.identity_claim,
                "identity_probe": self.affect.identity_probe,
                "negated": self.affect.negated,
                "verb_causal_valence": self.affect.verb_causal_valence,
            } if self.affect is not None else None,
            "repetition_context": self.repetition_context,
            "is_identity_probe": self.is_identity_probe,
        }
