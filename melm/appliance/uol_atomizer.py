"""Atomizer: converts FunctionalParse (flat slots) into UOL atoms (semantic graph).

This is the bridge between the existing weighted functional grammar and the
new language-agnostic UOL pipeline.  It preserves backward compatibility by
projecting atoms back into FunctionalParse when the old pipeline is active.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

from melm.contracts import load_function_words, load_causal_link_markers

from .functional_grammar import FunctionalParse
from .language_adapters import SyntaxGraph, function_word_entry, predicate_entry
from .uol_types import (
    AtomContext,
    AtomKind,
    AtomLinks,
    Modifier,
    PredicateRef,
    RoleAssignment,
    RoleStatus,
    TimeRef,
    UolAct,
    UolAtom,
)

# Map predicate_inventory semantic_class prefixes → atom kinds
_SEMANTIC_CLASS_TO_KIND: dict[str, AtomKind] = {
    "verb.consume": "event",
    "verb.move": "event",
    "verb.create": "event",
    "verb.communicate": "event",
    "verb.change": "change",
    "verb.cognition": "mental",
    "verb.perceive": "perception",
    "verb.possess": "relation",
    "verb.stative": "state",
    "action": "event",
    "unknown": "state",
}

# Map functional grammar token_roles → UOL semantic roles
_ROLE_MAP: dict[str, str] = {
    "grammatical_subject": "agent",
    "main_predicate": "predicate",
    "complement_predicate": "predicate",
    "secondary_predicate_candidate": "predicate",
    "semantic_object": "theme",
    "indirect_object": "beneficiary",
    "content_nominal": "theme",
    "interrogative": "information_target",
    "relation_marker": "relation",
    "determiner": "scope",
    "modal": "modality",
    "negation": "polarity",
    "frequency_modifier": "frequency",
    "politeness": "politeness",
    "discourse_greeting": "speech_act_marker",
}

# Map FunctionalParse speech_act → UolAct.act
_ACT_MAP: dict[str, str] = {
    "wh_question": "question",
    "yes_no_question": "question",
    "request": "request",
    "command": "command",
    "greeting": "greeting",
    "farewell": "farewell",
    "statement": "claim",
    "claim": "claim",
    "question": "question",
    "unknown": "unknown",
}

_FUNCTION_WORD_META: dict[tuple[str, str], dict[str, Any]] | None = None
_CAUSAL_LINK_MARKERS: dict[str, dict[str, Any]] | None = None
_MODIFIER_SEMCLASS_MAP: dict[str, str] | None = None


def _function_word_meta(lemma: str, language: str) -> dict[str, Any]:
    global _FUNCTION_WORD_META
    if _FUNCTION_WORD_META is None:
        payload = load_function_words()
        _FUNCTION_WORD_META = {}
        for entry in payload.get("entries", []):
            lang = str(entry.get("language", "en")).strip().lower()
            key = str(entry.get("lemma", "")).strip().lower()
            if lang and key:
                _FUNCTION_WORD_META[(lang, key)] = dict(entry)
    return _FUNCTION_WORD_META.get((language, lemma), {})


def _load_causal_link_markers() -> dict[str, dict[str, Any]]:
    global _CAUSAL_LINK_MARKERS
    if _CAUSAL_LINK_MARKERS is None:
        try:
            _CAUSAL_LINK_MARKERS = load_causal_link_markers()
        except Exception:
            _CAUSAL_LINK_MARKERS = {}
    return _CAUSAL_LINK_MARKERS


def _load_modifier_semclass_map() -> dict[str, str]:
    global _MODIFIER_SEMCLASS_MAP
    if _MODIFIER_SEMCLASS_MAP is None:
        try:
            from melm.contracts import load_modifier_atoms
            payload = load_modifier_atoms()
            _MODIFIER_SEMCLASS_MAP = {
                str(e["canonical_lemma"]).strip().lower(): str(e.get("semantic_class_id", "attribute"))
                for e in payload.get("entries", [])
                if e.get("canonical_lemma")
            }
        except Exception:
            _MODIFIER_SEMCLASS_MAP = {}
    return _MODIFIER_SEMCLASS_MAP


def _modifier_semantic_class(lemma: str) -> str:
    return _load_modifier_semclass_map().get(lemma.strip().lower(), "attribute")


def _extract_modifiers_from_parse(
    parse: FunctionalParse,
    index_to_role: dict[int, dict[str, Any]],
) -> list[tuple[int, str, str]]:
    """Return (adj_token_index, canonical_lemma, semantic_class_id) per ADJ token."""
    results: list[tuple[int, str, str]] = []
    for tr in parse.token_roles:
        if tr.get("role") == "adjectival_modifier":
            idx = int(tr["index"])
            lemma = str(tr.get("lemma", tr.get("token", ""))).strip().lower()
            sem_class = _modifier_semantic_class(lemma)
            results.append((idx, lemma, sem_class))
    return results


def _extract_modifiers_from_graph(
    graph: SyntaxGraph,
    allowed_indices: set[int],
) -> list[tuple[int, str, str]]:
    """Return (adj_token_index, canonical_lemma, semantic_class_id) per ADJ token."""
    amod_map: dict[int, int] = {}  # adj_index -> noun_index
    for edge in graph.dependencies:
        if edge.relation == "amod" and edge.dependent in allowed_indices:
            amod_map[edge.dependent] = edge.head
    results: list[tuple[int, str, str]] = []
    for adj_idx, noun_idx in amod_map.items():
        lemma = graph.lemmas[adj_idx].strip().lower()
        sem_class = _modifier_semantic_class(lemma)
        results.append((adj_idx, lemma, sem_class))
    return results


def _attach_modifiers(
    atom: UolAtom,
    modifiers: list[tuple[int, str, str]],
) -> UolAtom:
    """Return a new UolAtom with modifiers attached."""
    if not modifiers:
        return atom
    new_modifiers = [
        Modifier(lemma=lemma, semantic_class=sem_class, modifier_type="adjective")
        for _, lemma, sem_class in modifiers
    ]
    combined = list(atom.modifiers) + new_modifiers
    return replace(atom, modifiers=tuple(combined))


def _atom_kind(semantic_class: str) -> AtomKind:
    for prefix, kind in _SEMANTIC_CLASS_TO_KIND.items():
        if semantic_class.startswith(prefix):
            return kind
    return "state"


def _parse_predicate(meaning: str) -> tuple[str, str]:
    """Split 'eat:verb.consume' → ('eat', 'verb.consume')."""
    if ":" in meaning:
        return meaning.split(":", 1)
    return meaning, "unknown"


_NounEntityIndex: dict[str, dict[str, str]] | None = None


def _load_noun_entity_index() -> dict[str, dict[str, str]]:
    """Lazy-load noun_atoms.v1.json into a lemma → {semantic_class, entity_id} index."""
    global _NounEntityIndex
    if _NounEntityIndex is not None:
        return _NounEntityIndex
    try:
        from melm.contracts.validation import load_contract_json
        payload = load_contract_json("noun_atoms.v1.json")
        mapping: dict[str, dict[str, str]] = {}
        for entry in payload.get("entities", []):
            lemma = str(entry.get("canonical_lemma", "")).strip().lower()
            if lemma:
                mapping[lemma] = {
                    "semantic_class": str(entry.get("semantic_class_id", "")),
                    "entity_id": str(entry.get("entity_id", "")),
                }
        _NounEntityIndex = mapping
    except Exception:
        _NounEntityIndex = {}
    return _NounEntityIndex


def enrich_role_entities(
    roles: list[RoleAssignment],
) -> list[RoleAssignment]:
    """Enrich role values that match known noun entities with semantic_class + entity_id.
    
    Lazy-loads noun_atoms.v1.json — silently degrades to pass-through on error.
    """
    index = _load_noun_entity_index()
    if not index:
        return roles
    enriched: list[RoleAssignment] = []
    for role in roles:
        entry = index.get(role.value.strip().lower())
        if entry:
            enriched.append(replace(
                role,
                semantic_class=entry["semantic_class"],
                entity_id=entry["entity_id"],
            ))
        else:
            enriched.append(role)
    return enriched


_VerbAtomIndex: dict[str, str] | None = None

def _load_verb_atom_index() -> dict[str, str]:
    global _VerbAtomIndex
    if _VerbAtomIndex is not None:
        return _VerbAtomIndex
    try:
        from melm.contracts.validation import load_contract_json
        payload = load_contract_json("verb_atoms.v1.json")
        mapping: dict[str, str] = {}
        for entry in payload.get("entities", []):
            lemma = str(entry.get("canonical_lemma", "")).strip().lower()
            if lemma:
                mapping[lemma] = str(entry.get("entity_id", ""))
        _VerbAtomIndex = mapping
    except Exception:
        _VerbAtomIndex = {}
    return _VerbAtomIndex


def enrich_verb_predicate(predicate: PredicateRef) -> PredicateRef:
    """Enrich a predicate with entity_id from verb_atoms.v1.json.

    Matches by predicate.lemma (or predicate.id as fallback).
    Silently degrades to pass-through when contract is unavailable.
    """
    if predicate.entity_id:
        return predicate
    index = _load_verb_atom_index()
    if not index:
        return predicate
    key = (predicate.lemma or predicate.id).strip().lower()
    eid = index.get(key)
    if eid:
        return replace(predicate, entity_id=eid)
    return predicate


def atomize(parse: FunctionalParse | None, language: str = "en") -> UolAct | None:
    """Convert a FunctionalParse into a UolAct wrapping UolAtoms.

    Returns ``None`` when *parse* is ``None`` (fragment / unparsable).
    """
    if parse is None:
        return None

    act_type: str = _ACT_MAP.get(parse.speech_act, "unknown")
    atoms: list[UolAtom] = []

    # Build a map from token index → role info
    index_to_role: dict[int, dict[str, Any]] = {}
    for tr in parse.token_roles:
        index_to_role[tr["index"]] = dict(tr)

    # Main clause atom from the primary predicate
    main_pred = _extract_predicate(parse, language)
    if main_pred:
        main_pred = enrich_verb_predicate(main_pred)
        main_roles = _extract_roles(parse, index_to_role, main_pred.id)
        main_roles = enrich_role_entities(main_roles)
        ctx = _extract_context(parse, index_to_role, language=language)
        main_modifiers = _extract_modifiers_from_parse(parse, index_to_role)
        main_atom = UolAtom(
            id=_new_id(),
            kind=_atom_kind(main_pred.semantic_class),
            predicate=main_pred,
            roles=tuple(main_roles),
            context=ctx,
        )
        main_atom = _attach_modifiers(main_atom, main_modifiers)
        atoms.append(main_atom)

    # Complement predicate → separate atom (if different from main)
    if parse.complement_action and parse.complement_action != parse.action:
        comp_pred = enrich_verb_predicate(PredicateRef(
            id=parse.complement_action,
            semantic_class=_guess_semantic_class(parse.complement_action),
            language=language,
        ))
        comp_roles: list[RoleAssignment] = []
        comp_atom = UolAtom(
            id=_new_id(),
            kind=_atom_kind(comp_pred.semantic_class),
            predicate=comp_pred,
            roles=tuple(comp_roles),
            context=AtomContext(polarity=ctx.polarity if "ctx" in dir() else "positive"),
            links=AtomLinks(subordinate_atoms=(atoms[0].id,) if atoms else ()),
        )
        atoms.append(comp_atom)

    return UolAct(
        id=_new_id(),
        act=act_type,  # type: ignore[arg-type]
        speaker=parse.subject if parse.subject else "user",
        addressee=parse.target if parse.target else "assistant",
        content=tuple(atoms),
        expected_answer_type=_expected_answer_type(parse, index_to_role, language=language),
    )


def atomize_syntax_graph(graph: SyntaxGraph | None) -> UolAct | None:
    """Convert a SyntaxGraph into a UolAct without going through FunctionalParse."""
    if graph is None or not graph.tokens:
        return None

    root_index = 0
    for edge in graph.dependencies:
        if edge.relation == "root":
            root_index = edge.dependent
            break

    all_indices = set(range(len(graph.lemmas)))
    causal_markers = _load_causal_link_markers()
    subordinate_specs: list[tuple[int, dict[str, Any]]] = []
    for edge in graph.dependencies:
        if edge.relation != "advcl" or edge.head != root_index:
            continue
        subordinate_index = edge.dependent
        marker_info = None
        for mark_edge in graph.dependencies:
            if mark_edge.relation == "mark" and mark_edge.head == subordinate_index:
                marker_info = causal_markers.get(graph.lemmas[mark_edge.dependent])
                break
        if marker_info is None:
            continue
        subordinate_specs.append((subordinate_index, marker_info))

    subordinate_indices = set[int]()
    for subordinate_index, _ in subordinate_specs:
        subordinate_indices |= _clause_indices(graph, subordinate_index)
    main_indices = all_indices - subordinate_indices

    predicate = enrich_verb_predicate(_predicate_from_graph(graph, root_index))
    roles = _roles_from_graph(graph, root_index, main_indices)
    roles = enrich_role_entities(roles)
    context = _context_from_graph(graph)
    main_modifiers = _extract_modifiers_from_graph(graph, main_indices)
    main_atom = UolAtom(
        id=_new_id(),
        kind=_atom_kind(predicate.semantic_class),
        predicate=predicate,
        roles=tuple(roles),
        context=context,
    )
    main_atom = _attach_modifiers(main_atom, main_modifiers)
    atoms: list[UolAtom] = [main_atom]

    for subordinate_index, marker_info in subordinate_specs:
        sub_indices = _clause_indices(graph, subordinate_index)
        sub_predicate = enrich_verb_predicate(_predicate_from_graph(graph, subordinate_index))
        sub_roles = _roles_from_graph(graph, subordinate_index, sub_indices)
        sub_roles = enrich_role_entities(sub_roles)
        sub_context = _context_from_graph(graph)
        sub_modifiers = _extract_modifiers_from_graph(graph, sub_indices)
        sub_atom = UolAtom(
            id=_new_id(),
            kind=_atom_kind(sub_predicate.semantic_class),
            predicate=sub_predicate,
            roles=tuple(sub_roles),
            context=sub_context,
        )
        sub_atom = _attach_modifiers(sub_atom, sub_modifiers)
        atoms.append(sub_atom)
        main_atom, sub_atom = _link_causal_atoms(
            main_atom, sub_atom, marker_info.get("relation", ""), marker_info.get("introduces", "cause")
        )
        atoms[0] = main_atom
        atoms[-1] = sub_atom

    return UolAct(
        id=_new_id(),
        act=_speech_act_from_graph(graph),  # type: ignore[arg-type]
        speaker="user",
        addressee="assistant",
        content=tuple(atoms),
        expected_answer_type=_expected_answer_type_from_graph(graph),
    )


def _extract_predicate(parse: FunctionalParse, language: str) -> PredicateRef | None:
    """Pull the primary predicate from the FunctionalParse."""
    if not parse.action:
        return None
    # Try to find semantic_class from candidates
    semantic_class = "unknown"
    for cand in parse.candidates:
        if cand.get("action") == parse.action:
            semantic_class = cand.get("semantic_class", "unknown")
            break
    return PredicateRef(
        id=parse.action,
        semantic_class=semantic_class,
        language=language,
    )


def _predicate_from_graph(graph: SyntaxGraph, root_index: int) -> PredicateRef:
    lemma = graph.lemmas[root_index]
    entry = predicate_entry(graph.language, lemma)
    if entry:
        return PredicateRef(
            id=str(entry.get("predicate_id", lemma)),
            semantic_class=str(entry.get("semantic_class", "unknown")),
            lemma=lemma,
            language=graph.language,
        )
    return PredicateRef(
        id=lemma,
        semantic_class="unknown",
        lemma=lemma,
        language=graph.language,
    )


def _extract_roles(
    parse: FunctionalParse,
    index_to_role: dict[int, dict[str, Any]],
    predicate_id: str,
) -> list[RoleAssignment]:
    """Map token_roles to UOL RoleAssignments for the main clause."""
    roles: list[RoleAssignment] = []
    for idx, tr in index_to_role.items():
        role_name = tr.get("role", "")
        meaning = tr.get("meaning", "")
        weight = tr.get("weight", 0.5)

        if role_name == "main_predicate":
            roles.append(
                RoleAssignment(role="predicate", value=predicate_id, confidence=weight)
            )
        elif role_name == "grammatical_subject":
            referent = meaning if meaning else tr.get("lemma", "")
            roles.append(
                RoleAssignment(role="agent", value=referent, confidence=weight)
            )
        elif role_name == "semantic_object":
            obj = meaning if meaning else tr.get("lemma", "")
            roles.append(
                RoleAssignment(role="theme", value=obj, confidence=weight)
            )
        elif role_name == "indirect_object":
            iobj = meaning if meaning else tr.get("lemma", "")
            roles.append(
                RoleAssignment(role="beneficiary", value=iobj, confidence=weight)
            )
        elif role_name == "content_nominal":
            cn = meaning if meaning else tr.get("lemma", "")
            roles.append(
                RoleAssignment(role="theme", value=cn, confidence=weight)
            )
        elif role_name == "interrogative":
            roles.append(
                RoleAssignment(
                    role="information_target",
                    value=meaning if meaning else tr.get("lemma", ""),
                    confidence=weight,
                )
            )
        elif role_name == "negation":
            roles.append(
                RoleAssignment(
                    role="polarity",
                    value="negative",
                    confidence=weight,
                )
            )
        elif role_name == "modal":
            roles.append(
                RoleAssignment(
                    role="modality",
                    value=tr.get("lemma", ""),
                    confidence=weight,
                )
            )
        elif role_name == "relation_marker":
            roles.append(
                RoleAssignment(
                    role="relation",
                    value=tr.get("lemma", ""),
                    confidence=weight,
                )
            )
    return roles


def _clause_indices(graph: SyntaxGraph, root_index: int) -> set[int]:
    """Return all token indices syntactically governed by ``root_index``."""
    indices: set[int] = {root_index}
    stack = [root_index]
    while stack:
        head = stack.pop()
        for edge in graph.dependencies:
            if edge.head == head and edge.dependent not in indices:
                indices.add(edge.dependent)
                stack.append(edge.dependent)
    return indices


def _roles_from_graph(
    graph: SyntaxGraph, root_index: int, allowed_indices: set[int]
) -> list[RoleAssignment]:
    roles: list[RoleAssignment] = []
    root_predicate = _predicate_from_graph(graph, root_index)
    roles.append(RoleAssignment(role="predicate", value=root_predicate.id, confidence=1.0))
    # Tokens that are clause markers (e.g. "because", "if") should not become roles.
    marker_indices = {edge.dependent for edge in graph.dependencies if edge.relation == "mark"}
    for index, lemma in enumerate(graph.lemmas):
        if index not in allowed_indices or index == root_index or index in marker_indices:
            continue
        entry = function_word_entry(graph.language, lemma)
        role = str(entry.get("role", "")).strip().lower()
        subrole = str(entry.get("subrole", "")).strip().lower()
        referent = str(entry.get("referent", "")).strip().lower()
        status: RoleStatus = "asserted"
        confidence = 0.9
        if role == "wh_word":
            roles.append(
                RoleAssignment(
                    role="information_target",
                    value="requested_information_dimension",
                    confidence=confidence,
                )
            )
            continue
        if role == "pronoun" and subrole in {"agent", "agent_or_patient"}:
            roles.append(
                RoleAssignment(
                    role="agent",
                    value=referent or lemma,
                    confidence=confidence,
                )
            )
            continue
        if role == "pronoun" and subrole in {"patient", "human_collective", "human_indefinite"}:
            roles.append(
                RoleAssignment(
                    role="beneficiary",
                    value=referent or lemma,
                    confidence=confidence,
                )
            )
            continue
        if role == "modal":
            roles.append(RoleAssignment(role="modality", value=lemma, confidence=confidence))
            continue
        if role == "negation":
            roles.append(RoleAssignment(role="polarity", value="negative", confidence=confidence))
            continue
        if graph.pos_tags[index] == "NOUN":
            if not predicate_entry(graph.language, lemma) and not function_word_entry(graph.language, lemma):
                status = "unresolved" if root_predicate.semantic_class == "unknown" else "asserted"
            roles.append(
                RoleAssignment(
                    role="theme",
                    value=lemma,
                    status=status,
                    confidence=0.88,
                )
            )
    return roles


def _extract_context(
    parse: FunctionalParse,
    index_to_role: dict[int, dict[str, Any]],
    language: str = "en",
) -> AtomContext:
    """Derive AtomContext from negation/modal markers in token_roles."""
    polarity: str = "positive"
    modality: str = "assertive"
    negation_scope = False
    tense = "present"
    temporal_text = ""

    for tr in index_to_role.values():
        role = tr.get("role", "")
        if role == "negation":
            polarity = "negative"
            negation_scope = True
        elif role == "modal":
            lemma = tr.get("lemma", "")
            subrole = str(
                _function_word_meta(str(lemma), language).get("subrole", "")
            ).strip().lower()
            if subrole in {"possibility", "necessity", "obligation"}:
                modality = subrole
            elif subrole == "future":
                tense = "future"
                temporal_text = str(lemma)
        elif role in {"frequency_modifier", "relation_marker"}:
            lemma = str(tr.get("lemma", ""))
            meta = _function_word_meta(lemma, language)
            tense_hint = str(meta.get("tense", "")).strip().lower()
            if tense_hint in {"past", "future"}:
                tense = tense_hint
                temporal_text = lemma

    time_ref = TimeRef(text=temporal_text, tense=tense, anchor="utterance_time") if tense != "present" or temporal_text else None
    return AtomContext(
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        negation_scope=negation_scope,
        tense=tense,
        time=time_ref,
    )


def _context_from_graph(graph: SyntaxGraph) -> AtomContext:
    polarity = "positive"
    modality = "assertive"
    tense = "present"
    negation_scope = False
    temporal_text = ""
    for lemma in graph.lemmas:
        entry = function_word_entry(graph.language, lemma)
        role = str(entry.get("role", "")).strip().lower()
        subrole = str(entry.get("subrole", "")).strip().lower()
        if role == "negation":
            polarity = "negative"
            negation_scope = True
        elif role == "modal":
            if subrole in {"possibility", "necessity", "obligation"}:
                modality = subrole
            elif subrole == "future":
                tense = "future"
                temporal_text = lemma
        elif role == "temporal":
            tense_hint = subrole if subrole in {"past", "future"} else ""
            if tense_hint:
                tense = tense_hint
                temporal_text = lemma

    time_ref = TimeRef(text=temporal_text, tense=tense, anchor="utterance_time") if tense != "present" or temporal_text else None
    return AtomContext(
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        negation_scope=negation_scope,
        tense=tense,
        time=time_ref,
    )


def _expected_answer_type(
    parse: FunctionalParse,
    index_to_role: dict[int, dict[str, Any]],
    language: str = "en",
) -> str | None:
    """Infer expected answer type from wh_word subroles."""
    if parse.speech_act not in {"wh_question", "question"}:
        return None
    for tr in index_to_role.values():
        if tr.get("role") == "interrogative":
            lemma = tr.get("lemma", "")
            meta = _function_word_meta(str(lemma), language)
            answer_type = str(meta.get("answer_type", "")).strip().lower()
            return answer_type or str(meta.get("subrole", "")).strip().lower() or None
    return None


def _expected_answer_type_from_graph(graph: SyntaxGraph) -> str | None:
    for lemma in graph.lemmas:
        entry = function_word_entry(graph.language, lemma)
        if str(entry.get("role", "")).strip().lower() != "wh_word":
            continue
        answer_type = str(entry.get("answer_type", "")).strip().lower()
        if answer_type:
            return answer_type
        subrole = str(entry.get("subrole", "")).strip().lower()
        if subrole:
            return subrole
    return None


def _speech_act_from_graph(graph: SyntaxGraph) -> str:
    if len(graph.tokens) == 1 and str(function_word_entry(graph.language, graph.lemmas[0]).get("role", "")).strip().lower() == "greeting":
        return "greeting"
    if any(str(function_word_entry(graph.language, lemma).get("role", "")).strip().lower() == "wh_word" for lemma in graph.lemmas):
        return "question"
    if any(str(function_word_entry(graph.language, lemma).get("role", "")).strip().lower() == "modal" for lemma in graph.lemmas):
        return "question"
    root_index = 0
    for edge in graph.dependencies:
        if edge.relation == "root":
            root_index = edge.dependent
            break
    first_content_index = 0
    for index, lemma in enumerate(graph.lemmas):
        role = str(function_word_entry(graph.language, lemma).get("role", "")).strip().lower()
        if role not in {"politeness", "determiner"}:
            first_content_index = index
            break
    has_subject_before_root = any(
        edge.relation == "nsubj" and edge.head == root_index and edge.dependent < root_index
        for edge in graph.dependencies
    )
    has_subject_after_root = any(
        edge.relation in {"nsubj", "dep"} and edge.head == root_index and edge.dependent > root_index
        for edge in graph.dependencies
    )
    if (
        root_index == first_content_index == 0
        and graph.pos_tags
        and graph.pos_tags[root_index] == "AUX"
        and has_subject_after_root
    ):
        return "question"
    if any(str(function_word_entry(graph.language, lemma).get("role", "")).strip().lower() == "politeness" for lemma in graph.lemmas):
        if predicate_entry(graph.language, graph.lemmas[root_index]):
            return "request"
    if root_entry := predicate_entry(graph.language, graph.lemmas[root_index]):
        if str(root_entry.get("predicate_id", "")).strip():
            if root_index == first_content_index and not has_subject_before_root:
                if not has_subject_after_root:
                    return "command"
                return "request"
            return "claim"
    return "claim"


def _guess_semantic_class(action: str) -> str:
    """Best-effort semantic class via predicate_inventory contract lookup."""
    from .language_adapters import predicate_entry
    for lang in ("en",):
        entry = predicate_entry(lang, action)
        if entry:
            sc = str(entry.get("semantic_class", "")).strip()
            if sc:
                return sc
    return "unknown"


def _link_causal_atoms(
    main_atom: UolAtom,
    sub_atom: UolAtom,
    relation: str,
    introduces: str,
) -> tuple[UolAtom, UolAtom]:
    """Return (main_atom, sub_atom) with causal links populated.

    ``introduces`` is whether the subordinate clause introduced by the marker
    is the ``cause`` or the ``effect``.
    """
    if relation == "causes":
        if introduces == "cause":
            # Subordinate causes main.
            main_atom = replace(main_atom, links=replace(main_atom.links, caused_by=main_atom.links.caused_by + (sub_atom.id,)))
            sub_atom = replace(sub_atom, links=replace(sub_atom.links, causes=sub_atom.links.causes + (main_atom.id,)))
        else:
            # Main causes subordinate.
            main_atom = replace(main_atom, links=replace(main_atom.links, causes=main_atom.links.causes + (sub_atom.id,)))
            sub_atom = replace(sub_atom, links=replace(sub_atom.links, caused_by=sub_atom.links.caused_by + (main_atom.id,)))
    elif relation == "caused_by":
        if introduces == "cause":
            # Subordinate is the cause (same as "causes" introduces=cause).
            main_atom = replace(main_atom, links=replace(main_atom.links, caused_by=main_atom.links.caused_by + (sub_atom.id,)))
            sub_atom = replace(sub_atom, links=replace(sub_atom.links, causes=sub_atom.links.causes + (main_atom.id,)))
        else:
            # Main is the cause of the subordinate.
            main_atom = replace(main_atom, links=replace(main_atom.links, causes=main_atom.links.causes + (sub_atom.id,)))
            sub_atom = replace(sub_atom, links=replace(sub_atom.links, caused_by=sub_atom.links.caused_by + (main_atom.id,)))
    elif relation == "enables":
        main_atom = replace(main_atom, links=replace(main_atom.links, enables=main_atom.links.enables + (sub_atom.id,)))
        sub_atom = replace(sub_atom, links=replace(sub_atom.links, enables=sub_atom.links.enables + (main_atom.id,)))
    elif relation == "prevents":
        main_atom = replace(main_atom, links=replace(main_atom.links, prevents=main_atom.links.prevents + (sub_atom.id,)))
        sub_atom = replace(sub_atom, links=replace(sub_atom.links, prevents=sub_atom.links.prevents + (main_atom.id,)))
    return main_atom, sub_atom


def _new_id() -> str:
    return f"uol_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Backward-compatible projection
# ---------------------------------------------------------------------------


def atoms_to_functional_parse(act: UolAct) -> FunctionalParse | None:
    """Project UolAct atoms back into the legacy FunctionalParse shape.

    Used during migration so the router/synthesis layers can consume atoms
    while downstream code still expects FunctionalParse.
    """
    if not act.content:
        return None

    main_atom = act.content[0]
    pred = main_atom.predicate

    # Reconstruct token_roles from atom roles (best-effort)
    token_roles: list[dict[str, Any]] = []
    for r in main_atom.roles:
        if r.role == "predicate":
            token_roles.append(
                {
                    "index": 0,
                    "token": pred.lemma or pred.id,
                    "lemma": pred.lemma or pred.id,
                    "role": "main_predicate",
                    "meaning": f"{pred.id}:{pred.semantic_class}",
                    "weight": r.confidence,
                }
            )
        elif r.role == "agent":
            token_roles.append(
                {
                    "index": 1,
                    "token": r.value,
                    "lemma": r.value,
                    "role": "grammatical_subject",
                    "meaning": r.value,
                    "weight": r.confidence,
                }
            )
        elif r.role == "theme":
            token_roles.append(
                {
                    "index": 2,
                    "token": r.value,
                    "lemma": r.value,
                    "role": "semantic_object",
                    "meaning": r.value,
                    "weight": r.confidence,
                }
            )

    # Simple speech_act mapping back
    speech_act_map: dict[str, str] = {
        "question": "wh_question",
        "request": "request",
        "command": "command",
        "greeting": "greeting",
        "claim": "statement",
    }
    speech_act = speech_act_map.get(act.act, act.act)

    return FunctionalParse(
        speech_act=speech_act,
        subject=act.speaker,
        action=pred.id,
        object="",
        target=act.addressee or "",
        complement_action="",
        indirect_object="",
        modifiers={},
        relations=(),
        token_roles=tuple(token_roles),
        candidates=(),
        parse_score=0.8,
        syntactic_coverage=0.8,
        semantic_unknown_tokens=(),
        pattern="projected_from_atoms",
    )
