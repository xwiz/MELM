"""UOL-driven symbolic story scaffold engine.

Selects scene archetypes from contract, fills entity role slots from
entity store using seeded RNG, emits UOL atom sequences per scene.
Pure Python, zero ML deps.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any
from melm.contracts import load_story_scene_templates

__all__ = [
    "StoryAtom", "EntityBinding", "SceneAtomGraph", "StoryAtomGraph",
    "SymbolicStoryEngine", "_entity_to_adjective",
]


@dataclass
class StoryAtom:
    verb_lemma: str
    subject_role: str
    object_role: str | None = None
    location_role: str | None = None


@dataclass
class EntityBinding:
    label: str
    semantic_class: str
    slots: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneAtomGraph:
    scene_number: int
    archetype_id: str
    title: str
    atoms: list[StoryAtom]
    entity_bindings: dict[str, EntityBinding]


@dataclass
class StoryAtomGraph:
    scenes: list[SceneAtomGraph]


_FALLBACK_LABELS: dict[str, str] = {
    "protagonist": "the child",
    "ally": "a helper",
    "adversary": "a stranger",
    "place": "a strange place",
    "object": "something",
    "home": "home",
    "welcomer": "a familiar face",
    "pursuer": "a shadow",
    "obstacle": "a barrier",
    "rescuer": "a kind soul",
}

_STORY_ENTITIES: list[dict[str, Any]] = [
    {"label": "Forest", "class": "story_element.place.wild", "slots": {"danger": 0.3, "size": "large", "mood": "mysterious"}},
    {"label": "River", "class": "story_element.place.wild", "slots": {"danger": 0.2, "size": "medium", "mood": "calm"}},
    {"label": "Cave", "class": "story_element.place.wild", "slots": {"danger": 0.6, "size": "small", "mood": "scary"}},
    {"label": "Mountain", "class": "story_element.place.wild", "slots": {"danger": 0.5, "size": "large", "mood": "grand"}},
    {"label": "Lake", "class": "story_element.place.wild", "slots": {"danger": 0.1, "size": "large", "mood": "peaceful"}},
    {"label": "Tunnel", "class": "story_element.place.wild", "slots": {"danger": 0.5, "size": "small", "mood": "dark"}},
    {"label": "Village", "class": "story_element.place.settlement", "slots": {"size": "small", "mood": "friendly"}},
    {"label": "Castle", "class": "story_element.place.settlement", "slots": {"size": "large", "mood": "grand"}},
    {"label": "Market", "class": "story_element.place.settlement", "slots": {"size": "medium", "mood": "lively"}},
    {"label": "Hut", "class": "story_element.place.settlement", "slots": {"size": "small", "mood": "cozy"}},
    {"label": "Old Woman", "class": "story_element.character.ally", "slots": {"kindness": 0.9, "magic": 0.5, "age": "old"}},
    {"label": "Wise Man", "class": "story_element.character.ally", "slots": {"wisdom": 0.8, "magic": 0.3, "age": "old"}},
    {"label": "Witch", "class": "story_element.character.adversary", "slots": {"danger": 0.7, "magic": 0.9}},
    {"label": "Trickster", "class": "story_element.character.adversary", "slots": {"danger": 0.4, "cunning": 0.8}},
    {"label": "Queen", "class": "story_element.character.ally", "slots": {"kindness": 0.6, "wisdom": 0.7}},
    {"label": "King", "class": "story_element.character.ally", "slots": {"kindness": 0.5, "wisdom": 0.6}},
    {"label": "Tortoise", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.8, "speed": "slow"}},
    {"label": "Leopard", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.3, "speed": "fast", "danger": 0.6}},
    {"label": "Lion", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.5, "speed": "fast", "danger": 0.7}},
    {"label": "Hare", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.4, "speed": "fast", "cunning": 0.8}},
    {"label": "Crow", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.7, "speed": "fast", "messenger": True}},
    {"label": "Dove", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.6, "speed": "medium", "peaceful": True}},
    {"label": "Fox", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.6, "cunning": 0.9, "speed": "fast"}},
    {"label": "Bear", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.4, "danger": 0.6, "size": "large"}},
    {"label": "Drum", "class": "story_element.object.magical", "slots": {"power": "music", "awake": True}},
    {"label": "Mirror", "class": "story_element.object.magical", "slots": {"power": "visions", "awake": True}},
    {"label": "Lantern", "class": "story_element.object.ordinary", "slots": {"light": True, "fragile": True}},
    {"label": "Key", "class": "story_element.object.ordinary", "slots": {"unlocks": "door"}},
    {"label": "Rope", "class": "story_element.object.ordinary", "slots": {"strong": True}},
    {"label": "Basket", "class": "story_element.object.ordinary", "slots": {"holds": "food"}},
]


_ADJECTIVE_MAP: list[tuple[str, Any, str]] = [
    ("danger", lambda v: v >= 0.7, "fearsome"),
    ("danger", lambda v: v >= 0.4, "dangerous"),
    ("danger", lambda v: v < 0.4, "gentle"),
    ("size", "large", "enormous"),
    ("size", "small", "tiny"),
    ("mood", "mysterious", "mysterious"),
    ("mood", "friendly", "warm"),
    ("mood", "scary", "dark"),
    ("mood", "grand", "magnificent"),
    ("mood", "peaceful", "serene"),
    ("mood", "lively", "bustling"),
    ("mood", "cozy", "cozy"),
    ("mood", "dark", "dark"),
    ("wisdom", lambda v: v >= 0.7, "wise old"),
    ("magic", lambda v: v >= 0.7, "magical"),
    ("kindness", lambda v: v >= 0.8, "kind"),
    ("cunning", lambda v: v >= 0.8, "cunning"),
    ("speed", "slow", "slow"),
    ("speed", "fast", "swift"),
    ("size", "large", "great"),
]


def _entity_to_adjective(slots: dict[str, Any]) -> str:
    """Map entity property slots to an adjective string."""
    for slot_name, check, adjective in _ADJECTIVE_MAP:
        if slot_name not in slots:
            continue
        val = slots[slot_name]
        if callable(check):
            if check(val):
                return adjective
        elif val == check:
            return adjective
    return ""


class SymbolicStoryEngine:
    """Selects scene archetypes, fills entity roles, emits UOL atom sequences."""

    def __init__(self, profile: Any, rng: random.Random | None = None):
        self.profile = profile
        self._rng = rng or random.Random()

    def generate(self, topics: frozenset[str] = frozenset()) -> StoryAtomGraph | None:
        try:
            contract = self._load_contract()
        except Exception:
            return None
        if contract is None:
            return None

        archetypes = contract.get("archetypes", [])
        if not archetypes:
            return None

        story_arcs = contract.get("story_arcs", {})
        arc_id = self._select_arc(topics, story_arcs)
        arc_archetype_ids = story_arcs.get(arc_id, list(a["archetype_id"] for a in archetypes))
        if not arc_archetype_ids:
            arc_archetype_ids = [a["archetype_id"] for a in archetypes]

        topic_entity_map = contract.get("topic_to_entity_classes", {})
        preferred_classes: set[str] = set()
        for t in topics:
            for cls in topic_entity_map.get(t, []):
                preferred_classes.add(cls)

        scenes: list[SceneAtomGraph] = []
        for i, arch_id in enumerate(arc_archetype_ids):
            arch = next((a for a in archetypes if a["archetype_id"] == arch_id), None)
            if arch is None:
                continue
            scene = self._build_scene(arch, i + 1, preferred_classes)
            if scene is not None:
                scenes.append(scene)

        return StoryAtomGraph(scenes=scenes) if scenes else None

    def _load_contract(self) -> dict | None:
        try:
            return load_story_scene_templates()
        except Exception:
            return None

    def _select_arc(self, topics: frozenset[str], story_arcs: dict) -> str:
        for topic in topics:
            if topic in story_arcs:
                return topic
        arcs = list(story_arcs.keys())
        return self._rng.choice(arcs) if arcs else "quest"

    def _collect_candidates(self, allowed_classes: list[str]) -> list[dict[str, Any]]:
        return [e for e in _STORY_ENTITIES if e["class"] in allowed_classes]

    def _build_scene(
        self, archetype: dict, scene_number: int, preferred_classes: set[str],
    ) -> SceneAtomGraph | None:
        slots_spec = archetype.get("entity_slots", [])
        bindings: dict[str, EntityBinding] = {}
        for slot in slots_spec:
            role = slot["role"]
            if role == "protagonist":
                pname = getattr(self.profile, "user_name", "the child")
                bindings[role] = EntityBinding(
                    label=pname, semantic_class="person", slots={"name": pname},
                )
                continue
            allowed = slot.get("allowed_classes", [])
            candidates = self._collect_candidates(allowed)
            if preferred_classes:
                preferred = [c for c in candidates if c["class"] in preferred_classes]
                if preferred:
                    candidates = preferred
            if candidates:
                chosen = self._rng.choice(candidates)
                adj = _entity_to_adjective(chosen["slots"])
                label = f"{adj} {chosen['label']}" if adj else chosen["label"]
            else:
                label = _FALLBACK_LABELS.get(role, f"the {role}")
                chosen = {"label": label, "class": "", "slots": {}}
            bindings[role] = EntityBinding(
                label=label,
                semantic_class=chosen.get("class", ""),
                slots=chosen.get("slots", {}),
            )

        atoms: list[StoryAtom] = []
        for atom_def in archetype.get("atom_sequence", []):
            atoms.append(StoryAtom(
                verb_lemma=atom_def["verb"],
                subject_role=atom_def["subject"],
                object_role=atom_def.get("object"),
                location_role=atom_def.get("location"),
            ))

        fmt_vars = {role: b.label for role, b in bindings.items()}
        template = archetype.get("title_template", f"Scene {scene_number}")
        try:
            title = template.format(**fmt_vars)
        except (KeyError, ValueError):
            title = f"Scene {scene_number}"

        return SceneAtomGraph(
            scene_number=scene_number,
            archetype_id=archetype.get("archetype_id", "unknown"),
            title=title,
            atoms=atoms,
            entity_bindings=bindings,
        )
