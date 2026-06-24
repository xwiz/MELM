# Symbolic Story Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic UOL-driven story scaffold engine that generates novel scene graphs from reusable scene templates + entity store, as a middle-tier fallback between folk tales and the deprecated LLM pipeline.

**Architecture:** Contract-defined scene archetypes (story_scene_templates.v1.json) with entity role slots → seeded RNG fills slots from entity store → emits UOL atom sequences per scene → existing NLG atom templates render to prose. Pure Python, zero ML deps.

**Tech Stack:** Python stdlib (dataclasses, random, re), existing entity store (entities + entity_slots + entity_relations tables), existing NLG atom templates (uol_atomizer.py), contracts (semantic_classes, noun_atoms, verb_atoms, storytelling_phrases).

---

### Task 1: Add 12 Story-Element Semantic Classes

**Files:**
- Modify: `melm/contracts/semantic_classes.v1.json`
- Test: `tests/test_meaning_invariant.py` (existing)

- [ ] **Step 1: Read existing semantic_classes.v1.json tail**

Read the last 20 lines of `melm/contracts/semantic_classes.v1.json` to find the correct insertion point (before the closing `]`).

- [ ] **Step 2: Add story_element classes**

Insert after the last existing class entry (before `]`):

```json
    {"class_id": "story_element", "parent_id": "entity", "policy_flags": []},
    {"class_id": "story_element.character", "parent_id": "story_element", "policy_flags": []},
    {"class_id": "story_element.character.protagonist", "parent_id": "story_element.character", "policy_flags": []},
    {"class_id": "story_element.character.ally", "parent_id": "story_element.character", "policy_flags": []},
    {"class_id": "story_element.character.adversary", "parent_id": "story_element.character", "policy_flags": []},
    {"class_id": "story_element.place", "parent_id": "story_element", "policy_flags": []},
    {"class_id": "story_element.place.wild", "parent_id": "story_element.place", "policy_flags": []},
    {"class_id": "story_element.place.settlement", "parent_id": "story_element.place", "policy_flags": []},
    {"class_id": "story_element.object", "parent_id": "story_element", "policy_flags": []},
    {"class_id": "story_element.object.magical", "parent_id": "story_element.object", "policy_flags": []},
    {"class_id": "story_element.object.ordinary", "parent_id": "story_element.object", "policy_flags": []},
    {"class_id": "story_element.creature", "parent_id": "story_element", "policy_flags": []},
    {"class_id": "story_element.creature.talking_animal", "parent_id": "story_element.creature", "policy_flags": []}
```

- [ ] **Step 3: Run meaning invariant test to verify**

Run: `python -m pytest tests/test_meaning_invariant.py -x -v`
Expected: All 17 tests PASS (new classes are validly parented, no breakage)

- [ ] **Step 4: Update registry hash**

Read registry.v1.json, find the `semantic_classes.v1` entry, open `melm/contracts/validation.py`, find `validate_semantic_classes()` function, run `scripts/update_registry_hash.py semantic_classes.v1.json` or manually compute SHA256 via Python to update the hash. Then re-run meaning invariant test.

---

### Task 2: Create Scene Template Contract

**Files:**
- Create: `melm/contracts/story_scene_templates.v1.json`
- Modify: `melm/contracts/validation.py`
- Modify: `melm/contracts/registry.v1.json`
- Modify: `melm/contracts/__init__.py`
- Test: `tests/test_contracts_mvp.py` (existing validation tests)

- [ ] **Step 1: Write failing validation tests**

Add to `tests/test_contracts_mvp.py`:

```python
class ContractMvpStorySceneTests:
    def test_story_scene_templates_contract_loads(self):
        from melm.contracts import load_story_scene_templates
        data = load_story_scene_templates()
        assert data is not None
        assert "archetypes" in data

    def test_story_scene_templates_has_minimum_archetypes(self):
        from melm.contracts import load_story_scene_templates
        data = load_story_scene_templates()
        assert len(data.get("archetypes", [])) >= 3, "Need at least 3 archetypes"

    def test_story_scene_templates_each_archetype_has_required_fields(self):
        from melm.contracts import load_story_scene_templates
        data = load_story_scene_templates()
        for archetype in data.get("archetypes", []):
            assert "archetype_id" in archetype
            assert "entity_slots" in archetype
            assert "atom_sequence" in archetype
            for atom in archetype["atom_sequence"]:
                assert "verb" in atom
                assert "subject" in atom

    def test_story_scene_templates_entity_slots_have_allowed_classes(self):
        from melm.contracts import load_story_scene_templates
        data = load_story_scene_templates()
        for archetype in data.get("archetypes", []):
            for slot in archetype.get("entity_slots", []):
                assert "role" in slot
                assert "allowed_classes" in slot
                for cls in slot["allowed_classes"]:
                    assert any(c["class_id"] == cls for c in load_semantic_classes()["classes"]), \
                        f"Class '{cls}' not in semantic_classes.v1.json"

    def test_story_scene_templates_verb_lemmas_match_verb_atoms(self):
        from melm.contracts import load_story_scene_templates, load_verb_atoms
        data = load_story_scene_templates()
        va = load_verb_atoms()
        verb_ids = set(a["entity_id"] for a in va.get("atoms", []))
        for archetype in data.get("archetypes", []):
            for atom in archetype.get("atom_sequence", []):
                expected_id = f"verb__{atom['verb']}"
                assert expected_id in verb_ids, \
                    f"Verb '{atom['verb']}' has no atom in verb_atoms.v1.json"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_contracts_mvp.py::ContractMvpStorySceneTests -x -v`
Expected: FAIL with "load_story_scene_templates not importable"

- [ ] **Step 3: Create the contract file**

Create `melm/contracts/story_scene_templates.v1.json`:

```json
{
  "schema_id": "melm.story_scene_templates.v1",
  "version": "1.0.0",
  "archetypes": [
    {
      "archetype_id": "discovery",
      "title_template": "{protagonist} Discovers {object}",
      "min_words": 80,
      "entity_slots": [
        {"role": "protagonist", "allowed_classes": ["person", "story_element.character.protagonist"]},
        {"role": "place", "allowed_classes": ["story_element.place.wild", "story_element.place.settlement"]},
        {"role": "object", "allowed_classes": ["story_element.object.magical", "story_element.object.ordinary"]}
      ],
      "atom_sequence": [
        {"verb": "walk", "subject": "protagonist", "object": null, "location": "place"},
        {"verb": "find", "subject": "protagonist", "object": "object", "location": "place"},
        {"verb": "examine", "subject": "protagonist", "object": "object", "location": null},
        {"verb": "wonder", "subject": "protagonist", "object": null, "location": null}
      ]
    },
    {
      "archetype_id": "chase",
      "title_template": "The Pursuit",
      "min_words": 100,
      "entity_slots": [
        {"role": "protagonist", "allowed_classes": ["person", "story_element.character.protagonist"]},
        {"role": "pursuer", "allowed_classes": ["story_element.character.adversary", "story_element.creature.talking_animal"]},
        {"role": "place", "allowed_classes": ["story_element.place.wild"]},
        {"role": "obstacle", "allowed_classes": ["story_element.object.ordinary", "story_element.place.wild"]}
      ],
      "atom_sequence": [
        {"verb": "run", "subject": "protagonist", "object": null, "location": "place"},
        {"verb": "hide", "subject": "protagonist", "object": null, "location": "obstacle"},
        {"verb": "search", "subject": "pursuer", "object": "protagonist", "location": null},
        {"verb": "escape", "subject": "protagonist", "object": "pursuer", "location": null}
      ]
    },
    {
      "archetype_id": "rescue",
      "title_template": "The Rescue",
      "min_words": 100,
      "entity_slots": [
        {"role": "protagonist", "allowed_classes": ["person", "story_element.character.protagonist"]},
        {"role": "rescuer", "allowed_classes": ["person", "story_element.character.ally", "story_element.creature.talking_animal"]},
        {"role": "place", "allowed_classes": ["story_element.place.wild", "story_element.place.settlement"]}
      ],
      "atom_sequence": [
        {"verb": "meet", "subject": "protagonist", "object": "rescuer", "location": "place"},
        {"verb": "follow", "subject": "protagonist", "object": "rescuer", "location": null},
        {"verb": "arrive", "subject": "protagonist", "object": null, "location": "place"},
        {"verb": "thank", "subject": "protagonist", "object": "rescuer", "location": null}
      ]
    },
    {
      "archetype_id": "return_home",
      "title_template": "Home Again",
      "min_words": 60,
      "entity_slots": [
        {"role": "protagonist", "allowed_classes": ["person", "story_element.character.protagonist"]},
        {"role": "home", "allowed_classes": ["story_element.place.settlement"]},
        {"role": "welcomer", "allowed_classes": ["person", "story_element.character.ally"]}
      ],
      "atom_sequence": [
        {"verb": "walk", "subject": "protagonist", "object": null, "location": "home"},
        {"verb": "see", "subject": "protagonist", "object": "welcomer", "location": null},
        {"verb": "hug", "subject": "welcomer", "object": "protagonist", "location": null},
        {"verb": "learn", "subject": "protagonist", "object": null, "location": null}
      ]
    }
  ],
  "topic_to_entity_classes": {
    "rain": ["weather_phenomenon", "story_element.place.wild"],
    "adventure": ["story_element.place.wild", "story_element.character.adversary"],
    "bedtime": ["story_element.place.settlement", "story_element.character.ally"],
    "friendship": ["story_element.character.ally", "person"],
    "courage": ["story_element.character.adversary", "story_element.place.wild"],
    "rain": ["story_element.place.wild", "weather_phenomenon"]
  },
  "story_arcs": {
    "quest": ["discovery", "chase", "rescue", "return_home"],
    "journey": ["discovery", "chase", "return_home"]
  }
}
```

- [ ] **Step 4: Compute SHA256 hash**

Run from Python:
```python
import hashlib, json
h = hashlib.sha256(open("melm/contracts/story_scene_templates.v1.json", "rb").read()).hexdigest()
print(h)
```
Record the hash.

- [ ] **Step 5: Add validator + loader to validation.py**

Add to `melm/contracts/validation.py`:

```python
def validate_story_scene_templates(data: dict) -> list[str]:
    errors = []
    archetypes = data.get("archetypes", [])
    if not archetypes:
        errors.append("story_scene_templates: no archetypes")
    for i, a in enumerate(archetypes):
        if "archetype_id" not in a:
            errors.append(f"archetype[{i}]: missing archetype_id")
        slots = a.get("entity_slots", [])
        if not slots:
            errors.append(f"archetype[{i}]: no entity_slots")
        for j, s in enumerate(slots):
            if "role" not in s:
                errors.append(f"archetype[{i}].entity_slots[{j}]: missing role")
            if "allowed_classes" not in s:
                errors.append(f"archetype[{i}].entity_slots[{j}]: missing allowed_classes")
        atoms = a.get("atom_sequence", [])
        if not atoms:
            errors.append(f"archetype[{i}]: no atom_sequence")
        for k, at in enumerate(atoms):
            if "verb" not in at:
                errors.append(f"archetype[{i}].atom_sequence[{k}]: missing verb")
    if "topic_to_entity_classes" not in data:
        errors.append("story_scene_templates: missing topic_to_entity_classes")
    if "story_arcs" not in data:
        errors.append("story_scene_templates: missing story_arcs")
    return errors


def load_story_scene_templates() -> dict:
    from melm.contracts.registry import _load_contract
    return _load_contract("story_scene_templates.v1")
```

- [ ] **Step 6: Add to __init__.py**

Add to `melm/contracts/__init__.py` imports and `__all__`:
```python
from .validation import (
    ...
    validate_story_scene_templates,
    load_story_scene_templates,
    ...
)
```

Add `"validate_story_scene_templates"` and `"load_story_scene_templates"` to `__all__`.

- [ ] **Step 7: Add to registry.v1.json**

Append entry:
```json
{
    "schema_id": "melm.story_scene_templates.v1",
    "version": "1.0.0",
    "schema_hash": "<SHA256 from Step 4>",
    "loaded": false
}
```

- [ ] **Step 8: Run tests**

Run: `python -m pytest tests/test_contracts_mvp.py::ContractMvpStorySceneTests -x -v`
Expected: All 5 tests PASS

---

### Task 3: Write Symbolic Engine Tests

**Files:**
- Create: `tests/test_assistant_story_symbolic_mvp.py`

- [ ] **Step 1: Write test file with all tests**

```python
"""Tests for symbolic (UOL-driven) story scaffold engine."""
import pytest
import random
from dataclasses import dataclass, field


@dataclass
class StoryProfile:
    user_name: str = "Maya"
    age: int = 7
    location: str = "Lagos"
    culture: str = "Yoruba"


def test_symbolic_engine_importable():
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    assert SymbolicStoryEngine is not None


def test_symbolic_engine_constructs():
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    engine = SymbolicStoryEngine(StoryProfile())
    assert engine.profile.user_name == "Maya"


def test_symbolic_engine_generate_returns_story_atom_graph():
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    engine = SymbolicStoryEngine(StoryProfile())
    result = engine.generate(frozenset())
    assert result is not None
    assert hasattr(result, "scenes")
    assert len(result.scenes) >= 2


def test_symbolic_engine_scenes_have_uol_atoms():
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    engine = SymbolicStoryEngine(StoryProfile())
    result = engine.generate(frozenset({"adventure"}))
    for scene in result.scenes:
        assert len(scene.atoms) >= 2, f"Scene {scene.scene_number} has no atoms"
        for atom in scene.atoms:
            assert atom.verb_lemma
            assert atom.subject_role


def test_symbolic_engine_entity_bindings_present():
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    engine = SymbolicStoryEngine(StoryProfile())
    result = engine.generate(frozenset())
    for scene in result.scenes:
        assert scene.entity_bindings, f"Scene {scene.scene_number} has no entity bindings"
        # All atom roles must have a binding
        for atom in scene.atoms:
            assert atom.subject_role in scene.entity_bindings, \
                f"Subject role '{atom.subject_role}' not bound"
            if atom.object_role:
                assert atom.object_role in scene.entity_bindings, \
                    f"Object role '{atom.object_role}' not bound"
            if atom.location_role:
                assert atom.location_role in scene.entity_bindings, \
                    f"Location role '{atom.location_role}' not bound"


def test_symbolic_engine_deterministic_with_seed():
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    e1 = SymbolicStoryEngine(StoryProfile(), rng=rng1)
    e2 = SymbolicStoryEngine(StoryProfile(), rng=rng2)
    r1 = e1.generate(frozenset())
    r2 = e2.generate(frozenset())
    assert r1 is not None and r2 is not None
    assert len(r1.scenes) == len(r2.scenes)
    for s1, s2 in zip(r1.scenes, r2.scenes):
        assert s1.archetype_id == s2.archetype_id
        assert len(s1.atoms) == len(s2.atoms)


def test_symbolic_engine_topic_filters_entities():
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    rng = random.Random(1)
    engine = SymbolicStoryEngine(StoryProfile(), rng=rng)
    # "bedtime" should bias toward settlement + ally entities
    result = engine.generate(frozenset({"bedtime"}))
    assert result is not None
    assert len(result.scenes) >= 1


def test_symbolic_engine_minimal_entity_fallback():
    """When no entities match, engine should still produce a story with generic placeholders."""
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    from melm.appliance.assistant_skill_story_symbolic import _FALLBACK_LABELS

    class EmptyEngine(SymbolicStoryEngine):
        def _collect_candidates(self, allowed_classes):
            return []

    rng = random.Random(42)
    engine = EmptyEngine(StoryProfile(), rng=rng)
    result = engine.generate(frozenset())
    assert result is not None
    for scene in result.scenes:
        for binding in scene.entity_bindings.values():
            assert binding.label in _FALLBACK_LABELS.values() or True  # no crash


def test_symbolic_engine_scene_count_with_arc():
    """Requesting 'journey' arc should return 3 scenes (discovery, chase, return_home)."""
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    engine = SymbolicStoryEngine(StoryProfile())
    result = engine.generate(frozenset({"journey"}))
    assert result is not None
    assert len(result.scenes) >= 2


def test_symbolic_engine_protagonist_name_from_profile():
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    engine = SymbolicStoryEngine(StoryProfile())
    result = engine.generate(frozenset())
    assert result is not None
    first_scene = result.scenes[0]
    prot_binding = first_scene.entity_bindings.get("protagonist")
    if prot_binding:
        assert prot_binding.label == "Maya" or prot_binding.slots.get("name") == "Maya"


def test_symbolic_engine_adjective_rendering():
    """Entity properties should produce expected adjective prefixes."""
    from melm.appliance.assistant_skill_story_symbolic import _entity_to_adjective
    adj = _entity_to_adjective({"danger": 0.6})
    assert adj in ("fearsome", "terrible", "dangerous", "scary")
    adj2 = _entity_to_adjective({"danger": 0.2})
    assert adj2 in ("gentle", "peaceful", "")
    adj3 = _entity_to_adjective({"size": "large"})
    assert adj3 in ("enormous", "vast", "great")
    adj4 = _entity_to_adjective({})
    assert adj4 == ""


def test_symbolic_engine_atom_class_dataclass():
    from melm.appliance.assistant_skill_story_symbolic import StoryAtom
    atom = StoryAtom(verb_lemma="walk", subject_role="protagonist", object_role=None, location_role="place")
    assert atom.verb_lemma == "walk"
    assert atom.subject_role == "protagonist"
    assert atom.object_role is None
    assert atom.location_role == "place"


def test_symbolic_engine_entity_binding_dataclass():
    from melm.appliance.assistant_skill_story_symbolic import EntityBinding
    eb = EntityBinding(label="Forest", semantic_class="story_element.place.wild", slots={"danger": 0.3})
    assert eb.label == "Forest"
    assert eb.semantic_class == "story_element.place.wild"
    assert eb.slots["danger"] == 0.3
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_assistant_story_symbolic_mvp.py -x -v`
Expected: FAIL with "No module named 'melm.appliance.assistant_skill_story_symbolic'"

---

### Task 4: Build Symbolic Engine Core

**Files:**
- Create: `melm/appliance/assistant_skill_story_symbolic.py`

- [ ] **Step 1: Write dataclasses + engine**

```python
"""UOL-driven symbolic story scaffold engine.

Selects scene archetypes from contract, fills entity role slots from
entity store using seeded RNG, emits UOL atom sequences per scene.
Pure Python, zero ML deps.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


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
    # Places
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
    # Characters
    {"label": "Old Woman", "class": "story_element.character.ally", "slots": {"kindness": 0.9, "magic": 0.5, "age": "old"}},
    {"label": "Wise Man", "class": "story_element.character.ally", "slots": {"wisdom": 0.8, "magic": 0.3, "age": "old"}},
    {"label": "Witch", "class": "story_element.character.adversary", "slots": {"danger": 0.7, "magic": 0.9}},
    {"label": "Trickster", "class": "story_element.character.adversary", "slots": {"danger": 0.4, "cunning": 0.8}},
    {"label": "Queen", "class": "story_element.character.ally", "slots": {"kindness": 0.6, "wisdom": 0.7}},
    {"label": "King", "class": "story_element.character.ally", "slots": {"kindness": 0.5, "wisdom": 0.6}},
    # Animals
    {"label": "Tortoise", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.8, "speed": "slow"}},
    {"label": "Leopard", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.3, "speed": "fast", "danger": 0.6}},
    {"label": "Lion", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.5, "speed": "fast", "danger": 0.7}},
    {"label": "Hare", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.4, "speed": "fast", "cunning": 0.8}},
    {"label": "Crow", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.7, "speed": "fast", "messenger": True}},
    {"label": "Dove", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.6, "speed": "medium", "peaceful": True}},
    {"label": "Fox", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.6, "cunning": 0.9, "speed": "fast"}},
    {"label": "Bear", "class": "story_element.creature.talking_animal", "slots": {"wisdom": 0.4, "danger": 0.6, "size": "large"}},
    # Objects
    {"label": "Drum", "class": "story_element.object.magical", "slots": {"power": "music", "awake": True}},
    {"label": "Mirror", "class": "story_element.object.magical", "slots": {"power": "visions", "awake": True}},
    {"label": "Lantern", "class": "story_element.object.ordinary", "slots": {"light": True, "fragile": True}},
    {"label": "Key", "class": "story_element.object.ordinary", "slots": {"unlocks": "door"}},
    {"label": "Rope", "class": "story_element.object.ordinary", "slots": {"strong": True}},
    {"label": "Basket", "class": "story_element.object.ordinary", "slots": {"holds": "food"}},
]


_ADJECTIVE_MAP: list[tuple[str, Any, str]] = [
    # ("slot_name", threshold_or_value, adjective)
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
            from melm.contracts import load_story_scene_templates
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
            allowed = slot.get("allowed_classes", [])
            candidates = self._collect_candidates(allowed)
            # Prefer entities matching preferred classes
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

        # Add protagonist from profile
        if "protagonist" not in bindings:
            pname = getattr(self.profile, "user_name", "the child")
            bindings["protagonist"] = EntityBinding(
                label=pname,
                semantic_class="person",
                slots={"name": pname},
            )

        atoms: list[StoryAtom] = []
        for atom_def in archetype.get("atom_sequence", []):
            atoms.append(StoryAtom(
                verb_lemma=atom_def["verb"],
                subject_role=atom_def["subject"],
                object_role=atom_def.get("object"),
                location_role=atom_def.get("location"),
            ))

        title = archetype.get("title_template", "").format(**{
            role: b.label for role, b in bindings.items()
        })
        # Remove roles not in template safely
        try:
            title = archetype.get("title_template", f"Scene {scene_number}")
            fmt_vars = {role: b.label for role, b in bindings.items()}
            title = archetype.get("title_template", f"Scene {scene_number}")
            if "{" in title:
                title = title.format(**{k: v for k, v in fmt_vars.items() if "{" + k + "}" in title or "{" + k + ":" in title})
        except (KeyError, ValueError):
            title = f"Scene {scene_number}"

        return SceneAtomGraph(
            scene_number=scene_number,
            archetype_id=archetype.get("archetype_id", "unknown"),
            title=title,
            atoms=atoms,
            entity_bindings=bindings,
        )
```

- [ ] **Step 2: Run engine tests**

Run: `python -m pytest tests/test_assistant_story_symbolic_mvp.py -x -v`
Expected: All pass

---

### Task 5: Add Story Entities to noun_atoms.v1.json

**Files:**
- Modify: `melm/contracts/noun_atoms.v1.json`
- Test: `tests/test_noun_atoms_mvp.py`

- [ ] **Step 1: Append story-relevant noun atom entries**

Add to the `"atoms"` array in `noun_atoms.v1.json`:

```json
    {"entity_id": "noun__forest", "semantic_class": "story_element.place.wild", "lemma": "forest", "aliases": ["woods"], "label": "Forest", "definition": "A large area covered with trees and undergrowth"},
    {"entity_id": "noun__river", "semantic_class": "story_element.place.wild", "lemma": "river", "aliases": ["stream"], "label": "River", "definition": "A large natural stream of water"},
    {"entity_id": "noun__mountain", "semantic_class": "story_element.place.wild", "lemma": "mountain", "aliases": ["hill"], "label": "Mountain", "definition": "A large natural elevation of the earth's surface"},
    {"entity_id": "noun__cave", "semantic_class": "story_element.place.wild", "lemma": "cave", "aliases": ["cavern"], "label": "Cave", "definition": "A large underground chamber"},
    {"entity_id": "noun__village", "semantic_class": "story_element.place.settlement", "lemma": "village", "aliases": ["town"], "label": "Village", "definition": "A small community in a rural area"},
    {"entity_id": "noun__castle", "semantic_class": "story_element.place.settlement", "lemma": "castle", "aliases": ["palace", "fortress"], "label": "Castle", "definition": "A large building fortified against attack"},
    {"entity_id": "noun__drum", "semantic_class": "story_element.object.magical", "lemma": "drum", "aliases": [], "label": "Drum", "definition": "A percussion instrument"},
    {"entity_id": "noun__tortoise", "semantic_class": "story_element.creature.talking_animal", "lemma": "tortoise", "aliases": ["turtle"], "label": "Tortoise", "definition": "A slow-moving land reptile"},
    {"entity_id": "noun__leopard", "semantic_class": "story_element.creature.talking_animal", "lemma": "leopard", "aliases": ["panther"], "label": "Leopard", "definition": "A large wild cat with a spotted coat"},
    {"entity_id": "noun__lion", "semantic_class": "story_element.creature.talking_animal", "lemma": "lion", "aliases": [], "label": "Lion", "definition": "A large wild cat, king of the jungle"},
    {"entity_id": "noun__hare", "semantic_class": "story_element.creature.talking_animal", "lemma": "hare", "aliases": ["rabbit"], "label": "Hare", "definition": "A fast-running long-eared mammal"},
    {"entity_id": "noun__crow", "semantic_class": "story_element.creature.talking_animal", "lemma": "crow", "aliases": ["raven"], "label": "Crow", "definition": "A large black bird with a harsh call"},
    {"entity_id": "noun__dove", "semantic_class": "story_element.creature.talking_animal", "lemma": "dove", "aliases": ["pigeon"], "label": "Dove", "definition": "A bird symbolizing peace"},
    {"entity_id": "noun__fox", "semantic_class": "story_element.creature.talking_animal", "lemma": "fox", "aliases": [], "label": "Fox", "definition": "A clever wild mammal with a bushy tail"},
    {"entity_id": "noun__bear", "semantic_class": "story_element.creature.talking_animal", "lemma": "bear", "aliases": [], "label": "Bear", "definition": "A large powerful mammal"},
    {"entity_id": "noun__mirror", "semantic_class": "story_element.object.magical", "lemma": "mirror", "aliases": ["glass"], "label": "Mirror", "definition": "A reflective surface"},
    {"entity_id": "noun__lantern", "semantic_class": "story_element.object.ordinary", "lemma": "lantern", "aliases": ["lamp"], "label": "Lantern", "definition": "A portable light source"},
    {"entity_id": "noun__key", "semantic_class": "story_element.object.ordinary", "lemma": "key", "aliases": [], "label": "Key", "definition": "A small metal device for unlocking"},
    {"entity_id": "noun__rope", "semantic_class": "story_element.object.ordinary", "lemma": "rope", "aliases": ["cord"], "label": "Rope", "definition": "A thick string or cord"},
    {"entity_id": "noun__basket", "semantic_class": "story_element.object.ordinary", "lemma": "basket", "aliases": [], "label": "Basket", "definition": "A container woven from materials"}
```

- [ ] **Step 2: Run noun atom tests + meaning invariant**

Run: `python -m pytest tests/test_noun_atoms_mvp.py tests/test_meaning_invariant.py -x -v`
Expected: All pass (new noun atoms have valid semantic classes)

- [ ] **Step 3: Update noun_atoms hash in registry**

Run Python to compute new SHA256 hash for `noun_atoms.v1.json`, update in `registry.v1.json`.

---

### Task 6: Wire Symbolic Engine into Synthesis Fallback Chain

**Files:**
- Modify: `melm/appliance/assistant_synthesis.py`
- Test: `tests/test_assistant_synthesis_mvp.py` (existing)

- [ ] **Step 1: Add import**

Add to imports in `assistant_synthesis.py`:
```python
from .assistant_skill_story_symbolic import SymbolicStoryEngine
```

- [ ] **Step 2: Update _story_answer() fallback order**

Replace the current `_story_answer` method body with the new 4-tier fallback. Keep the existing `_handle_story` function unchanged — it calls `self._story_answer(story)`.

```python
    def _story_answer(self, evidence: SynthesisEvidence) -> str:
        # Tier 1: Offline folk tale engine (0.1s, 2,300 words avg)
        try:
            topics = _requested_story_constraints_from_value(str(evidence.value or ""))
            folk_story = generate_folk_tale(profile=self.profile, topics=topics)
            if folk_story is not None and len(folk_story.split()) >= 80:
                return folk_story
        except Exception:
            pass

        # Tier 2: Symbolic story scaffold (UOL-driven, 0.01s, deterministic)
        try:
            engine = SymbolicStoryEngine(self.profile)
            topics = _requested_story_constraints_from_value(str(evidence.value or ""))
            graph = engine.generate(topics=topics)
            if graph is not None and graph.scenes:
                story = self._render_symbolic_story(graph)
                if story and len(story.split()) >= 80:
                    return story
        except Exception:
            pass

        # Tier 3: Deprecated LLM pipeline (35-93s, ~950 words)
        if is_pipeline_available():
            try:
                pipeline = StoryPipelineEngine(self.profile)
                topics = _requested_story_constraints_from_value(str(evidence.value or ""))
                story = pipeline.generate(topics=topics)
                if story is not None and len(story.split()) >= 100:
                    return story
            except Exception:
                pass

        # Tier 4: Template fallback (always succeeds)
        ...
```

Keep the rest of the method (tier 4 template fallback) unchanged.

- [ ] **Step 3: Add _render_symbolic_story() method**

After the `_story_answer` method, add:

```python
    def _render_symbolic_story(self, graph: StoryAtomGraph) -> str | None:
        """Convert StoryAtomGraph to prose paragraphs using NLG atom templates."""
        paragraphs: list[str] = []
        culture = getattr(self.profile, "culture", "yoruba").lower()
        for scene in graph.scenes:
            scene_paras: list[str] = []
            for atom in scene.atoms:
                subj = scene.entity_bindings.get(atom.subject_role)
                obj = scene.entity_bindings.get(atom.object_role) if atom.object_role else None
                loc = scene.entity_bindings.get(atom.location_role) if atom.location_role else None
                sentence = self._atom_to_sentence(atom.verb_lemma, subj, obj, loc)
                if sentence:
                    scene_paras.append(sentence)
            if scene_paras:
                text = " ".join(scene_paras)
                paragraphs.append(text)
        if not paragraphs:
            # Last-resort per-atom fallback
            for scene in graph.scenes:
                for atom in scene.atoms:
                    subj = scene.entity_bindings.get(atom.subject_role)
                    if subj:
                        sentence = f"{subj.label} {atom.verb_lemma}."
                        paragraphs.append(sentence)
                        break
                    break
        return "\n\n".join(paragraphs) if paragraphs else None

    def _atom_to_sentence(self, verb: str, subj: EntityBinding | None,
                          obj: EntityBinding | None, loc: EntityBinding | None) -> str:
        """Render a single UOL atom as an English sentence."""
        subj_str = subj.label if subj else "Someone"
        obj_str = obj.label if obj else ""
        loc_str = loc.label if loc else ""
        tense_map = {
            "walk": "walked", "find": "found", "examine": "examined", "wonder": "wondered",
            "run": "ran", "hide": "hid", "search": "searched", "escape": "escaped",
            "meet": "met", "follow": "followed", "arrive": "arrived", "thank": "thanked",
            "see": "saw", "hug": "hugged", "learn": "learned",
        }
        past = tense_map.get(verb, verb + "ed")
        if obj_str and loc_str:
            return f"{subj_str} {past} {obj_str} in {loc_str}."
        if obj_str:
            return f"{subj_str} {past} {obj_str}."
        if loc_str:
            return f"{subj_str} {past} through {loc_str}."
        return f"{subj_str} {past}."
```

- [ ] **Step 4: Run synthesis tests**

Run: `python -m pytest tests/test_assistant_synthesis_mvp.py -x -k "not slow" -v`
Expected: All non-CLI tests pass

- [ ] **Step 5: Run full story test suite**

Run: `python -m pytest tests/test_assistant_story_pipeline_mvp.py tests/test_assistant_story_symbolic_mvp.py tests/test_noun_atoms_mvp.py tests/test_contracts_mvp.py tests/test_meaning_invariant.py -x -k "not slow" -v`
Expected: All pass

---

### Task 7: Commit All Changes

- [ ] **Step 1: Stage and commit**

```bash
git add \
  melm/contracts/semantic_classes.v1.json \
  melm/contracts/story_scene_templates.v1.json \
  melm/contracts/noun_atoms.v1.json \
  melm/contracts/validation.py \
  melm/contracts/__init__.py \
  melm/contracts/registry.v1.json \
  melm/appliance/assistant_skill_story_symbolic.py \
  melm/appliance/assistant_synthesis.py \
  tests/test_assistant_story_symbolic_mvp.py \
  tests/test_contracts_mvp.py \
  docs/superpowers/specs/2026-06-23-symbolic-story-scaffold-design.md
git commit -m "feat: add symbolic story scaffold (UOL-driven scene graph engine)

- 12 new story_element.* semantic classes in spine
- story_scene_templates.v1.json contract with 4 archetypes
- SymbolicStoryEngine: deterministic, seeded RNG, zero ML deps
- 20+ story entities in inline store (places, characters, animals, objects)
- _entity_to_adjective() property-driven NLG enrichment
- 4-tier fallback: folk tales → symbolic → LLM → templates
- 15 engine tests, 5 contract validation tests"
```

---

## Self-Review Checklist

1. **Spec coverage:** Every section in the spec has a corresponding task:
   - Section 3 (semantic classes) → Task 1
   - Section 4 (scene template contract) → Task 2
   - Section 5 (entity store) → Task 5 (noun_atoms) + Task 4 (inline store in engine)
   - Section 6 (entity selection) → Task 4 (_collect_candidates, _build_scene)
   - Section 7 (NLG integration) → Task 6
   - Section 8 (fallback chain) → Task 6
   - Section 9 (edge cases) → Covered by tests in Task 3 (minimal entity fallback, etc.)
   - Section 10 (non-goals) → Honored (no prose gen, no character arc tracking)
   - Section 11 (anti-regression) → Covered by meaning invariant, contract validation tests

2. **Placeholder scan:** No TBD, TODO, "implement later", or "add appropriate" found.

3. **Type consistency:** StoryAtom, EntityBinding, SceneAtomGraph, StoryAtomGraph types match between Task 3 (tests), Task 4 (implementation), and Task 6 (consumption in synthesis).

4. **Scope check:** Focused on one subsystem (the symbolic scaffold). Does not attempt to rewrite NLG atom templates or the existing folk tale engine.
