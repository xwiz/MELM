# Symbolic Story Scaffold (UOL-Driven)

**Date**: 2026-06-23  
**Status**: Draft  
**Supersedes**: `docs/superpowers/specs/2026-06-23-story-pipeline-design.md` (LLM pipeline deprecated for story generation)  

## 1. Purpose

Generate novel story scene graphs from reusable scene templates + entity store, expressed as UOL atom sequences, rendered by existing NLG atom templates. Middle tier between folk tales (fastest, highest quality) and the deprecated LLM pipeline (last resort).

## 2. Architecture

```
User Profile + Topics
        │
        v
 SymbolicStoryEngine.generate()
        │
        ├── 1. Topic → entity-type mapping
        ├── 2. Select scene archetypes from contract (3-5)
        ├── 3. Fill entity role slots from store (seeded RNG)
        └── 4. Emit StoryAtomGraph (UOL atoms per scene)
                │
                v
        Existing NLG atom templates → prose
```

### Files

| File | Role |
|------|------|
| `melm/appliance/assistant_skill_story_symbolic.py` | Engine: entity selection, scene composition, UOL emission |
| `melm/contracts/story_scene_templates.v1.json` | Scene archetypes with UOL atom slot sequences + entity role slots |
| `melm/contracts/semantic_classes.v1.json` | Extended with 12 new `story_element.*` classes |
| `melm/contracts/noun_atoms.v1.json` | Extended with ~50 story-relevant entity entries |
| `tests/test_assistant_story_symbolic_mvp.py` | Tests |

### Data Types

```python
@dataclass
class StoryAtomGraph:
    scenes: list[SceneAtomGraph]

@dataclass
class SceneAtomGraph:
    scene_number: int
    archetype_id: str
    title: str
    atoms: list[StoryAtom]  # UOL atoms with entity refs
    entity_bindings: dict[str, EntityBinding]  # role → bound entity

@dataclass
class StoryAtom:
    verb_lemma: str
    subject_role: str      # references entity_bindings key
    object_role: str | None
    location_role: str | None

@dataclass
class EntityBinding:
    label: str
    semantic_class: str
    slots: dict[str, Any]  # narrative properties
```

## 3. New Semantic Classes (12 entries)

Added to `semantic_classes.v1.json` parallel to existing classes — NOT replacing them:

```
story_element                       (parent: entity)
  story_element.character           (parent: story_element)
    story_element.character.protagonist
    story_element.character.ally
    story_element.character.adversary
  story_element.place               (parent: story_element)
    story_element.place.wild        (forest, river, mountain, cave, well, dungeon, tunnel)
    story_element.place.settlement  (village, castle, market, hut, kingdom)
  story_element.object              (parent: story_element)
    story_element.object.magical    (magic mirror, flying carpet, talking drum)
    story_element.object.ordinary   (rope, basket, lantern, key)
  story_element.creature            (parent: story_element)
    story_element.creature.talking_animal  (tortoise, leopard, lion, hare, crow, dove)
```

These live alongside the existing `animal`, `location.public_place`, `person` classes. An entity can have multiple semantic classes (e.g., `person` + `story_element.character.adversary`).

## 4. Scene Template Contract

`story_scene_templates.v1.json`:

```json
{
  "schema_id": "melm.story_scene_templates.v1",
  "version": "1.0.0",
  "archetypes": [
    {
      "archetype_id": "discovery",
      "title_template": "{protagonist} discovers {object}",
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

Note: `verb` values are lemmas — they must match entries in `verb_atoms.v1.json` or `verb_states.v1.json` for NLG rendering. The engine does not validate this at runtime; the contract author is responsible for consistency.
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
    "rain": ["story_element.place.wild", "weather_phenomenon"],
    "adventure": ["story_element.place.wild", "story_element.character.adversary"],
    "bedtime": ["story_element.place.settlement", "story_element.character.ally"],
    "friendship": ["story_element.character.ally", "person"],
    "courage": ["story_element.character.adversary", "story_element.place.wild"]
  },
  "story_arcs": {
    "quest": ["discovery", "chase", "rescue", "return_home"],
    "journey": ["discovery", "chase", "return_home"]
  }
}
```

## 5. Entity Store Population

Entities are seeded into the store on first use. Each entity carries narrative property slots:

| Entity | semantic_class | Example Slots |
|--------|---------------|---------------|
| Forest | `story_element.place.wild` | `danger: 0.3`, `size: large`, `mood: mysterious` |
| River | `story_element.place.wild` | `danger: 0.2`, `size: medium`, `mood: calm` |
| Cave | `story_element.place.wild` | `danger: 0.6`, `size: small`, `mood: scary` |
| Village | `story_element.place.settlement` | `size: small`, `mood: friendly` |
| Castle | `story_element.place.settlement` | `size: large`, `mood: grand` |
| Tortoise | `story_element.creature.talking_animal` | `wisdom: 0.8`, `speed: slow` |
| Leopard | `story_element.creature.talking_animal` | `wisdom: 0.3`, `speed: fast`, `danger: 0.6` |
| Old Woman | `story_element.character.ally` | `kindness: 0.9`, `magic: 0.5` |
| Witch | `story_element.character.adversary` | `danger: 0.7`, `magic: 0.9` |
| Magic Drum | `story_element.object.magical` | `power: music`, `awake: true` |

Properties influence NLG via a `_entity_to_adjective()` mapping in the engine:

| Slot | Value | Adjective |
|------|-------|-----------|
| `danger` | ≥0.7 | "fearsome", "terrible" |
| `danger` | ≥0.4 | "dangerous", "scary" |
| `danger` | <0.4 | "gentle", "peaceful" |
| `size` | large | "enormous", "vast", "great" |
| `size` | small | "tiny", "little" |
| `mood` | mysterious | "mysterious", "strange" |
| `mood` | friendly | "warm", "welcoming" |
| `mood` | scary | "dark", "frightening" |
| `wisdom` | ≥0.7 | "wise old" |
| `magic` | ≥0.7 | "magical", "enchanted" |

The adjective is prefixed to the entity label during NLG rendering (e.g., `danger: 0.6` + `"Cave"` → `"the fearsome Cave"`). Entities with no matching properties render as plain label (`"the Forest"`).

## 6. Entity Selection Strategy

```
topics = {"rain", "adventure"}

1. Topic → entity-type mapping (from topic_entity_map section in contract):
   "rain"   → ["weather_phenomenon", "story_element.place.wild"]
   "adventure" → ["story_element.place.wild", "story_element.character.adversary"]

2. For each scene archetype's entity_slots:
   a. Score candidate entities by class match + topic relevance
   b. Select highest-scoring (seeded RNG breaks ties)
   c. Fallback: any entity matching allowed_classes
   d. Last resort: generic placeholder ("a strange creature")

3. No topic topics → all-entities pool (pure RNG selection)
```

## 7. NLG Integration

The `StoryAtomGraph` is consumed by a new `_render_symbolic_story()` function in synthesis:

1. Each `atom` maps to existing atom NLG templates via verb_lemma lookup
2. Entity bindings provide subject/object/location surface text + property-driven adjectives
3. `storytelling_phrases.v1.json` provides cultural transition phrases between scenes
4. The final output is prose paragraphs (one per scene)

## 8. Fallback Chain

`_story_answer()` updated to:

```
folk_tales (0.1s, 2300 words)
    → symbolic_scaffold (0.01s, 200-400 words per scene)
        → LLM_pipeline (deprecated, 35-93s)
            → format_story_answer (template fallback)
```

## 9. Edge Cases

| Case | Behavior |
|------|----------|
| No entities match scene slot | Use generic placeholder phrase ("a strange place", "someone") |
| Fewer scenes than needed (topics narrow pool) | Repeat most versatile archetype with different entities |
| All entities exhausted | Reuse entities with different property rolls |
| Topic set empty | All-entities pool, pure seeded RNG |
| Entity store empty | Seed minimal entity set on first call (10-15 defaults) |
| NLG rendering fails for an atom | Skip atom, log warning, continue |

## 10. Non-Goals

- Full prose generation (that's the NLG layer's job)
- Story plot novelty (scene templates constrain structure — novelty comes from entity combination)
- Character development arc tracking across scenes
- Discourse coherence beyond scene-level (each scene is independent UOL atoms)

## 11. Anti-Regression Checklist (per architecture.md §17)

- [ ] All new semantic classes documented in `semantic_classes.v1.json`
- [ ] scene template contract registered in `registry.v1.json`
- [ ] Validator + loader for scene template contract in `validation.py`
- [ ] Entity seeding function (called on first engine use)
- [ ] No new intents in keyword pipeline (uses existing `story` intent)
- [ ] Tests use reproducible PRNG seeds
- [ ] Engine is pure Python, zero ML deps
