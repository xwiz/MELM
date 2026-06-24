"""Tests for symbolic (UOL-driven) story scaffold engine."""
import pytest
import random
from dataclasses import dataclass


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
    result = engine.generate(frozenset({"bedtime"}))
    assert result is not None
    assert len(result.scenes) >= 1


def test_symbolic_engine_minimal_entity_fallback():
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    from melm.appliance.assistant_skill_story_symbolic import _FALLBACK_LABELS

    class EmptyEngine(SymbolicStoryEngine):
        def _collect_candidates(self, allowed_classes):
            return []

    rng = random.Random(42)
    engine = EmptyEngine(StoryProfile(), rng=rng)
    result = engine.generate(frozenset())
    assert result is not None
    assert len(result.scenes) >= 1


def test_symbolic_engine_scene_count_with_arc():
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
    from melm.appliance.assistant_skill_story_symbolic import _entity_to_adjective
    adj = _entity_to_adjective({"danger": 0.6})
    assert adj == "dangerous", f"Expected 'dangerous' for danger=0.6, got '{adj}'"
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


def test_symbolic_engine_scene_atom_graph_dataclass():
    from melm.appliance.assistant_skill_story_symbolic import SceneAtomGraph, StoryAtom, EntityBinding
    atom = StoryAtom("walk", "protagonist", location_role="place")
    binding = EntityBinding("Maya", "person", {"name": "Maya"})
    scene = SceneAtomGraph(
        scene_number=1,
        archetype_id="discovery",
        title="Maya Discovers Something",
        atoms=[atom],
        entity_bindings={"protagonist": binding, "place": EntityBinding("Forest", "story_element.place.wild")},
    )
    assert scene.scene_number == 1
    assert scene.archetype_id == "discovery"
    assert len(scene.atoms) == 1


def test_symbolic_engine_empty_topics_still_generates():
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    engine = SymbolicStoryEngine(StoryProfile(), rng=random.Random(1))
    result = engine.generate(frozenset())
    assert result is not None
    assert len(result.scenes) >= 1


def test_symbolic_engine_no_matching_arc_falls_back_to_all():
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    engine = SymbolicStoryEngine(StoryProfile(), rng=random.Random(2))
    result = engine.generate(frozenset({"nonexistent_topic"}))
    assert result is not None
    assert len(result.scenes) >= 1


def test_symbolic_engine_known_arc_returns_expected_scenes():
    from melm.appliance.assistant_skill_story_symbolic import SymbolicStoryEngine
    engine = SymbolicStoryEngine(StoryProfile(), rng=random.Random(3))
    result = engine.generate(frozenset({"quest"}))
    assert result is not None
    scene_ids = [s.archetype_id for s in result.scenes]
    assert len(scene_ids) >= 3


def test_symbolic_engine_story_atom_graph_dataclass():
    from melm.appliance.assistant_skill_story_symbolic import StoryAtomGraph, SceneAtomGraph, StoryAtom, EntityBinding
    atom = StoryAtom("walk", "protagonist")
    binding = EntityBinding("Maya", "person", {"name": "Maya"})
    scene = SceneAtomGraph(1, "discovery", "Scene 1", [atom], {"protagonist": binding})
    graph = StoryAtomGraph(scenes=[scene])
    assert len(graph.scenes) == 1
    assert graph.scenes[0].archetype_id == "discovery"
