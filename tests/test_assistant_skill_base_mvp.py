"""Tests for the formal Skill protocol and registry."""

import unittest

from melm.appliance.assistant_skill_base import (
    SkillManifest,
    SkillRegistry,
    get_skill_registry,
    register_skill,
)


class SkillManifestMvpTests(unittest.TestCase):
    """SkillManifest dataclass — metadata container for skills."""

    def test_requires_family(self) -> None:
        m = SkillManifest(family="test_skill")
        self.assertEqual(m.family, "test_skill")

    def test_default_frames_empty(self) -> None:
        m = SkillManifest(family="test_skill")
        self.assertEqual(m.frames, ())

    def test_default_knowledge_refs_empty(self) -> None:
        m = SkillManifest(family="test_skill")
        self.assertEqual(m.knowledge_refs, ())

    def test_has_knowledge_returns_true_for_registered(self) -> None:
        m = SkillManifest(family="test", knowledge_refs=("a.v1.json", "b.v1.json"))
        self.assertTrue(m.has_knowledge("a.v1.json"))
        self.assertTrue(m.has_knowledge("b.v1.json"))

    def test_has_knowledge_returns_false_for_unregistered(self) -> None:
        m = SkillManifest(family="test", knowledge_refs=("a.v1.json",))
        self.assertFalse(m.has_knowledge("missing.v1.json"))

    def test_has_frame_returns_true_for_registered(self) -> None:
        m = SkillManifest(family="test", frames=("frame_a", "frame_b"))
        self.assertTrue(m.has_frame("frame_a"))
        self.assertTrue(m.has_frame("frame_b"))

    def test_has_frame_returns_false_for_unregistered(self) -> None:
        m = SkillManifest(family="test", frames=("frame_a",))
        self.assertFalse(m.has_frame("frame_b"))


class SkillRegistryMvpTests(unittest.TestCase):
    """SkillRegistry — registration and lookup."""

    def test_empty_registry_has_zero_skills(self) -> None:
        r = SkillRegistry()
        self.assertEqual(len(r), 0)

    def test_register_adds_skill(self) -> None:
        r = SkillRegistry()
        m = SkillManifest(family="test", frames=("frame_a",))
        r.register(m)
        self.assertEqual(len(r), 1)
        self.assertIn("test", r)

    def test_by_family_returns_manifest(self) -> None:
        r = SkillRegistry()
        m = SkillManifest(family="test", knowledge_refs=("data.v1.json",))
        r.register(m)
        self.assertIs(r.by_family("test"), m)

    def test_by_family_returns_none_for_unknown(self) -> None:
        r = SkillRegistry()
        self.assertIsNone(r.by_family("nonexistent"))

    def test_by_frame_returns_manifest(self) -> None:
        r = SkillRegistry()
        m = SkillManifest(family="test", frames=("frame_a", "frame_b"))
        r.register(m)
        self.assertIs(r.by_frame("frame_a"), m)
        self.assertIs(r.by_frame("frame_b"), m)

    def test_by_frame_returns_none_for_unregistered_frame(self) -> None:
        r = SkillRegistry()
        m = SkillManifest(family="test", frames=("frame_a",))
        r.register(m)
        self.assertIsNone(r.by_frame("frame_b"))

    def test_families_property_returns_sorted(self) -> None:
        r = SkillRegistry()
        r.register(SkillManifest(family="z_skill"))
        r.register(SkillManifest(family="a_skill"))
        self.assertEqual(r.families, ("a_skill", "z_skill"))

    def test_frames_property_returns_all_frames(self) -> None:
        r = SkillRegistry()
        r.register(SkillManifest(family="a", frames=("x", "y")))
        r.register(SkillManifest(family="b", frames=("z",)))
        self.assertEqual(r.frames, ("x", "y", "z"))


class SkillRegistrationIntegrationTests(unittest.TestCase):
    """The 3 production skill modules register on import."""

    def test_meal_skill_registers(self) -> None:
        from melm.appliance import assistant_skill_meal  # noqa: F401

        m = get_skill_registry().by_family("meal_suggestion")
        self.assertIsNotNone(m, "meal_suggestion should be registered")
        assert m is not None
        self.assertEqual(m.family, "meal_suggestion")
        self.assertTrue(m.has_knowledge("food_tags.v1.json"))
        self.assertTrue(m.has_knowledge("meal_scopes.v1.json"))
        self.assertTrue(m.has_frame("meal_suggestion"))

    def test_story_skill_registers(self) -> None:
        from melm.appliance import assistant_skill_story  # noqa: F401

        m = get_skill_registry().by_family("story")
        self.assertIsNotNone(m, "story should be registered")
        assert m is not None
        self.assertEqual(m.family, "story")
        self.assertTrue(m.has_knowledge("story_components.v1.json"))
        self.assertTrue(m.has_frame("story"))

    def test_memory_skill_registers(self) -> None:
        from melm.appliance import assistant_skill_memory  # noqa: F401

        m = get_skill_registry().by_family("memory")
        self.assertIsNotNone(m, "memory should be registered")
        assert m is not None
        self.assertEqual(m.family, "memory")
        self.assertTrue(m.has_knowledge("memory_insights.v1.json"))
        self.assertTrue(m.has_frame("personal_memory"))
        self.assertTrue(m.has_frame("autobiographical_memory"))

    def test_all_skills_have_valid_manifests(self) -> None:
        from melm.appliance import (
            assistant_skill_meal,
            assistant_skill_memory,
            assistant_skill_story,
        )

        for mod in (assistant_skill_meal, assistant_skill_memory, assistant_skill_story):
            m = getattr(mod, "MANIFEST", None)
            self.assertIsInstance(m, SkillManifest, f"{mod.__name__}.MANIFEST missing")
            assert isinstance(m, SkillManifest)
            self.assertTrue(m.family, f"{mod.__name__}.MANIFEST.family empty")
            self.assertTrue(m.knowledge_refs, f"{mod.__name__}.MANIFEST has no knowledge_refs")


if __name__ == "__main__":
    unittest.main()
