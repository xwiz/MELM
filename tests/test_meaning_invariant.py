"""Cross-layer semantic class spine invariant test.

Every class ID referenced anywhere in the system must be defined in
semantic_classes.v1.json (the spine). This prevents meaning fragmentation
between UOL, frame templates, entity store, and contracts.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from melm.contracts import CONTRACT_ROOT, load_contract_json


def _load_json(name: str) -> dict[str, Any]:
    path = CONTRACT_ROOT / name
    return json.loads(path.read_text(encoding="utf-8"))


def _spine_class_ids() -> set[str]:
    """Return set of all class IDs defined in semantic_classes.v1.json."""
    payload = _load_json("semantic_classes.v1.json")
    return {c["class_id"] for c in payload["classes"]}


def _frame_template_class_ids() -> set[str]:
    """All semantic class IDs referenced by frame template activation sets.

    Only checks class-related fields. action_tokens are surface verb/noun
    forms, NOT semantic class IDs — they are excluded from the invariant.
    """
    payload = _load_json("frame_templates.v1.json")
    ids: set[str] = set()
    for tmpl in payload.get("templates", {}).values():
        act = tmpl.get("activation", {})
        for key in ("required_classes", "required_all_classes", "optional_classes", "exclude_classes"):
            ids.update(act.get(key, []))
    return ids


def _uol_inline_class_ids() -> set[str]:
    """Verb semantic classes from the UOL _VERBS inline dict (transitional).

    These are known to be partially outside the spine and are tracked
    separately. The permanent architecture reads verb classes from the
    lexical_senses table, which uses taxonomy class IDs.
    """
    return {
        "communication", "guidance", "state", "transfer", "acquisition",
        "creation", "generic_action", "consumption", "knowledge_transfer",
        "motion", "development", "possession_or_experience", "support",
        "cognition", "preference", "necessity", "activity", "iteration",
        "desire", "function",
    }


def _uol_inline_noun_classes() -> set[str]:
    """Noun domains from the UOL _KNOWN_NOMINAL_DOMAINS inline dict (transitional)."""
    return {
        "career", "health", "memory", "people", "person", "purpose", "thing", "work",
    }


def _contract_class_ids() -> set[str]:
    """All class IDs referenced by non-spine, non-frame contracts."""
    ids: set[str] = set()
    # wn_supersense_map.v1.json — flat dict: key=wn_supersense, value=class_id
    wn = _load_json("wn_supersense_map.v1.json")
    for v in wn.get("mappings", {}).values():
        if isinstance(v, str):
            ids.add(v)
    # verbnet_map.v1.json — flat dict: key=verbnet_class, value=class_id
    vn = _load_json("verbnet_map.v1.json")
    for v in vn.get("mappings", {}).values():
        if isinstance(v, str):
            ids.add(v)
    # router_lexicon_families.v1.json
    rlf = _load_json("router_lexicon_families.v1.json")
    for family in rlf.get("families", {}).values():
        ids.update(family.get("allowed_classes", []))
    # frame_minimal_pairs.v1.json — lexicon entries
    fmp = _load_json("frame_minimal_pairs.v1.json")
    for classes in fmp.get("lexicon", {}).values():
        ids.update(classes)
    return ids


def _store_class_schema_ids() -> set[str]:
    """Class IDs seeded by the entity store's seed_class_schemas().

    Note: 'place' maps to taxonomy 'location'; 'object' maps to taxonomy
    'physical_object'. These are intentionally different names because the
    entity store uses entity-kind discriminators, not semantic class IDs.
    'competition' and 'personal_experience' are entity-kind-only and have
    no semantic class equivalents.
    """
    return {
        "entity", "person", "event", "place", "object",
        "competition", "personal_experience",
    }


_ENTITY_KIND_ONLY = frozenset({"competition", "personal_experience"})
"""Entity kinds that are entity-store-only discriminators, not semantic classes."""


_STORE_TO_TAXONOMY_MISALIGNMENT: dict[str, str] = {
    "place": "location",
    "object": "physical_object",
}
"""Known naming misalignment between entity store kind names and taxonomy class IDs."""


def _uol_transitional_exceptions() -> set[str]:
    """Known UOL inline class IDs that are NOT in the spine yet.

    These are transitional — they will be removed when UOL reads verb
    classes from the lexical_senses table instead of the inline _VERBS dict.
    """
    return _uol_inline_class_ids() | _uol_inline_noun_classes()


class SemanticSpineInvariantTests(unittest.TestCase):
    """Every class ID referenced by any layer must exist in semantic_classes.v1.json."""

    def setUp(self) -> None:
        self.spine = _spine_class_ids()

    def test_spine_is_not_empty(self) -> None:
        self.assertGreater(len(self.spine), 70)

    def test_frame_template_class_ids_all_in_spine(self) -> None:
        frame_ids = _frame_template_class_ids()
        missing = frame_ids - self.spine
        self.assertEqual(
            missing, set(),
            f"Frame templates reference class IDs not in spine: {missing}",
        )

    def test_contract_class_ids_all_in_spine(self) -> None:
        contract_ids = _contract_class_ids()
        missing = contract_ids - self.spine
        self.assertEqual(
            missing, set(),
            f"Contracts reference class IDs not in spine: {missing}",
        )

    def test_uol_inline_verb_classes_not_in_spine_documented(self) -> None:
        """Transitional: document which UOL inline verb classes are outside the spine.

        The inline dict will be removed entirely when UOL reads verbs from the
        lexical_senses table. At that point, the verb classes used will be the
        taxonomy classes (verb.communicate, verb.perceive, etc.), not the UOL
        private classes (communication, guidance, etc.).
        """
        inline = _uol_inline_class_ids()
        outside = inline - self.spine
        in_spine = inline & self.spine
        self.assertGreater(
            len(outside), 0,
            f"All UOL inline verb classes in spine ({in_spine}) — inline dict can be removed!",
        )

    def test_uol_inline_noun_classes_not_in_spine_documented(self) -> None:
        """Transitional: document which UOL inline noun domains are outside the spine."""
        inline = _uol_inline_noun_classes()
        outside = inline - self.spine
        self.assertGreater(
            len(outside), 0,
            "All UOL inline noun classes in spine — inline dict can be removed!",
        )

    def test_uol_inline_and_spine_overlap_documented(self) -> None:
        """Document known overlap: 3 UOL classes share names with taxonomy classes.

        These must be reconciled — either the UOL class means the same thing
        as the taxonomy class (and should be removed from the inline dict), or
        it means something different (and should be renamed).
        """
        spine = self.spine
        verb = _uol_inline_class_ids()
        noun = _uol_inline_noun_classes()
        all_inline = verb | noun
        overlap = all_inline & spine
        # Accept current: 3 overlapping classes (state, cognition, person).
        self.assertIn("state", overlap, "'state' is in both UOL and spine")
        self.assertIn("cognition", overlap, "'cognition' is in both UOL and spine")
        self.assertIn("person", overlap, "'person' is in both UOL and spine")
        self.assertEqual(
            len(overlap), 3,
            f"Expected exactly 3 overlapping UOL/spine classes, got {len(overlap)}: {overlap}",
        )

    def test_store_class_ids_that_overlap_are_all_in_spine(self) -> None:
        """Entity store class schemas that share names with taxonomy must exist."""
        store_ids = _store_class_schema_ids()
        excluded = _ENTITY_KIND_ONLY | set(_STORE_TO_TAXONOMY_MISALIGNMENT.keys())
        checkable = store_ids - excluded
        missing = checkable - self.spine
        self.assertEqual(
            missing, set(),
            f"Store class IDs not in spine: {missing}. "
            f"(Excluded kinds: {sorted(excluded)})",
        )

    def test_store_entity_kind_misalignments_documented(self) -> None:
        """Document the known mapping between entity kinds and taxonomy classes."""
        for kind, expected_class in _STORE_TO_TAXONOMY_MISALIGNMENT.items():
            self.assertIn(
                expected_class, self.spine,
                f"Entity kind '{kind}' maps to taxonomy class '{expected_class}' "
                f"which does not exist in the spine.",
            )

    def test_personal_experience_and_competition_are_entity_kinds_only(self) -> None:
        """These entity kinds are runtime discriminators, not semantic classes."""
        for kind in sorted(_ENTITY_KIND_ONLY):
            self.assertNotIn(
                kind, self.spine,
                f"'{kind}' is an entity-kind-only discriminator and should not "
                f"be in the semantic class taxonomy.",
            )

class CrossLayerConversationMeaningTests(unittest.TestCase):
    """Simulate realistic conversations and verify meaning representation holds."""

    def _run_through_pipeline(self, tokens: tuple[str, ...]) -> dict[str, Any]:
        """Run utterance through UOL parse + frame linker + reranker.

        Returns captured meaning across layers.
        """
        from melm.appliance.functional_grammar import parse_functional_relations
        from melm.appliance.assistant_frame_linker import FrameLinker
        from melm.appliance.assistant_frame_ranker import E3CandidateReranker

        # T1: Utterance meaning from UOL
        parse = parse_functional_relations(tokens, question_mark="?" in " ".join(tokens))

        # T1: Frame linker candidate scores
        linker = FrameLinker()
        lexicon = self._test_lexicon()
        candidates = linker.score(tokens, lexicon)

        # T1: Reranker with UOL token roles
        reranker = E3CandidateReranker()
        reranked = reranker.rerank(
            candidates, tokens, lexicon,
            token_roles=parse.token_roles if parse else None,
        )

        return {
            "parse": parse,
            "candidates": candidates,
            "reranked": reranked,
            "tokens": tokens,
        }

    def _test_lexicon(self) -> dict[str, frozenset[str]]:
        return {
            "tell": frozenset({"narrative_content", "verb.communicate"}),
            "story": frozenset({"narrative_content"}),
            "about": frozenset({"relation"}),
            "rain": frozenset({"weather_phenomenon"}),
            "today": frozenset({"temporal_descriptor"}),
            "play": frozenset({"activity", "verb.perform_media"}),
            "music": frozenset({"media_content"}),
            "feel": frozenset({"verb.emotion"}),
            "sick": frozenset({"health_condition"}),
            "help": frozenset({"advice_action", "verb.cognition"}),
            "call": frozenset({"verb.communicate", "contact_action"}),
            "mom": frozenset({"social_relation", "person"}),
            "eat": frozenset({"verb.consume", "food_item"}),
            "pasta": frozenset({"food_item"}),
            "weather": frozenset({"weather_phenomenon"}),
            "how": frozenset({"verb.cognition", "temporal_descriptor"}),
            "are": frozenset({"verb.stative"}),
            "you": frozenset({"person"}),
            "good": frozenset({"evaluative_expression"}),
            "memory": frozenset({"memory_recall"}),
            "cook": frozenset({"verb.create"}),
            "want": frozenset({"verb.cognition"}),
            "I": frozenset({"person"}),
            "my": frozenset({"possession"}),
        }

    def test_tell_story_about_rain(self) -> None:
        """'tell me a story about rain' — story frame should win."""
        result = self._run_through_pipeline(("tell", "me", "a", "story", "about", "rain"))
        self.assertIsNotNone(result["parse"], "UOL parse should succeed")
        self.assertGreater(len(result["reranked"]), 0, "Reranker should produce candidates")
        top = result["reranked"][0]
        self.assertEqual(top.intent, "story",
                         f"Expected story intent, got {top.intent}")

    def test_play_some_music(self) -> None:
        """'play some music for me' — media_playback frame should win."""
        result = self._run_through_pipeline(("play", "some", "music", "for", "me"))
        self.assertIsNotNone(result["parse"])
        self.assertGreater(len(result["reranked"]), 0)
        top = result["reranked"][0]
        self.assertEqual(top.intent, "media_playback",
                         f"Expected media_playback, got {top.intent}")

    def test_i_feel_sick(self) -> None:
        """'I feel sick' — health_advice should win (even if UOL parse is None)."""
        result = self._run_through_pipeline(("I", "feel", "sick"))
        # UOL may not parse single-token utterances; fallback to linker + reranker.
        if result["parse"] is None:
            self.assertGreater(len(result["reranked"]), 0,
                               "Reranker should produce candidates even without UOL parse")
        top = result["reranked"][0]
        self.assertEqual(top.intent, "health_advice",
                         f"Expected health_advice, got {top.intent}")

    def test_i_want_to_eat_pasta(self) -> None:
        """'I want to eat pasta' — meal_suggestion should win."""
        result = self._run_through_pipeline(("I", "want", "to", "eat", "pasta"))
        self.assertIsNotNone(result["parse"])
        self.assertGreater(len(result["reranked"]), 0)
        top = result["reranked"][0]
        self.assertEqual(top.intent, "meal_suggestion",
                         f"Expected meal_suggestion, got {top.intent}")

    def test_call_mom(self) -> None:
        """'call mom' — social_contact should win."""
        result = self._run_through_pipeline(("call", "mom"))
        self.assertIsNotNone(result["parse"])
        self.assertGreater(len(result["reranked"]), 0)
        top = result["reranked"][0]
        self.assertEqual(top.intent, "social_contact",
                         f"Expected social_contact, got {top.intent}")

    def test_how_is_the_weather_today(self) -> None:
        """'how is the weather today' — weather should win."""
        result = self._run_through_pipeline(("how", "is", "the", "weather", "today"))
        self.assertIsNotNone(result["parse"])
        self.assertGreater(len(result["reranked"]), 0)
        top = result["reranked"][0]
        self.assertEqual(top.intent, "weather",
                         f"Expected weather, got {top.intent}")

    def test_uol_token_roles_use_taxonomy_class_ids(self) -> None:
        """UOL token_roles meaning field uses class IDs compatible with taxonomy.

        This test verifies that the meaning annotations in UOL output can be
        mapped to taxonomy class IDs — the bridge that allows T1 (UOL) to
        feed T2 (entity store personal_experience).
        """
        from melm.appliance.functional_grammar import parse_functional_relations
        parse = parse_functional_relations(("tell", "me", "a", "story"))
        self.assertIsNotNone(parse)
        # Verify meaning field exists on main_predicate
        roles = parse.token_roles if parse else ()
        pred_roles = [r for r in roles if r.get("role") == "main_predicate"]
        self.assertGreater(len(pred_roles), 0, "UOL should find main_predicate")
        meaning = pred_roles[0].get("meaning", "")
        self.assertIn(":", meaning,
                      f"main_predicate meaning should be 'canonical:class', got '{meaning}'")

    def test_uol_parse_and_reranker_agree_on_meaningful_sentence(self) -> None:
        """For a meaningful utterance, UOL parse, frame linker, and reranker
        must all agree on the intent (T1 consistency across layers)."""
        result = self._run_through_pipeline(("tell", "me", "a", "story", "about", "rain"))
        parse = result["parse"]
        self.assertIsNotNone(parse)
        # UOL speech act
        self.assertEqual(parse.speech_act, "request",
                         f"Expected 'request' speech act, got '{parse.speech_act}'")
        # UOL action
        self.assertEqual(parse.action, "tell",
                         f"Expected action 'tell', got '{parse.action}'")
        # Reranker agrees
        top = result["reranked"][0]
        self.assertEqual(top.intent, "story",
                         f"Reranker produced '{top.intent}', not 'story'")


class MinimalPairTokenRolesTests(unittest.TestCase):
    """Every minimal pair's token_roles must match the UOL parser's output.

    The contract stores expected UOL token role annotations for each pair.
    If the UOL parser changes, this test fails — signaling the contract
    must be updated.
    """

    def setUp(self) -> None:
        from melm.appliance.functional_grammar import parse_functional_relations
        self._parse = parse_functional_relations
        self.pairs = _load_json("frame_minimal_pairs.v1.json")["pairs"]
        self.parsed: dict[str, dict[str, Any]] = {}
        for p in self.pairs:
            tokens = tuple(p["tokens"])
            parse = self._parse(tokens, question_mark="?" in p.get("utterance", ""))
            self.parsed[p["id"]] = {
                "parse": parse,
                "roles": list(parse.token_roles) if parse else [],
            }

    def test_all_pairs_have_token_roles(self) -> None:
        missing = [p["id"] for p in self.pairs if "token_roles" not in p]
        self.assertEqual(
            missing, [],
            f"Pairs missing token_roles field: {missing}",
        )

    def test_token_roles_match_uol_parse(self) -> None:
        mismatches: list[str] = []
        for p in self.pairs:
            pid = p["id"]
            expected = p.get("token_roles", [])
            actual = self.parsed[pid]["roles"]
            # Normalize: only compare index, token, lemma, role (weights change with implementation)
            expected_slim = [
                {k: r[k] for k in ("index", "token", "lemma", "role") if k in r}
                for r in expected
            ]
            actual_slim = [
                {k: r[k] for k in ("index", "token", "lemma", "role") if k in r}
                for r in actual
            ]
            # Both empty = parse returned None, which is fine
            if not expected_slim and not actual_slim:
                continue
            if expected_slim != actual_slim:
                mismatches.append(
                    f"{pid} ({p.get('utterance', '')}):\n"
                    f"  expected: {expected_slim}\n"
                    f"  actual:   {actual_slim}"
                )
        if mismatches:
            self.fail(
                f"{len(mismatches)} pair(s) have token_roles that don't match UOL parser:\n"
                + "\n".join(mismatches)
            )

    def test_precision_target_pairs_have_competing_frames(self) -> None:
        """Precision-target pairs should have at least one competing frame
        in top-3, otherwise precision is trivially 1.0."""
        from melm.appliance.assistant_frame_linker import FrameLinker

        linker = FrameLinker()
        lexicon = _pair_set_lexicon()

        for p in self.pairs:
            if not p.get("precision_target", False):
                continue
            tokens = tuple(p["tokens"])
            candidates = linker.score(
                tokens, lexicon,
                is_question_like=p["is_question_like"],
                is_request_like=p["is_request_like"],
            )
            top3 = [c.frame_id for c in candidates[:3]]
            expected = p["expected_top_frame"]
            if expected is None:
                continue
            competitors = [f for f in top3 if f != expected]
            self.assertGreater(
                len(competitors), 0,
                f"{p['id']}: precision_target pair '{p['utterance']}' has "
                f"no competing frames in top-3 ({top3}) — precision is trivial",
            )


def _pair_set_lexicon() -> dict[str, frozenset[str]]:
    """Build lexicon from the frame_minimal_pairs contract for testing."""
    payload = _load_json("frame_minimal_pairs.v1.json")
    return {
        token: frozenset(classes)
        for token, classes in payload.get("lexicon", {}).items()
    }


if __name__ == "__main__":
    unittest.main()
